from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from simulator.config import SimulatorConfig, load_config


class FeatureError(ValueError):
    """Raised when telemetry cannot be converted into detector features."""


@dataclass(frozen=True)
class FeatureBatch:
    """A validated feature matrix with the source rows used to create it."""

    matrix: np.ndarray
    source: pd.DataFrame
    feature_names: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return int(self.matrix.shape[0])


class FeaturePipeline:
    """Load, validate, and sample GPU telemetry for anomaly detection."""

    REQUIRED_CONTEXT_COLUMNS = (
        "timestamp",
        "node_id",
        "gpu_id",
        "operating_state",
        "anomaly_label",
        "incident_id",
        "xid_error_code",
    )

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or load_config()
        ordered = self.config.detection["features"]["ordered"]
        self.feature_names = tuple(str(name) for name in ordered)
        self.required_columns = tuple(
            dict.fromkeys(self.REQUIRED_CONTEXT_COLUMNS + self.feature_names)
        )
        self.training_config = self.config.detection["training"]
        self.preprocessing_config = self.config.detection["preprocessing"]

    def telemetry_path(self) -> Path:
        return self.config.output_path("gpu_telemetry")

    def validate_source_file(self, path: Path | None = None) -> Path:
        source = (path or self.telemetry_path()).resolve()
        if not source.exists():
            raise FeatureError(f"GPU telemetry file does not exist: {source}")
        if not source.is_file():
            raise FeatureError(f"GPU telemetry path is not a file: {source}")
        if source.stat().st_size == 0:
            raise FeatureError(f"GPU telemetry file is empty: {source}")
        return source

    def iter_csv_chunks(
        self,
        path: Path | None = None,
        *,
        chunksize: int = 10000,
    ) -> Iterator[pd.DataFrame]:
        if isinstance(chunksize, bool) or not isinstance(chunksize, int):
            raise FeatureError("chunksize must be a positive integer")
        if chunksize <= 0:
            raise FeatureError("chunksize must be a positive integer")

        source = self.validate_source_file(path)
        try:
            reader = pd.read_csv(
                source,
                usecols=list(self.required_columns),
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk in reader:
                yield self.validate_frame(chunk)
        except ValueError as exc:
            raise FeatureError(
                f"GPU telemetry schema is incompatible with the detector: {exc}"
            ) from exc
        except (OSError, pd.errors.ParserError) as exc:
            raise FeatureError(f"Unable to read GPU telemetry: {source}") from exc

    def validate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise FeatureError("Telemetry input must be a pandas DataFrame")
        if frame.empty:
            raise FeatureError("Telemetry input cannot be empty")

        missing = sorted(set(self.required_columns).difference(frame.columns))
        if missing:
            raise FeatureError(
                "Telemetry is missing required columns: " + ", ".join(missing)
            )

        validated = frame.loc[:, self.required_columns].copy()
        for column in self.feature_names + ("xid_error_code",):
            validated[column] = pd.to_numeric(validated[column], errors="coerce")

        invalid_numeric_strategy = str(
            self.preprocessing_config["invalid_numeric_strategy"]
        )
        numeric_columns = list(self.feature_names) + ["xid_error_code"]
        invalid_mask = validated[numeric_columns].isna().any(axis=1)

        if invalid_mask.any() and invalid_numeric_strategy == "reject":
            invalid_count = int(invalid_mask.sum())
            raise FeatureError(
                f"Telemetry contains {invalid_count} rows with invalid numeric values"
            )

        values = validated[numeric_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise FeatureError("Telemetry contains non-finite numeric values")

        if bool(self.preprocessing_config["clamp_percentages"]):
            for column in (
                "gpu_utilization_percent",
                "memory_utilization_percent",
            ):
                validated[column] = validated[column].clip(0.0, 100.0)

        if bool(self.preprocessing_config["prevent_negative_values"]):
            negative_columns = [
                column
                for column in numeric_columns
                if column not in {"xid_error_code"}
            ]
            if (validated[negative_columns] < 0).any().any():
                raise FeatureError("Telemetry contains negative numeric values")

        return validated

    def to_feature_batch(self, frame: pd.DataFrame) -> FeatureBatch:
        validated = self.validate_frame(frame)
        matrix = validated.loc[:, self.feature_names].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise FeatureError("Feature matrix has an unexpected shape")

        return FeatureBatch(
            matrix=matrix,
            source=validated,
            feature_names=self.feature_names,
        )

    def load_training_frame(
        self,
        path: Path | None = None,
        *,
        chunksize: int = 10000,
    ) -> pd.DataFrame:
        target_state = str(self.training_config["source_operating_state"])
        maximum_rows = int(self.training_config["maximum_training_rows"])
        minimum_rows = int(self.training_config["minimum_training_rows"])
        random_seed = int(self.training_config["random_seed"])

        candidates: list[pd.DataFrame] = []
        for chunk in self.iter_csv_chunks(path, chunksize=chunksize):
            healthy = chunk[
                (chunk["operating_state"] == target_state)
                & (chunk["anomaly_label"] == "normal")
                & (chunk["incident_id"].fillna("") == "")
                & (chunk["xid_error_code"] == 0)
                & (chunk["ecc_uncorrectable_errors"] == 0)
            ]
            if not healthy.empty:
                candidates.append(healthy)

        if not candidates:
            raise FeatureError("No healthy training records were found")

        training_frame = pd.concat(candidates, ignore_index=True)
        if len(training_frame) < minimum_rows:
            raise FeatureError(
                f"Only {len(training_frame)} healthy rows are available; "
                f"at least {minimum_rows} are required"
            )

        if len(training_frame) > maximum_rows:
            training_frame = training_frame.sample(
                n=maximum_rows,
                replace=False,
                random_state=random_seed,
            )

        return training_frame.sort_values(
            by=["timestamp", "node_id", "gpu_id"],
            kind="stable",
        ).reset_index(drop=True)

    def iter_evaluation_batches(
        self,
        path: Path | None = None,
        *,
        chunksize: int = 10000,
    ) -> Iterator[FeatureBatch]:
        for chunk in self.iter_csv_chunks(path, chunksize=chunksize):
            yield self.to_feature_batch(chunk)

    def feature_order_matches(self, names: Sequence[str]) -> bool:
        return tuple(names) == self.feature_names
