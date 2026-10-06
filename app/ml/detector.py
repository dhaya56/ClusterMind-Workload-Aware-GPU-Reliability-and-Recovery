from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.ml.features import FeatureBatch, FeatureError, FeaturePipeline
from simulator.config import SimulatorConfig, load_config


class DetectorError(RuntimeError):
    """Raised when the anomaly detector cannot be trained, saved, or used."""


@dataclass(frozen=True)
class DetectionBatch:
    """Detector outputs aligned with the source telemetry rows."""

    results: pd.DataFrame

    @property
    def row_count(self) -> int:
        return len(self.results)


class IsolationForestDetector:
    """Configuration-driven Isolation Forest detector for GPU telemetry."""

    RESULT_COLUMNS = (
        "timestamp",
        "node_id",
        "gpu_id",
        "anomaly_score",
        "model_is_anomaly",
        "severity",
        "override_reason",
        "predicted_anomaly",
        "ground_truth_label",
        "incident_id",
    )

    def __init__(
        self,
        config: SimulatorConfig | None = None,
        feature_pipeline: FeaturePipeline | None = None,
    ) -> None:
        self.config = config or load_config()
        self.features = feature_pipeline or FeaturePipeline(self.config)
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None
        self.score_low: float | None = None
        self.score_high: float | None = None
        self.training_rows: int = 0
        self.trained_at_utc: str | None = None

    @property
    def is_fitted(self) -> bool:
        return (
            self.model is not None
            and self.scaler is not None
            and self.score_low is not None
            and self.score_high is not None
        )

    def fit(self, training_frame: pd.DataFrame) -> None:
        batch = self.features.to_feature_batch(training_frame)
        training = self.config.detection["training"]

        minimum_rows = int(training["minimum_training_rows"])
        if batch.row_count < minimum_rows:
            raise DetectorError(
                f"At least {minimum_rows} training rows are required; "
                f"received {batch.row_count}"
            )

        scaler = StandardScaler()
        scaled = scaler.fit_transform(batch.matrix)

        model = IsolationForest(
            n_estimators=int(training["n_estimators"]),
            max_samples=training["max_samples"],
            contamination=float(training["contamination"]),
            max_features=float(training["max_features"]),
            bootstrap=bool(training["bootstrap"]),
            n_jobs=int(training["n_jobs"]),
            random_state=int(training["random_seed"]),
        )
        model.fit(scaled)

        raw_anomaly = -model.decision_function(scaled)
        score_low = float(np.quantile(raw_anomaly, 0.01))
        score_high = float(np.quantile(raw_anomaly, 0.99))

        if not np.isfinite(score_low) or not np.isfinite(score_high):
            raise DetectorError("Training produced non-finite score calibration")
        if score_high <= score_low:
            raise DetectorError("Training produced a degenerate score range")

        self.scaler = scaler
        self.model = model
        self.score_low = score_low
        self.score_high = score_high
        self.training_rows = batch.row_count
        self.trained_at_utc = datetime.now(timezone.utc).replace(
            microsecond=0
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

    def fit_from_csv(
        self,
        path: Path | None = None,
        *,
        chunksize: int = 10000,
    ) -> None:
        training_frame = self.features.load_training_frame(
            path,
            chunksize=chunksize,
        )
        self.fit(training_frame)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise DetectorError("Detector is not fitted")

    def _normalized_scores(self, batch: FeatureBatch) -> np.ndarray:
        self._require_fitted()
        assert self.scaler is not None
        assert self.model is not None
        assert self.score_low is not None
        assert self.score_high is not None

        scaled = self.scaler.transform(batch.matrix)
        raw_anomaly = -self.model.decision_function(scaled)
        normalized = (raw_anomaly - self.score_low) / (
            self.score_high - self.score_low
        )
        return np.clip(normalized, 0.0, 1.0)

    def _severity_for_score(self, score: float) -> str:
        thresholds = self.config.detection["scoring"]["severity_thresholds"]
        if score >= float(thresholds["critical"]):
            return "critical"
        if score >= float(thresholds["high"]):
            return "high"
        if score >= float(thresholds["medium"]):
            return "medium"
        if score >= float(thresholds["low"]):
            return "low"
        return "normal"

    @staticmethod
    def _severity_rank(severity: str) -> int:
        ranks = {
            "normal": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
        }
        return ranks[severity]

    def _apply_override(
        self,
        row: pd.Series,
        severity: str,
    ) -> tuple[str, str]:
        reasons: list[str] = []
        minimum = severity

        if int(row["ecc_uncorrectable_errors"]) > 0:
            reasons.append("uncorrectable_ecc_error")
            minimum = self._max_severity(minimum, "critical")

        if int(row["xid_error_code"]) != 0:
            reasons.append("xid_error")
            minimum = self._max_severity(minimum, "high")

        if float(row["gpu_temperature_c"]) >= 90.0:
            reasons.append("extreme_temperature")
            minimum = self._max_severity(minimum, "critical")

        if float(row["memory_temperature_c"]) >= 95.0:
            reasons.append("severe_memory_temperature")
            minimum = self._max_severity(minimum, "critical")

        return minimum, ",".join(reasons)

    def _max_severity(self, first: str, second: str) -> str:
        return (
            first
            if self._severity_rank(first) >= self._severity_rank(second)
            else second
        )

    def predict_frame(self, frame: pd.DataFrame) -> DetectionBatch:
        batch = self.features.to_feature_batch(frame)
        scores = self._normalized_scores(batch)
        minimum_incident = str(
            self.config.detection["scoring"]["minimum_incident_severity"]
        )
        minimum_rank = self._severity_rank(minimum_incident)

        output_rows: list[dict[str, Any]] = []
        for position, (_, source_row) in enumerate(batch.source.iterrows()):
            score = float(scores[position])
            severity = self._severity_for_score(score)
            severity, override_reason = self._apply_override(
                source_row,
                severity,
            )
            predicted_anomaly = self._severity_rank(severity) >= minimum_rank

            output_rows.append(
                {
                    "timestamp": source_row["timestamp"],
                    "node_id": source_row["node_id"],
                    "gpu_id": source_row["gpu_id"],
                    "anomaly_score": round(score, 6),
                    "model_is_anomaly": bool(
                        self.model.predict(
                            self.scaler.transform(
                                batch.matrix[position : position + 1]
                            )
                        )[0]
                        == -1
                    ),
                    "severity": severity,
                    "override_reason": override_reason,
                    "predicted_anomaly": predicted_anomaly,
                    "ground_truth_label": source_row["anomaly_label"],
                    "incident_id": (
                        ""
                        if pd.isna(source_row["incident_id"])
                        else str(source_row["incident_id"])
                    ),
                }
            )

        result = pd.DataFrame(output_rows, columns=self.RESULT_COLUMNS)
        return DetectionBatch(results=result)

    def predict_csv(
        self,
        path: Path | None = None,
        *,
        chunksize: int = 10000,
    ) -> pd.DataFrame:
        outputs: list[pd.DataFrame] = []
        for batch in self.features.iter_evaluation_batches(
            path,
            chunksize=chunksize,
        ):
            outputs.append(self.predict_frame(batch.source).results)

        if not outputs:
            raise DetectorError("No telemetry rows were available for prediction")
        return pd.concat(outputs, ignore_index=True)

    def artifact_paths(self) -> dict[str, Path]:
        artifacts = self.config.detection["artifacts"]
        directory = (
            self.config.project_root / str(artifacts["directory"])
        ).resolve()
        return {
            "directory": directory,
            "model": directory / str(artifacts["model_file"]),
            "scaler": directory / str(artifacts["scaler_file"]),
            "metadata": directory / str(artifacts["metadata_file"]),
            "feature_order": directory / str(artifacts["feature_order_file"]),
        }

    def save(self) -> dict[str, Path]:
        self._require_fitted()
        paths = self.artifact_paths()
        directory = paths["directory"]
        directory.mkdir(parents=True, exist_ok=True)

        assert self.model is not None
        assert self.scaler is not None
        assert self.score_low is not None
        assert self.score_high is not None

        metadata = {
            "model_name": self.config.detection["detection"]["model_name"],
            "model_version": self.config.detection["detection"]["model_version"],
            "algorithm": "isolation_forest",
            "trained_at_utc": self.trained_at_utc,
            "training_rows": self.training_rows,
            "configuration_hash": self.config.configuration_hash,
            "score_calibration": {
                "low_quantile": self.score_low,
                "high_quantile": self.score_high,
            },
        }

        temporary_paths = {
            name: path.with_name(path.name + ".tmp")
            for name, path in paths.items()
            if name != "directory"
        }

        try:
            joblib.dump(self.model, temporary_paths["model"])
            joblib.dump(self.scaler, temporary_paths["scaler"])
            temporary_paths["metadata"].write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_paths["feature_order"].write_text(
                json.dumps(list(self.features.feature_names), indent=2) + "\n",
                encoding="utf-8",
            )
            for name in ("model", "scaler", "metadata", "feature_order"):
                os.replace(temporary_paths[name], paths[name])
        except OSError as exc:
            for temporary in temporary_paths.values():
                temporary.unlink(missing_ok=True)
            raise DetectorError("Unable to save detector artifacts") from exc

        return paths

    @classmethod
    def load(
        cls,
        config: SimulatorConfig | None = None,
    ) -> IsolationForestDetector:
        detector = cls(config=config)
        paths = detector.artifact_paths()

        required = ("model", "scaler", "metadata", "feature_order")
        missing = [str(paths[name]) for name in required if not paths[name].is_file()]
        if missing:
            raise DetectorError("Missing detector artifacts: " + ", ".join(missing))

        try:
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            feature_order = json.loads(
                paths["feature_order"].read_text(encoding="utf-8")
            )
            model = joblib.load(paths["model"])
            scaler = joblib.load(paths["scaler"])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise DetectorError("Unable to load detector artifacts") from exc

        if metadata.get("configuration_hash") != detector.config.configuration_hash:
            raise DetectorError("Detector configuration hash does not match")
        if not detector.features.feature_order_matches(feature_order):
            raise DetectorError("Detector feature order does not match configuration")
        if not isinstance(model, IsolationForest):
            raise DetectorError("Loaded model is not an IsolationForest")
        if not isinstance(scaler, StandardScaler):
            raise DetectorError("Loaded scaler is not a StandardScaler")

        calibration = metadata.get("score_calibration", {})
        try:
            score_low = float(calibration["low_quantile"])
            score_high = float(calibration["high_quantile"])
            training_rows = int(metadata["training_rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DetectorError("Detector metadata is incomplete") from exc

        if not np.isfinite(score_low) or not np.isfinite(score_high):
            raise DetectorError("Detector score calibration is invalid")
        if score_high <= score_low:
            raise DetectorError("Detector score calibration is degenerate")

        detector.model = model
        detector.scaler = scaler
        detector.score_low = score_low
        detector.score_high = score_high
        detector.training_rows = training_rows
        detector.trained_at_utc = metadata.get("trained_at_utc")
        return detector
