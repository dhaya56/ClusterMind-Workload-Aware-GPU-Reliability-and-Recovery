from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import Settings, get_settings
from app.services.runbooks import Runbook, RunbookCatalog, get_runbook_catalog


class IncidentRepositoryError(RuntimeError):
    """Raised when incident evidence is missing, invalid, or inconsistent."""


class IncidentRecord(BaseModel):
    """Validated ground-truth incident metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    incident_id: str = Field(min_length=1, max_length=100)
    node_id: str = Field(min_length=1, max_length=100)
    gpu_id: str = Field(min_length=1, max_length=100)
    incident_type: str = Field(min_length=1, max_length=100)
    severity: str = Field(pattern=r"^(low|medium|high|critical)$")
    start_timestamp: datetime
    end_timestamp: datetime
    root_cause: str = Field(min_length=1, max_length=300)
    affected_workload: str = Field(min_length=1, max_length=100)
    gpu_hours_lost: float = Field(ge=0.0)
    recovery_action: str = Field(min_length=1, max_length=300)
    incident_status: str = Field(min_length=1, max_length=100)

    @field_validator("start_timestamp", "end_timestamp", mode="before")
    @classmethod
    def parse_utc_timestamp(cls, value: Any) -> datetime:
        try:
            parsed = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid incident timestamp: {value!r}") from exc

        if pd.isna(parsed):
            raise ValueError("incident timestamp cannot be missing")

        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("UTC")
        else:
            parsed = parsed.tz_convert("UTC")

        return parsed.to_pydatetime()

    @model_validator(mode="after")
    def validate_time_window(self) -> "IncidentRecord":
        if self.end_timestamp < self.start_timestamp:
            raise ValueError("end_timestamp cannot precede start_timestamp")
        return self


class IncidentSummary(BaseModel):
    """Compact incident representation for API and UI listing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    incident_id: str
    incident_type: str
    severity: str
    node_id: str
    gpu_id: str
    affected_workload: str
    start_timestamp: datetime
    end_timestamp: datetime
    incident_status: str


class IncidentEvidencePackage(BaseModel):
    """Immutable correlated evidence consumed by the agent workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    incident: IncidentRecord
    runbook: Runbook
    gpu_baseline: tuple[Mapping[str, Any], ...]
    gpu_incident_window: tuple[Mapping[str, Any], ...]
    workload_baseline: tuple[Mapping[str, Any], ...]
    workload_incident_window: tuple[Mapping[str, Any], ...]
    node_incident_window: tuple[Mapping[str, Any], ...]
    detector_incident_window: tuple[Mapping[str, Any], ...]
    sampling_interval_minutes: int = Field(ge=1, le=1440)

    @model_validator(mode="after")
    def validate_evidence(self) -> "IncidentEvidencePackage":
        if not self.gpu_baseline:
            raise ValueError("GPU baseline evidence cannot be empty")
        if not self.gpu_incident_window:
            raise ValueError("GPU incident-window evidence cannot be empty")
        if not self.workload_baseline:
            raise ValueError("workload baseline evidence cannot be empty")
        if not self.workload_incident_window:
            raise ValueError("workload incident-window evidence cannot be empty")
        if not self.node_incident_window:
            raise ValueError("node incident-window evidence cannot be empty")
        if not self.detector_incident_window:
            raise ValueError("detector incident-window evidence cannot be empty")
        return self


@dataclass(frozen=True)
class _SourcePaths:
    incidents: Path
    gpu: Path
    node: Path
    workloads: Path
    detections: Path

    def all(self) -> tuple[Path, ...]:
        return (
            self.incidents,
            self.gpu,
            self.node,
            self.workloads,
            self.detections,
        )


class IncidentRepository:
    """Thread-safe, read-only incident and evidence repository over local artifacts."""

    INCIDENT_COLUMNS = frozenset(
        {
            "incident_id",
            "node_id",
            "gpu_id",
            "incident_type",
            "severity",
            "start_timestamp",
            "end_timestamp",
            "root_cause",
            "affected_workload",
            "gpu_hours_lost",
            "recovery_action",
            "incident_status",
        }
    )
    GPU_COLUMNS = frozenset(
        {
            "timestamp",
            "node_id",
            "gpu_id",
            "operating_state",
            "anomaly_label",
            "incident_id",
        }
    )
    NODE_COLUMNS = frozenset(
        {"timestamp", "node_id", "node_health_status", "incident_id"}
    )
    WORKLOAD_COLUMNS = frozenset(
        {
            "timestamp",
            "job_id",
            "job_type",
            "assigned_node",
            "workload_status",
            "checkpoint_available",
            "incident_id",
        }
    )
    DETECTION_COLUMNS = frozenset(
        {
            "timestamp",
            "node_id",
            "gpu_id",
            "anomaly_score",
            "severity",
            "predicted_anomaly",
            "ground_truth_label",
            "incident_id",
        }
    )

    def __init__(
        self,
        settings: Settings | None = None,
        runbooks: RunbookCatalog | None = None,
        *,
        baseline_samples: int = 12,
    ) -> None:
        if isinstance(baseline_samples, bool) or baseline_samples < 3:
            raise IncidentRepositoryError(
                "baseline_samples must be an integer greater than or equal to 3"
            )

        self.settings = settings or get_settings()
        self.runbooks = runbooks or get_runbook_catalog(self.settings)
        self.baseline_samples = int(baseline_samples)
        self._lock = RLock()
        self._paths = _SourcePaths(
            incidents=self.settings.data_directory / "incidents.csv",
            gpu=self.settings.data_directory / "gpu_telemetry.csv",
            node=self.settings.data_directory / "node_telemetry.csv",
            workloads=self.settings.data_directory / "workloads.csv",
            detections=self.settings.artifact_directory / "detections.csv",
        )
        self._validate_source_files()
        self._load()

    def _validate_source_files(self) -> None:
        missing = [str(path) for path in self._paths.all() if not path.is_file()]
        if missing:
            raise IncidentRepositoryError(
                "Missing incident evidence files: " + ", ".join(missing)
            )
        empty = [str(path) for path in self._paths.all() if path.stat().st_size == 0]
        if empty:
            raise IncidentRepositoryError(
                "Empty incident evidence files: " + ", ".join(empty)
            )

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, low_memory=False)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise IncidentRepositoryError(f"Unable to read CSV file: {path}") from exc

    @staticmethod
    def _validate_columns(
        frame: pd.DataFrame,
        required: frozenset[str],
        source_name: str,
    ) -> None:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise IncidentRepositoryError(
                f"{source_name} is missing required columns: {missing}"
            )

    @staticmethod
    def _parse_timestamp_column(frame: pd.DataFrame, source_name: str) -> None:
        try:
            parsed = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        except (TypeError, ValueError) as exc:
            raise IncidentRepositoryError(
                f"{source_name} contains invalid timestamps"
            ) from exc
        if parsed.isna().any():
            raise IncidentRepositoryError(
                f"{source_name} contains missing timestamps"
            )
        frame["timestamp"] = parsed

    def _load(self) -> None:
        with self._lock:
            incidents = self._read_csv(self._paths.incidents)
            gpu = self._read_csv(self._paths.gpu)
            node = self._read_csv(self._paths.node)
            workloads = self._read_csv(self._paths.workloads)
            detections = self._read_csv(self._paths.detections)

            self._validate_columns(incidents, self.INCIDENT_COLUMNS, "incidents")
            self._validate_columns(gpu, self.GPU_COLUMNS, "GPU telemetry")
            self._validate_columns(node, self.NODE_COLUMNS, "node telemetry")
            self._validate_columns(
                workloads,
                self.WORKLOAD_COLUMNS,
                "workload telemetry",
            )
            self._validate_columns(
                detections,
                self.DETECTION_COLUMNS,
                "detector results",
            )

            for frame, name in (
                (gpu, "GPU telemetry"),
                (node, "node telemetry"),
                (workloads, "workload telemetry"),
                (detections, "detector results"),
            ):
                self._parse_timestamp_column(frame, name)

            if incidents.empty:
                raise IncidentRepositoryError("incident dataset cannot be empty")
            if incidents["incident_id"].isna().any():
                raise IncidentRepositoryError("incident IDs cannot be missing")
            if incidents["incident_id"].duplicated().any():
                duplicates = sorted(
                    incidents.loc[
                        incidents["incident_id"].duplicated(keep=False),
                        "incident_id",
                    ].astype(str).unique()
                )
                raise IncidentRepositoryError(
                    f"duplicate incident IDs found: {duplicates}"
                )

            records: dict[str, IncidentRecord] = {}
            for raw in incidents.to_dict(orient="records"):
                try:
                    record = IncidentRecord.model_validate(raw)
                except Exception as exc:
                    raise IncidentRepositoryError(
                        f"invalid incident record {raw.get('incident_id')!r}: {exc}"
                    ) from exc
                if record.incident_type not in self.runbooks.scenarios():
                    raise IncidentRepositoryError(
                        f"incident {record.incident_id} has no runbook: "
                        f"{record.incident_type}"
                    )
                if not self.runbooks.supports_severity(
                    record.incident_type,
                    record.severity,
                ):
                    raise IncidentRepositoryError(
                        f"incident {record.incident_id} uses unsupported severity "
                        f"'{record.severity}' for {record.incident_type}"
                    )
                records[record.incident_id] = record

            known_ids = frozenset(records)
            for frame, source_name in (
                (gpu, "GPU telemetry"),
                (node, "node telemetry"),
                (workloads, "workload telemetry"),
                (detections, "detector results"),
            ):
                references = set(
                    frame["incident_id"].dropna().astype(str).loc[
                        lambda values: values.str.strip().ne("")
                    ]
                )
                unknown = references.difference(known_ids)
                if unknown:
                    raise IncidentRepositoryError(
                        f"{source_name} references unknown incidents: "
                        f"{sorted(unknown)}"
                    )

            self._incident_records: Mapping[str, IncidentRecord] = (
                MappingProxyType(records)
            )
            self._gpu = gpu.sort_values("timestamp", kind="stable").reset_index(
                drop=True
            )
            self._node = node.sort_values("timestamp", kind="stable").reset_index(
                drop=True
            )
            self._workloads = workloads.sort_values(
                "timestamp", kind="stable"
            ).reset_index(drop=True)
            self._detections = detections.sort_values(
                "timestamp", kind="stable"
            ).reset_index(drop=True)

            self._sampling_interval_minutes = self._infer_sampling_interval()
            self._validate_referential_integrity()

    def _infer_sampling_interval(self) -> int:
        times = (
            self._gpu.loc[
                (self._gpu["node_id"] == self._gpu.iloc[0]["node_id"])
                & (self._gpu["gpu_id"] == self._gpu.iloc[0]["gpu_id"]),
                "timestamp",
            ]
            .drop_duplicates()
            .sort_values()
        )
        differences = times.diff().dropna().dt.total_seconds().div(60.0)
        if differences.empty:
            raise IncidentRepositoryError("unable to infer sampling interval")
        mode = differences.mode()
        if mode.empty or mode.iloc[0] <= 0:
            raise IncidentRepositoryError("invalid telemetry sampling interval")
        return int(round(float(mode.iloc[0])))

    def _validate_referential_integrity(self) -> None:
        for incident in self._incident_records.values():
            gpu_rows = self._gpu.loc[
                (self._gpu["incident_id"] == incident.incident_id)
                & (self._gpu["node_id"] == incident.node_id)
                & (self._gpu["gpu_id"] == incident.gpu_id)
            ]
            workload_rows = self._workloads.loc[
                (self._workloads["incident_id"] == incident.incident_id)
                & (self._workloads["job_id"] == incident.affected_workload)
            ]
            detector_rows = self._detections.loc[
                (self._detections["incident_id"] == incident.incident_id)
                & (self._detections["node_id"] == incident.node_id)
                & (self._detections["gpu_id"] == incident.gpu_id)
            ]
            if gpu_rows.empty:
                raise IncidentRepositoryError(
                    f"incident {incident.incident_id} has no matching GPU telemetry"
                )
            if workload_rows.empty:
                raise IncidentRepositoryError(
                    f"incident {incident.incident_id} has no matching workload telemetry"
                )
            if detector_rows.empty:
                raise IncidentRepositoryError(
                    f"incident {incident.incident_id} has no matching detector results"
                )

    @staticmethod
    def _clean_value(value: Any) -> Any:
        if isinstance(value, pd.Timestamp):
            return value.isoformat().replace("+00:00", "Z")
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(value, "item"):
            return value.item()
        return value

    @classmethod
    def _records(cls, frame: pd.DataFrame) -> tuple[Mapping[str, Any], ...]:
        output: list[Mapping[str, Any]] = []
        for raw in frame.to_dict(orient="records"):
            cleaned = {key: cls._clean_value(value) for key, value in raw.items()}
            output.append(MappingProxyType(cleaned))
        return tuple(output)

    def list_incidents(
        self,
        *,
        severity: str | None = None,
        incident_type: str | None = None,
        node_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[IncidentSummary, ...]:
        if isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise IncidentRepositoryError("limit must be between 1 and 1000")
        if isinstance(offset, bool) or offset < 0:
            raise IncidentRepositoryError("offset must be non-negative")

        records = list(self._incident_records.values())
        if severity is not None:
            records = [item for item in records if item.severity == severity]
        if incident_type is not None:
            records = [
                item for item in records if item.incident_type == incident_type
            ]
        if node_id is not None:
            records = [item for item in records if item.node_id == node_id]

        records.sort(
            key=lambda item: (item.start_timestamp, item.incident_id),
            reverse=True,
        )
        selected = records[offset : offset + limit]
        return tuple(
            IncidentSummary(
                incident_id=item.incident_id,
                incident_type=item.incident_type,
                severity=item.severity,
                node_id=item.node_id,
                gpu_id=item.gpu_id,
                affected_workload=item.affected_workload,
                start_timestamp=item.start_timestamp,
                end_timestamp=item.end_timestamp,
                incident_status=item.incident_status,
            )
            for item in selected
        )

    def get_incident(self, incident_id: str) -> IncidentRecord:
        normalized = incident_id.strip()
        if not normalized:
            raise IncidentRepositoryError("incident_id cannot be blank")
        try:
            return self._incident_records[normalized]
        except KeyError as exc:
            raise IncidentRepositoryError(
                f"Unknown incident_id: {normalized}"
            ) from exc

    def build_evidence(self, incident_id: str) -> IncidentEvidencePackage:
        incident = self.get_incident(incident_id)
        baseline_duration = pd.Timedelta(
            minutes=self._sampling_interval_minutes * self.baseline_samples
        )
        baseline_start = incident.start_timestamp - baseline_duration

        gpu_baseline = self._gpu.loc[
            (self._gpu["node_id"] == incident.node_id)
            & (self._gpu["gpu_id"] == incident.gpu_id)
            & (self._gpu["timestamp"] >= baseline_start)
            & (self._gpu["timestamp"] < incident.start_timestamp)
            & (self._gpu["incident_id"].fillna("").astype(str).str.strip() == "")
        ].tail(self.baseline_samples)

        gpu_window = self._gpu.loc[
            (self._gpu["node_id"] == incident.node_id)
            & (self._gpu["gpu_id"] == incident.gpu_id)
            & (self._gpu["timestamp"] >= incident.start_timestamp)
            & (self._gpu["timestamp"] <= incident.end_timestamp)
            & (self._gpu["incident_id"] == incident.incident_id)
        ]

        workload_baseline = self._workloads.loc[
            (self._workloads["job_id"] == incident.affected_workload)
            & (self._workloads["timestamp"] >= baseline_start)
            & (self._workloads["timestamp"] < incident.start_timestamp)
            & (
                self._workloads["incident_id"]
                .fillna("")
                .astype(str)
                .str.strip()
                == ""
            )
        ].tail(self.baseline_samples)

        workload_window = self._workloads.loc[
            (self._workloads["job_id"] == incident.affected_workload)
            & (self._workloads["timestamp"] >= incident.start_timestamp)
            & (self._workloads["timestamp"] <= incident.end_timestamp)
            & (self._workloads["incident_id"] == incident.incident_id)
        ]

        node_window = self._node.loc[
            (self._node["node_id"] == incident.node_id)
            & (self._node["timestamp"] >= incident.start_timestamp)
            & (self._node["timestamp"] <= incident.end_timestamp)
        ]

        detector_window = self._detections.loc[
            (self._detections["node_id"] == incident.node_id)
            & (self._detections["gpu_id"] == incident.gpu_id)
            & (self._detections["timestamp"] >= incident.start_timestamp)
            & (self._detections["timestamp"] <= incident.end_timestamp)
            & (self._detections["incident_id"] == incident.incident_id)
        ]

        try:
            return IncidentEvidencePackage(
                incident=incident,
                runbook=self.runbooks.get(incident.incident_type),
                gpu_baseline=self._records(gpu_baseline),
                gpu_incident_window=self._records(gpu_window),
                workload_baseline=self._records(workload_baseline),
                workload_incident_window=self._records(workload_window),
                node_incident_window=self._records(node_window),
                detector_incident_window=self._records(detector_window),
                sampling_interval_minutes=self._sampling_interval_minutes,
            )
        except Exception as exc:
            raise IncidentRepositoryError(
                f"Unable to build complete evidence for {incident.incident_id}: {exc}"
            ) from exc

    def overview(self) -> dict[str, Any]:
        records = tuple(self._incident_records.values())
        severity_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for incident in records:
            severity_counts[incident.severity] = (
                severity_counts.get(incident.severity, 0) + 1
            )
            type_counts[incident.incident_type] = (
                type_counts.get(incident.incident_type, 0) + 1
            )
        return {
            "incident_count": len(records),
            "severity_counts": dict(sorted(severity_counts.items())),
            "incident_type_counts": dict(sorted(type_counts.items())),
            "node_count": int(self._gpu["node_id"].nunique()),
            "gpu_count": int(
                self._gpu[["node_id", "gpu_id"]].drop_duplicates().shape[0]
            ),
            "workload_count": int(self._workloads["job_id"].nunique()),
            "sampling_interval_minutes": self._sampling_interval_minutes,
        }


@lru_cache(maxsize=2)
def _cached_repository(
    data_directory: str,
    artifact_directory: str,
    runbook_directory: str,
    baseline_samples: int,
) -> IncidentRepository:
    settings = get_settings().model_copy(
        update={
            "data_directory": Path(data_directory),
            "artifact_directory": Path(artifact_directory),
            "runbook_directory": Path(runbook_directory),
        }
    )
    return IncidentRepository(settings, baseline_samples=baseline_samples)


def get_incident_repository(
    settings: Settings | None = None,
    *,
    baseline_samples: int = 12,
) -> IncidentRepository:
    runtime = settings or get_settings()
    return _cached_repository(
        str(runtime.data_directory.resolve()),
        str(runtime.artifact_directory.resolve()),
        str(runtime.runbook_directory.resolve()),
        baseline_samples,
    )


def clear_incident_repository_cache() -> None:
    _cached_repository.cache_clear()
