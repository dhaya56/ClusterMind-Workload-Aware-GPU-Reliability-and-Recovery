from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.client.exceptions import InfluxDBError

from app.core.config import Settings, get_settings


class InfluxDBServiceError(RuntimeError):
    """Raised when ClusterMind cannot communicate with InfluxDB safely."""


@dataclass(frozen=True)
class WriteSummary:
    """Result of a batched InfluxDB write operation."""

    measurement: str
    source_rows: int
    written_points: int
    skipped_rows: int
    batches: int


class InfluxDBService:
    """Connection and batched telemetry-writing service for InfluxDB OSS v2."""

    GPU_TAGS = (
        "node_id",
        "gpu_id",
        "operating_state",
        "anomaly_label",
        "incident_id",
    )
    GPU_FIELDS = (
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "gpu_temperature_c",
        "memory_temperature_c",
        "power_draw_watts",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "ecc_correctable_errors",
        "ecc_uncorrectable_errors",
        "xid_error_code",
        "nvlink_tx_mbps",
        "nvlink_rx_mbps",
    )

    NODE_TAGS = (
        "node_id",
        "node_health_status",
        "incident_id",
    )
    NODE_FIELDS = (
        "cpu_utilization_percent",
        "host_memory_percent",
        "disk_io_percent",
        "network_tx_mbps",
        "network_rx_mbps",
        "fan_speed_percent",
        "active_workloads",
        "healthy_gpu_count",
    )

    WORKLOAD_TAGS = (
        "job_id",
        "job_type",
        "priority",
        "assigned_node",
        "workload_status",
        "incident_id",
    )
    WORKLOAD_FIELDS = (
        "allocated_gpus",
        "checkpoint_available",
        "training_throughput",
        "tokens_per_second",
        "step_duration_ms",
        "training_loss",
        "inference_latency_ms",
        "error_rate_percent",
        "restart_count",
    )

    INCIDENT_TAGS = (
        "incident_id",
        "node_id",
        "gpu_id",
        "incident_type",
        "severity",
        "root_cause",
        "affected_workload",
        "recovery_action",
        "incident_status",
    )
    INCIDENT_FIELDS = (
        "gpu_hours_lost",
        "duration_minutes",
    )

    DETECTION_TAGS = (
        "node_id",
        "gpu_id",
        "severity",
        "override_reason",
        "ground_truth_label",
        "incident_id",
    )
    DETECTION_FIELDS = (
        "anomaly_score",
        "model_is_anomaly",
        "predicted_anomaly",
    )

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: InfluxDBClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_client = client is None
        self.client = client or InfluxDBClient(
            url=self.settings.influxdb_url,
            token=self.settings.influxdb_token,
            org=self.settings.influxdb_org,
            timeout=self.settings.influxdb_timeout_ms,
            enable_gzip=True,
        )

    def __enter__(self) -> "InfluxDBService":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception as exc:
            raise InfluxDBServiceError("InfluxDB ping failed") from exc

    def health(self) -> dict[str, Any]:
        try:
            health = self.client.health()
        except Exception as exc:
            raise InfluxDBServiceError("InfluxDB health check failed") from exc

        return {
            "status": str(getattr(health, "status", "unknown")),
            "message": str(getattr(health, "message", "")),
            "version": str(getattr(health, "version", "")),
        }

    def verify_bucket(self) -> None:
        """Confirm that the configured organization and bucket are available."""

        try:
            bucket = self.client.buckets_api().find_bucket_by_name(
                self.settings.influxdb_bucket
            )
        except Exception as exc:
            raise InfluxDBServiceError("Unable to query InfluxDB buckets") from exc

        if bucket is None:
            raise InfluxDBServiceError(
                f"InfluxDB bucket does not exist: {self.settings.influxdb_bucket}"
            )

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        try:
            result = pd.isna(value)
        except (TypeError, ValueError):
            return False
        return bool(result) if isinstance(result, (bool, type(pd.NA))) else False

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = pd.to_datetime(value, utc=True, errors="raise").to_pydatetime()
            except (TypeError, ValueError, OverflowError) as exc:
                raise InfluxDBServiceError(
                    f"Invalid telemetry timestamp: {value!r}"
                ) from exc

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @classmethod
    def _field_value(cls, value: Any) -> bool | int | float | str | None:
        if cls._is_missing(value):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int,)):
            return int(value)
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return float(value)
        if hasattr(value, "item"):
            return cls._field_value(value.item())
        return str(value)

    @classmethod
    def _tag_value(cls, value: Any) -> str | None:
        if cls._is_missing(value):
            return None
        text = str(value).strip()
        return text or None

    def _record_to_point(
        self,
        record: Mapping[str, Any],
        *,
        measurement: str,
        timestamp_field: str,
        tag_fields: Sequence[str],
        value_fields: Sequence[str],
    ) -> Point | None:
        if timestamp_field not in record:
            raise InfluxDBServiceError(
                f"Record for '{measurement}' is missing timestamp field "
                f"'{timestamp_field}'"
            )

        point = Point(measurement).time(
            self._parse_timestamp(record[timestamp_field]),
            WritePrecision.NS,
        )

        for field_name in tag_fields:
            if field_name not in record:
                raise InfluxDBServiceError(
                    f"Record for '{measurement}' is missing tag '{field_name}'"
                )
            value = self._tag_value(record[field_name])
            if value is not None:
                point = point.tag(field_name, value)

        field_count = 0
        for field_name in value_fields:
            if field_name not in record:
                raise InfluxDBServiceError(
                    f"Record for '{measurement}' is missing field '{field_name}'"
                )
            value = self._field_value(record[field_name])
            if value is not None:
                point = point.field(field_name, value)
                field_count += 1

        return point if field_count else None

    @staticmethod
    def _chunks(items: Sequence[Point], size: int) -> Iterator[Sequence[Point]]:
        for start in range(0, len(items), size):
            yield items[start : start + size]

    def write_records(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        measurement: str,
        timestamp_field: str,
        tag_fields: Sequence[str],
        value_fields: Sequence[str],
        batch_size: int = 2000,
    ) -> WriteSummary:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise InfluxDBServiceError("batch_size must be a positive integer")
        if batch_size <= 0:
            raise InfluxDBServiceError("batch_size must be a positive integer")
        if not measurement.strip():
            raise InfluxDBServiceError("measurement cannot be empty")

        points: list[Point] = []
        source_rows = 0
        skipped_rows = 0

        for record in records:
            source_rows += 1
            point = self._record_to_point(
                record,
                measurement=measurement,
                timestamp_field=timestamp_field,
                tag_fields=tag_fields,
                value_fields=value_fields,
            )
            if point is None:
                skipped_rows += 1
            else:
                points.append(point)

        if not points:
            return WriteSummary(
                measurement=measurement,
                source_rows=source_rows,
                written_points=0,
                skipped_rows=skipped_rows,
                batches=0,
            )

        write_api = self.client.write_api(write_options=SYNCHRONOUS)
        batches = 0

        try:
            for point_batch in self._chunks(points, batch_size):
                write_api.write(
                    bucket=self.settings.influxdb_bucket,
                    org=self.settings.influxdb_org,
                    record=list(point_batch),
                    write_precision=WritePrecision.NS,
                )
                batches += 1
        except (InfluxDBError, OSError, RuntimeError) as exc:
            raise InfluxDBServiceError(
                f"Failed to write measurement '{measurement}'"
            ) from exc
        finally:
            write_api.close()

        return WriteSummary(
            measurement=measurement,
            source_rows=source_rows,
            written_points=len(points),
            skipped_rows=skipped_rows,
            batches=batches,
        )

    def write_dataframe_chunks(
        self,
        csv_path: Path,
        *,
        measurement: str,
        timestamp_field: str,
        tag_fields: Sequence[str],
        value_fields: Sequence[str],
        csv_chunksize: int = 5000,
        batch_size: int = 2000,
    ) -> WriteSummary:
        source = csv_path.resolve()
        if not source.is_file():
            raise InfluxDBServiceError(f"CSV file does not exist: {source}")
        if isinstance(csv_chunksize, bool) or csv_chunksize <= 0:
            raise InfluxDBServiceError("csv_chunksize must be positive")

        totals = WriteSummary(measurement, 0, 0, 0, 0)
        try:
            chunks = pd.read_csv(source, chunksize=csv_chunksize, low_memory=False)
            for frame in chunks:
                summary = self.write_records(
                    frame.to_dict(orient="records"),
                    measurement=measurement,
                    timestamp_field=timestamp_field,
                    tag_fields=tag_fields,
                    value_fields=value_fields,
                    batch_size=batch_size,
                )
                totals = WriteSummary(
                    measurement=measurement,
                    source_rows=totals.source_rows + summary.source_rows,
                    written_points=totals.written_points + summary.written_points,
                    skipped_rows=totals.skipped_rows + summary.skipped_rows,
                    batches=totals.batches + summary.batches,
                )
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            raise InfluxDBServiceError(f"Unable to ingest CSV file: {source}") from exc

        return totals

    def write_gpu_csv(self, path: Path, **kwargs: Any) -> WriteSummary:
        return self.write_dataframe_chunks(
            path,
            measurement="gpu_telemetry",
            timestamp_field="timestamp",
            tag_fields=self.GPU_TAGS,
            value_fields=self.GPU_FIELDS,
            **kwargs,
        )

    def write_node_csv(self, path: Path, **kwargs: Any) -> WriteSummary:
        return self.write_dataframe_chunks(
            path,
            measurement="node_telemetry",
            timestamp_field="timestamp",
            tag_fields=self.NODE_TAGS,
            value_fields=self.NODE_FIELDS,
            **kwargs,
        )

    def write_workload_csv(self, path: Path, **kwargs: Any) -> WriteSummary:
        return self.write_dataframe_chunks(
            path,
            measurement="workload_telemetry",
            timestamp_field="timestamp",
            tag_fields=self.WORKLOAD_TAGS,
            value_fields=self.WORKLOAD_FIELDS,
            **kwargs,
        )

    def write_incident_csv(self, path: Path, **kwargs: Any) -> WriteSummary:
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            return WriteSummary("gpu_incidents", 0, 0, 0, 0)

        start = pd.to_datetime(frame["start_timestamp"], utc=True, errors="raise")
        end = pd.to_datetime(frame["end_timestamp"], utc=True, errors="raise")
        frame = frame.copy()
        frame["duration_minutes"] = (end - start).dt.total_seconds() / 60.0

        return self.write_records(
            frame.to_dict(orient="records"),
            measurement="gpu_incidents",
            timestamp_field="start_timestamp",
            tag_fields=self.INCIDENT_TAGS,
            value_fields=self.INCIDENT_FIELDS,
            **kwargs,
        )

    def write_detection_csv(self, path: Path, **kwargs: Any) -> WriteSummary:
        return self.write_dataframe_chunks(
            path,
            measurement="detector_results",
            timestamp_field="timestamp",
            tag_fields=self.DETECTION_TAGS,
            value_fields=self.DETECTION_FIELDS,
            **kwargs,
        )

    def delete_measurements(self, measurements: Sequence[str]) -> None:
        """Delete only named ClusterMind measurements from the configured bucket."""

        allowed = {
            "gpu_telemetry",
            "node_telemetry",
            "workload_telemetry",
            "gpu_incidents",
            "detector_results",
        }
        requested = set(measurements)
        unknown = requested.difference(allowed)
        if unknown:
            raise InfluxDBServiceError(
                "Refusing to delete unknown measurements: "
                + ", ".join(sorted(unknown))
            )
        if not requested:
            return

        start = "1970-01-01T00:00:00Z"
        stop = "2100-01-01T00:00:00Z"
        delete_api = self.client.delete_api()
        try:
            for measurement in sorted(requested):
                delete_api.delete(
                    start,
                    stop,
                    f'_measurement="{measurement}"',
                    bucket=self.settings.influxdb_bucket,
                    org=self.settings.influxdb_org,
                )
        except Exception as exc:
            raise InfluxDBServiceError(
                "Unable to clear ClusterMind measurements"
            ) from exc

    def measurement_count(self, measurement: str) -> int:
        """Count logical points by using one required field per measurement."""

        count_fields = {
            "gpu_telemetry": "gpu_utilization_percent",
            "node_telemetry": "cpu_utilization_percent",
            "workload_telemetry": "allocated_gpus",
            "gpu_incidents": "gpu_hours_lost",
            "detector_results": "anomaly_score",
        }
        field_name = count_fields.get(measurement)
        if field_name is None:
            raise InfluxDBServiceError(
                f"Unknown ClusterMind measurement: {measurement}"
            )

        flux = f'''
from(bucket: "{self.settings.influxdb_bucket}")
  |> range(start: 1970-01-01T00:00:00Z, stop: 2100-01-01T00:00:00Z)
  |> filter(fn: (r) => r._measurement == "{measurement}")
  |> filter(fn: (r) => r._field == "{field_name}")
  |> count(column: "_value")
'''.strip()

        try:
            tables = self.client.query_api().query(
                query=flux,
                org=self.settings.influxdb_org,
            )
        except Exception as exc:
            raise InfluxDBServiceError(
                f"Unable to count measurement '{measurement}'"
            ) from exc

        values = [
            int(record.get_value())
            for table in tables
            for record in table.records
            if record.get_value() is not None
        ]
        return sum(values)
