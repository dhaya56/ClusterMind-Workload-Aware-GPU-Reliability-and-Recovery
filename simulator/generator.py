from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import pi, sin
from typing import Any, Iterator

import numpy as np

from simulator.config import SimulatorConfig, load_config
from simulator.scenarios import SCENARIO_METADATA, apply_scenario


class GenerationError(RuntimeError):
    """Raised when a valid synthetic dataset cannot be generated."""


@dataclass(frozen=True)
class WorkloadDefinition:
    job_id: str
    job_type: str
    priority: str
    assigned_node: str
    allocated_gpus: int
    checkpoint_available: bool


@dataclass(frozen=True)
class ScenarioWindow:
    window_id: str
    scenario_name: str
    severity: str
    node_id: str
    gpu_id: str
    workload_id: str
    start_index: int
    end_index: int
    is_healthy_control: bool = False

    def contains(self, timestamp_index: int) -> bool:
        return self.start_index <= timestamp_index <= self.end_index

    def progress(self, timestamp_index: int) -> float:
        """Return triangular progress: onset -> peak -> recovery."""

        if not self.contains(timestamp_index):
            raise GenerationError(
                f"Timestamp index {timestamp_index} is outside {self.window_id}"
            )

        duration = self.end_index - self.start_index
        if duration <= 0:
            return 1.0

        position = (timestamp_index - self.start_index) / duration
        return float(np.clip(1.0 - abs((2.0 * position) - 1.0), 0.0, 1.0))


class DatasetGenerator:
    """Generate deterministic, correlated ClusterMind telemetry in chunks."""

    def __init__(
        self,
        config: SimulatorConfig | None = None,
        *,
        profile_name: str = "development",
        start_time: datetime | None = None,
    ) -> None:
        self.config = config or load_config()
        self.profile_name = profile_name
        self.profile = self.config.profile(profile_name)
        self.start_time = self._normalize_start_time(start_time)
        self.interval = timedelta(
            minutes=int(self.profile["sampling_interval_minutes"])
        )
        self.timestamp_count = self._calculate_timestamp_count()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.node_ids = self._build_node_ids()
        self.gpu_ids = self._build_gpu_ids()
        self.workloads = self._build_workloads()
        self._workload_by_node = {
            workload.assigned_node: workload for workload in self.workloads
        }
        self.windows = self._schedule_windows()
        self._windows_by_node = self._index_windows_by_node()

    @staticmethod
    def _normalize_start_time(value: datetime | None) -> datetime:
        if value is None:
            return datetime(2026, 1, 1, tzinfo=timezone.utc)

        if value.tzinfo is None:
            raise GenerationError("start_time must include timezone information")

        return value.astimezone(timezone.utc).replace(microsecond=0)

    def _calculate_timestamp_count(self) -> int:
        duration_days = int(self.profile["duration_days"])
        interval_minutes = int(self.profile["sampling_interval_minutes"])
        total_minutes = duration_days * 24 * 60

        if duration_days <= 0 or interval_minutes <= 0:
            raise GenerationError("Duration and sampling interval must be positive")
        if total_minutes % interval_minutes != 0:
            raise GenerationError("Sampling interval must divide the profile duration")

        return total_minutes // interval_minutes

    def _build_node_ids(self) -> tuple[str, ...]:
        identifiers = self.config.cluster["cluster"]["identifiers"]
        prefix = str(identifiers["node_prefix"])
        width = int(identifiers["node_number_width"])
        return tuple(
            f"{prefix}-{index:0{width}d}"
            for index in range(1, self.config.node_count + 1)
        )

    def _build_gpu_ids(self) -> tuple[str, ...]:
        identifiers = self.config.cluster["cluster"]["identifiers"]
        prefix = str(identifiers["gpu_prefix"])
        width = int(identifiers["gpu_number_width"])
        return tuple(
            f"{prefix}-{index:0{width}d}"
            for index in range(self.config.gpus_per_node)
        )

    def _build_workloads(self) -> tuple[WorkloadDefinition, ...]:
        workload_config = self.config.cluster["workloads"]
        workload_types = tuple(workload_config["enabled_types"])
        priorities = tuple(workload_config["priorities"])
        checkpoint_probability = float(
            workload_config["checkpoint_probability"]
        )
        minimum_gpus = int(workload_config["minimum_gpus_per_job"])
        maximum_gpus = min(
            int(workload_config["maximum_gpus_per_job"]),
            self.config.gpus_per_node,
        )

        definitions: list[WorkloadDefinition] = []
        for index, node_id in enumerate(self.node_ids, start=1):
            job_type = workload_types[(index - 1) % len(workload_types)]
            priority = priorities[(index - 1) % len(priorities)]
            allocated_gpus = int(
                self.rng.integers(minimum_gpus, maximum_gpus + 1)
            )
            checkpoint_available = bool(
                self.rng.random() < checkpoint_probability
            )
            definitions.append(
                WorkloadDefinition(
                    job_id=f"job-{index:04d}",
                    job_type=job_type,
                    priority=priority,
                    assigned_node=node_id,
                    allocated_gpus=allocated_gpus,
                    checkpoint_available=checkpoint_available,
                )
            )

        return tuple(definitions)

    def _duration_steps(self) -> tuple[int, int]:
        incident_config = self.config.dataset["incidents"]
        interval_minutes = int(self.profile["sampling_interval_minutes"])
        minimum = max(
            3,
            int(np.ceil(
                int(incident_config["minimum_duration_minutes"])
                / interval_minutes
            )),
        )
        maximum = max(
            minimum,
            int(np.floor(
                int(incident_config["maximum_duration_minutes"])
                / interval_minutes
            )),
        )
        return minimum, maximum

    def _gap_steps(self) -> int:
        gap_minutes = int(
            self.config.dataset["incidents"]["minimum_gap_minutes"]
        )
        interval_minutes = int(self.profile["sampling_interval_minutes"])
        return max(1, int(np.ceil(gap_minutes / interval_minutes)))

    def _scenario_choice(self) -> str:
        distribution = self.config.dataset["scenario_distribution"]
        names = tuple(distribution)
        probabilities = np.array(
            [float(distribution[name]) for name in names],
            dtype=float,
        )
        return str(self.rng.choice(names, p=probabilities))

    def _severity_choice(self, scenario_name: str) -> str:
        supported = SCENARIO_METADATA[scenario_name].supported_severities
        base_weights = {
            "low": 0.15,
            "medium": 0.35,
            "high": 0.35,
            "critical": 0.15,
        }
        weights = np.array(
            [base_weights[level] for level in supported],
            dtype=float,
        )
        weights /= weights.sum()
        return str(self.rng.choice(supported, p=weights))

    def _window_fits(
        self,
        existing: list[tuple[int, int]],
        start_index: int,
        end_index: int,
        gap_steps: int,
    ) -> bool:
        return all(
            end_index + gap_steps < existing_start
            or start_index - gap_steps > existing_end
            for existing_start, existing_end in existing
        )

    def _schedule_windows(self) -> tuple[ScenarioWindow, ...]:
        incident_count = int(self.profile["incident_count"])
        control_config = self.config.dataset["healthy_control_scenarios"]
        control_count = (
            int(control_config["healthy_high_utilization_count"][self.profile_name])
            if bool(control_config["enabled"])
            else 0
        )
        minimum_steps, maximum_steps = self._duration_steps()
        gap_steps = self._gap_steps()
        occupied_by_node: dict[str, list[tuple[int, int]]] = {
            node_id: [] for node_id in self.node_ids
        }
        windows: list[ScenarioWindow] = []

        total_windows = incident_count + control_count
        max_attempts = max(1000, total_windows * 200)
        attempts = 0

        while len(windows) < total_windows and attempts < max_attempts:
            attempts += 1
            is_control = len(windows) >= incident_count
            scenario_name = (
                "healthy_high_utilization"
                if is_control
                else self._scenario_choice()
            )
            severity = "low" if is_control else self._severity_choice(scenario_name)
            node_id = str(self.rng.choice(self.node_ids))
            gpu_id = str(self.rng.choice(self.gpu_ids))
            duration_steps = int(
                self.rng.integers(minimum_steps, maximum_steps + 1)
            )

            latest_start = self.timestamp_count - duration_steps - 1
            if latest_start <= 1:
                raise GenerationError("Profile is too short for incident scheduling")

            start_index = int(self.rng.integers(1, latest_start + 1))
            end_index = start_index + duration_steps

            if not self._window_fits(
                occupied_by_node[node_id],
                start_index,
                end_index,
                gap_steps,
            ):
                continue

            occupied_by_node[node_id].append((start_index, end_index))
            workload_id = self._workload_by_node[node_id].job_id
            prefix = "CTRL" if is_control else "INC"
            ordinal = len(windows) - incident_count + 1 if is_control else len(windows) + 1
            windows.append(
                ScenarioWindow(
                    window_id=f"{prefix}-2026-{ordinal:04d}",
                    scenario_name=scenario_name,
                    severity=severity,
                    node_id=node_id,
                    gpu_id=gpu_id,
                    workload_id=workload_id,
                    start_index=start_index,
                    end_index=end_index,
                    is_healthy_control=is_control,
                )
            )

        if len(windows) != total_windows:
            raise GenerationError(
                f"Scheduled {len(windows)} of {total_windows} requested windows "
                f"after {attempts} attempts"
            )

        return tuple(
            sorted(windows, key=lambda item: (item.start_index, item.window_id))
        )

    def _index_windows_by_node(self) -> dict[str, tuple[ScenarioWindow, ...]]:
        index: dict[str, list[ScenarioWindow]] = {
            node_id: [] for node_id in self.node_ids
        }
        for window in self.windows:
            index[window.node_id].append(window)
        return {
            node_id: tuple(sorted(items, key=lambda item: item.start_index))
            for node_id, items in index.items()
        }

    def _active_window(
        self,
        node_id: str,
        timestamp_index: int,
    ) -> ScenarioWindow | None:
        for window in self._windows_by_node[node_id]:
            if window.start_index > timestamp_index:
                break
            if window.contains(timestamp_index):
                return window
        return None

    def _timestamp(self, timestamp_index: int) -> datetime:
        return self.start_time + (self.interval * timestamp_index)

    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _baseline_workload(
        self,
        definition: WorkloadDefinition,
        timestamp_index: int,
    ) -> dict[str, Any]:
        samples_per_day = int(
            timedelta(days=1).total_seconds() / self.interval.total_seconds()
        )
        phase = (2.0 * pi * timestamp_index / samples_per_day) + (
            self.node_ids.index(definition.assigned_node) * 0.31
        )
        load_factor = float(np.clip(0.72 + (0.16 * sin(phase)), 0.45, 0.95))
        is_inference = definition.job_type == "batch_inference"

        return {
            "job_type": definition.job_type,
            "workload_status": "running",
            "checkpoint_available": definition.checkpoint_available,
            "training_throughput": None if is_inference else round(
                max(50.0, 1800.0 * load_factor + self.rng.normal(0.0, 25.0)),
                3,
            ),
            "tokens_per_second": round(
                max(50.0, 5200.0 * load_factor + self.rng.normal(0.0, 65.0)),
                3,
            ),
            "step_duration_ms": None if is_inference else round(
                max(40.0, 360.0 / load_factor + self.rng.normal(0.0, 5.0)),
                3,
            ),
            "training_loss": None if is_inference else round(
                max(0.2, 2.8 * np.exp(-timestamp_index / 1400.0) + 0.25),
                5,
            ),
            "inference_latency_ms": round(
                max(5.0, 45.0 / load_factor + self.rng.normal(0.0, 1.0)),
                3,
            ) if is_inference else None,
            "error_rate_percent": round(
                max(0.0, self.rng.normal(0.15, 0.05)),
                4,
            ),
            "restart_count": 0,
        }

    def _baseline_gpu(
        self,
        workload: dict[str, Any],
        gpu_index: int,
    ) -> dict[str, Any]:
        useful_rate = workload["tokens_per_second"] or workload["training_throughput"]
        utilization = float(np.clip(
            40.0 + (float(useful_rate) / 110.0) + self.rng.normal(0.0, 3.0),
            25.0,
            98.0,
        ))
        memory_utilization = float(np.clip(
            38.0 + (0.45 * utilization) + self.rng.normal(0.0, 4.0),
            20.0,
            95.0,
        ))
        temperature = float(np.clip(
            34.0 + (0.42 * utilization) + self.rng.normal(0.0, 2.0),
            35.0,
            82.0,
        ))
        memory_temperature = float(np.clip(
            temperature + 4.0 + self.rng.normal(0.0, 1.5),
            38.0,
            88.0,
        ))
        power = float(np.clip(
            65.0 + (2.65 * utilization) + self.rng.normal(0.0, 8.0),
            70.0,
            330.0,
        ))
        active_fraction = float(np.clip((utilization - 30.0) / 70.0, 0.0, 1.0))
        sm_clock = float(np.clip(
            850.0 + (900.0 * active_fraction) + self.rng.normal(0.0, 35.0),
            900.0,
            1800.0,
        ))
        memory_clock = float(np.clip(
            1200.0 + (700.0 * active_fraction) + self.rng.normal(0.0, 25.0),
            1200.0,
            2000.0,
        ))
        nvlink_base = float(np.clip(
            9000.0 + (390.0 * utilization) + self.rng.normal(0.0, 1200.0),
            8000.0,
            50000.0,
        ))
        direction_factor = float(self.rng.uniform(0.94, 1.06))

        return {
            "gpu_utilization_percent": round(utilization, 3),
            "memory_utilization_percent": round(memory_utilization, 3),
            "gpu_temperature_c": round(temperature, 3),
            "memory_temperature_c": round(memory_temperature, 3),
            "power_draw_watts": round(power, 3),
            "sm_clock_mhz": round(sm_clock, 3),
            "memory_clock_mhz": round(memory_clock, 3),
            "ecc_correctable_errors": 0,
            "ecc_uncorrectable_errors": 0,
            "xid_error_code": 0,
            "nvlink_tx_mbps": round(nvlink_base, 3),
            "nvlink_rx_mbps": round(
                float(np.clip(nvlink_base * direction_factor, 8000.0, 50000.0)),
                3,
            ),
        }

    def _workload_output_record(
        self,
        definition: WorkloadDefinition,
        timestamp: str,
        workload: dict[str, Any],
        incident_id: str,
    ) -> dict[str, Any]:
        return {
            "timestamp": timestamp,
            "job_id": definition.job_id,
            "job_type": workload["job_type"],
            "priority": definition.priority,
            "assigned_node": definition.assigned_node,
            "allocated_gpus": definition.allocated_gpus,
            "workload_status": workload["workload_status"],
            "checkpoint_available": workload["checkpoint_available"],
            "training_throughput": workload["training_throughput"],
            "tokens_per_second": workload["tokens_per_second"],
            "step_duration_ms": workload["step_duration_ms"],
            "training_loss": workload["training_loss"],
            "inference_latency_ms": workload["inference_latency_ms"],
            "error_rate_percent": workload["error_rate_percent"],
            "restart_count": workload["restart_count"],
            "incident_id": incident_id,
        }

    def _generate_timestamp_records(
        self,
        timestamp_index: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        timestamp = self._format_timestamp(self._timestamp(timestamp_index))
        gpu_records: list[dict[str, Any]] = []
        node_records: list[dict[str, Any]] = []
        workload_records: list[dict[str, Any]] = []

        for definition in self.workloads:
            node_id = definition.assigned_node
            active_window = self._active_window(node_id, timestamp_index)
            workload = self._baseline_workload(definition, timestamp_index)
            node_gpu_records: list[dict[str, Any]] = []
            incident_id = ""

            for gpu_index, gpu_id in enumerate(self.gpu_ids):
                gpu = self._baseline_gpu(workload, gpu_index)
                operating_state = "normal"
                anomaly_label = "normal"
                gpu_incident_id = ""

                if active_window is not None and active_window.gpu_id == gpu_id:
                    result = apply_scenario(
                        active_window.scenario_name,
                        gpu,
                        workload,
                        progress=active_window.progress(timestamp_index),
                        severity=active_window.severity,
                        rng=self.rng,
                    )
                    gpu = result.gpu_telemetry
                    workload = result.workload
                    anomaly_label = active_window.scenario_name
                    if active_window.is_healthy_control:
                        operating_state = "normal"
                    else:
                        operating_state = (
                            "failure"
                            if active_window.severity == "critical"
                            else "degraded"
                        )
                        gpu_incident_id = active_window.window_id
                        incident_id = active_window.window_id

                record = {
                    "timestamp": timestamp,
                    "node_id": node_id,
                    "gpu_id": gpu_id,
                    **gpu,
                    "operating_state": operating_state,
                    "anomaly_label": anomaly_label,
                    "incident_id": gpu_incident_id,
                }
                gpu_records.append(record)
                node_gpu_records.append(record)

            workload_records.append(
                self._workload_output_record(
                    definition,
                    timestamp,
                    workload,
                    incident_id,
                )
            )

            temperatures = [
                float(record["gpu_temperature_c"]) for record in node_gpu_records
            ]
            failure_count = sum(
                record["operating_state"] == "failure"
                for record in node_gpu_records
            )
            degraded_count = sum(
                record["operating_state"] == "degraded"
                for record in node_gpu_records
            )
            node_status = (
                "unhealthy" if failure_count else "degraded" if degraded_count else "healthy"
            )
            node_records.append(
                {
                    "timestamp": timestamp,
                    "node_id": node_id,
                    "cpu_utilization_percent": round(float(np.clip(
                        18.0 + (0.55 * np.mean([
                            float(record["gpu_utilization_percent"])
                            for record in node_gpu_records
                        ])) + self.rng.normal(0.0, 3.0),
                        5.0,
                        100.0,
                    )), 3),
                    "host_memory_percent": round(float(np.clip(
                        30.0 + (4.5 * definition.allocated_gpus)
                        + self.rng.normal(0.0, 4.0),
                        10.0,
                        100.0,
                    )), 3),
                    "disk_io_percent": round(float(np.clip(
                        35.0 + self.rng.normal(0.0, 8.0),
                        0.0,
                        100.0,
                    )), 3),
                    "network_tx_mbps": round(float(np.clip(
                        sum(float(record["nvlink_tx_mbps"]) for record in node_gpu_records)
                        * 0.22 + self.rng.normal(0.0, 250.0),
                        0.0,
                        50000.0,
                    )), 3),
                    "network_rx_mbps": round(float(np.clip(
                        sum(float(record["nvlink_rx_mbps"]) for record in node_gpu_records)
                        * 0.22 + self.rng.normal(0.0, 250.0),
                        0.0,
                        50000.0,
                    )), 3),
                    "fan_speed_percent": round(float(np.clip(
                        25.0 + (0.75 * (max(temperatures) - 35.0))
                        + self.rng.normal(0.0, 2.0),
                        20.0,
                        100.0,
                    )), 3),
                    "node_health_status": node_status,
                    "active_workloads": 1,
                    "healthy_gpu_count": self.config.gpus_per_node
                    - failure_count
                    - degraded_count,
                    "incident_id": incident_id,
                }
            )

        return gpu_records, node_records, workload_records

    def iter_chunks(
        self,
        *,
        chunk_timestamp_count: int = 288,
    ) -> Iterator[dict[str, list[dict[str, Any]]]]:
        if isinstance(chunk_timestamp_count, bool) or chunk_timestamp_count <= 0:
            raise GenerationError("chunk_timestamp_count must be positive")

        for chunk_start in range(0, self.timestamp_count, chunk_timestamp_count):
            chunk_end = min(
                chunk_start + chunk_timestamp_count,
                self.timestamp_count,
            )
            chunk = {
                "gpu_telemetry": [],
                "node_telemetry": [],
                "workloads": [],
            }
            for timestamp_index in range(chunk_start, chunk_end):
                gpu, node, workload = self._generate_timestamp_records(
                    timestamp_index
                )
                chunk["gpu_telemetry"].extend(gpu)
                chunk["node_telemetry"].extend(node)
                chunk["workloads"].extend(workload)
            yield chunk

    def incident_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for window in self.windows:
            if window.is_healthy_control:
                continue
            metadata = SCENARIO_METADATA[window.scenario_name]
            duration_hours = (
                (window.end_index - window.start_index + 1)
                * self.interval.total_seconds()
                / 3600.0
            )
            records.append(
                {
                    "incident_id": window.window_id,
                    "node_id": window.node_id,
                    "gpu_id": window.gpu_id,
                    "incident_type": window.scenario_name,
                    "severity": window.severity,
                    "start_timestamp": self._format_timestamp(
                        self._timestamp(window.start_index)
                    ),
                    "end_timestamp": self._format_timestamp(
                        self._timestamp(window.end_index)
                    ),
                    "root_cause": metadata.root_cause,
                    "affected_workload": window.workload_id,
                    "gpu_hours_lost": round(duration_hours, 3),
                    "recovery_action": metadata.recovery_action,
                    "incident_status": "resolved",
                }
            )
        return records

    def agent_evaluation_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for window in self.windows:
            if window.is_healthy_control:
                continue
            metadata = SCENARIO_METADATA[window.scenario_name]
            detection_index = min(window.start_index + 1, window.end_index)
            records.append(
                {
                    "incident_id": window.window_id,
                    "expected_root_cause": metadata.root_cause,
                    "expected_severity": window.severity,
                    "expected_recovery_action": metadata.recovery_action,
                    "expected_workload_impact": metadata.workload_impact,
                    "detection_timestamp": self._format_timestamp(
                        self._timestamp(detection_index)
                    ),
                    "ground_truth_label": window.scenario_name,
                }
            )
        return records

    def summary(self) -> dict[str, Any]:
        incident_windows = [
            window for window in self.windows if not window.is_healthy_control
        ]
        control_windows = [
            window for window in self.windows if window.is_healthy_control
        ]
        return {
            "profile": self.profile_name,
            "start_timestamp": self._format_timestamp(self.start_time),
            "end_timestamp": self._format_timestamp(
                self._timestamp(self.timestamp_count - 1)
            ),
            "timestamp_count": self.timestamp_count,
            "node_count": self.config.node_count,
            "gpus_per_node": self.config.gpus_per_node,
            "expected_gpu_rows": (
                self.timestamp_count
                * self.config.node_count
                * self.config.gpus_per_node
            ),
            "expected_node_rows": self.timestamp_count * self.config.node_count,
            "expected_workload_rows": self.timestamp_count * len(self.workloads),
            "incident_count": len(incident_windows),
            "healthy_control_count": len(control_windows),
            "configuration_hash": self.config.configuration_hash,
            "random_seed": self.config.random_seed,
        }
