from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


class ScenarioError(ValueError):
    """Raised when a scenario cannot be applied safely."""


@dataclass(frozen=True)
class ScenarioMetadata:
    """Ground-truth metadata associated with a simulated scenario."""

    name: str
    root_cause: str
    recovery_action: str
    workload_impact: str
    supported_severities: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioResult:
    """Telemetry and workload values produced by one scenario application."""

    gpu_telemetry: dict[str, Any]
    workload: dict[str, Any]
    metadata: ScenarioMetadata


SEVERITY_FACTORS: dict[str, float] = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75,
    "critical": 1.00,
}

SCENARIO_METADATA: dict[str, ScenarioMetadata] = {
    "thermal_throttling": ScenarioMetadata(
        name="thermal_throttling",
        root_cause="gpu_thermal_throttling",
        recovery_action="resume_from_checkpoint_and_quarantine_node",
        workload_impact="training_or_inference_performance_degradation",
        supported_severities=("medium", "high", "critical"),
    ),
    "ecc_memory_degradation": ScenarioMetadata(
        name="ecc_memory_degradation",
        root_cause="gpu_memory_degradation",
        recovery_action="quarantine_gpu_and_migrate_workload",
        workload_impact="job_instability_and_failure_risk",
        supported_severities=("medium", "high", "critical"),
    ),
    "memory_saturation": ScenarioMetadata(
        name="memory_saturation",
        root_cause="gpu_memory_capacity_exhaustion",
        recovery_action="reduce_batch_size_and_restart_workload",
        workload_impact="increased_step_time_and_out_of_memory_risk",
        supported_severities=("low", "medium", "high", "critical"),
    ),
    "power_capping": ScenarioMetadata(
        name="power_capping",
        root_cause="gpu_power_limit_enforcement",
        recovery_action="review_workload_power_profile_and_continue_monitoring",
        workload_impact="reduced_compute_throughput",
        supported_severities=("low", "medium", "high"),
    ),
    "nvlink_degradation": ScenarioMetadata(
        name="nvlink_degradation",
        root_cause="inter_gpu_link_degradation",
        recovery_action="migrate_distributed_workload",
        workload_impact="distributed_training_synchronization_delay",
        supported_severities=("medium", "high", "critical"),
    ),
    "idle_gpu_waste": ScenarioMetadata(
        name="idle_gpu_waste",
        root_cause="allocated_gpu_without_useful_work",
        recovery_action="release_allocation_and_reschedule_workload",
        workload_impact="wasted_gpu_capacity",
        supported_severities=("high",),
    ),
    "healthy_high_utilization": ScenarioMetadata(
        name="healthy_high_utilization",
        root_cause="none",
        recovery_action="none",
        workload_impact="none",
        supported_severities=("low",),
    ),
}

REQUIRED_GPU_FIELDS: frozenset[str] = frozenset(
    {
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
    }
)

REQUIRED_WORKLOAD_FIELDS: frozenset[str] = frozenset(
    {
        "job_type",
        "workload_status",
        "checkpoint_available",
        "training_throughput",
        "tokens_per_second",
        "step_duration_ms",
        "training_loss",
        "inference_latency_ms",
        "error_rate_percent",
        "restart_count",
    }
)


def available_scenarios() -> tuple[str, ...]:
    """Return scenario names in deterministic alphabetical order."""

    return tuple(sorted(SCENARIO_METADATA))


def _require_fields(
    values: Mapping[str, Any],
    required_fields: frozenset[str],
    object_name: str,
) -> None:
    missing = sorted(required_fields.difference(values))

    if missing:
        missing_text = ", ".join(missing)
        raise ScenarioError(
            f"{object_name} is missing required fields: {missing_text}"
        )


def _validate_numeric_fields(
    values: Mapping[str, Any],
    fields: frozenset[str],
    object_name: str,
) -> None:
    for field_name in fields:
        value = values[field_name]

        if field_name in {"job_type", "workload_status"}:
            continue

        if field_name == "checkpoint_available":
            if not isinstance(value, bool):
                raise ScenarioError(
                    f"{object_name}.{field_name} must be boolean"
                )
            continue

        if value is None:
            continue

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise ScenarioError(
                f"{object_name}.{field_name} must be numeric or null"
            )

        if not np.isfinite(float(value)):
            raise ScenarioError(
                f"{object_name}.{field_name} must be finite"
            )


def _validate_progress(progress: float) -> float:
    if isinstance(progress, bool) or not isinstance(
        progress,
        (int, float, np.integer, np.floating),
    ):
        raise ScenarioError("Scenario progress must be numeric")

    numeric_progress = float(progress)

    if not np.isfinite(numeric_progress):
        raise ScenarioError("Scenario progress must be finite")

    if not 0.0 <= numeric_progress <= 1.0:
        raise ScenarioError(
            "Scenario progress must be between 0.0 and 1.0"
        )

    return numeric_progress


def _severity_factor(
    severity: str,
    metadata: ScenarioMetadata,
) -> float:
    normalized_severity = severity.strip().lower()

    if normalized_severity not in SEVERITY_FACTORS:
        available = ", ".join(SEVERITY_FACTORS)
        raise ScenarioError(
            f"Unknown scenario severity '{severity}'. "
            f"Available severities: {available}"
        )

    if normalized_severity not in metadata.supported_severities:
        supported = ", ".join(metadata.supported_severities)
        raise ScenarioError(
            f"Scenario '{metadata.name}' does not support severity "
            f"'{normalized_severity}'. Supported severities: {supported}"
        )

    return SEVERITY_FACTORS[normalized_severity]


def _effective_intensity(
    progress: float,
    severity_factor: float,
) -> float:
    """Return smooth monotonic fault intensity from onset to peak."""

    smooth_progress = progress * progress * (3.0 - (2.0 * progress))
    return float(np.clip(smooth_progress * severity_factor, 0.0, 1.0))


def _clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return float(np.clip(value, minimum, maximum))


def _round_float(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _degrade_positive_metric(
    value: float | int | None,
    reduction_fraction: float,
) -> float | None:
    if value is None:
        return None

    numeric_value = max(0.0, float(value))
    reduction = _clamp(reduction_fraction, 0.0, 0.95)

    return _round_float(numeric_value * (1.0 - reduction))


def _increase_positive_metric(
    value: float | int | None,
    increase_fraction: float,
) -> float | None:
    if value is None:
        return None

    numeric_value = max(0.0, float(value))
    increase = max(0.0, float(increase_fraction))

    return _round_float(numeric_value * (1.0 + increase))


def _mark_degraded_workload(
    workload: dict[str, Any],
    *,
    throughput_reduction: float,
    latency_increase: float,
    step_duration_increase: float,
    error_rate_increase: float,
) -> None:
    workload["workload_status"] = "degraded"

    workload["training_throughput"] = _degrade_positive_metric(
        workload["training_throughput"],
        throughput_reduction,
    )
    workload["tokens_per_second"] = _degrade_positive_metric(
        workload["tokens_per_second"],
        throughput_reduction,
    )
    workload["step_duration_ms"] = _increase_positive_metric(
        workload["step_duration_ms"],
        step_duration_increase,
    )
    workload["inference_latency_ms"] = _increase_positive_metric(
        workload["inference_latency_ms"],
        latency_increase,
    )

    existing_error_rate = float(workload["error_rate_percent"] or 0.0)
    workload["error_rate_percent"] = _round_float(
        _clamp(
            existing_error_rate + error_rate_increase,
            0.0,
            100.0,
        )
    )


def _apply_thermal_throttling(
    gpu: dict[str, Any],
    workload: dict[str, Any],
    intensity: float,
    rng: np.random.Generator,
) -> None:
    if intensity <= 0.0:
        return

    gpu["gpu_temperature_c"] = _round_float(
        _clamp(
            float(gpu["gpu_temperature_c"])
            + (14.0 * intensity)
            + rng.normal(0.0, 0.5 * intensity),
            0.0,
            96.0,
        )
    )
    gpu["memory_temperature_c"] = _round_float(
        _clamp(
            float(gpu["memory_temperature_c"])
            + (12.0 * intensity)
            + rng.normal(0.0, 0.5 * intensity),
            0.0,
            101.0,
        )
    )
    gpu["power_draw_watts"] = _round_float(
        _clamp(
            float(gpu["power_draw_watts"]) + (20.0 * intensity),
            0.0,
            350.0,
        )
    )
    gpu["sm_clock_mhz"] = _round_float(
        _clamp(
            float(gpu["sm_clock_mhz"]) * (1.0 - (0.38 * intensity)),
            300.0,
            1800.0,
        )
    )
    gpu["memory_clock_mhz"] = _round_float(
        _clamp(
            float(gpu["memory_clock_mhz"]) * (1.0 - (0.18 * intensity)),
            600.0,
            2000.0,
        )
    )

    _mark_degraded_workload(
        workload,
        throughput_reduction=0.40 * intensity,
        latency_increase=0.45 * intensity,
        step_duration_increase=0.55 * intensity,
        error_rate_increase=2.0 * intensity,
    )


def _apply_ecc_memory_degradation(
    gpu: dict[str, Any],
    workload: dict[str, Any],
    intensity: float,
    rng: np.random.Generator,
) -> None:
    if intensity <= 0.0:
        return

    gpu["ecc_correctable_errors"] = max(
        int(gpu["ecc_correctable_errors"]),
        int(np.ceil(35.0 * intensity)),
    )

    if intensity >= 0.72:
        gpu["ecc_uncorrectable_errors"] = max(
            1,
            int(gpu["ecc_uncorrectable_errors"]),
        )
        gpu["xid_error_code"] = 48
    elif intensity >= 0.42:
        gpu["xid_error_code"] = 48

    _mark_degraded_workload(
        workload,
        throughput_reduction=0.22 * intensity,
        latency_increase=0.25 * intensity,
        step_duration_increase=0.30 * intensity,
        error_rate_increase=12.0 * intensity,
    )

    if intensity >= 0.75:
        workload["restart_count"] = max(
            1,
            int(workload["restart_count"]) + 1,
        )


def _apply_memory_saturation(
    gpu: dict[str, Any],
    workload: dict[str, Any],
    intensity: float,
    rng: np.random.Generator,
) -> None:
    if intensity <= 0.0:
        return

    baseline_memory = float(gpu["memory_utilization_percent"])
    target_memory = baseline_memory + ((100.0 - baseline_memory) * intensity)

    gpu["memory_utilization_percent"] = _round_float(
        _clamp(
            target_memory + rng.normal(0.0, 0.6 * intensity),
            0.0,
            100.0,
        )
    )
    gpu["memory_temperature_c"] = _round_float(
        _clamp(
            float(gpu["memory_temperature_c"]) + (7.0 * intensity),
            0.0,
            96.0,
        )
    )

    _mark_degraded_workload(
        workload,
        throughput_reduction=0.28 * intensity,
        latency_increase=0.35 * intensity,
        step_duration_increase=0.65 * intensity,
        error_rate_increase=8.0 * intensity,
    )

    if intensity >= 0.75:
        workload["restart_count"] = max(
            1,
            int(workload["restart_count"]) + 1,
        )


def _apply_power_capping(
    gpu: dict[str, Any],
    workload: dict[str, Any],
    intensity: float,
    rng: np.random.Generator,
) -> None:
    if intensity <= 0.0:
        return

    oscillation = np.sin(intensity * np.pi * 8.0)
    oscillation += rng.normal(0.0, 0.12 * intensity)
    baseline_power = float(gpu["power_draw_watts"])
    target_power = baseline_power + ((345.0 - baseline_power) * intensity)

    gpu["power_draw_watts"] = _round_float(
        _clamp(target_power, 0.0, 350.0)
    )
    gpu["sm_clock_mhz"] = _round_float(
        _clamp(
            float(gpu["sm_clock_mhz"])
            * (1.0 - (0.20 * intensity))
            * (1.0 + (0.045 * oscillation)),
            500.0,
            1800.0,
        )
    )

    _mark_degraded_workload(
        workload,
        throughput_reduction=0.25 * intensity,
        latency_increase=0.20 * intensity,
        step_duration_increase=0.30 * intensity,
        error_rate_increase=1.0 * intensity,
    )


def _apply_nvlink_degradation(
    gpu: dict[str, Any],
    workload: dict[str, Any],
    intensity: float,
    rng: np.random.Generator,
) -> None:
    if intensity <= 0.0:
        return

    throughput_reduction = 0.90 * intensity

    gpu["nvlink_tx_mbps"] = _round_float(
        _clamp(
            float(gpu["nvlink_tx_mbps"])
            * (1.0 - throughput_reduction),
            100.0,
            50000.0,
        )
    )
    gpu["nvlink_rx_mbps"] = _round_float(
        _clamp(
            float(gpu["nvlink_rx_mbps"])
            * (1.0 - throughput_reduction)
            * rng.uniform(1.0 - (0.12 * intensity), 1.0),
            100.0,
            50000.0,
        )
    )
    gpu["gpu_utilization_percent"] = _round_float(
        _clamp(
            float(gpu["gpu_utilization_percent"])
            - (20.0 * intensity)
            + rng.normal(0.0, 5.0 * intensity),
            0.0,
            100.0,
        )
    )

    _mark_degraded_workload(
        workload,
        throughput_reduction=0.38 * intensity,
        latency_increase=0.55 * intensity,
        step_duration_increase=0.85 * intensity,
        error_rate_increase=4.0 * intensity,
    )


def _apply_idle_gpu_waste(
    gpu: dict[str, Any],
    workload: dict[str, Any],
    intensity: float,
    rng: np.random.Generator,
) -> None:
    if intensity <= 0.0:
        return

    baseline_utilization = float(gpu["gpu_utilization_percent"])
    utilization_reduction = min(0.98, 1.20 * intensity)
    gpu["gpu_utilization_percent"] = _round_float(
        _clamp(
            baseline_utilization * (1.0 - utilization_reduction)
            + rng.normal(0.0, 0.8 * intensity),
            0.0,
            100.0,
        )
    )

    baseline_memory = float(gpu["memory_utilization_percent"])
    retained_memory = max(20.0, baseline_memory * 0.90)
    gpu["memory_utilization_percent"] = _round_float(
        _clamp(
            baseline_memory
            + ((retained_memory - baseline_memory) * intensity),
            0.0,
            100.0,
        )
    )

    baseline_power = float(gpu["power_draw_watts"])
    target_power = 65.0
    gpu["power_draw_watts"] = _round_float(
        _clamp(
            baseline_power + ((target_power - baseline_power) * intensity)
            + rng.normal(0.0, 2.0 * intensity),
            45.0,
            350.0,
        )
    )

    baseline_clock = float(gpu["sm_clock_mhz"])
    target_clock = 850.0
    gpu["sm_clock_mhz"] = _round_float(
        _clamp(
            baseline_clock + ((target_clock - baseline_clock) * intensity)
            + rng.normal(0.0, 30.0 * intensity),
            500.0,
            1800.0,
        )
    )

    workload["workload_status"] = "degraded"
    workload["training_throughput"] = _degrade_positive_metric(
        workload["training_throughput"],
        0.98 * intensity,
    )
    workload["tokens_per_second"] = _degrade_positive_metric(
        workload["tokens_per_second"],
        0.98 * intensity,
    )
    workload["step_duration_ms"] = _increase_positive_metric(
        workload["step_duration_ms"],
        2.50 * intensity,
    )
    workload["inference_latency_ms"] = _increase_positive_metric(
        workload["inference_latency_ms"],
        1.50 * intensity,
    )


def _apply_healthy_high_utilization(
    gpu: dict[str, Any],
    workload: dict[str, Any],
    intensity: float,
    rng: np.random.Generator,
) -> None:
    if intensity <= 0.0:
        return

    baseline_utilization = float(gpu["gpu_utilization_percent"])
    target_utilization = 97.0
    gpu["gpu_utilization_percent"] = _round_float(
        _clamp(
            baseline_utilization
            + ((target_utilization - baseline_utilization) * intensity)
            + rng.normal(0.0, 1.0 * intensity),
            0.0,
            99.0,
        )
    )

    baseline_memory = float(gpu["memory_utilization_percent"])
    target_memory = 90.0
    gpu["memory_utilization_percent"] = _round_float(
        _clamp(
            baseline_memory + ((target_memory - baseline_memory) * intensity),
            0.0,
            95.0,
        )
    )

    baseline_temperature = float(gpu["gpu_temperature_c"])
    target_temperature = 78.0
    gpu["gpu_temperature_c"] = _round_float(
        _clamp(
            baseline_temperature
            + ((target_temperature - baseline_temperature) * intensity)
            + rng.normal(0.0, 0.8 * intensity),
            35.0,
            82.0,
        )
    )
    gpu["memory_temperature_c"] = _round_float(
        _clamp(
            float(gpu["gpu_temperature_c"])
            + 4.0
            + rng.normal(0.0, 0.5 * intensity),
            38.0,
            88.0,
        )
    )

    baseline_power = float(gpu["power_draw_watts"])
    gpu["power_draw_watts"] = _round_float(
        _clamp(
            baseline_power + ((320.0 - baseline_power) * intensity)
            + rng.normal(0.0, 4.0 * intensity),
            70.0,
            330.0,
        )
    )

    baseline_clock = float(gpu["sm_clock_mhz"])
    gpu["sm_clock_mhz"] = _round_float(
        _clamp(
            baseline_clock + ((1750.0 - baseline_clock) * intensity)
            + rng.normal(0.0, 20.0 * intensity),
            900.0,
            1800.0,
        )
    )
    gpu["ecc_correctable_errors"] = 0
    gpu["ecc_uncorrectable_errors"] = 0
    gpu["xid_error_code"] = 0

    workload["workload_status"] = "running"
    workload["training_throughput"] = _increase_positive_metric(
        workload["training_throughput"],
        0.20 * intensity,
    )
    workload["tokens_per_second"] = _increase_positive_metric(
        workload["tokens_per_second"],
        0.20 * intensity,
    )

    if workload["step_duration_ms"] is not None:
        workload["step_duration_ms"] = _degrade_positive_metric(
            workload["step_duration_ms"],
            0.12 * intensity,
        )

    if workload["inference_latency_ms"] is not None:
        workload["inference_latency_ms"] = _degrade_positive_metric(
            workload["inference_latency_ms"],
            0.10 * intensity,
        )

    workload["error_rate_percent"] = _round_float(
        min(float(workload["error_rate_percent"] or 0.0), 0.5)
    )


ScenarioHandler = Callable[
    [
        dict[str, Any],
        dict[str, Any],
        float,
        np.random.Generator,
    ],
    None,
]

SCENARIO_HANDLERS: dict[str, ScenarioHandler] = {
    "thermal_throttling": _apply_thermal_throttling,
    "ecc_memory_degradation": _apply_ecc_memory_degradation,
    "memory_saturation": _apply_memory_saturation,
    "power_capping": _apply_power_capping,
    "nvlink_degradation": _apply_nvlink_degradation,
    "idle_gpu_waste": _apply_idle_gpu_waste,
    "healthy_high_utilization": _apply_healthy_high_utilization,
}


def apply_scenario(
    scenario_name: str,
    gpu_telemetry: Mapping[str, Any],
    workload: Mapping[str, Any],
    *,
    progress: float,
    severity: str,
    rng: np.random.Generator,
) -> ScenarioResult:
    """
    Apply one scenario to copies of GPU and workload records.

    The input mappings are never modified. Progress must be between zero and
    one and represents the position within the incident window.
    """

    normalized_name = scenario_name.strip().lower()

    if normalized_name not in SCENARIO_HANDLERS:
        available = ", ".join(available_scenarios())
        raise ScenarioError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available scenarios: {available}"
        )

    if not isinstance(rng, np.random.Generator):
        raise ScenarioError(
            "rng must be an instance of numpy.random.Generator"
        )

    _require_fields(
        gpu_telemetry,
        REQUIRED_GPU_FIELDS,
        "gpu_telemetry",
    )
    _require_fields(
        workload,
        REQUIRED_WORKLOAD_FIELDS,
        "workload",
    )
    _validate_numeric_fields(
        gpu_telemetry,
        REQUIRED_GPU_FIELDS,
        "gpu_telemetry",
    )
    _validate_numeric_fields(
        workload,
        REQUIRED_WORKLOAD_FIELDS,
        "workload",
    )

    validated_progress = _validate_progress(progress)
    metadata = SCENARIO_METADATA[normalized_name]
    severity_factor = _severity_factor(severity, metadata)
    intensity = _effective_intensity(
        validated_progress,
        severity_factor,
    )

    transformed_gpu = dict(gpu_telemetry)
    transformed_workload = dict(workload)

    handler = SCENARIO_HANDLERS[normalized_name]
    handler(
        transformed_gpu,
        transformed_workload,
        intensity,
        rng,
    )

    transformed_gpu["gpu_utilization_percent"] = _round_float(
        _clamp(
            float(transformed_gpu["gpu_utilization_percent"]),
            0.0,
            100.0,
        )
    )
    transformed_gpu["memory_utilization_percent"] = _round_float(
        _clamp(
            float(transformed_gpu["memory_utilization_percent"]),
            0.0,
            100.0,
        )
    )
    transformed_gpu["ecc_correctable_errors"] = max(
        0,
        int(transformed_gpu["ecc_correctable_errors"]),
    )
    transformed_gpu["ecc_uncorrectable_errors"] = max(
        0,
        int(transformed_gpu["ecc_uncorrectable_errors"]),
    )
    transformed_gpu["xid_error_code"] = max(
        0,
        int(transformed_gpu["xid_error_code"]),
    )
    transformed_workload["restart_count"] = max(
        0,
        int(transformed_workload["restart_count"]),
    )
    transformed_workload["error_rate_percent"] = _round_float(
        _clamp(
            float(transformed_workload["error_rate_percent"] or 0.0),
            0.0,
            100.0,
        )
    )

    return ScenarioResult(
        gpu_telemetry=transformed_gpu,
        workload=transformed_workload,
        metadata=metadata,
    )