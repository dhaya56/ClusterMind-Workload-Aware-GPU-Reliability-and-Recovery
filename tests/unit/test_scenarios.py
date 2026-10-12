from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from simulator.scenarios import (
    ScenarioError,
    apply_scenario,
    available_scenarios,
)


def make_gpu_record() -> dict[str, float | int]:
    return {
        "gpu_utilization_percent": 88.0,
        "memory_utilization_percent": 72.0,
        "gpu_temperature_c": 70.0,
        "memory_temperature_c": 74.0,
        "power_draw_watts": 285.0,
        "sm_clock_mhz": 1700.0,
        "memory_clock_mhz": 1900.0,
        "ecc_correctable_errors": 0,
        "ecc_uncorrectable_errors": 0,
        "xid_error_code": 0,
        "nvlink_tx_mbps": 32000.0,
        "nvlink_rx_mbps": 31500.0,
    }


def make_training_workload() -> dict[str, str | bool | float | int | None]:
    return {
        "job_type": "llm_training",
        "workload_status": "running",
        "checkpoint_available": True,
        "training_throughput": 1800.0,
        "tokens_per_second": 4500.0,
        "step_duration_ms": 300.0,
        "training_loss": 1.75,
        "inference_latency_ms": None,
        "error_rate_percent": 0.0,
        "restart_count": 0,
    }


def make_inference_workload() -> dict[str, str | bool | float | int | None]:
    return {
        "job_type": "batch_inference",
        "workload_status": "running",
        "checkpoint_available": False,
        "training_throughput": None,
        "tokens_per_second": 6200.0,
        "step_duration_ms": None,
        "training_loss": None,
        "inference_latency_ms": 42.0,
        "error_rate_percent": 0.2,
        "restart_count": 0,
    }


def assert_gpu_output_bounds(gpu: dict[str, object]) -> None:
    assert 0.0 <= float(gpu["gpu_utilization_percent"]) <= 100.0
    assert 0.0 <= float(gpu["memory_utilization_percent"]) <= 100.0
    assert float(gpu["gpu_temperature_c"]) >= 0.0
    assert float(gpu["memory_temperature_c"]) >= 0.0
    assert float(gpu["power_draw_watts"]) >= 0.0
    assert float(gpu["sm_clock_mhz"]) >= 0.0
    assert float(gpu["memory_clock_mhz"]) >= 0.0
    assert int(gpu["ecc_correctable_errors"]) >= 0
    assert int(gpu["ecc_uncorrectable_errors"]) >= 0
    assert int(gpu["xid_error_code"]) >= 0
    assert float(gpu["nvlink_tx_mbps"]) >= 0.0
    assert float(gpu["nvlink_rx_mbps"]) >= 0.0


def assert_workload_output_bounds(workload: dict[str, object]) -> None:
    assert 0.0 <= float(workload["error_rate_percent"]) <= 100.0
    assert int(workload["restart_count"]) >= 0

    for field_name in (
        "training_throughput",
        "tokens_per_second",
        "step_duration_ms",
        "training_loss",
        "inference_latency_ms",
    ):
        value = workload[field_name]

        if value is not None:
            assert float(value) >= 0.0


def test_available_scenarios_are_complete_and_sorted() -> None:
    expected = (
        "ecc_memory_degradation",
        "healthy_high_utilization",
        "idle_gpu_waste",
        "memory_saturation",
        "nvlink_degradation",
        "power_capping",
        "thermal_throttling",
    )

    assert available_scenarios() == expected


@pytest.mark.parametrize(
    ("scenario_name", "severity"),
    [
        ("thermal_throttling", "high"),
        ("ecc_memory_degradation", "critical"),
        ("memory_saturation", "high"),
        ("power_capping", "high"),
        ("nvlink_degradation", "high"),
        ("idle_gpu_waste", "high"),
        ("healthy_high_utilization", "low"),
    ],
)
def test_all_scenarios_return_bounded_outputs(
    scenario_name: str,
    severity: str,
) -> None:
    result = apply_scenario(
        scenario_name,
        make_gpu_record(),
        make_training_workload(),
        progress=1.0,
        severity=severity,
        rng=np.random.default_rng(42),
    )

    assert result.metadata.name == scenario_name
    assert_gpu_output_bounds(result.gpu_telemetry)
    assert_workload_output_bounds(result.workload)


def test_thermal_throttling_has_correlated_effects() -> None:
    gpu = make_gpu_record()
    workload = make_training_workload()

    result = apply_scenario(
        "thermal_throttling",
        gpu,
        workload,
        progress=1.0,
        severity="critical",
        rng=np.random.default_rng(42),
    )

    assert result.gpu_telemetry["gpu_temperature_c"] > gpu["gpu_temperature_c"]
    assert (
        result.gpu_telemetry["memory_temperature_c"]
        > gpu["memory_temperature_c"]
    )
    assert result.gpu_telemetry["sm_clock_mhz"] < gpu["sm_clock_mhz"]
    assert result.workload["training_throughput"] < workload["training_throughput"]
    assert result.workload["step_duration_ms"] > workload["step_duration_ms"]
    assert result.workload["workload_status"] == "degraded"


def test_ecc_memory_degradation_emits_hardware_errors() -> None:
    result = apply_scenario(
        "ecc_memory_degradation",
        make_gpu_record(),
        make_training_workload(),
        progress=1.0,
        severity="critical",
        rng=np.random.default_rng(42),
    )

    assert result.gpu_telemetry["ecc_correctable_errors"] > 0
    assert result.gpu_telemetry["ecc_uncorrectable_errors"] > 0
    assert result.gpu_telemetry["xid_error_code"] == 48
    assert result.workload["restart_count"] > 0


def test_memory_saturation_raises_memory_pressure() -> None:
    gpu = make_gpu_record()
    workload = make_training_workload()

    result = apply_scenario(
        "memory_saturation",
        gpu,
        workload,
        progress=1.0,
        severity="critical",
        rng=np.random.default_rng(42),
    )

    assert result.gpu_telemetry["memory_utilization_percent"] >= 99.0
    assert result.workload["step_duration_ms"] > workload["step_duration_ms"]
    assert result.workload["restart_count"] > 0


def test_power_capping_preserves_hardware_error_counters() -> None:
    gpu = make_gpu_record()

    result = apply_scenario(
        "power_capping",
        gpu,
        make_training_workload(),
        progress=1.0,
        severity="high",
        rng=np.random.default_rng(42),
    )

    assert result.gpu_telemetry["power_draw_watts"] >= gpu["power_draw_watts"]
    assert result.gpu_telemetry["sm_clock_mhz"] < gpu["sm_clock_mhz"]
    assert result.gpu_telemetry["ecc_correctable_errors"] == 0
    assert result.gpu_telemetry["ecc_uncorrectable_errors"] == 0
    assert result.gpu_telemetry["xid_error_code"] == 0


def test_nvlink_degradation_reduces_link_throughput() -> None:
    gpu = make_gpu_record()
    workload = make_training_workload()

    result = apply_scenario(
        "nvlink_degradation",
        gpu,
        workload,
        progress=1.0,
        severity="critical",
        rng=np.random.default_rng(42),
    )

    assert result.gpu_telemetry["nvlink_tx_mbps"] < gpu["nvlink_tx_mbps"]
    assert result.gpu_telemetry["nvlink_rx_mbps"] < gpu["nvlink_rx_mbps"]
    assert result.workload["step_duration_ms"] > workload["step_duration_ms"]


def test_idle_gpu_waste_retains_memory_allocation() -> None:
    gpu = make_gpu_record()
    workload = make_training_workload()

    result = apply_scenario(
        "idle_gpu_waste",
        gpu,
        workload,
        progress=1.0,
        severity="high",
        rng=np.random.default_rng(42),
    )

    assert result.gpu_telemetry["gpu_utilization_percent"] <= 10.0
    assert result.gpu_telemetry["memory_utilization_percent"] >= 20.0
    assert result.workload["training_throughput"] < workload["training_throughput"]


def test_healthy_high_utilization_improves_useful_work() -> None:
    gpu = make_gpu_record()
    workload = make_training_workload()

    result = apply_scenario(
        "healthy_high_utilization",
        gpu,
        workload,
        progress=1.0,
        severity="low",
        rng=np.random.default_rng(42),
    )

    assert result.gpu_telemetry["gpu_utilization_percent"] >= 88.0
    assert result.gpu_telemetry["gpu_temperature_c"] <= 82.0
    assert result.gpu_telemetry["ecc_correctable_errors"] == 0
    assert result.gpu_telemetry["ecc_uncorrectable_errors"] == 0
    assert result.gpu_telemetry["xid_error_code"] == 0
    assert result.workload["training_throughput"] > workload["training_throughput"]
    assert result.workload["workload_status"] == "running"


def test_inference_workload_handles_null_training_metrics() -> None:
    result = apply_scenario(
        "thermal_throttling",
        make_gpu_record(),
        make_inference_workload(),
        progress=1.0,
        severity="high",
        rng=np.random.default_rng(42),
    )

    assert result.workload["training_throughput"] is None
    assert result.workload["step_duration_ms"] is None
    assert result.workload["inference_latency_ms"] > 42.0
    assert result.workload["tokens_per_second"] < 6200.0


def test_scenario_application_does_not_mutate_inputs() -> None:
    gpu = make_gpu_record()
    workload = make_training_workload()
    original_gpu = deepcopy(gpu)
    original_workload = deepcopy(workload)

    apply_scenario(
        "thermal_throttling",
        gpu,
        workload,
        progress=1.0,
        severity="high",
        rng=np.random.default_rng(42),
    )

    assert gpu == original_gpu
    assert workload == original_workload


def test_scenario_is_deterministic_with_same_seed() -> None:
    first = apply_scenario(
        "ecc_memory_degradation",
        make_gpu_record(),
        make_training_workload(),
        progress=0.8,
        severity="high",
        rng=np.random.default_rng(42),
    )
    second = apply_scenario(
        "ecc_memory_degradation",
        make_gpu_record(),
        make_training_workload(),
        progress=0.8,
        severity="high",
        rng=np.random.default_rng(42),
    )

    assert first == second


@pytest.mark.parametrize("invalid_progress", [-0.01, 1.01, float("inf")])
def test_invalid_progress_is_rejected(invalid_progress: float) -> None:
    with pytest.raises(ScenarioError):
        apply_scenario(
            "thermal_throttling",
            make_gpu_record(),
            make_training_workload(),
            progress=invalid_progress,
            severity="high",
            rng=np.random.default_rng(42),
        )


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(ScenarioError, match="Unknown scenario"):
        apply_scenario(
            "unknown_scenario",
            make_gpu_record(),
            make_training_workload(),
            progress=0.5,
            severity="high",
            rng=np.random.default_rng(42),
        )


def test_unsupported_severity_is_rejected() -> None:
    with pytest.raises(ScenarioError, match="does not support severity"):
        apply_scenario(
            "thermal_throttling",
            make_gpu_record(),
            make_training_workload(),
            progress=0.5,
            severity="low",
            rng=np.random.default_rng(42),
        )


def test_missing_gpu_field_is_rejected() -> None:
    gpu = make_gpu_record()
    gpu.pop("gpu_temperature_c")

    with pytest.raises(ScenarioError, match="missing required fields"):
        apply_scenario(
            "thermal_throttling",
            gpu,
            make_training_workload(),
            progress=0.5,
            severity="high",
            rng=np.random.default_rng(42),
        )


def test_non_finite_numeric_value_is_rejected() -> None:
    gpu = make_gpu_record()
    gpu["power_draw_watts"] = float("nan")

    with pytest.raises(ScenarioError, match="must be finite"):
        apply_scenario(
            "power_capping",
            gpu,
            make_training_workload(),
            progress=0.5,
            severity="medium",
            rng=np.random.default_rng(42),
        )


def test_invalid_rng_is_rejected() -> None:
    with pytest.raises(ScenarioError, match="numpy.random.Generator"):
        apply_scenario(
            "thermal_throttling",
            make_gpu_record(),
            make_training_workload(),
            progress=0.5,
            severity="high",
            rng=None,
        )