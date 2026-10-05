from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import pytest

from simulator.generator import DatasetGenerator, GenerationError, ScenarioWindow


def test_development_summary_has_expected_counts() -> None:
    generator = DatasetGenerator(profile_name="development")
    summary = generator.summary()

    assert summary["profile"] == "development"
    assert summary["timestamp_count"] == 2016
    assert summary["node_count"] == 10
    assert summary["gpus_per_node"] == 4
    assert summary["expected_gpu_rows"] == 80640
    assert summary["expected_node_rows"] == 20160
    assert summary["expected_workload_rows"] == 20160
    assert summary["incident_count"] == 35
    assert summary["healthy_control_count"] == 10
    assert summary["random_seed"] == 42
    assert len(summary["configuration_hash"]) == 64


def test_production_demo_summary_has_expected_counts() -> None:
    generator = DatasetGenerator(profile_name="production_demo")
    summary = generator.summary()

    assert summary["timestamp_count"] == 43200
    assert summary["expected_gpu_rows"] == 1728000
    assert summary["expected_node_rows"] == 432000
    assert summary["expected_workload_rows"] == 432000
    assert summary["incident_count"] == 150
    assert summary["healthy_control_count"] == 40


def test_default_time_range_is_exactly_seven_development_days() -> None:
    generator = DatasetGenerator(profile_name="development")
    summary = generator.summary()

    assert summary["start_timestamp"] == "2026-01-01T00:00:00Z"
    assert summary["end_timestamp"] == "2026-01-07T23:55:00Z"


def test_node_and_gpu_identifiers_are_complete_and_unique() -> None:
    generator = DatasetGenerator(profile_name="development")

    assert len(generator.node_ids) == 10
    assert len(set(generator.node_ids)) == 10
    assert generator.node_ids[0] == "gpu-node-01"
    assert generator.node_ids[-1] == "gpu-node-10"

    assert len(generator.gpu_ids) == 4
    assert len(set(generator.gpu_ids)) == 4
    assert generator.gpu_ids == ("gpu-00", "gpu-01", "gpu-02", "gpu-03")


def test_one_workload_is_assigned_to_each_node() -> None:
    generator = DatasetGenerator(profile_name="development")

    assert len(generator.workloads) == len(generator.node_ids)
    assert {item.assigned_node for item in generator.workloads} == set(
        generator.node_ids
    )
    assert len({item.job_id for item in generator.workloads}) == 10

    for workload in generator.workloads:
        assert 1 <= workload.allocated_gpus <= generator.config.gpus_per_node


def test_generation_is_deterministic_for_same_configuration() -> None:
    first = DatasetGenerator(profile_name="development")
    second = DatasetGenerator(profile_name="development")

    assert first.workloads == second.workloads
    assert first.windows == second.windows
    assert first.summary() == second.summary()

    first_chunk = next(first.iter_chunks(chunk_timestamp_count=2))
    second_chunk = next(second.iter_chunks(chunk_timestamp_count=2))
    assert first_chunk == second_chunk


def test_scheduled_window_counts_and_identifiers_are_correct() -> None:
    generator = DatasetGenerator(profile_name="development")
    incident_windows = [
        window for window in generator.windows if not window.is_healthy_control
    ]
    control_windows = [
        window for window in generator.windows if window.is_healthy_control
    ]

    assert len(incident_windows) == 35
    assert len(control_windows) == 10
    assert len({window.window_id for window in generator.windows}) == 45
    assert all(window.window_id.startswith("INC-2026-") for window in incident_windows)
    assert all(window.window_id.startswith("CTRL-2026-") for window in control_windows)
    assert all(
        window.scenario_name == "healthy_high_utilization"
        for window in control_windows
    )


def test_windows_do_not_overlap_on_the_same_node() -> None:
    generator = DatasetGenerator(profile_name="development")
    windows_by_node: dict[str, list[ScenarioWindow]] = defaultdict(list)

    for window in generator.windows:
        windows_by_node[window.node_id].append(window)

    for windows in windows_by_node.values():
        ordered = sorted(windows, key=lambda item: item.start_index)
        for previous, current in zip(ordered, ordered[1:]):
            assert previous.end_index < current.start_index


def test_window_targets_reference_known_entities() -> None:
    generator = DatasetGenerator(profile_name="development")
    workload_ids = {item.job_id for item in generator.workloads}

    for window in generator.windows:
        assert window.node_id in generator.node_ids
        assert window.gpu_id in generator.gpu_ids
        assert window.workload_id in workload_ids
        assert 0 <= window.start_index <= window.end_index
        assert window.end_index < generator.timestamp_count


def test_window_progress_is_bounded_and_peaks_near_middle() -> None:
    window = ScenarioWindow(
        window_id="INC-TEST-0001",
        scenario_name="thermal_throttling",
        severity="high",
        node_id="gpu-node-01",
        gpu_id="gpu-00",
        workload_id="job-0001",
        start_index=10,
        end_index=20,
    )

    assert window.progress(10) == pytest.approx(0.0)
    assert window.progress(15) == pytest.approx(1.0)
    assert window.progress(20) == pytest.approx(0.0)

    with pytest.raises(GenerationError, match="outside"):
        window.progress(9)


def test_small_chunk_has_expected_row_counts_and_schemas() -> None:
    generator = DatasetGenerator(profile_name="development")
    chunk = next(generator.iter_chunks(chunk_timestamp_count=2))

    assert len(chunk["gpu_telemetry"]) == 80
    assert len(chunk["node_telemetry"]) == 20
    assert len(chunk["workloads"]) == 20

    expected_gpu_fields = set(generator.config.dataset["gpu_telemetry"]["fields"])
    expected_node_fields = set(generator.config.dataset["node_telemetry"]["fields"])
    expected_workload_fields = set(generator.config.dataset["workloads"]["fields"])

    assert set(chunk["gpu_telemetry"][0]) == expected_gpu_fields
    assert set(chunk["node_telemetry"][0]) == expected_node_fields
    assert set(chunk["workloads"][0]) == expected_workload_fields


def test_first_chunk_timestamps_are_ordered_and_utc() -> None:
    generator = DatasetGenerator(profile_name="development")
    chunk = next(generator.iter_chunks(chunk_timestamp_count=2))

    gpu_timestamps = sorted({row["timestamp"] for row in chunk["gpu_telemetry"]})
    assert gpu_timestamps == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:05:00Z",
    ]

    for timestamp in gpu_timestamps:
        parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
        assert parsed.year == 2026


def test_generated_gpu_and_node_values_are_bounded() -> None:
    generator = DatasetGenerator(profile_name="development")
    chunk = next(generator.iter_chunks(chunk_timestamp_count=3))

    for row in chunk["gpu_telemetry"]:
        assert 0.0 <= row["gpu_utilization_percent"] <= 100.0
        assert 0.0 <= row["memory_utilization_percent"] <= 100.0
        assert row["gpu_temperature_c"] >= 0.0
        assert row["memory_temperature_c"] >= 0.0
        assert row["power_draw_watts"] >= 0.0
        assert row["sm_clock_mhz"] >= 0.0
        assert row["memory_clock_mhz"] >= 0.0
        assert row["ecc_correctable_errors"] >= 0
        assert row["ecc_uncorrectable_errors"] >= 0
        assert row["xid_error_code"] >= 0
        assert row["nvlink_tx_mbps"] >= 0.0
        assert row["nvlink_rx_mbps"] >= 0.0
        assert row["operating_state"] in {"normal", "degraded", "failure"}

    for row in chunk["node_telemetry"]:
        assert 0.0 <= row["cpu_utilization_percent"] <= 100.0
        assert 0.0 <= row["host_memory_percent"] <= 100.0
        assert 0.0 <= row["disk_io_percent"] <= 100.0
        assert 0.0 <= row["fan_speed_percent"] <= 100.0
        assert row["network_tx_mbps"] >= 0.0
        assert row["network_rx_mbps"] >= 0.0
        assert 0 <= row["healthy_gpu_count"] <= generator.config.gpus_per_node
        assert row["node_health_status"] in {"healthy", "degraded", "unhealthy"}


def test_incident_records_match_schema_and_are_referentially_valid() -> None:
    generator = DatasetGenerator(profile_name="development")
    records = generator.incident_records()
    expected_fields = set(generator.config.dataset["incidents"]["fields"])
    incident_window_ids = {
        window.window_id
        for window in generator.windows
        if not window.is_healthy_control
    }
    workload_ids = {item.job_id for item in generator.workloads}

    assert len(records) == 35
    assert {record["incident_id"] for record in records} == incident_window_ids

    for record in records:
        assert set(record) == expected_fields
        assert record["node_id"] in generator.node_ids
        assert record["gpu_id"] in generator.gpu_ids
        assert record["affected_workload"] in workload_ids
        assert record["gpu_hours_lost"] > 0.0
        assert record["incident_status"] == "resolved"
        assert record["start_timestamp"] < record["end_timestamp"]


def test_agent_evaluation_records_match_incidents() -> None:
    generator = DatasetGenerator(profile_name="development")
    incidents = generator.incident_records()
    evaluations = generator.agent_evaluation_records()
    expected_fields = set(generator.config.dataset["agent_evaluation"]["fields"])

    incident_ids = {record["incident_id"] for record in incidents}
    evaluation_ids = {record["incident_id"] for record in evaluations}

    assert len(evaluations) == 35
    assert evaluation_ids == incident_ids
    assert all(set(record) == expected_fields for record in evaluations)


def test_incident_window_is_visible_in_generated_telemetry() -> None:
    generator = DatasetGenerator(profile_name="development")
    target = next(
        window for window in generator.windows if not window.is_healthy_control
    )
    gpu_rows, node_rows, workload_rows = generator._generate_timestamp_records(
        target.start_index
    )

    matching_gpu_rows = [
        row
        for row in gpu_rows
        if row["node_id"] == target.node_id and row["gpu_id"] == target.gpu_id
    ]
    assert len(matching_gpu_rows) == 1
    assert matching_gpu_rows[0]["incident_id"] == target.window_id
    assert matching_gpu_rows[0]["anomaly_label"] == target.scenario_name

    matching_node_rows = [
        row for row in node_rows if row["node_id"] == target.node_id
    ]
    assert len(matching_node_rows) == 1
    assert matching_node_rows[0]["incident_id"] == target.window_id

    matching_workload_rows = [
        row for row in workload_rows if row["job_id"] == target.workload_id
    ]
    assert len(matching_workload_rows) == 1
    assert matching_workload_rows[0]["incident_id"] == target.window_id


def test_healthy_control_is_not_written_as_incident() -> None:
    generator = DatasetGenerator(profile_name="development")
    target = next(window for window in generator.windows if window.is_healthy_control)
    gpu_rows, node_rows, workload_rows = generator._generate_timestamp_records(
        target.start_index
    )

    matching_gpu = next(
        row
        for row in gpu_rows
        if row["node_id"] == target.node_id and row["gpu_id"] == target.gpu_id
    )
    matching_node = next(
        row for row in node_rows if row["node_id"] == target.node_id
    )
    matching_workload = next(
        row for row in workload_rows if row["job_id"] == target.workload_id
    )

    assert matching_gpu["anomaly_label"] == "healthy_high_utilization"
    assert matching_gpu["operating_state"] == "normal"
    assert matching_gpu["incident_id"] == ""
    assert matching_node["incident_id"] == ""
    assert matching_workload["incident_id"] == ""


def test_chunk_iteration_covers_full_profile_without_loading_all_rows() -> None:
    generator = DatasetGenerator(profile_name="development")
    gpu_count = 0
    node_count = 0
    workload_count = 0
    chunk_count = 0

    for chunk in generator.iter_chunks(chunk_timestamp_count=500):
        chunk_count += 1
        gpu_count += len(chunk["gpu_telemetry"])
        node_count += len(chunk["node_telemetry"])
        workload_count += len(chunk["workloads"])

    assert chunk_count == 5
    assert gpu_count == 80640
    assert node_count == 20160
    assert workload_count == 20160


@pytest.mark.parametrize("invalid_size", [0, -1, False])
def test_invalid_chunk_size_is_rejected(invalid_size: int) -> None:
    generator = DatasetGenerator(profile_name="development")

    with pytest.raises(GenerationError, match="must be positive"):
        next(generator.iter_chunks(chunk_timestamp_count=invalid_size))


def test_naive_start_time_is_rejected() -> None:
    with pytest.raises(GenerationError, match="timezone"):
        DatasetGenerator(
            profile_name="development",
            start_time=datetime(2026, 1, 1),
        )


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown generation profile"):
        DatasetGenerator(profile_name="unknown")
