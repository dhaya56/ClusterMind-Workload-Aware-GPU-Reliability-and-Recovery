from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from simulator.config import SimulatorConfig, load_config
from simulator.exporter import DatasetExporter, ExportError
from simulator.generator import DatasetGenerator


def make_test_config(tmp_path: Path) -> SimulatorConfig:
    base = load_config()
    dataset = deepcopy(base.dataset)
    dataset["output"]["directory"] = str(tmp_path / "generated")

    return SimulatorConfig(
        project_root=base.project_root,
        cluster=deepcopy(base.cluster),
        dataset=dataset,
        detection=deepcopy(base.detection),
        configuration_hash=base.configuration_hash,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def completed_export(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("exporter-complete")
    config = make_test_config(root)
    generator = DatasetGenerator(config=config, profile_name="development")
    exporter = DatasetExporter(generator)
    result = exporter.export(chunk_timestamp_count=500)
    return generator, exporter, result


def test_complete_export_creates_all_expected_files(completed_export) -> None:
    _, _, result = completed_export

    expected_names = {
        "gpu_telemetry",
        "node_telemetry",
        "workloads",
        "incidents",
        "agent_evaluation",
    }

    assert {item.dataset_name for item in result.files} == expected_names
    assert result.manifest_path.is_file()
    assert all(item.path.is_file() for item in result.files)
    assert not (result.output_directory / ".export-tmp").exists()


def test_exported_row_counts_match_generator_summary(completed_export) -> None:
    generator, _, result = completed_export
    summary = generator.summary()

    expected_counts = {
        "gpu_telemetry": summary["expected_gpu_rows"],
        "node_telemetry": summary["expected_node_rows"],
        "workloads": summary["expected_workload_rows"],
        "incidents": summary["incident_count"],
        "agent_evaluation": summary["incident_count"],
    }

    assert {
        item.dataset_name: item.row_count for item in result.files
    } == expected_counts


def test_csv_physical_row_counts_include_one_header(completed_export) -> None:
    _, _, result = completed_export

    for item in result.files:
        with item.path.open("r", encoding="utf-8", newline="") as handle:
            physical_rows = sum(1 for _ in handle)
        assert physical_rows == item.row_count + 1


def test_csv_headers_follow_configured_order(completed_export) -> None:
    generator, _, result = completed_export

    for item in result.files:
        expected_fields = generator.config.dataset[item.dataset_name]["fields"]
        with item.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
        assert header == expected_fields


def test_exported_file_metadata_is_correct(completed_export) -> None:
    _, _, result = completed_export

    for item in result.files:
        assert item.size_bytes == item.path.stat().st_size
        assert item.size_bytes > 0
        assert len(item.sha256) == 64
        assert item.sha256 == sha256(item.path)


def test_manifest_contains_integrity_metadata(completed_export) -> None:
    generator, _, result = completed_export
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["dataset_name"] == "clustermind_synthetic_gpu_telemetry"
    assert manifest["dataset_version"] == "1.0.0"
    assert manifest["synthetic"] is True
    assert manifest["profile"] == "development"
    assert manifest["configuration_hash"] == generator.config.configuration_hash
    assert manifest["random_seed"] == 42
    assert manifest["summary"] == generator.summary()
    assert "synthetic" in manifest["statement"].lower()

    for item in result.files:
        entry = manifest["files"][item.dataset_name]
        assert entry["filename"] == item.path.name
        assert entry["row_count"] == item.row_count
        assert entry["size_bytes"] == item.size_bytes
        assert entry["sha256"] == item.sha256


def test_manifest_is_valid_utf8_json(completed_export) -> None:
    _, _, result = completed_export
    raw = result.manifest_path.read_bytes()
    decoded = raw.decode("utf-8")
    parsed = json.loads(decoded)

    assert isinstance(parsed, dict)
    assert decoded.endswith("\n")


def test_export_result_file_lookup(completed_export) -> None:
    _, _, result = completed_export

    gpu_file = result.file("gpu_telemetry")
    assert gpu_file.dataset_name == "gpu_telemetry"
    assert gpu_file.path.name == "gpu_telemetry.csv"

    with pytest.raises(ExportError, match="Unknown exported dataset"):
        result.file("unknown")


def test_overwrite_protection_rejects_existing_outputs(completed_export) -> None:
    generator, _, result = completed_export
    original_manifest = result.manifest_path.read_bytes()
    exporter = DatasetExporter(generator, overwrite=False)

    with pytest.raises(ExportError, match="overwrite is disabled"):
        exporter.export(chunk_timestamp_count=500)

    assert result.manifest_path.read_bytes() == original_manifest
    assert not (result.output_directory / ".export-tmp").exists()


def test_explicit_overwrite_replaces_outputs(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    first_generator = DatasetGenerator(config=config, profile_name="development")
    first = DatasetExporter(first_generator).export(chunk_timestamp_count=1000)
    first_hash = first.file("gpu_telemetry").sha256

    second_generator = DatasetGenerator(config=config, profile_name="development")
    second = DatasetExporter(
        second_generator,
        overwrite=True,
    ).export(chunk_timestamp_count=700)

    assert second.file("gpu_telemetry").sha256 == first_hash
    assert second.manifest_path.is_file()
    assert not (second.output_directory / ".export-tmp").exists()


def test_invalid_chunk_size_is_rejected_before_generation(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    generator = DatasetGenerator(config=config, profile_name="development")
    exporter = DatasetExporter(generator)

    for invalid_size in (0, -1, False, 1.5):
        with pytest.raises(ExportError, match="positive integer"):
            exporter.export(chunk_timestamp_count=invalid_size)


def test_invalid_generator_type_is_rejected() -> None:
    with pytest.raises(ExportError, match="DatasetGenerator"):
        DatasetExporter(object())


def test_schema_mismatch_is_rejected(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    generator = DatasetGenerator(config=config, profile_name="development")
    exporter = DatasetExporter(generator)

    broken_record = {
        field: ""
        for field in config.dataset["gpu_telemetry"]["fields"]
        if field != "gpu_temperature_c"
    }

    with pytest.raises(ExportError, match="Schema mismatch"):
        exporter._validate_record_schema(broken_record, "gpu_telemetry")


def test_row_count_mismatch_is_rejected(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    generator = DatasetGenerator(config=config, profile_name="development")
    exporter = DatasetExporter(generator)

    incorrect_counts = {
        "gpu_telemetry": 1,
        "node_telemetry": 1,
        "workloads": 1,
        "incidents": 1,
        "agent_evaluation": 1,
    }

    with pytest.raises(ExportError, match="Row count mismatch"):
        exporter._validate_row_counts(incorrect_counts)
