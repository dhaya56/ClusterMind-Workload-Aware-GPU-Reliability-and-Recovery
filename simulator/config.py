from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when ClusterMind configuration is missing or inconsistent."""


@dataclass(frozen=True)
class SimulatorConfig:
    """Validated configuration required by the telemetry simulator."""

    project_root: Path
    cluster: dict[str, Any]
    dataset: dict[str, Any]
    detection: dict[str, Any]
    configuration_hash: str

    @property
    def node_count(self) -> int:
        return int(self.cluster["cluster"]["node_count"])

    @property
    def gpus_per_node(self) -> int:
        return int(self.cluster["cluster"]["gpus_per_node"])

    @property
    def total_gpus(self) -> int:
        return int(self.cluster["cluster"]["total_gpus"])

    @property
    def random_seed(self) -> int:
        return int(self.dataset["dataset"]["random_seed"])

    @property
    def output_directory(self) -> Path:
        configured_path = Path(self.dataset["output"]["directory"])
        return (self.project_root / configured_path).resolve()

    def profile(self, profile_name: str) -> dict[str, Any]:
        profiles = self.dataset["generation_profiles"]

        if profile_name not in profiles:
            available = ", ".join(sorted(profiles))
            raise ConfigurationError(
                f"Unknown generation profile '{profile_name}'. "
                f"Available profiles: {available}"
            )

        return profiles[profile_name]

    def output_path(self, dataset_name: str) -> Path:
        output_files = self.dataset["output"]["files"]

        if dataset_name not in output_files:
            available = ", ".join(sorted(output_files))
            raise ConfigurationError(
                f"Unknown dataset output '{dataset_name}'. "
                f"Available outputs: {available}"
            )

        return self.output_directory / output_files[dataset_name]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    if not path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"Unable to read configuration file: {path}"
        ) from exc

    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML syntax in configuration file: {path}"
        ) from exc

    if not isinstance(loaded, dict):
        raise ConfigurationError(
            f"Configuration root must be a mapping: {path}"
        )

    return loaded


def _required_mapping(
    source: dict[str, Any],
    key: str,
    source_name: str,
) -> dict[str, Any]:
    value = source.get(key)

    if not isinstance(value, dict):
        raise ConfigurationError(
            f"'{key}' must be a mapping in {source_name}"
        )

    return value


def _required_positive_int(
    source: dict[str, Any],
    key: str,
    source_name: str,
) -> int:
    value = source.get(key)

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(
            f"'{key}' must be a positive integer in {source_name}"
        )

    return value


def _validate_cluster(cluster_config: dict[str, Any]) -> None:
    cluster = _required_mapping(cluster_config, "cluster", "cluster.yaml")

    node_count = _required_positive_int(
        cluster,
        "node_count",
        "cluster.yaml",
    )
    gpus_per_node = _required_positive_int(
        cluster,
        "gpus_per_node",
        "cluster.yaml",
    )
    total_gpus = _required_positive_int(
        cluster,
        "total_gpus",
        "cluster.yaml",
    )

    calculated_total = node_count * gpus_per_node

    if calculated_total != total_gpus:
        raise ConfigurationError(
            "GPU count mismatch in cluster.yaml: "
            f"{node_count} nodes x {gpus_per_node} GPUs = "
            f"{calculated_total}, but total_gpus is {total_gpus}"
        )

    workloads = _required_mapping(
        cluster_config,
        "workloads",
        "cluster.yaml",
    )

    minimum_jobs = _required_positive_int(
        workloads,
        "minimum_active_jobs",
        "cluster.yaml",
    )
    maximum_jobs = _required_positive_int(
        workloads,
        "maximum_active_jobs",
        "cluster.yaml",
    )

    if minimum_jobs > maximum_jobs:
        raise ConfigurationError(
            "minimum_active_jobs cannot exceed maximum_active_jobs"
        )

    minimum_gpus = _required_positive_int(
        workloads,
        "minimum_gpus_per_job",
        "cluster.yaml",
    )
    maximum_gpus = _required_positive_int(
        workloads,
        "maximum_gpus_per_job",
        "cluster.yaml",
    )

    if minimum_gpus > maximum_gpus:
        raise ConfigurationError(
            "minimum_gpus_per_job cannot exceed maximum_gpus_per_job"
        )

    if maximum_gpus > total_gpus:
        raise ConfigurationError(
            "maximum_gpus_per_job cannot exceed total cluster GPUs"
        )

    checkpoint_probability = workloads.get("checkpoint_probability")

    if not isinstance(checkpoint_probability, (int, float)):
        raise ConfigurationError(
            "checkpoint_probability must be numeric in cluster.yaml"
        )

    if not 0.0 <= float(checkpoint_probability) <= 1.0:
        raise ConfigurationError(
            "checkpoint_probability must be between 0 and 1"
        )


def _expected_gpu_rows(
    node_count: int,
    gpus_per_node: int,
    duration_days: int,
    sampling_interval_minutes: int,
) -> int:
    minutes_per_day = 24 * 60

    if minutes_per_day % sampling_interval_minutes != 0:
        raise ConfigurationError(
            "sampling_interval_minutes must divide evenly into 1440 minutes"
        )

    samples_per_day = minutes_per_day // sampling_interval_minutes

    return (
        node_count
        * gpus_per_node
        * duration_days
        * samples_per_day
    )


def _validate_dataset(
    cluster_config: dict[str, Any],
    dataset_config: dict[str, Any],
) -> None:
    dataset_metadata = _required_mapping(
        dataset_config,
        "dataset",
        "dataset.yaml",
    )

    if dataset_metadata.get("synthetic") is not True:
        raise ConfigurationError(
            "dataset.synthetic must be true for ClusterMind"
        )

    profiles = _required_mapping(
        dataset_config,
        "generation_profiles",
        "dataset.yaml",
    )

    cluster = cluster_config["cluster"]
    node_count = int(cluster["node_count"])
    gpus_per_node = int(cluster["gpus_per_node"])

    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ConfigurationError(
                f"Profile '{profile_name}' must be a mapping"
            )

        duration_days = _required_positive_int(
            profile,
            "duration_days",
            f"dataset.yaml profile '{profile_name}'",
        )
        interval = _required_positive_int(
            profile,
            "sampling_interval_minutes",
            f"dataset.yaml profile '{profile_name}'",
        )
        configured_rows = _required_positive_int(
            profile,
            "expected_gpu_rows",
            f"dataset.yaml profile '{profile_name}'",
        )
        _required_positive_int(
            profile,
            "incident_count",
            f"dataset.yaml profile '{profile_name}'",
        )

        calculated_rows = _expected_gpu_rows(
            node_count=node_count,
            gpus_per_node=gpus_per_node,
            duration_days=duration_days,
            sampling_interval_minutes=interval,
        )

        if configured_rows != calculated_rows:
            raise ConfigurationError(
                f"Profile '{profile_name}' expects {configured_rows} GPU "
                f"rows, but the cluster and sampling configuration produce "
                f"{calculated_rows}"
            )

    for distribution_name in ("scenario_distribution",):
        distribution = _required_mapping(
            dataset_config,
            distribution_name,
            "dataset.yaml",
        )

        if not distribution:
            raise ConfigurationError(
                f"{distribution_name} cannot be empty"
            )

        for label, probability in distribution.items():
            if not isinstance(probability, (int, float)):
                raise ConfigurationError(
                    f"Probability for '{label}' in {distribution_name} "
                    "must be numeric"
                )

            if float(probability) < 0.0:
                raise ConfigurationError(
                    f"Probability for '{label}' in {distribution_name} "
                    "cannot be negative"
                )

        total_probability = sum(float(value) for value in distribution.values())

        if abs(total_probability - 1.0) > 1e-9:
            raise ConfigurationError(
                f"{distribution_name} must total 1.0, "
                f"but totals {total_probability}"
            )

    output = _required_mapping(dataset_config, "output", "dataset.yaml")
    output_files = _required_mapping(output, "files", "dataset.yaml")

    required_outputs = {
        "gpu_telemetry",
        "node_telemetry",
        "workloads",
        "incidents",
        "agent_evaluation",
        "generation_manifest",
    }

    missing_outputs = required_outputs.difference(output_files)

    if missing_outputs:
        missing = ", ".join(sorted(missing_outputs))
        raise ConfigurationError(
            f"dataset.yaml is missing output files: {missing}"
        )

    output_names = list(output_files.values())

    if len(output_names) != len(set(output_names)):
        raise ConfigurationError(
            "Dataset output filenames must be unique"
        )


def _validate_detection(
    dataset_config: dict[str, Any],
    detection_config: dict[str, Any],
) -> None:
    features = _required_mapping(
        detection_config,
        "features",
        "detection.yaml",
    )

    ordered_features = features.get("ordered")

    if not isinstance(ordered_features, list) or not ordered_features:
        raise ConfigurationError(
            "features.ordered must be a non-empty list"
        )

    if len(ordered_features) != len(set(ordered_features)):
        raise ConfigurationError(
            "features.ordered contains duplicate feature names"
        )

    normal_ranges = _required_mapping(
        detection_config,
        "normal_operating_ranges",
        "detection.yaml",
    )

    if set(ordered_features) != set(normal_ranges):
        missing_ranges = set(ordered_features).difference(normal_ranges)
        extra_ranges = set(normal_ranges).difference(ordered_features)

        raise ConfigurationError(
            "Detection features and normal ranges are inconsistent. "
            f"Missing ranges: {sorted(missing_ranges)}. "
            f"Unexpected ranges: {sorted(extra_ranges)}"
        )

    for feature_name, limits in normal_ranges.items():
        if not isinstance(limits, dict):
            raise ConfigurationError(
                f"Range for '{feature_name}' must be a mapping"
            )

        minimum = limits.get("minimum")
        maximum = limits.get("maximum")

        if not isinstance(minimum, (int, float)):
            raise ConfigurationError(
                f"Minimum for '{feature_name}' must be numeric"
            )

        if not isinstance(maximum, (int, float)):
            raise ConfigurationError(
                f"Maximum for '{feature_name}' must be numeric"
            )

        if float(minimum) > float(maximum):
            raise ConfigurationError(
                f"Minimum exceeds maximum for '{feature_name}'"
            )

    training = _required_mapping(
        detection_config,
        "training",
        "detection.yaml",
    )

    minimum_rows = _required_positive_int(
        training,
        "minimum_training_rows",
        "detection.yaml",
    )
    maximum_rows = _required_positive_int(
        training,
        "maximum_training_rows",
        "detection.yaml",
    )

    if minimum_rows > maximum_rows:
        raise ConfigurationError(
            "minimum_training_rows cannot exceed maximum_training_rows"
        )

    contamination = training.get("contamination")

    if not isinstance(contamination, (int, float)):
        raise ConfigurationError(
            "training.contamination must be numeric"
        )

    if not 0.0 < float(contamination) < 0.5:
        raise ConfigurationError(
            "training.contamination must be greater than 0 and less than 0.5"
        )

    scoring = _required_mapping(
        detection_config,
        "scoring",
        "detection.yaml",
    )
    thresholds = _required_mapping(
        scoring,
        "severity_thresholds",
        "detection.yaml",
    )

    expected_severities = ["low", "medium", "high", "critical"]

    if set(thresholds) != set(expected_severities):
        raise ConfigurationError(
            "Severity thresholds must define low, medium, high, and critical"
        )

    ordered_thresholds = [
        float(thresholds[level]) for level in expected_severities
    ]

    if not (
        0.0
        < ordered_thresholds[0]
        < ordered_thresholds[1]
        < ordered_thresholds[2]
        < ordered_thresholds[3]
        <= 1.0
    ):
        raise ConfigurationError(
            "Severity thresholds must increase from low to critical "
            "and remain within the range 0 to 1"
        )

    evaluation = _required_mapping(
        detection_config,
        "evaluation",
        "detection.yaml",
    )
    positive_labels = set(evaluation.get("positive_labels", []))
    negative_labels = set(evaluation.get("negative_labels", []))

    scenario_labels = set(dataset_config["scenario_distribution"])

    if positive_labels != scenario_labels:
        raise ConfigurationError(
            "Detection positive labels must exactly match dataset scenarios"
        )

    if positive_labels.intersection(negative_labels):
        raise ConfigurationError(
            "Positive and negative evaluation labels must not overlap"
        )


def _configuration_hash(configurations: dict[str, dict[str, Any]]) -> str:
    canonical_json = json.dumps(
        configurations,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def load_config(project_root: Path | None = None) -> SimulatorConfig:
    """Load and validate all ClusterMind simulator configuration files."""

    root = (
        project_root.resolve()
        if project_root is not None
        else _project_root()
    )

    if not root.exists():
        raise ConfigurationError(f"Project root does not exist: {root}")

    config_directory = root / "configs"

    cluster_config = _read_yaml(config_directory / "cluster.yaml")
    dataset_config = _read_yaml(config_directory / "dataset.yaml")
    detection_config = _read_yaml(config_directory / "detection.yaml")

    _validate_cluster(cluster_config)
    _validate_dataset(cluster_config, dataset_config)
    _validate_detection(dataset_config, detection_config)

    configurations = {
        "cluster": cluster_config,
        "dataset": dataset_config,
        "detection": detection_config,
    }

    return SimulatorConfig(
        project_root=root,
        cluster=cluster_config,
        dataset=dataset_config,
        detection=detection_config,
        configuration_hash=_configuration_hash(configurations),
    )