from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulator.generator import DatasetGenerator, GenerationError


class ExportError(RuntimeError):
    """Raised when generated dataset files cannot be exported safely."""


@dataclass(frozen=True)
class ExportedFile:
    """Metadata describing one finalized dataset artifact."""

    dataset_name: str
    path: Path
    row_count: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ExportResult:
    """Result returned after a complete and validated export."""

    profile_name: str
    output_directory: Path
    files: tuple[ExportedFile, ...]
    manifest_path: Path

    def file(self, dataset_name: str) -> ExportedFile:
        for exported_file in self.files:
            if exported_file.dataset_name == dataset_name:
                return exported_file
        available = ", ".join(item.dataset_name for item in self.files)
        raise ExportError(
            f"Unknown exported dataset '{dataset_name}'. Available: {available}"
        )


class DatasetExporter:
    """Stream ClusterMind datasets to validated CSV files and a manifest."""

    STREAMED_DATASETS = (
        "gpu_telemetry",
        "node_telemetry",
        "workloads",
    )
    STATIC_DATASETS = (
        "incidents",
        "agent_evaluation",
    )

    def __init__(
        self,
        generator: DatasetGenerator,
        *,
        overwrite: bool | None = None,
    ) -> None:
        if not isinstance(generator, DatasetGenerator):
            raise ExportError("generator must be a DatasetGenerator instance")

        self.generator = generator
        configured_overwrite = bool(
            self.generator.config.dataset["output"]["overwrite_existing"]
        )
        self.overwrite = (
            configured_overwrite if overwrite is None else bool(overwrite)
        )
        self.output_directory = self.generator.config.output_directory
        self._temporary_directory = self.output_directory / ".export-tmp"

    def export(
        self,
        *,
        chunk_timestamp_count: int = 288,
    ) -> ExportResult:
        """Generate, validate, and atomically finalize all dataset artifacts."""

        if (
            isinstance(chunk_timestamp_count, bool)
            or not isinstance(chunk_timestamp_count, int)
            or chunk_timestamp_count <= 0
        ):
            raise ExportError("chunk_timestamp_count must be a positive integer")

        self._prepare_directories()
        self._validate_destination_policy()

        row_counts = {
            dataset_name: 0
            for dataset_name in self.STREAMED_DATASETS + self.STATIC_DATASETS
        }

        try:
            writers, handles = self._open_streamed_writers()
            try:
                for chunk in self.generator.iter_chunks(
                    chunk_timestamp_count=chunk_timestamp_count
                ):
                    for dataset_name in self.STREAMED_DATASETS:
                        records = chunk[dataset_name]
                        self._write_records(
                            writers[dataset_name],
                            records,
                            dataset_name,
                        )
                        row_counts[dataset_name] += len(records)
            finally:
                for handle in handles.values():
                    handle.close()

            static_records = {
                "incidents": self.generator.incident_records(),
                "agent_evaluation": (
                    self.generator.agent_evaluation_records()
                ),
            }
            for dataset_name, records in static_records.items():
                self._write_static_dataset(dataset_name, records)
                row_counts[dataset_name] = len(records)

            self._validate_row_counts(row_counts)
            finalized_files = self._finalize_csv_files(row_counts)
            manifest_path = self._write_manifest(finalized_files)
            self._remove_temporary_directory()

            return ExportResult(
                profile_name=self.generator.profile_name,
                output_directory=self.output_directory,
                files=tuple(finalized_files),
                manifest_path=manifest_path,
            )
        except Exception as exc:
            self._remove_temporary_directory()
            if isinstance(exc, (ExportError, GenerationError)):
                raise
            raise ExportError("Dataset export failed") from exc

    def _prepare_directories(self) -> None:
        try:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            if self._temporary_directory.exists():
                shutil.rmtree(self._temporary_directory)
            self._temporary_directory.mkdir(parents=False, exist_ok=False)
        except OSError as exc:
            raise ExportError(
                f"Unable to prepare output directory: {self.output_directory}"
            ) from exc

    def _validate_destination_policy(self) -> None:
        existing = [
            self.generator.config.output_path(dataset_name)
            for dataset_name in self.STREAMED_DATASETS + self.STATIC_DATASETS
            if self.generator.config.output_path(dataset_name).exists()
        ]
        manifest_path = self.generator.config.output_path(
            "generation_manifest"
        )
        if manifest_path.exists():
            existing.append(manifest_path)

        if existing and not self.overwrite:
            paths = ", ".join(path.name for path in existing)
            self._remove_temporary_directory()
            raise ExportError(
                "Output files already exist and overwrite is disabled: "
                f"{paths}"
            )

    def _fieldnames(self, dataset_name: str) -> list[str]:
        section = self.generator.config.dataset.get(dataset_name)
        if not isinstance(section, dict):
            raise ExportError(
                f"Missing dataset configuration section: {dataset_name}"
            )

        fields = section.get("fields")
        if not isinstance(fields, list) or not fields:
            raise ExportError(
                f"Dataset '{dataset_name}' must define a non-empty fields list"
            )
        if len(fields) != len(set(fields)):
            raise ExportError(
                f"Dataset '{dataset_name}' contains duplicate field names"
            )

        return [str(field) for field in fields]

    def _temporary_csv_path(self, dataset_name: str) -> Path:
        final_path = self.generator.config.output_path(dataset_name)
        return self._temporary_directory / final_path.name

    def _open_streamed_writers(
        self,
    ) -> tuple[dict[str, csv.DictWriter], dict[str, Any]]:
        writers: dict[str, csv.DictWriter] = {}
        handles: dict[str, Any] = {}

        try:
            for dataset_name in self.STREAMED_DATASETS:
                path = self._temporary_csv_path(dataset_name)
                handle = path.open(
                    "w",
                    encoding=str(
                        self.generator.config.dataset["output"]["encoding"]
                    ),
                    newline="",
                )
                writer = csv.DictWriter(
                    handle,
                    fieldnames=self._fieldnames(dataset_name),
                    extrasaction="raise",
                    lineterminator="\n",
                )
                writer.writeheader()
                handles[dataset_name] = handle
                writers[dataset_name] = writer
        except Exception:
            for handle in handles.values():
                handle.close()
            raise

        return writers, handles

    def _validate_record_schema(
        self,
        record: Mapping[str, Any],
        dataset_name: str,
    ) -> None:
        expected = set(self._fieldnames(dataset_name))
        actual = set(record)
        if actual != expected:
            missing = sorted(expected.difference(actual))
            extra = sorted(actual.difference(expected))
            raise ExportError(
                f"Schema mismatch for '{dataset_name}'. "
                f"Missing fields: {missing}. Extra fields: {extra}"
            )

    def _write_records(
        self,
        writer: csv.DictWriter,
        records: Iterable[Mapping[str, Any]],
        dataset_name: str,
    ) -> None:
        for record in records:
            self._validate_record_schema(record, dataset_name)
            writer.writerow(dict(record))

    def _write_static_dataset(
        self,
        dataset_name: str,
        records: list[dict[str, Any]],
    ) -> None:
        path = self._temporary_csv_path(dataset_name)
        try:
            with path.open(
                "w",
                encoding=str(
                    self.generator.config.dataset["output"]["encoding"]
                ),
                newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=self._fieldnames(dataset_name),
                    extrasaction="raise",
                    lineterminator="\n",
                )
                writer.writeheader()
                self._write_records(writer, records, dataset_name)
        except OSError as exc:
            raise ExportError(
                f"Unable to write dataset '{dataset_name}'"
            ) from exc

    def _expected_row_counts(self) -> dict[str, int]:
        summary = self.generator.summary()
        return {
            "gpu_telemetry": int(summary["expected_gpu_rows"]),
            "node_telemetry": int(summary["expected_node_rows"]),
            "workloads": int(summary["expected_workload_rows"]),
            "incidents": int(summary["incident_count"]),
            "agent_evaluation": int(summary["incident_count"]),
        }

    def _validate_row_counts(self, actual_counts: dict[str, int]) -> None:
        expected_counts = self._expected_row_counts()
        for dataset_name, expected in expected_counts.items():
            actual = actual_counts.get(dataset_name)
            if actual != expected:
                raise ExportError(
                    f"Row count mismatch for '{dataset_name}': "
                    f"expected {expected}, generated {actual}"
                )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise ExportError(f"Unable to hash file: {path}") from exc
        return digest.hexdigest()

    def _finalize_csv_files(
        self,
        row_counts: dict[str, int],
    ) -> list[ExportedFile]:
        finalized: list[ExportedFile] = []
        dataset_names = self.STREAMED_DATASETS + self.STATIC_DATASETS

        try:
            for dataset_name in dataset_names:
                temporary_path = self._temporary_csv_path(dataset_name)
                final_path = self.generator.config.output_path(dataset_name)

                if not temporary_path.is_file():
                    raise ExportError(
                        f"Temporary output is missing: {temporary_path}"
                    )

                if final_path.exists() and self.overwrite:
                    final_path.unlink()

                os.replace(temporary_path, final_path)
                finalized.append(
                    ExportedFile(
                        dataset_name=dataset_name,
                        path=final_path,
                        row_count=row_counts[dataset_name],
                        size_bytes=final_path.stat().st_size,
                        sha256=self._sha256(final_path),
                    )
                )
        except Exception as exc:
            for exported_file in finalized:
                try:
                    exported_file.path.unlink(missing_ok=True)
                except OSError:
                    pass
            if isinstance(exc, ExportError):
                raise
            raise ExportError("Unable to finalize generated CSV files") from exc

        return finalized

    def _write_manifest(
        self,
        finalized_files: list[ExportedFile],
    ) -> Path:
        manifest_path = self.generator.config.output_path(
            "generation_manifest"
        )
        temporary_manifest = self._temporary_directory / manifest_path.name
        summary = self.generator.summary()
        created_at = datetime.now(timezone.utc).replace(microsecond=0)

        manifest = {
            "dataset_name": self.generator.config.dataset["dataset"]["name"],
            "dataset_version": self.generator.config.dataset["dataset"]["version"],
            "synthetic": True,
            "profile": self.generator.profile_name,
            "created_at_utc": created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "configuration_hash": self.generator.config.configuration_hash,
            "random_seed": self.generator.config.random_seed,
            "summary": summary,
            "files": {
                item.dataset_name: {
                    "filename": item.path.name,
                    "row_count": item.row_count,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in finalized_files
            },
            "statement": (
                "All telemetry, workloads, incidents, labels, and recovery "
                "outcomes in this dataset are synthetic and intended for "
                "research and demonstration."
            ),
        }

        try:
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if manifest_path.exists() and self.overwrite:
                manifest_path.unlink()
            os.replace(temporary_manifest, manifest_path)
        except OSError as exc:
            for exported_file in finalized_files:
                try:
                    exported_file.path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise ExportError("Unable to write generation manifest") from exc

        return manifest_path

    def _remove_temporary_directory(self) -> None:
        if self._temporary_directory.exists():
            try:
                shutil.rmtree(self._temporary_directory)
            except OSError:
                pass
