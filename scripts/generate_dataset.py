from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.config import ConfigurationError, load_config
from simulator.exporter import DatasetExporter, ExportError
from simulator.generator import DatasetGenerator, GenerationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the synthetic ClusterMind GPU telemetry datasets, "
            "ground-truth incidents, evaluation records, and manifest."
        )
    )
    parser.add_argument(
        "--profile",
        default="development",
        choices=("development", "production_demo"),
        help="Dataset generation profile. Default: development.",
    )
    parser.add_argument(
        "--chunk-timestamps",
        type=int,
        default=288,
        metavar="COUNT",
        help=(
            "Number of timestamps generated per in-memory chunk. "
            "Default: 288."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated dataset files.",
    )
    return parser


def _human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")

    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{size_bytes} B"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.chunk_timestamps <= 0:
        parser.error("--chunk-timestamps must be a positive integer")

    try:
        config = load_config(PROJECT_ROOT)
        generator = DatasetGenerator(
            config=config,
            profile_name=args.profile,
        )
        exporter = DatasetExporter(
            generator,
            overwrite=args.overwrite,
        )

        summary = generator.summary()
        print("ClusterMind synthetic dataset generation")
        print(f"Profile: {summary['profile']}")
        print(f"Time range: {summary['start_timestamp']} to {summary['end_timestamp']}")
        print(f"Expected GPU rows: {summary['expected_gpu_rows']:,}")
        print(f"Expected node rows: {summary['expected_node_rows']:,}")
        print(f"Expected workload rows: {summary['expected_workload_rows']:,}")
        print(f"Incidents: {summary['incident_count']}")
        print(f"Healthy controls: {summary['healthy_control_count']}")
        print(f"Output directory: {config.output_directory}")
        print("Generating files...")

        result = exporter.export(
            chunk_timestamp_count=args.chunk_timestamps
        )

        print("Generation completed successfully.")
        for exported_file in result.files:
            print(
                f"- {exported_file.path.name}: "
                f"{exported_file.row_count:,} rows, "
                f"{_human_size(exported_file.size_bytes)}, "
                f"sha256={exported_file.sha256}"
            )
        print(f"- {result.manifest_path.name}: generation manifest")
        return 0
    except (ConfigurationError, GenerationError, ExportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ERROR: generation interrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
