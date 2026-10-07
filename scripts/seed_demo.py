from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.influxdb import (
    InfluxDBService,
    InfluxDBServiceError,
    WriteSummary,
)


MEASUREMENTS = (
    "gpu_telemetry",
    "node_telemetry",
    "workload_telemetry",
    "gpu_incidents",
    "detector_results",
)

EXPECTED_COUNTS = {
    "gpu_telemetry": 80640,
    "node_telemetry": 20160,
    "workload_telemetry": 20160,
    "gpu_incidents": 35,
    "detector_results": 80640,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seed the ClusterMind development datasets and detector results "
            "into the configured InfluxDB OSS bucket."
        )
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Delete only the five known ClusterMind measurements before "
            "writing the development dataset."
        ),
    )
    parser.add_argument(
        "--csv-chunksize",
        type=int,
        default=5000,
        metavar="COUNT",
        help="CSV rows read per chunk. Default: 5000.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        metavar="COUNT",
        help="InfluxDB points written per batch. Default: 2000.",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip post-write measurement count verification.",
    )
    return parser


def _require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise InfluxDBServiceError(f"Missing {label} file: {resolved}")
    if resolved.stat().st_size == 0:
        raise InfluxDBServiceError(f"Empty {label} file: {resolved}")
    return resolved


def _print_summary(summary: WriteSummary) -> None:
    print(
        f"- {summary.measurement}: "
        f"source={summary.source_rows:,}, "
        f"written={summary.written_points:,}, "
        f"skipped={summary.skipped_rows:,}, "
        f"batches={summary.batches:,}"
    )


def _validate_positive_integer(value: int, option_name: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise InfluxDBServiceError(
            f"{option_name} must be a positive integer"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_positive_integer(args.csv_chunksize, "--csv-chunksize")
        _validate_positive_integer(args.batch_size, "--batch-size")

        get_settings.cache_clear()
        settings = get_settings()

        gpu_path = _require_file(
            settings.data_directory / "gpu_telemetry.csv",
            "GPU telemetry",
        )
        node_path = _require_file(
            settings.data_directory / "node_telemetry.csv",
            "node telemetry",
        )
        workload_path = _require_file(
            settings.data_directory / "workloads.csv",
            "workload telemetry",
        )
        incident_path = _require_file(
            settings.data_directory / "incidents.csv",
            "incident",
        )
        detection_path = _require_file(
            settings.artifact_directory / "detections.csv",
            "detector result",
        )

        print("ClusterMind InfluxDB demo seeding")
        print(f"InfluxDB URL: {settings.influxdb_url}")
        print(f"Organization: {settings.influxdb_org}")
        print(f"Bucket: {settings.influxdb_bucket}")
        print(f"Replace existing measurements: {args.replace}")

        with InfluxDBService(settings) as service:
            if not service.ping():
                raise InfluxDBServiceError("InfluxDB ping returned false")
            service.verify_bucket()
            print("InfluxDB connection and bucket verification succeeded.")

            if args.replace:
                service.delete_measurements(MEASUREMENTS)
                print("Existing ClusterMind measurements were cleared.")

            common = {
                "csv_chunksize": args.csv_chunksize,
                "batch_size": args.batch_size,
            }
            summaries = [
                service.write_gpu_csv(gpu_path, **common),
                service.write_node_csv(node_path, **common),
                service.write_workload_csv(workload_path, **common),
                service.write_incident_csv(
                    incident_path,
                    batch_size=args.batch_size,
                ),
                service.write_detection_csv(detection_path, **common),
            ]

            print("Write summaries:")
            for summary in summaries:
                _print_summary(summary)

            if not args.skip_verification:
                print("Verifying logical measurement counts:")
                verification_errors: list[str] = []
                for measurement in MEASUREMENTS:
                    actual = service.measurement_count(measurement)
                    expected = EXPECTED_COUNTS[measurement]
                    status = "OK" if actual == expected else "MISMATCH"
                    print(
                        f"- {measurement}: actual={actual:,}, "
                        f"expected={expected:,}, status={status}"
                    )
                    if actual != expected:
                        verification_errors.append(
                            f"{measurement}: expected {expected}, found {actual}"
                        )

                if verification_errors:
                    raise InfluxDBServiceError(
                        "Post-write verification failed: "
                        + "; ".join(verification_errors)
                    )

        print("SUCCESS: ClusterMind demo data is available in InfluxDB.")
        return 0
    except InfluxDBServiceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ERROR: seeding interrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
