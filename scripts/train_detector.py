from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.detector import DetectorError, IsolationForestDetector
from app.ml.features import FeatureError, FeaturePipeline
from simulator.config import ConfigurationError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train, evaluate, and persist the ClusterMind Isolation Forest "
            "GPU anomaly detector."
        )
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10000,
        metavar="COUNT",
        help="CSV rows processed per evaluation chunk. Default: 10000.",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Do not save row-level detector predictions.",
    )
    return parser


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_predictions(
    predictions: pd.DataFrame,
    positive_labels: set[str],
) -> dict[str, object]:
    actual = predictions["ground_truth_label"].isin(positive_labels)
    predicted = predictions["predicted_anomaly"].astype(bool)

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        average="binary",
        zero_division=0,
    )

    true_positive = int((actual & predicted).sum())
    true_negative = int((~actual & ~predicted).sum())
    false_positive = int((~actual & predicted).sum())
    false_negative = int((actual & ~predicted).sum())
    negative_count = int((~actual).sum())

    by_label: dict[str, dict[str, object]] = {}
    for label, group in predictions.groupby("ground_truth_label", sort=True):
        label_predicted = group["predicted_anomaly"].astype(bool)
        by_label[str(label)] = {
            "rows": int(len(group)),
            "detections": int(label_predicted.sum()),
            "detection_rate": round(float(label_predicted.mean()), 6),
            "mean_anomaly_score": round(
                float(group["anomaly_score"].mean()),
                6,
            ),
        }

    return {
        "rows_evaluated": int(len(predictions)),
        "positive_rows": int(actual.sum()),
        "negative_rows": negative_count,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1_score": round(float(f1), 6),
        "false_positive_rate": round(
            _safe_divide(false_positive, negative_count),
            6,
        ),
        "confusion_matrix": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "severity_counts": {
            str(key): int(value)
            for key, value in predictions["severity"].value_counts().items()
        },
        "by_ground_truth_label": by_label,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.chunksize <= 0:
        parser.error("--chunksize must be a positive integer")

    try:
        config = load_config(PROJECT_ROOT)
        pipeline = FeaturePipeline(config)
        detector = IsolationForestDetector(
            config=config,
            feature_pipeline=pipeline,
        )

        print("ClusterMind Isolation Forest training")
        print(f"Telemetry source: {pipeline.telemetry_path()}")
        training_frame = pipeline.load_training_frame(
            chunksize=args.chunksize
        )
        print(f"Healthy training rows: {len(training_frame):,}")
        print(f"Feature count: {len(pipeline.feature_names)}")
        print("Training detector...")

        detector.fit(training_frame)
        artifact_paths = detector.save()

        print("Evaluating full development telemetry...")
        predictions = detector.predict_csv(chunksize=args.chunksize)
        positive_labels = set(
            config.detection["evaluation"]["positive_labels"]
        )
        metrics = evaluate_predictions(predictions, positive_labels)

        artifact_directory = artifact_paths["directory"]
        metrics_path = artifact_directory / "evaluation_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        predictions_path = config.project_root / "data" / "artifacts" / "detections.csv"
        if not args.skip_predictions:
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(predictions_path, index=False, encoding="utf-8")

        print("Training and evaluation completed successfully.")
        print(f"Precision: {metrics['precision']:.6f}")
        print(f"Recall: {metrics['recall']:.6f}")
        print(f"F1 score: {metrics['f1_score']:.6f}")
        print(f"False-positive rate: {metrics['false_positive_rate']:.6f}")
        print(f"Model: {artifact_paths['model']}")
        print(f"Scaler: {artifact_paths['scaler']}")
        print(f"Metadata: {artifact_paths['metadata']}")
        print(f"Metrics: {metrics_path}")
        if not args.skip_predictions:
            print(f"Predictions: {predictions_path}")
        return 0
    except (
        ConfigurationError,
        FeatureError,
        DetectorError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ERROR: training interrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
