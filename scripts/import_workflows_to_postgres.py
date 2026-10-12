from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.db import DatabaseRuntime
from app.db.repositories import PostgresWorkflowRepository
from app.schemas.agents import WorkflowResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import validated ClusterMind workflow JSON into PostgreSQL."
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("data/artifacts/workflows"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.directory.glob("WF-*.json"))
    if not files:
        raise SystemExit(f"No workflow files found in {args.directory}")

    runtime = DatabaseRuntime(get_settings())
    repository = PostgresWorkflowRepository(runtime)
    try:
        for path in files:
            original = WorkflowResult.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            repository.save(original)
            persisted = repository.get(original.workflow_id)
            if persisted.model_dump(mode="json") != original.model_dump(mode="json"):
                raise RuntimeError(
                    f"Round-trip mismatch after importing {path.name}"
                )
            print("IMPORTED", original.workflow_id)
        print(f"SUCCESS: imported and round-trip validated {len(files)} workflow(s)")
    finally:
        runtime.dispose()


if __name__ == "__main__":
    main()
