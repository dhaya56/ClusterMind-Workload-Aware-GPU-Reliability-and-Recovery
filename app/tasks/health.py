from __future__ import annotations

import sys

from app.tasks.celery_app import celery_app


def worker_is_healthy(timeout: float = 5.0) -> bool:
    responses = celery_app.control.inspect(timeout=timeout).ping() or {}
    return any(reply.get("ok") == "pong" for reply in responses.values())


def main() -> None:
    healthy = worker_is_healthy()
    print("healthy" if healthy else "unavailable")
    raise SystemExit(0 if healthy else 1)


if __name__ == "__main__":
    main()
