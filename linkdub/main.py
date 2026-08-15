from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

from .pipeline import Pipeline
from .queue import WorkerApiClient


LOG = logging.getLogger("linkdub")


def _heartbeat(api: WorkerApiClient, job_id: str, stop: threading.Event) -> None:
    while not stop.wait(120):
        try:
            api.heartbeat(job_id)
        except Exception:
            LOG.exception("Heartbeat failed; the active pipeline will continue")


def process_once() -> bool:
    api = WorkerApiClient()
    job = api.claim()
    if not job:
        LOG.info("No queued LinkDub jobs")
        return False

    job_id = str(job["id"])
    LOG.info("Claimed job %s", job_id)
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=(api, job_id, stop),
        name="linkdub-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    try:
        with tempfile.TemporaryDirectory(prefix=f"linkdub-{job_id[:8]}-") as raw_dir:
            Pipeline(job, api, Path(raw_dir)).run()
        return True
    except Exception as exc:
        LOG.exception("Job %s failed", job_id)
        message = f"{type(exc).__name__}: {exc}"[:2000]
        try:
            api.update(
                job_id,
                status="failed",
                stage="Failed",
                error=message,
            )
        except Exception:
            LOG.exception("Could not report job failure")
        raise
    finally:
        stop.set()
        heartbeat.join(timeout=5)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    process_once()


if __name__ == "__main__":
    main()

