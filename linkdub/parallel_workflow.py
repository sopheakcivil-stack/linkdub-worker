from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import SETTINGS
from .main import _heartbeat, failure_update_fields
from .media import (
    download_video,
    extract_speech_audio,
    probe_duration,
    split_speech_audio,
)
from .models import Segment
from .pipeline import Pipeline
from .queue import WorkerApiClient
from .services import transcribe_chinese


LOG = logging.getLogger("linkdub.parallel")


def _write_output(name: str, value: str) -> None:
    output = os.getenv("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as destination:
            destination.write(f"{name}={value}\n")


def _read_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def heartbeat(api: WorkerApiClient, job_id: str) -> Iterator[None]:
    stop = threading.Event()
    thread = threading.Thread(
        target=_heartbeat,
        args=(api, job_id, stop),
        name="linkdub-parallel-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


def prepare(state_path: Path, workdir: Path) -> bool:
    api = WorkerApiClient()
    job = api.claim()
    if not job:
        LOG.info("No queued LinkDub jobs")
        _write_output("has_job", "false")
        return False

    job_id = str(job["id"])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"job": job}), encoding="utf-8")
    _write_output("has_job", "true")
    _write_output("job_id", job_id)

    try:
        with heartbeat(api, job_id):
            api.update(
                job_id,
                status="processing",
                stage="Downloading",
                progress=3,
            )
            source = download_video(
                str(job["source_url"]),
                str(job["media_url"]) if job.get("media_url") else None,
                workdir,
            )
            duration = probe_duration(source, SETTINGS)
            if duration > SETTINGS.max_video_seconds:
                raise RuntimeError(
                    f"Video is {duration / 3600:.1f} hours; the current limit is "
                    f"{SETTINGS.max_video_seconds / 3600:.1f} hours"
                )

            api.update(
                job_id,
                status="processing",
                stage="Extracting Audio",
                progress=14,
                duration_seconds=round(duration, 3),
            )
            speech = workdir / "speech.wav"
            extract_speech_audio(source, speech, SETTINGS)
            chunks = split_speech_audio(
                speech,
                workdir / "chunks",
                duration,
                SETTINGS.transcription_chunks,
                SETTINGS,
            )
            state = {
                "job": job,
                "duration": duration,
                "chunks": [
                    {"index": index, "file": path.name, "offset": offset}
                    for index, (path, offset) in enumerate(chunks)
                ],
            }
            state_path.write_text(json.dumps(state), encoding="utf-8")
            api.update(
                job_id,
                status="processing",
                stage="Transcribing in parallel",
                progress=24,
            )
        return True
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:2000]
        try:
            api.update(job_id, **failure_update_fields(job, message))
        except Exception:
            LOG.exception("Could not report prepare failure")
        raise


def transcribe_chunk(state_path: Path, workdir: Path, chunk_index: int) -> Path:
    state = _read_state(state_path)
    job = state["job"]
    job_id = str(job["id"])
    chunks = state["chunks"]
    if chunk_index < 0 or chunk_index >= len(chunks):
        raise ValueError(f"Invalid transcription chunk: {chunk_index}")
    chunk = chunks[chunk_index]
    audio = workdir / "chunks" / str(chunk["file"])
    if not audio.exists():
        raise FileNotFoundError(audio)

    api = WorkerApiClient()
    with heartbeat(api, job_id):
        segments = transcribe_chinese(
            audio,
            SETTINGS,
            time_offset=float(chunk["offset"]),
        )
        output = workdir / f"transcript-{chunk_index}.json"
        output.write_text(
            json.dumps([segment.to_api() for segment in segments], ensure_ascii=False),
            encoding="utf-8",
        )
        progress = 24 + int(18 * (chunk_index + 1) / len(chunks))
        api.update(
            job_id,
            status="processing",
            stage=f"Transcribing in parallel ({chunk_index + 1}/{len(chunks)})",
            progress=progress,
        )
    return output


def finalize(state_path: Path, workdir: Path) -> dict[str, str]:
    state = _read_state(state_path)
    job = state["job"]
    job_id = str(job["id"])
    segments: list[Segment] = []
    for chunk in state["chunks"]:
        path = workdir / f"transcript-{chunk['index']}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        segments.extend(Segment(**row) for row in rows)
    segments.sort(key=lambda segment: (segment.start, segment.end))
    if not segments:
        raise RuntimeError("Parallel transcription returned no Chinese speech")

    api = WorkerApiClient()
    with heartbeat(api, job_id):
        api.update(
            job_id,
            status="processing",
            stage="Splitting",
            progress=43,
        )
        source = download_video(
            str(job["source_url"]),
            str(job["media_url"]) if job.get("media_url") else None,
            workdir,
        )
        pipeline = Pipeline(job, api, workdir, SETTINGS)
        return pipeline.finish(source, float(state["duration"]), segments)


def fail(state_path: Path, message: str) -> None:
    state = _read_state(state_path)
    job = state["job"]
    WorkerApiClient().update(
        str(job["id"]),
        **failure_update_fields(job, message[:2000]),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--state", type=Path, required=True)
    prepare_parser.add_argument("--workdir", type=Path, required=True)

    transcribe_parser = subparsers.add_parser("transcribe")
    transcribe_parser.add_argument("--state", type=Path, required=True)
    transcribe_parser.add_argument("--workdir", type=Path, required=True)
    transcribe_parser.add_argument("--chunk", type=int, required=True)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--state", type=Path, required=True)
    finalize_parser.add_argument("--workdir", type=Path, required=True)

    fail_parser = subparsers.add_parser("fail")
    fail_parser.add_argument("--state", type=Path, required=True)
    fail_parser.add_argument("--message", required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.state, args.workdir)
    elif args.command == "transcribe":
        transcribe_chunk(args.state, args.workdir, args.chunk)
    elif args.command == "finalize":
        finalize(args.state, args.workdir)
    else:
        fail(args.state, args.message)


if __name__ == "__main__":
    main()
