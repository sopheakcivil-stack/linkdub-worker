from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import SETTINGS, Settings
from .media import (
    build_timed_voice_track,
    download_video,
    extract_speech_audio,
    fit_voice_to_segment,
    isolate_background,
    probe_duration,
    render_final_video,
)
from .models import Segment
from .queue import WorkerApiClient
from .services import (
    generate_voice,
    transcribe_chinese,
    translate_segments,
    write_srt,
)


LOG = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        job: dict[str, Any],
        api: WorkerApiClient,
        workdir: Path,
        settings: Settings = SETTINGS,
    ) -> None:
        self.job = job
        self.api = api
        self.workdir = workdir
        self.settings = settings
        self.job_id = str(job["id"])
        self.target_language = str(job.get("target_language") or "English")

    def stage(self, name: str, progress: int, **extra: Any) -> None:
        LOG.info("[%s%%] %s", progress, name)
        self.api.update(
            self.job_id,
            status="processing",
            stage=name,
            progress=progress,
            **extra,
        )

    def run(self) -> dict[str, str]:
        self.stage("Downloading", 3)
        source = download_video(
            str(self.job["source_url"]),
            str(self.job["media_url"]) if self.job.get("media_url") else None,
            self.workdir,
        )
        duration = probe_duration(source, self.settings)
        if duration > self.settings.max_video_seconds:
            raise RuntimeError(
                f"Video is {duration / 3600:.1f} hours; the current limit is "
                f"{self.settings.max_video_seconds / 3600:.1f} hours"
            )

        self.stage("Extracting Audio", 14, duration_seconds=round(duration, 3))
        speech_audio = self.workdir / "speech.wav"
        extract_speech_audio(source, speech_audio, self.settings)

        self.stage("Transcribing", 24)
        segments = transcribe_chinese(speech_audio, self.settings)

        self.stage("Splitting", 43)
        source_srt = self.workdir / "source-zh.srt"
        write_srt(segments, source_srt, translated=False)

        self.stage("Translating", 50)
        translate_segments(segments, self.target_language)
        translated_srt = self.workdir / "translated.srt"
        write_srt(segments, translated_srt, translated=True)
        self.api.save_segments(self.job_id, [segment.to_api() for segment in segments])

        self.stage("Generating Voices", 60)
        fitted_voices = self._generate_voices(segments)

        self.stage("Mixing Audio", 81)
        voice_track = self.workdir / "dubbed-voice.wav"
        build_timed_voice_track(segments, fitted_voices, duration, voice_track)
        background = isolate_background(source, self.workdir, self.settings)

        self.stage("Rendering", 88)
        final_video = self.workdir / "linkdub-final.mp4"
        render_final_video(
            source,
            voice_track,
            translated_srt,
            final_video,
            duration,
            self.target_language,
            background=background,
            settings=self.settings,
        )

        self.stage("Uploading", 95)
        video = self.api.upload_artifact(self.job_id, "video", final_video)
        translated = self.api.upload_artifact(
            self.job_id, "translated_subtitles", translated_srt
        )
        source_subtitles = self.api.upload_artifact(
            self.job_id, "source_subtitles", source_srt
        )

        self.api.update(
            self.job_id,
            status="completed",
            stage="Completed",
            progress=100,
            error=None,
            output_url=video["public_url"],
            output_path=video["path"],
            translated_subtitles_url=translated["public_url"],
            translated_subtitles_path=translated["path"],
            source_subtitles_url=source_subtitles["public_url"],
            source_subtitles_path=source_subtitles["path"],
        )
        LOG.info("Job %s completed: %s", self.job_id, video["public_url"])
        return {
            "video": video["public_url"],
            "translated_subtitles": translated["public_url"],
            "source_subtitles": source_subtitles["public_url"],
        }

    def _generate_voices(self, segments: list[Segment]) -> list[Path]:
        fitted: list[Path] = []
        last_reported = 60
        for index, segment in enumerate(segments):
            raw_path = self.workdir / f"voice-{index:05d}.mp3"
            fitted_path = self.workdir / f"voice-{index:05d}.wav"
            generate_voice(segment.translated_text, self.target_language, raw_path)
            fit_voice_to_segment(
                raw_path, fitted_path, segment.duration, settings=self.settings
            )
            raw_path.unlink(missing_ok=True)
            fitted.append(fitted_path)
            progress = 60 + int(19 * (index + 1) / len(segments))
            if progress >= last_reported + 3:
                self.stage("Generating Voices", progress)
                last_reported = progress
        return fitted

