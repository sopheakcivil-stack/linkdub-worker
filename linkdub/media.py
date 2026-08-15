from __future__ import annotations

import ipaddress
import json
import shutil
import socket
import subprocess
import sys
import wave
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import yt_dlp

from .config import SETTINGS, Settings
from .models import Segment


class MediaError(RuntimeError):
    pass


def validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise MediaError("Only public HTTP or HTTPS video URLs are supported")
    if parsed.username or parsed.password:
        raise MediaError("Video URLs containing credentials are not supported")
    try:
        addresses = {
            result[4][0]
            for result in socket.getaddrinfo(parsed.hostname, parsed.port or 443)
        }
    except socket.gaierror as exc:
        raise MediaError(f"Could not resolve video host: {parsed.hostname}") from exc
    for raw in addresses:
        address = ipaddress.ip_address(raw)
        if not address.is_global:
            raise MediaError("Private, local, or reserved network URLs are not allowed")


def run(command: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        rendered = " ".join(command[:5])
        raise MediaError(f"Command failed ({rendered}): {result.stderr[-4000:]}")
    return result


def download_video(
    source_url: str,
    media_url: str | None,
    workdir: Path,
) -> Path:
    validate_public_url(source_url)
    target_template = str(workdir / "source.%(ext)s")
    options = {
        "format": "bv*+ba/b",
        "outtmpl": target_template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "continuedl": True,
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
        "quiet": True,
        "no_warnings": False,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.extract_info(source_url, download=True)
        candidates = [
            path
            for path in workdir.glob("source.*")
            if path.suffix not in {".part", ".ytdl", ".json"}
        ]
        if candidates:
            return max(candidates, key=lambda item: item.stat().st_size)
    except Exception as exc:
        if not media_url:
            raise MediaError(f"The public video could not be imported: {exc}") from exc

    if not media_url:
        raise MediaError("The public video importer returned no media file")
    return _download_direct(media_url, workdir / "source.mp4")


def _download_direct(url: str, destination: Path) -> Path:
    current = url
    for _ in range(6):
        validate_public_url(current)
        response = requests.get(current, stream=True, timeout=(20, 120), allow_redirects=False)
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("location")
            if not location:
                raise MediaError("Video server returned an invalid redirect")
            current = urljoin(current, location)
            continue
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if not (content_type.startswith("video/") or content_type == "application/octet-stream"):
            raise MediaError(f"Resolved URL is not a video ({content_type or 'unknown type'})")
        with destination.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
        return destination
    raise MediaError("Video URL redirected too many times")


def probe_duration(path: Path, settings: Settings = SETTINGS) -> float:
    result = run(
        [
            settings.ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaError("Could not determine video duration") from exc


def has_audio(path: Path, settings: Settings = SETTINGS) -> bool:
    result = run(
        [
            settings.ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return bool(result.stdout.strip())


def extract_speech_audio(source: Path, destination: Path, settings: Settings = SETTINGS) -> None:
    run(
        [
            settings.ffmpeg_bin,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )


def _atempo_chain(factor: float) -> str:
    filters: list[str] = []
    while factor > 2.0:
        filters.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        filters.append("atempo=0.5")
        factor /= 0.5
    filters.append(f"atempo={factor:.6f}")
    return ",".join(filters)


def fit_voice_to_segment(
    raw_voice: Path,
    fitted_voice: Path,
    target_seconds: float,
    settings: Settings = SETTINGS,
) -> None:
    source_seconds = max(0.05, probe_duration(raw_voice, settings))
    speed_factor = source_seconds / max(0.05, target_seconds)
    filters: list[str] = []
    if speed_factor > 1.02:
        filters.append(_atempo_chain(speed_factor))
    filters.extend([f"apad=whole_dur={target_seconds:.6f}", f"atrim=duration={target_seconds:.6f}"])
    run(
        [
            settings.ffmpeg_bin,
            "-y",
            "-i",
            str(raw_voice),
            "-af",
            ",".join(filters),
            "-ar",
            "24000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(fitted_voice),
        ]
    )


def build_timed_voice_track(
    segments: list[Segment],
    fitted_voices: list[Path],
    total_seconds: float,
    destination: Path,
) -> None:
    sample_rate = 24000
    sample_width = 2
    total_samples = max(1, round(total_seconds * sample_rate))
    cursor = 0

    def write_silence(output: wave.Wave_write, count: int) -> None:
        block = b"\0" * (sample_rate * sample_width)
        while count > 0:
            take = min(count, sample_rate)
            output.writeframesraw(block[: take * sample_width])
            count -= take

    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(sample_width)
        output.setframerate(sample_rate)
        for segment, voice_path in zip(segments, fitted_voices, strict=True):
            start_sample = max(cursor, round(segment.start * sample_rate))
            write_silence(output, start_sample - cursor)
            cursor = start_sample
            with wave.open(str(voice_path), "rb") as voice:
                if (
                    voice.getnchannels() != 1
                    or voice.getsampwidth() != sample_width
                    or voice.getframerate() != sample_rate
                ):
                    raise MediaError("Fitted voice has an unexpected PCM format")
                frames = voice.readframes(voice.getnframes())
            remaining = max(0, total_samples - cursor)
            frames = frames[: remaining * sample_width]
            output.writeframesraw(frames)
            cursor += len(frames) // sample_width
        write_silence(output, max(0, total_samples - cursor))


def isolate_background(
    source: Path,
    workdir: Path,
    settings: Settings = SETTINGS,
) -> Path | None:
    if not settings.enable_demucs:
        return None
    if not shutil.which("demucs") and not shutil.which("python"):
        return None
    audio = workdir / "separation-input.wav"
    run(
        [
            settings.ffmpeg_bin,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(audio),
        ]
    )
    output_root = workdir / "demucs"
    run(
        [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-n",
            "htdemucs",
            "-o",
            str(output_root),
            str(audio),
        ]
    )
    candidate = output_root / "htdemucs" / audio.stem / "no_vocals.wav"
    return candidate if candidate.exists() else None


def render_final_video(
    source: Path,
    voice_track: Path,
    translated_srt: Path,
    destination: Path,
    duration_seconds: float,
    target_language: str,
    background: Path | None = None,
    settings: Settings = SETTINGS,
) -> None:
    audio_present = has_audio(source, settings)
    command = [settings.ffmpeg_bin, "-y", "-i", str(source)]
    if background:
        command.extend(["-i", str(background), "-i", str(voice_track), "-i", str(translated_srt)])
        mix = "[1:a][2:a]amix=inputs=2:duration=longest:normalize=0[mix]"
        subtitle_input = "3:0"
    else:
        command.extend(["-i", str(voice_track), "-i", str(translated_srt)])
        if audio_present:
            mix = (
                f"[0:a]volume={settings.original_audio_volume:.4f}[bed];"
                "[bed][1:a]amix=inputs=2:duration=longest:normalize=0[mix]"
            )
        else:
            mix = "[1:a]anull[mix]"
        subtitle_input = "2:0"

    command.extend(
        [
            "-filter_complex",
            mix,
            "-map",
            "0:v:0",
            "-map",
            "[mix]",
            "-map",
            subtitle_input,
            "-c:v",
            "copy" if not settings.burn_subtitles else "libx264",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            f"title={target_language} subtitles",
            "-t",
            f"{duration_seconds:.3f}",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    try:
        run(command)
    except MediaError:
        retry = command.copy()
        codec_index = retry.index("copy") if "copy" in retry else -1
        if codec_index < 0:
            raise
        retry[codec_index] = "libx264"
        retry[codec_index + 1 : codec_index + 1] = ["-preset", "veryfast", "-crf", "21"]
        run(retry)

