from pathlib import Path

from linkdub.queue import WorkerApiClient


def test_large_video_uses_github_release_storage(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LINKDUB_GITHUB_TOKEN", "test-token")
    video = tmp_path / "result.mp4"
    video.write_bytes(b"video")
    client = WorkerApiClient()
    calls: list[tuple[str, Path, str]] = []

    def fake_upload(job_id: str, path: Path, token: str) -> dict[str, str]:
        calls.append((job_id, path, token))
        return {"path": "release/result.mp4", "public_url": "https://example.com/result.mp4"}

    monkeypatch.setattr(client, "_upload_github_release", fake_upload)

    result = client.upload_artifact("job-1", "video", video)

    assert calls == [("job-1", video, "test-token")]
    assert result["public_url"] == "https://example.com/result.mp4"
