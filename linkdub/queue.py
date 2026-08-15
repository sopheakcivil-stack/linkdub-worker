from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from tusclient import client as tus_client

from .config import SETTINGS, Settings


class WorkerApiError(RuntimeError):
    pass


class WorkerApiClient:
    """Secure queue client authenticated by GitHub Actions OIDC.

    No Supabase service key is present in the repository or workflow. The
    Supabase Edge Function verifies the GitHub-signed identity token and then
    performs narrowly scoped queue/storage operations with its server key.
    """

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self.settings = settings
        self.http = requests.Session()
        self.http.headers.update({"user-agent": "LinkDub-Worker/0.1"})

    def _oidc_token(self) -> str:
        request_url = os.getenv("ACTIONS_ID_TOKEN_REQUEST_URL")
        request_token = os.getenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
        if not request_url or not request_token:
            raise WorkerApiError(
                "GitHub Actions OIDC variables are unavailable. Run this worker from "
                "the production workflow with id-token: write permission."
            )

        parts = urlsplit(request_url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["audience"] = self.settings.oidc_audience
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
        response = self.http.get(
            url,
            headers={"authorization": f"Bearer {request_token}"},
            timeout=30,
        )
        response.raise_for_status()
        token = response.json().get("value")
        if not token:
            raise WorkerApiError("GitHub OIDC endpoint did not return a token")
        return str(token)

    def _post(self, payload: dict[str, Any], attempts: int = 4) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.http.post(
                    self.settings.worker_api_url,
                    json=payload,
                    headers={"authorization": f"Bearer {self._oidc_token()}"},
                    timeout=60,
                )
                if response.status_code >= 400:
                    raise WorkerApiError(
                        f"Worker API returned {response.status_code}: {response.text[:1000]}"
                    )
                return response.json()
            except (requests.RequestException, WorkerApiError) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(2**attempt)
        raise WorkerApiError(str(last_error or "Worker API request failed"))

    def claim(self) -> dict[str, Any] | None:
        return self._post({"action": "claim"}).get("job")

    def heartbeat(self, job_id: str) -> None:
        self._post({"action": "heartbeat", "job_id": job_id}, attempts=2)

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        return self._post({"action": "update", "job_id": job_id, "fields": fields})[
            "job"
        ]

    def save_segments(self, job_id: str, segments: list[dict[str, object]]) -> None:
        self._post({"action": "segments", "job_id": job_id, "segments": segments})

    def upload_artifact(self, job_id: str, kind: str, path: Path) -> dict[str, str]:
        signed = self._post({"action": "upload-url", "job_id": job_id, "kind": kind})
        uploader = tus_client.TusClient(
            self.settings.storage_tus_url,
            headers={"x-signature": signed["token"], "x-upsert": "true"},
        ).uploader(
            file_path=str(path),
            chunk_size=6 * 1024 * 1024,
            metadata={
                "bucketName": "linkdub-results",
                "objectName": signed["path"],
                "contentType": signed["content_type"],
                "cacheControl": "3600",
            },
        )
        uploader.upload()
        return {
            "path": str(signed["path"]),
            "public_url": str(signed["public_url"]),
        }

