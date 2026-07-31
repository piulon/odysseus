"""Read-only HTTP client for the Homelab Operator API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


class HomelabClientError(RuntimeError):
    """Base error raised by the Homelab Operator client."""


class HomelabAuthenticationError(
    HomelabClientError
):
    """Raised when the Operator rejects the token."""


class HomelabConnectionError(HomelabClientError):
    """Raised when the Operator cannot be reached."""


class HomelabClient:
    """Read-only client for the Homelab Operator."""

    DEFAULT_URL = "http://homelab-operator:8765"

    def __init__(
        self,
        base_url: Optional[str] = None,
        read_token: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("HOMELAB_OPERATOR_URL")
            or self.DEFAULT_URL
        ).rstrip("/")

        self.read_token = (
            read_token
            or os.getenv(
                "HOMELAB_OPERATOR_READ_TOKEN",
                "",
            )
        ).strip()

        self.timeout = timeout

        if not self.read_token:
            raise HomelabAuthenticationError(
                "HOMELAB_OPERATOR_READ_TOKEN "
                "is not configured"
            )

    def _request(
        self,
        path: str,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"

        request = urllib.request.Request(
            url=url,
            headers={
                "Accept": "application/json",
                "X-Homelab-Read-Token":
                    self.read_token,
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                payload = (
                    response
                    .read()
                    .decode("utf-8")
                )

                return (
                    json.loads(payload)
                    if payload
                    else {}
                )

        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise HomelabAuthenticationError(
                    "Homelab Operator rejected "
                    "the read-only token"
                ) from exc

            raise HomelabClientError(
                "Homelab Operator returned "
                f"HTTP {exc.code}"
            ) from exc

        except urllib.error.URLError as exc:
            raise HomelabConnectionError(
                "Cannot reach the Homelab Operator"
            ) from exc

        except json.JSONDecodeError as exc:
            raise HomelabClientError(
                "Homelab Operator returned "
                "invalid JSON"
            ) from exc

    def health(self) -> Dict[str, Any]:
        return self._request("/v1/health")

    def status(self) -> Dict[str, Any]:
        return self._request("/v1/status")

    def doctor(self) -> Dict[str, Any]:
        return self._request("/v1/doctor")

    def service(
        self,
        service: str,
    ) -> Dict[str, Any]:
        safe_service = urllib.parse.quote(
            service,
            safe="",
        )

        return self._request(
            f"/v1/services/{safe_service}"
        )
