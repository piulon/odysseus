"""Restricted action client for the Homelab Operator API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from src.services.homelab.client import (
    HomelabAuthenticationError,
    HomelabClientError,
    HomelabConnectionError,
)


class HomelabActionClient:
    """Client restricted to explicitly implemented Operator actions."""

    DEFAULT_URL = "http://homelab-operator:8765"

    def __init__(
        self,
        base_url: Optional[str] = None,
        action_token: Optional[str] = None,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("HOMELAB_OPERATOR_URL")
            or self.DEFAULT_URL
        ).rstrip("/")

        self.action_token = (
            action_token
            or os.getenv(
                "HOMELAB_OPERATOR_ACTION_TOKEN",
                "",
            )
        ).strip()

        self.timeout = timeout

        if not self.action_token:
            raise HomelabAuthenticationError(
                "HOMELAB_OPERATOR_ACTION_TOKEN "
                "is not configured"
            )

    def _post(
        self,
        path: str,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"

        request = urllib.request.Request(
            url=url,
            data=b"",
            headers={
                "Accept": "application/json",
                "X-Homelab-Action-Token":
                    self.action_token,
            },
            method="POST",
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
                    "the restricted action token"
                ) from exc

            raise HomelabClientError(
                "Homelab Operator returned "
                f"HTTP {exc.code} for the action"
            ) from exc

        except urllib.error.URLError as exc:
            raise HomelabConnectionError(
                "Cannot reach the Homelab Operator "
                "action endpoint"
            ) from exc

        except json.JSONDecodeError as exc:
            raise HomelabClientError(
                "Homelab Operator returned invalid "
                "JSON for the action"
            ) from exc

    def create_palworld_backup(
        self,
    ) -> Dict[str, Any]:
        return self._post(
            "/v1/palworld/backups/create"
        )


    def start_palworld(
        self,
    ) -> Dict[str, Any]:
        """Start only the dedicated Palworld service."""

        return self._post(
            "/v1/palworld/start"
        )

    def stop_palworld(
        self,
    ) -> Dict[str, Any]:
        """Stop only the dedicated Palworld service."""

        return self._post(
            "/v1/palworld/stop"
        )

    def restart_palworld(
        self,
    ) -> Dict[str, Any]:
        """Restart only the dedicated Palworld service."""

        return self._post(
            "/v1/palworld/restart"
        )
