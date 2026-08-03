import asyncio
import json
import unittest
from unittest.mock import Mock, patch

from src.agent_tools.homelab_tools import (
    HomelabTool,
    format_homelab_gpu,
    format_homelab_service,
    format_palworld_backups,
    format_palworld_status,
)
from src.services.homelab.client import (
    HomelabClient,
)


class HomelabSpecificFormatterTests(
    unittest.TestCase
):
    def test_formats_service_status(self):
        text = format_homelab_service({
            "ok": True,
            "service": {
                "service": "grafana",
                "container": "grafana",
                "category": "monitoring",
                "expected_running": True,
                "present": True,
                "running": True,
                "status": "running",
                "health": None,
                "image": "grafana/grafana-oss:latest",
                "networks": [
                    "management_net",
                    "proxy_net",
                ],
            },
        })

        self.assertIn(
            "## Estado de Grafana",
            text,
        )
        self.assertIn("**OK**", text)
        self.assertIn("management_net", text)

    def test_formats_gpu_and_vram(self):
        text = format_homelab_gpu({
            "gpu": {
                "available": True,
                "source": "dcgm-exporter",
                "devices": [
                    {
                        "index": 0,
                        "model": (
                            "NVIDIA GeForce RTX 3080"
                        ),
                        "temperature_c": 50,
                        "utilization_percent": 23,
                        "memory_used_mib": 9066,
                        "memory_free_mib": 801,
                        "memory_reserved_mib": 371,
                        "memory_total_mib": 10238,
                        "power_w": 38.982,
                        "driver_version": "595.84",
                    }
                ],
            },
        })

        self.assertIn(
            "NVIDIA GeForce RTX 3080",
            text,
        )
        self.assertIn("9066/10238 MiB", text)
        self.assertIn("50 °C", text)

    def test_formats_palworld_status(self):
        text = format_palworld_status({
            "ok": True,
            "status": "online",
            "server": "PalHub",
            "version": "v1.0",
            "players_display": "2 / 32",
            "fps": 59,
            "uptime": "16 min",
            "world_days": 266,
        })

        self.assertIn(
            "## Estado de Palworld",
            text,
        )
        self.assertIn("EN LÍNEA", text)
        self.assertIn("2 / 32", text)
        self.assertIn("59", text)

    def test_formats_palworld_backups(self):
        text = format_palworld_backups({
            "ok": True,
            "status": "healthy",
            "count": 12,
            "latest_backup": "backup.tar.gz",
            "age": "5 h",
            "size": "4.8 MiB",
            "integrity": "correcta",
            "retention": "14 días",
            "schedule": "04:15–04:25",
        })

        self.assertIn(
            "## Copias de Palworld",
            text,
        )
        self.assertIn("12", text)
        self.assertIn("correcta", text)
        self.assertIn("backup.tar.gz", text)

    def test_client_uses_dedicated_paths(self):
        client = object.__new__(
            HomelabClient
        )

        client._request = Mock(
            return_value={"ok": True}
        )

        client.palworld_status()

        client._request.assert_called_once_with(
            "/v1/palworld/status"
        )

        client._request.reset_mock()

        client.palworld_backups()

        client._request.assert_called_once_with(
            "/v1/palworld/backups"
        )

    def test_specific_tool_actions_are_terminal(self):
        cases = (
            (
                {
                    "action": "service",
                    "service": "grafana",
                },
                "service",
                {
                    "ok": True,
                    "service": {
                        "service": "grafana",
                        "running": True,
                        "present": True,
                        "status": "running",
                    },
                },
            ),
            (
                {"action": "gpu"},
                "status",
                {
                    "gpu": {
                        "available": False,
                        "devices": [],
                    }
                },
            ),
            (
                {"action": "palworld_status"},
                "palworld_status",
                {
                    "status": "online",
                    "players_display": "0 / 32",
                },
            ),
            (
                {"action": "palworld_backups"},
                "palworld_backups",
                {
                    "status": "healthy",
                    "count": 12,
                },
            ),
        )

        for command, method, response in cases:
            with self.subTest(command=command):
                with patch(
                    "src.agent_tools.homelab_tools."
                    "HomelabClient"
                ) as client_class:
                    client = client_class.return_value

                    getattr(
                        client,
                        method,
                    ).return_value = response

                    result = asyncio.run(
                        HomelabTool().execute(
                            json.dumps(command),
                            {},
                        )
                    )

                self.assertEqual(
                    result.get("exit_code"),
                    0,
                )

                self.assertTrue(
                    result.get(
                        "terminal_response"
                    )
                )

                self.assertTrue(
                    result.get(
                        "direct_response"
                    )
                )


if __name__ == "__main__":
    unittest.main()
