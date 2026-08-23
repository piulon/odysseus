import unittest

from src.agent_tools.homelab_tools import (
    format_homelab_status,
)


PAYLOAD = {
    "ok": True,
    "count": 4,
    "summary": {
        "inventory": {
            "total": 4,
            "running": 3,
            "stopped": 0,
            "missing": 1,
            "unhealthy": 0,
        },
        "docker": {
            "running": 5,
            "total": 5,
            "outside_inventory_running_count": 2,
            "outside_inventory_running": [
                "homelab-operator",
                "palworld-control",
            ],
        },
        "categories": {
            "ai": {
                "total": 1,
                "running": 1,
            },
            "infrastructure": {
                "total": 1,
                "running": 1,
            },
            "monitoring": {
                "total": 1,
                "running": 1,
            },
            "games": {
                "total": 1,
                "running": 0,
            },
        },
        "expected_issues": [],
        "optional_missing": ["palworld"],
    },
    "services": [
        {
            "service": "grafana",
            "container": "grafana",
            "category": "monitoring",
            "expected_running": True,
            "present": True,
            "running": True,
            "health": None,
        },
        {
            "service": "palworld",
            "container": "palworld",
            "category": "games",
            "expected_running": False,
            "present": False,
            "running": False,
            "health": None,
        },
        {
            "service": "ollama",
            "container": "ollama",
            "category": "ai",
            "expected_running": True,
            "present": True,
            "running": True,
            "health": None,
        },
        {
            "service": "caddy",
            "container": "net-caddy",
            "category": "infrastructure",
            "expected_running": True,
            "present": True,
            "running": True,
            "health": None,
        },
    ],
    "gpu": {
        "available": True,
        "count": 1,
        "devices": [
            {
                "index": 0,
                "model": "NVIDIA GeForce RTX 3080",
                "temperature_c": 49,
                "utilization_percent": 3,
                "memory_used_mib": 8888,
                "memory_total_mib": 10239,
                "power_w": 31.33,
            },
        ],
    },
}


class HomelabStatusFormatterTests(unittest.TestCase):
    def test_lists_every_inventory_service_once(
        self,
    ) -> None:
        text = format_homelab_status(PAYLOAD)

        for service in (
            "ollama",
            "caddy",
            "grafana",
            "palworld",
        ):
            self.assertEqual(
                text.count(f"**{service}**"),
                1,
            )

    def test_uses_fixed_category_order(
        self,
    ) -> None:
        text = format_homelab_status(PAYLOAD)

        positions = [
            text.index("### IA"),
            text.index("### Infraestructura"),
            text.index("### Monitorización"),
            text.index("### Juegos"),
        ]

        self.assertEqual(
            positions,
            sorted(positions),
        )

    def test_formats_summary_and_gpu(
        self,
    ) -> None:
        text = format_homelab_status(PAYLOAD)

        self.assertIn(
            "Inventario: **3/4**",
            text,
        )

        self.assertIn(
            "Docker: **5/5**",
            text,
        )

        self.assertIn(
            "[AUSENTE OPCIONAL] **palworld**",
            text,
        )

        self.assertIn(
            "**homelab-operator**",
            text,
        )

        self.assertIn(
            "NVIDIA GeForce RTX 3080",
            text,
        )

        self.assertIn(
            "VRAM 8888/10239 MiB",
            text,
        )


if __name__ == "__main__":
    unittest.main()
