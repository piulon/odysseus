import unittest

from src.agent_tools.homelab_tools import (
    classify_direct_homelab_request,
)


class HomelabFastPathTests(unittest.TestCase):
    def assert_command(
        self,
        text,
        expected,
        domains=None,
    ):
        self.assertEqual(
            classify_direct_homelab_request(
                text,
                {"homelab"}
                if domains is None
                else domains,
            ),
            expected,
            text,
        )

    def assert_normal_agent(
        self,
        text,
        domains=None,
    ):
        self.assertIsNone(
            classify_direct_homelab_request(
                text,
                {"homelab"}
                if domains is None
                else domains,
            ),
            text,
        )

    def test_general_status_commands(self):
        for text in (
            "Estado del homelab",
            "¿Cuál es el estado del homelab?",
            "¿Cómo está el homelab?",
            "Homelab status",
            "Quin és l'estat del homelab?",
        ):
            with self.subTest(text=text):
                self.assert_command(
                    text,
                    {"action": "status"},
                )

    def test_specific_status_commands(self):
        cases = (
            (
                "Estado de Grafana",
                {},
                {
                    "action": "service",
                    "service": "grafana",
                },
            ),
            (
                "Estado de Palworld",
                {"homelab"},
                {"action": "palworld_status"},
            ),
            (
                "Estado de las copias de Palworld",
                {"homelab"},
                {"action": "palworld_backups"},
            ),
            (
                "Estado de la GPU",
                {"homelab"},
                {"action": "gpu"},
            ),
            (
                "Uso de VRAM",
                {"homelab"},
                {"action": "gpu"},
            ),
            (
                "Quanta VRAM lliure hi ha?",
                {"homelab"},
                {"action": "gpu"},
            ),
        )

        for text, domains, expected in cases:
            with self.subTest(text=text):
                self.assert_command(
                    text,
                    expected,
                    domains=domains,
                )

    def test_complex_requests_use_normal_agent(self):
        for text in (
            "¿Por qué usa tanta VRAM el homelab?",
            "Diagnostica la GPU",
            "Reinicia Palworld",
            "Crea una copia de Palworld",
            "Estado del homelab y de Palworld",
            "Estado de los servicios del homelab",
            "Estado de Grafana y Prometheus",
        ):
            with self.subTest(text=text):
                self.assert_normal_agent(text)

    def test_continuations_are_not_direct(self):
        self.assertIsNone(
            classify_direct_homelab_request(
                "sí, revísalo",
                {"homelab"},
                continuation=True,
            )
        )

    def test_other_domains_are_not_direct(self):
        self.assert_normal_agent(
            "Estado del homelab",
            domains={"homelab", "web"},
        )


if __name__ == "__main__":
    unittest.main()
