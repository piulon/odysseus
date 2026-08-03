import unittest

from src.agent_tools.homelab_tools import (
    is_direct_homelab_status_request,
)


class HomelabFastPathTests(unittest.TestCase):
    def assert_direct(self, text):
        self.assertTrue(
            is_direct_homelab_status_request(
                text,
                {"homelab"},
            ),
            text,
        )

    def assert_normal_agent(self, text):
        self.assertFalse(
            is_direct_homelab_status_request(
                text,
                {"homelab"},
            ),
            text,
        )

    def test_simple_status_requests_are_direct(self):
        for text in (
            "Estado del homelab",
            "¿Cuál es el estado del homelab?",
            "¿Cómo está el homelab?",
            "Homelab status",
            "Quin és l'estat del homelab?",
        ):
            with self.subTest(text=text):
                self.assert_direct(text)

    def test_specific_requests_use_normal_agent(self):
        for text in (
            "Estado de Palworld",
            "Revisa Grafana",
            "¿Por qué usa tanta VRAM el homelab?",
            "Diagnostica el homelab",
            "Reinicia el homelab",
            "Estado de los servicios del homelab",
            "Estado del homelab y de Palworld",
        ):
            with self.subTest(text=text):
                self.assert_normal_agent(text)

    def test_continuations_are_not_direct(self):
        self.assertFalse(
            is_direct_homelab_status_request(
                "sí, revísalo",
                {"homelab"},
                continuation=True,
            )
        )

    def test_other_domains_are_not_direct(self):
        self.assertFalse(
            is_direct_homelab_status_request(
                "Estado del homelab",
                {"homelab", "web"},
            )
        )


if __name__ == "__main__":
    unittest.main()
