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

    def test_named_service_status_commands(self):
        cases = (
            ("Estado de Grafana", "grafana"),
            ("Estado de Prometheus", "prometheus"),
            ("¿Cómo está Ollama?", "ollama"),
            ("Caddy status", "caddy"),
            ("Estado de ChromaDB", "chromadb"),
            ("Estado de Chroma DB", "chromadb"),
            ("Estado de SearXNG", "searxng"),
            ("Estado de SearX NG", "searxng"),
            ("Estado de Portainer", "portainer"),
            ("Estado de Homepage", "homepage"),
            ("Estado de Home Page", "homepage"),
            ("Estado de Open WebUI", "open-webui"),
            ("Estado de Open Web UI", "open-webui"),
            ("Estado de OpenWebUI", "open-webui"),
            ("Estado de ComfyUI", "comfyui"),
            ("Estado de Comfy UI", "comfyui"),
            ("Com està Grafana?", "grafana"),
            ("Estat de Prometheus", "prometheus"),
        )

        for text, service in cases:
            with self.subTest(text=text):
                self.assert_command(
                    text,
                    {
                        "action": "service",
                        "service": service,
                    },
                    domains=set(),
                )


    def test_service_alias_accepts_cookbook_domain(self):
        self.assert_command(
            "¿Cómo está Ollama?",
            {
                "action": "service",
                "service": "ollama",
            },
            domains={"cookbook"},
        )

    def test_open_webui_accepts_ui_domain(self):
        self.assert_command(
            "Estado de Open WebUI",
            {
                "action": "service",
                "service": "open-webui",
            },
            domains={"ui"},
        )

    def test_service_domain_overrides_are_isolated(self):
        cases = (
            ("Estado de Grafana", {"cookbook"}),
            ("Estado de Grafana", {"ui"}),
            ("Estado de ComfyUI", {"ui"}),
            ("Estado de Open WebUI", {"cookbook"}),
            ("Estado de Ollama", {"ui"}),
        )

        for text, domains in cases:
            with self.subTest(
                text=text,
                domains=domains,
            ):
                self.assert_normal_agent(
                    text,
                    domains=domains,
                )


    def test_cookbook_domain_does_not_open_other_paths(self):
        for text in (
            "Estado del homelab",
            "Estado de la GPU",
            "Estado de Palworld",
            "Estado de las copias de Palworld",
        ):
            with self.subTest(text=text):
                self.assert_normal_agent(
                    text,
                    domains={"cookbook"},
                )

    def test_web_domain_does_not_open_service_path(self):
        self.assert_normal_agent(
            "Estado de Ollama",
            domains={"web"},
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
            "¿Por qué está caído Prometheus?",
            "Diagnostica Ollama",
            "Reinicia Caddy",
            "Detén Portainer",
            "Actualiza Open WebUI",
            "Revisa los logs de ChromaDB",
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
