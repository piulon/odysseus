"""Guard the centralized SearXNG architecture against unsafe regression.

SearXNG is no longer part of the local Odysseus compose stack.  The base
compose must not reintroduce a local SearXNG container, and Odysseus must keep
using the centralized service through the external ``ai_net`` network.
"""

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILES = (
    ROOT / "docker-compose.yml",
    ROOT / "docker-compose.gpu-nvidia.yml",
    ROOT / "docker-compose.gpu-amd.yml",
)


def _load_compose(compose_file):
    return yaml.safe_load(compose_file.read_text(encoding="utf-8"))


def _environment_map(service):
    result = {}
    for entry in service.get("environment", []):
        if isinstance(entry, str):
            key, sep, value = entry.partition("=")
            if sep:
                result[key] = value
        elif isinstance(entry, dict):
            result.update(entry)
    return result


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_compose_does_not_run_local_searxng_image(compose_file):
    compose = _load_compose(compose_file)
    services = compose.get("services", {})

    assert "searxng" not in services, (
        "SearXNG must remain a centralized service, not a local compose service"
    )
    assert all(
        "searxng/searxng" not in str(service.get("image", "")).lower()
        for service in services.values()
    ), f"{compose_file.name} must not reintroduce a local SearXNG image"

    volumes = compose.get("volumes", {}) or {}
    assert "searxng-data" not in volumes, (
        "local searxng-data must not return while SearXNG is centralized"
    )


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_odysseus_keeps_centralized_searxng_connection(compose_file):
    compose = _load_compose(compose_file)

    odysseus = compose["services"]["odysseus"]
    environment = _environment_map(odysseus)

    assert environment.get("SEARXNG_INSTANCE") == "http://searxng:8080"
    assert "ai_net" in odysseus.get("networks", [])

    ai_net = (compose.get("networks", {}) or {}).get("ai_net")
    assert isinstance(ai_net, dict)
    assert ai_net.get("external") is True
