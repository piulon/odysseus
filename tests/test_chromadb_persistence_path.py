"""Regression guard for ChromaDB persistence.

The pinned Chroma image runs with ``persist_path: /data`` in ``/config.yaml``.
Mounting the named volume anywhere else leaves the real Chroma database in the
container writable layer, where it is lost when the container is recreated.
"""

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent

COMPOSE_FILES = (
    "docker-compose.yml",
    "docker-compose.gpu-nvidia.yml",
    "docker-compose.gpu-amd.yml",
    "docker-compose.msi.yml",
)


@pytest.mark.parametrize("compose_name", COMPOSE_FILES)
def test_chromadb_named_volume_persists_actual_data_path(compose_name):
    compose = yaml.safe_load(
        (ROOT / compose_name).read_text(encoding="utf-8")
    )

    chromadb = compose["services"]["chromadb"]
    volumes = chromadb.get("volumes", [])

    assert "chromadb-data:/data" in volumes, (
        f"{compose_name}: Chroma persists to /data; "
        "the chromadb-data volume must be mounted there"
    )

    assert all(
        not str(volume).endswith(":/chroma/chroma")
        for volume in volumes
    ), (
        f"{compose_name}: /chroma/chroma is not the persist_path "
        "of the pinned Chroma image"
    )

    assert "chromadb-data" in (compose.get("volumes", {}) or {})


def test_no_tracked_compose_keeps_obsolete_chroma_mount():
    tracked = [
        path
        for path in ROOT.glob("docker-compose*.yml")
        if path.is_file()
    ]

    offenders = [
        path.name
        for path in tracked
        if "chromadb-data:/chroma/chroma"
        in path.read_text(encoding="utf-8")
    ]

    assert offenders == [], (
        "obsolete Chroma persistence mount remains in: "
        + ", ".join(offenders)
    )
