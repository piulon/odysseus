import json

from src import settings
from src.adaptive_routing import RoutingCandidate
from src.adaptive_routing_snapshot import (
    clear_adaptive_routing_snapshot,
    get_adaptive_routing_snapshot,
    publish_adaptive_routing_snapshot,
)


def _candidate():
    return RoutingCandidate(
        endpoint_id="ep-1",
        model="model-1",
        endpoint_url="http://127.0.0.1:11434/api/chat",
        node="ep-1",
        scope="local",
        reachable=True,
    )


def _configure_tmp_settings(monkeypatch, tmp_path, enabled):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"adaptive_routing_enabled": enabled}))
    monkeypatch.setattr(settings, "SETTINGS_FILE", str(path))
    settings._settings_cache = None
    return path


def _publish():
    return publish_adaptive_routing_snapshot(
        "alice",
        (_candidate(),),
        generated_at=100.0,
    )


def _current():
    return get_adaptive_routing_snapshot(
        "alice",
        max_age_seconds=60.0,
        now=101.0,
    )


def setup_function():
    clear_adaptive_routing_snapshot()
    settings._settings_cache = None


def teardown_function():
    clear_adaptive_routing_snapshot()
    settings._settings_cache = None


def test_false_to_true_clears_existing_snapshots(monkeypatch, tmp_path):
    _configure_tmp_settings(monkeypatch, tmp_path, False)
    _publish()

    updated = settings.load_settings().copy()
    updated["adaptive_routing_enabled"] = True
    settings.save_settings(updated)

    assert _current() is None


def test_true_to_false_clears_existing_snapshots(monkeypatch, tmp_path):
    _configure_tmp_settings(monkeypatch, tmp_path, True)
    _publish()

    updated = settings.load_settings().copy()
    updated["adaptive_routing_enabled"] = False
    settings.save_settings(updated)

    assert _current() is None


def test_unrelated_save_does_not_clear_snapshot_when_gate_stays_false(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_settings(monkeypatch, tmp_path, False)
    published = _publish()

    updated = settings.load_settings().copy()
    updated["document_writing_style"] = "concise"
    settings.save_settings(updated)

    assert _current() == published


def test_unrelated_save_does_not_clear_snapshot_when_gate_stays_true(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_settings(monkeypatch, tmp_path, True)
    published = _publish()

    updated = settings.load_settings().copy()
    updated["document_writing_style"] = "concise"
    settings.save_settings(updated)

    assert _current() == published


def test_only_literal_true_counts_as_enabled(monkeypatch, tmp_path):
    _configure_tmp_settings(monkeypatch, tmp_path, "true")
    _publish()

    updated = settings.load_settings().copy()
    updated["adaptive_routing_enabled"] = True
    settings.save_settings(updated)

    assert _current() is None
