import json
from pathlib import Path

from src.adaptive_routing_app import (
    adaptive_routing_owners_from_auth_file,
    adaptive_routing_refresh_enabled,
)


def test_adaptive_refresh_is_disabled_by_default():
    assert adaptive_routing_refresh_enabled({}) is False
    assert adaptive_routing_refresh_enabled({"ODYSSEUS_ADAPTIVE_ROUTING_REFRESH": "0"}) is False
    assert adaptive_routing_refresh_enabled({"ODYSSEUS_ADAPTIVE_ROUTING_REFRESH": "false"}) is False


def test_adaptive_refresh_accepts_explicit_truthy_values():
    for value in ("1", "true", "TRUE", "yes", "on"):
        assert adaptive_routing_refresh_enabled(
            {"ODYSSEUS_ADAPTIVE_ROUTING_REFRESH": value}
        ) is True


def test_missing_auth_file_uses_single_user_owner(tmp_path):
    assert adaptive_routing_owners_from_auth_file(
        tmp_path / "missing.json"
    ) == ("",)


def test_empty_users_uses_single_user_owner(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"users": {}}), encoding="utf-8")

    assert adaptive_routing_owners_from_auth_file(path) == ("",)


def test_auth_users_are_normalized_and_sorted(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps(
            {
                "users": {
                    " bob ": {},
                    "alice": {},
                    "": {},
                }
            }
        ),
        encoding="utf-8",
    )

    assert adaptive_routing_owners_from_auth_file(path) == (
        "alice",
        "bob",
    )


def test_malformed_auth_json_fails_closed(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text("{bad json", encoding="utf-8")

    assert adaptive_routing_owners_from_auth_file(path) == ()


def test_invalid_users_shape_fails_closed(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"users": ["alice"]}), encoding="utf-8")

    assert adaptive_routing_owners_from_auth_file(path) == ()


def test_app_wires_default_off_refresher_as_tracked_startup_task():
    source = Path("app.py").read_text(encoding="utf-8")
    compact = "".join(source.split())

    assert "adaptive_routing_refresh_enabled" in source
    assert "if adaptive_routing_refresh_enabled():" in source
    assert (
        "_startup_tasks.append(asyncio.create_task("
        "adaptive_routing_refresh_loop(_adaptive_routing_owners)))"
        in compact
    )


def test_app_owner_provider_uses_auth_file():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "adaptive_routing_owners_from_auth_file(AUTH_FILE)" in source
