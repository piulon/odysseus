from types import SimpleNamespace

import pytest

from routes import chat_routes
from src import settings as settings_module
from src.chat_model_router import ChatRoute, RouteTarget


def _route(reason: str) -> ChatRoute:
    return ChatRoute(
        auto=True,
        lane="chat",
        target=RouteTarget(
            endpoint_id="endpoint-1",
            model="model-1",
        ),
        reason=reason,
    )


def test_effective_auto_route_defaults_to_legacy(monkeypatch):
    session = SimpleNamespace(auto_route=True)
    legacy = _route("legacy")

    legacy_calls = []
    adaptive_calls = []

    def resolve_legacy(sess, *, owner, agent_mode):
        legacy_calls.append((sess, owner, agent_mode))
        return legacy

    def resolve_adaptive(*args, **kwargs):
        adaptive_calls.append((args, kwargs))
        pytest.fail("Adaptive resolver must not run while disabled")

    monkeypatch.setattr(
        settings_module,
        "get_setting",
        lambda key, default=None: {
            "adaptive_routing_enabled": False,
            "adaptive_routing_snapshot_ttl_seconds": 60,
        }.get(key, default),
    )
    monkeypatch.setattr(chat_routes, "resolve_chat_route", resolve_legacy)
    monkeypatch.setattr(
        chat_routes,
        "resolve_adaptive_chat_route",
        resolve_adaptive,
    )

    result = chat_routes._resolve_effective_auto_route(
        session,
        owner="alice",
        agent_mode=False,
    )

    assert result is legacy
    assert legacy_calls == [(session, "alice", False)]
    assert adaptive_calls == []


@pytest.mark.parametrize("agent_mode", [False, True])
def test_effective_auto_route_opt_in_uses_adaptive(
    monkeypatch,
    agent_mode,
):
    session = SimpleNamespace(auto_route=True)
    adaptive = _route("adaptive")
    calls = []

    def resolve_legacy(*args, **kwargs):
        pytest.fail("Legacy resolver must not be called directly by the wiring helper")

    def resolve_adaptive(
        sess,
        *,
        owner,
        agent_mode,
        enabled,
        snapshot_ttl_seconds,
    ):
        calls.append(
            {
                "session": sess,
                "owner": owner,
                "agent_mode": agent_mode,
                "enabled": enabled,
                "snapshot_ttl_seconds": snapshot_ttl_seconds,
            }
        )
        return adaptive

    monkeypatch.setattr(
        settings_module,
        "get_setting",
        lambda key, default=None: {
            "adaptive_routing_enabled": True,
            "adaptive_routing_snapshot_ttl_seconds": 37,
        }.get(key, default),
    )
    monkeypatch.setattr(chat_routes, "resolve_chat_route", resolve_legacy)
    monkeypatch.setattr(
        chat_routes,
        "resolve_adaptive_chat_route",
        resolve_adaptive,
    )

    result = chat_routes._resolve_effective_auto_route(
        session,
        owner="alice",
        agent_mode=agent_mode,
    )

    assert result is adaptive
    assert calls == [
        {
            "session": session,
            "owner": "alice",
            "agent_mode": agent_mode,
            "enabled": True,
            "snapshot_ttl_seconds": 37.0,
        }
    ]


def test_adaptive_runtime_settings_are_global_and_disabled_by_default():
    assert settings_module.DEFAULT_SETTINGS["adaptive_routing_enabled"] is False
    assert (
        settings_module.DEFAULT_SETTINGS["adaptive_routing_snapshot_ttl_seconds"]
        == 60
    )
    assert "adaptive_routing_enabled" not in settings_module._PER_USER_KEYS
    assert (
        "adaptive_routing_snapshot_ttl_seconds"
        not in settings_module._PER_USER_KEYS
    )


@pytest.mark.parametrize("raw_ttl", [None, "", "invalid", 0, -1])
def test_adaptive_runtime_invalid_ttl_falls_back_safely(monkeypatch, raw_ttl):
    def get_setting(key, default=None):
        if key == "adaptive_routing_enabled":
            return True
        if key == "adaptive_routing_snapshot_ttl_seconds":
            return raw_ttl
        return default

    monkeypatch.setattr(settings_module, "get_setting", get_setting)

    enabled, ttl = chat_routes._adaptive_routing_runtime_config()

    assert enabled is True
    assert ttl == chat_routes._ADAPTIVE_ROUTING_SNAPSHOT_TTL_SECONDS_DEFAULT


@pytest.mark.parametrize("malformed_enabled", ["true", "1", 1, {}, []])
def test_adaptive_runtime_requires_literal_true(monkeypatch, malformed_enabled):
    monkeypatch.setattr(
        settings_module,
        "get_setting",
        lambda key, default=None: (
            malformed_enabled
            if key == "adaptive_routing_enabled"
            else default
        ),
    )

    enabled, _ttl = chat_routes._adaptive_routing_runtime_config()

    assert enabled is False


def test_adaptive_runtime_settings_failure_falls_back_to_disabled(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(settings_module, "get_setting", fail)

    enabled, ttl = chat_routes._adaptive_routing_runtime_config()

    assert enabled is False
    assert ttl == chat_routes._ADAPTIVE_ROUTING_SNAPSHOT_TTL_SECONDS_DEFAULT
