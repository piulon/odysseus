from types import SimpleNamespace

import pytest

from routes import chat_routes
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

    monkeypatch.setattr(chat_routes, "_ADAPTIVE_ROUTING_ENABLED", False)
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

    monkeypatch.setattr(chat_routes, "_ADAPTIVE_ROUTING_ENABLED", True)
    monkeypatch.setattr(
        chat_routes,
        "_ADAPTIVE_ROUTING_SNAPSHOT_TTL_SECONDS",
        37.0,
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


def test_adaptive_runtime_flag_is_disabled_by_default():
    assert chat_routes._ADAPTIVE_ROUTING_ENABLED is False
