from dataclasses import replace
from types import SimpleNamespace

import pytest

from src import adaptive_chat_router
from src import model_capabilities as mc
from src.adaptive_routing import RoutingCandidate
from src.chat_model_router import ChatRoute, RouteTarget


def _session(*, auto_route=True, model="manual-model", endpoint_url="http://manual/chat"):
    return SimpleNamespace(
        auto_route=auto_route,
        model=model,
        endpoint_url=endpoint_url,
        headers={"Authorization": "Bearer secret"},
    )


def _legacy(*, auto=True, lane="chat", reason=None, fallback=True):
    if not auto:
        return ChatRoute(
            auto=False,
            lane="manual",
            target=RouteTarget(model="manual-model", endpoint_url="http://manual/chat"),
            reason="manual",
        )
    reason = reason or f"auto_{lane}"
    return ChatRoute(
        auto=True,
        lane=lane,
        target=RouteTarget(endpoint_id=f"configured-{lane}", model=f"configured-{lane}-model"),
        reason=reason,
        manual_fallback=(
            RouteTarget(model="manual-model", endpoint_url="http://manual/chat")
            if fallback
            else None
        ),
    )


def _candidate(
    *,
    endpoint_id="adaptive-endpoint",
    model="adaptive-model",
    url="http://adaptive/chat",
    preference=0,
    capabilities=(),
):
    return RoutingCandidate(
        endpoint_id=endpoint_id,
        endpoint_url=url,
        model=model,
        node=endpoint_id,
        scope="local",
        preference=preference,
        capabilities=tuple(capabilities),
    )


@pytest.fixture
def legacy(monkeypatch):
    route = _legacy()
    monkeypatch.setattr(adaptive_chat_router, "resolve_chat_route", lambda *a, **k: route)
    return route


def publish(monkeypatch, *, owner="alice", candidates=()):
    snapshot = SimpleNamespace(owner=owner, candidates=tuple(candidates))
    calls = []

    def lookup(actual_owner, *, max_age_seconds):
        calls.append((actual_owner, max_age_seconds))
        return snapshot

    monkeypatch.setattr(adaptive_chat_router, "get_adaptive_routing_snapshot", lookup)
    return calls


def test_manual_session_returns_exact_legacy_route(monkeypatch):
    route = _legacy(auto=False)
    monkeypatch.setattr(adaptive_chat_router, "resolve_chat_route", lambda *a, **k: route)
    assert adaptive_chat_router.resolve_adaptive_chat_route(_session(auto_route=False), enabled=True) is route


def test_disabled_returns_exact_legacy_route(legacy):
    assert adaptive_chat_router.resolve_adaptive_chat_route(_session(), enabled=False) is legacy


@pytest.mark.parametrize("snapshot", [None, SimpleNamespace(owner="alice", candidates=())])
def test_missing_or_empty_snapshot_returns_exact_legacy_route(monkeypatch, legacy, snapshot):
    monkeypatch.setattr(
        adaptive_chat_router,
        "get_adaptive_routing_snapshot",
        lambda *a, **k: snapshot,
    )
    assert adaptive_chat_router.resolve_adaptive_chat_route(_session(), owner="alice", enabled=True) is legacy


def test_stale_snapshot_returns_exact_legacy_route(monkeypatch, legacy):
    calls = []

    def stale_lookup(owner, *, max_age_seconds):
        calls.append((owner, max_age_seconds))
        return None

    monkeypatch.setattr(adaptive_chat_router, "get_adaptive_routing_snapshot", stale_lookup)
    assert adaptive_chat_router.resolve_adaptive_chat_route(_session(), owner="alice", enabled=True) is legacy
    assert calls == [("alice", adaptive_chat_router.DEFAULT_SNAPSHOT_TTL_SECONDS)]


def test_no_viable_adaptive_candidate_returns_exact_legacy_route(monkeypatch, legacy):
    publish(monkeypatch, candidates=[_candidate(preference=0, url="http://adaptive/chat")])
    # Make the only candidate unreachable through the scoring input.
    publish(monkeypatch, candidates=[replace(_candidate(), reachable=False)])
    assert adaptive_chat_router.resolve_adaptive_chat_route(_session(), owner="alice", enabled=True) is legacy


def test_internal_adaptive_exception_returns_exact_legacy_route(monkeypatch, legacy):
    def fail(*args, **kwargs):
        raise RuntimeError("do not expose")

    monkeypatch.setattr(adaptive_chat_router, "get_adaptive_routing_snapshot", fail)
    assert adaptive_chat_router.resolve_adaptive_chat_route(_session(), owner="alice", enabled=True) is legacy


def test_snapshot_lookup_is_owner_scoped_and_ttl_is_positive(monkeypatch, legacy):
    calls = publish(monkeypatch, owner="alice", candidates=[])
    adaptive_chat_router.resolve_adaptive_chat_route(
        _session(), owner="alice", enabled=True, snapshot_ttl_seconds=37
    )
    assert calls == [("alice", 37.0)]


def test_foreign_owner_snapshot_is_ignored(monkeypatch, legacy):
    publish(monkeypatch, owner="bob", candidates=[_candidate()])
    assert adaptive_chat_router.resolve_adaptive_chat_route(_session(), owner="alice", enabled=True) is legacy


@pytest.mark.parametrize(
    ("agent_mode", "workload"),
    [(False, "chat"), (True, "agent")],
)
def test_request_profile_workload(monkeypatch, legacy, agent_mode, workload):
    legacy = _legacy(lane=workload)
    monkeypatch.setattr(
        adaptive_chat_router,
        "resolve_chat_route",
        lambda *a, **k: legacy,
    )
    publish(monkeypatch, candidates=[_candidate()])
    captured = {}

    def decision(profile, candidates):
        captured["profile"] = profile
        return SimpleNamespace(primary=next(iter(candidates)))

    monkeypatch.setattr(adaptive_chat_router, "build_routing_decision", decision)
    route = adaptive_chat_router.resolve_adaptive_chat_route(
        _session(), owner="alice", agent_mode=agent_mode, enabled=True
    )
    assert route.reason == f"adaptive_{workload}"
    assert captured["profile"].workload == workload
    assert captured["profile"].required_capabilities == (
        (mc.CAP_TOOL_CALL,) if agent_mode else ()
    )
    assert captured["profile"].preferred_capabilities == ()


@pytest.mark.parametrize("agent_mode", [False, True])
def test_legacy_target_preference(monkeypatch, legacy, agent_mode):
    legacy = _legacy(lane="agent" if agent_mode else "chat")
    monkeypatch.setattr(adaptive_chat_router, "resolve_chat_route", lambda *a, **k: legacy)
    publish(monkeypatch, candidates=[_candidate()])
    captured = {}

    def decision(profile, candidates):
        captured["profile"] = profile
        return SimpleNamespace(primary=next(iter(candidates)))
    monkeypatch.setattr(adaptive_chat_router, "build_routing_decision", decision)
    adaptive_chat_router.resolve_adaptive_chat_route(_session(), owner="alice", agent_mode=agent_mode, enabled=True)
    assert captured["profile"].target_preferences == (
        (legacy.target.endpoint_id, legacy.target.model, adaptive_chat_router.LEGACY_TARGET_PREFERENCE_BONUS),
    )


def test_unconfigured_legacy_target_has_no_preference(monkeypatch):
    legacy = _legacy(reason="auto_chat_unconfigured")
    monkeypatch.setattr(adaptive_chat_router, "resolve_chat_route", lambda *a, **k: legacy)
    publish(monkeypatch, candidates=[_candidate()])
    captured = {}
    def decision(profile, candidates):
        captured["profile"] = profile
        return SimpleNamespace(primary=next(iter(candidates)))
    monkeypatch.setattr(adaptive_chat_router, "build_routing_decision", decision)
    adaptive_chat_router.resolve_adaptive_chat_route(_session(), owner="alice", enabled=True)
    assert captured["profile"].target_preferences == ()


def test_adaptive_target_contains_only_endpoint_id_and_model(monkeypatch, legacy):
    publish(monkeypatch, candidates=[_candidate(url="http://secret-probe-url/chat")])
    route = adaptive_chat_router.resolve_adaptive_chat_route(_session(), owner="alice", enabled=True)
    assert route.target.endpoint_id == "adaptive-endpoint"
    assert route.target.model == "adaptive-model"
    assert route.target.endpoint_url is None
    assert "secret-probe-url" not in route.reason


def test_existing_manual_fallback_is_preserved(monkeypatch, legacy):
    publish(monkeypatch, candidates=[_candidate()])
    route = adaptive_chat_router.resolve_adaptive_chat_route(_session(), owner="alice", enabled=True)
    assert route.manual_fallback is legacy.manual_fallback


def test_snapshot_url_cannot_suppress_manual_fallback(monkeypatch, legacy):
    publish(
        monkeypatch,
        candidates=[_candidate(model="manual-model", url="http://manual/chat")],
    )
    route = adaptive_chat_router.resolve_adaptive_chat_route(
        _session(),
        owner="alice",
        enabled=True,
    )
    assert route.manual_fallback is legacy.manual_fallback


def test_session_is_not_mutated(monkeypatch, legacy):
    session = _session()
    before = (session.auto_route, session.model, session.endpoint_url, dict(session.headers))
    publish(monkeypatch, candidates=[_candidate()])
    adaptive_chat_router.resolve_adaptive_chat_route(session, owner="alice", enabled=True)
    assert (session.auto_route, session.model, session.endpoint_url, dict(session.headers)) == before



def test_agent_rejects_non_tool_candidate_even_with_higher_preference(
    monkeypatch,
):
    legacy = _legacy(lane="agent")
    monkeypatch.setattr(
        adaptive_chat_router,
        "resolve_chat_route",
        lambda *a, **k: legacy,
    )

    publish(
        monkeypatch,
        candidates=[
            _candidate(
                endpoint_id="plain-endpoint",
                model="plain-model",
                preference=1000,
                capabilities=(),
            ),
            _candidate(
                endpoint_id="tool-endpoint",
                model="tool-model",
                preference=0,
                capabilities=(mc.CAP_TOOL_CALL,),
            ),
        ],
    )

    route = adaptive_chat_router.resolve_adaptive_chat_route(
        _session(),
        owner="alice",
        agent_mode=True,
        enabled=True,
    )

    assert route.reason == "adaptive_agent"
    assert route.target.endpoint_id == "tool-endpoint"
    assert route.target.model == "tool-model"


def test_agent_without_tool_capable_candidate_returns_exact_legacy_route(
    monkeypatch,
):
    legacy = _legacy(lane="agent")
    monkeypatch.setattr(
        adaptive_chat_router,
        "resolve_chat_route",
        lambda *a, **k: legacy,
    )

    publish(
        monkeypatch,
        candidates=[
            _candidate(
                endpoint_id="plain-endpoint",
                model="plain-model",
                preference=1000,
                capabilities=(),
            )
        ],
    )

    route = adaptive_chat_router.resolve_adaptive_chat_route(
        _session(),
        owner="alice",
        agent_mode=True,
        enabled=True,
    )

    assert route is legacy
