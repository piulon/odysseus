"""Focused safety and correlation coverage for request routing telemetry."""

from types import SimpleNamespace
import time

import pytest

from src import llm_core
from src import adaptive_chat_router
from routes import chat_routes
from src.adaptive_routing import RoutingCandidate
from src.adaptive_routing_snapshot import AdaptiveRoutingSnapshot
from src.chat_model_router import ChatRoute, RouteTarget
from src.chat_route_authorizer import ChatRouteAuthorizationError
from src.routing_observability import (
    log_llm_dispatch,
    log_manual_authorized,
    log_routing_authorized,
    log_routing_decision,
    new_routing_trace,
)


def _route(*, reason, model, auto=True, lane="chat", endpoint_id="ep-1"):
    return SimpleNamespace(
        reason=reason,
        model=model,
        auto=auto,
        lane=lane,
        target=SimpleNamespace(endpoint_id=endpoint_id, model=model),
        manual_fallback=None,
    )


def _candidate(model="qwen3:14b", endpoint_id="ep-1"):
    return SimpleNamespace(
        endpoint_id=endpoint_id,
        model=model,
        endpoint_url="http://127.0.0.1:11434/v1/chat/completions",
    )


def test_trace_is_opaque_and_route_events_are_safe(caplog):
    trace = new_routing_trace()
    route = _route(reason="adaptive_chat", model="qwen3:14b")
    candidate = _candidate()

    with caplog.at_level("INFO", logger="src.routing_observability"):
        log_routing_decision(trace, route)
        log_routing_authorized(trace, candidate)
        log_llm_dispatch(
            trace,
            lane="chat",
            endpoint_id=candidate.endpoint_id,
            model=candidate.model,
            endpoint_url=candidate.endpoint_url,
        )

    assert len(trace) == 16
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=routing_decision" in text
    assert "event=routing_authorized" in text
    assert "event=llm_dispatch" in text
    assert text.count(trace) == 3
    for forbidden in (
        "prompt text",
        "sk-test-api-key",
        "Bearer secret-token",
        "user@example.test",
        "session-secret",
        "Authorization",
        "api-key",
    ):
        assert forbidden not in text
    assert "127.0.0.1" not in text


@pytest.mark.asyncio
async def test_stream_fallback_logs_same_trace_and_exact_attempted_models(monkeypatch, caplog):
    trace = new_routing_trace()

    async def fake_stream(url, model, messages, **kwargs):
        if model == "qwen3:14b":
            yield 'event: error\ndata: {"status": 502}\n\n'
            return
        yield 'data: {"delta":"ok"}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(llm_core, "stream_llm", fake_stream)
    with caplog.at_level("INFO"):
        chunks = [
            chunk
            async for chunk in llm_core.stream_llm_with_fallback(
                [
                    ("http://first.invalid/v1/chat/completions", "qwen3:14b", {}),
                    ("http://second.invalid/v1/chat/completions", "qwen3:4b-nothink", {}),
                ],
                [{"role": "user", "content": "prompt text"}],
                _routing_trace=trace,
                _routing_lane="chat",
            )
        ]

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert chunks[-1] == "data: [DONE]\n\n"
    assert f"event=llm_dispatch routing_trace={trace}" in text
    assert "dispatch_model=qwen3:14b" in text
    assert "dispatch_model=qwen3:4b-nothink" in text
    assert f"event=routing_fallback routing_trace={trace}" in text
    assert "from_model=qwen3:14b" in text
    assert "to_model=qwen3:4b-nothink" in text
    assert "prompt text" not in text


def test_ui_session_model_and_effective_adaptive_dispatch_are_distinct(caplog):
    """The persisted picker label may differ from the request target."""
    trace = new_routing_trace()
    route = _route(reason="adaptive_chat", model="qwen3:14b")
    candidate = _candidate(model="qwen3:14b")

    with caplog.at_level("INFO", logger="src.routing_observability"):
        log_routing_decision(trace, route)
        log_routing_authorized(trace, candidate)
        log_llm_dispatch(
            trace,
            lane="chat",
            endpoint_id=candidate.endpoint_id,
            model=candidate.model,
            endpoint_url=candidate.endpoint_url,
        )

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "qwen3:14b" in text
    assert "qwen3:4b-nothink" not in text
    assert "adaptive_chat" in text


def test_real_adaptive_resolver_chain_preserves_effective_target(monkeypatch, caplog):
    """Pin the production mismatch: picker model differs from Auto target."""
    session = SimpleNamespace(auto_route=True, model="qwen3:4b-nothink")
    legacy = _route(reason="auto_chat", model="qwen3:4b-nothink")
    candidate = RoutingCandidate(
        endpoint_id="ep-1",
        endpoint_url="http://127.0.0.1:11434/v1/chat/completions",
        model="qwen3:14b",
        node="ep-1",
        scope="local",
        capabilities=("tool_call", "reasoning"),
        context_tokens=40960,
    )
    snapshot = AdaptiveRoutingSnapshot(
        owner="alice",
        candidates=(candidate,),
        generated_at=time.time(),
    )
    monkeypatch.setattr(adaptive_chat_router, "resolve_chat_route", lambda *a, **k: legacy)
    monkeypatch.setattr(adaptive_chat_router, "get_adaptive_routing_snapshot", lambda *a, **k: snapshot)

    route = adaptive_chat_router.resolve_adaptive_chat_route(
        session,
        owner="alice",
        enabled=True,
        snapshot_ttl_seconds=60,
    )
    trace = new_routing_trace()
    with caplog.at_level("INFO", logger="src.routing_observability"):
        log_routing_decision(trace, route)
        log_routing_authorized(trace, candidate)
        log_llm_dispatch(
            trace,
            lane="chat",
            endpoint_id=candidate.endpoint_id,
            model=candidate.model,
            endpoint_url=candidate.endpoint_url,
        )

    events = "\n".join(record.getMessage() for record in caplog.records)
    assert route.reason == "adaptive_chat"
    assert route.target.model == "qwen3:14b"
    assert "selected_model=qwen3:14b" in events
    assert "authorized_model=qwen3:14b" in events
    assert "dispatch_model=qwen3:14b" in events
    assert "qwen3:4b-nothink" not in events


def test_legacy_route_reason_is_reported_without_adaptive(caplog):
    trace = new_routing_trace()
    with caplog.at_level("INFO", logger="src.routing_observability"):
        log_routing_decision(
            trace,
            _route(reason="auto_agent", model="qwen3:14b", lane="agent"),
        )
    text = caplog.records[0].getMessage()
    assert "reason=auto_agent" in text
    assert "adaptive_agent" not in text


@pytest.mark.parametrize("selector,reason", [
    (chat_routes._select_auto_stream_context_candidate, "chat"),
    (chat_routes._select_auto_agent_context_candidate, "agent"),
])
def test_context_authorization_fallback_logs_telemetry(monkeypatch, caplog, selector, reason):
    primary = _route(reason=f"auto_{reason}", model="primary-model")
    manual = ChatRoute(
        auto=False,
        lane="manual",
        target=RouteTarget(model="manual-model", endpoint_url="https://manual.invalid"),
        reason="manual_fallback",
    )
    candidate = _candidate(model="manual-model", endpoint_id="manual")
    monkeypatch.setattr(
        chat_routes,
        "_resolve_effective_auto_route_for_request",
        lambda *args, **kwargs: primary,
    )
    monkeypatch.setattr(chat_routes, "_manual_fallback_route", lambda *args, **kwargs: manual)

    def authorize(route, sess, auth):
        if route is primary:
            raise ChatRouteAuthorizationError("endpoint_unavailable")
        return candidate

    monkeypatch.setattr(chat_routes, "authorize_chat_route", authorize)
    with caplog.at_level("INFO", logger="src.routing_observability"):
        selector(SimpleNamespace(model="manual-model"), owner="alice", auth=object(), routing_trace="trace")
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=routing_fallback routing_trace=trace" in text
    assert "from_model=primary-model" in text
    assert "to_model=manual-model" in text
    assert "reason=authorization_unavailable" in text


def test_agent_empty_completion_fallback_logs_telemetry(caplog):
    with caplog.at_level("INFO", logger="src.routing_observability"):
        from src.routing_observability import log_routing_fallback

        log_routing_fallback(
            "trace",
            from_model="primary-model",
            to_model="manual-model",
            reason="empty_completion",
        )
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=routing_fallback routing_trace=trace" in text
    assert "from_model=primary-model" in text
    assert "to_model=manual-model" in text
    assert "reason=empty_completion" in text


def test_routing_telemetry_helpers_are_noops_without_trace(caplog):
    route = _route(reason="adaptive_chat", model="qwen3:14b")
    candidate = _candidate()
    with caplog.at_level("INFO", logger="src.routing_observability"):
        log_routing_decision(None, route)
        log_routing_authorized(None, candidate)
        log_manual_authorized(None, "qwen3:4b-nothink")
    assert not caplog.records
