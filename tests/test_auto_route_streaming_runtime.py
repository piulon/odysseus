import asyncio
import json
from types import SimpleNamespace

import pytest

from core.models import Session
from routes import chat_routes
from src.chat_model_router import ChatRoute, RouteTarget
from src.chat_route_authorizer import (
    AuthorizedChatRoute,
    ChatRouteAuthorizationError,
)
from src.llm_core import ChatDispatchError


def _route():
    return ChatRoute(
        auto=True,
        lane="chat",
        target=RouteTarget(endpoint_id="auto", model="auto-model"),
        reason="auto_chat",
        manual_fallback=RouteTarget(
            model="manual-model",
            endpoint_url="https://manual.invalid/v1/chat/completions",
        ),
    )


def _fallback_route():
    return ChatRoute(
        auto=False,
        lane="manual",
        target=RouteTarget(
            model="manual-model",
            endpoint_url="https://manual.invalid/v1/chat/completions",
        ),
        reason="manual_fallback",
    )


def _authorized(model, marker, *, auto=True):
    return AuthorizedChatRoute(
        auto=auto,
        lane="chat" if auto else "manual",
        reason="auto_chat" if auto else "manual_fallback",
        model=model,
        endpoint_id=marker,
        endpoint_url=f"https://{marker}.invalid/v1/chat/completions",
        headers={"Authorization": f"Bearer {marker}"},
    )


async def _collect(stream):
    return [chunk async for chunk in stream]


def _run(monkeypatch, per_model, *, authorize=None, context_route=None):
    primary = _route()
    fallback = _fallback_route()
    session = Session(
        "s1",
        "Auto",
        "https://manual.invalid/v1/chat/completions",
        "manual-model",
        auto_route=True,
        headers={"kept": "manual"},
        owner="alice",
    )
    calls = []
    auth_calls = []

    def default_authorize(route, sess, auth):
        auth_calls.append(route.reason)
        if route is primary:
            return _authorized("auto-model", "primary")
        return _authorized("manual-model", "fallback", auto=False)

    async def fake_stream(url, model, messages, headers=None, **kwargs):
        calls.append((url, model, dict(headers or {}), kwargs))
        outcome = per_model(model)
        if isinstance(outcome, BaseException):
            raise outcome
        for chunk in outcome:
            yield chunk

    monkeypatch.setattr(chat_routes, "authorize_chat_route", authorize or default_authorize)
    monkeypatch.setattr(chat_routes, "stream_llm", fake_stream)
    state = chat_routes._AutoStreamState(requested_model="auto-model")
    stream = chat_routes._stream_auto_chat_with_fallback(
        selected_primary=primary,
        context_route=context_route or primary,
        manual_fallback=fallback,
        sess=session,
        auth=SimpleNamespace(),
        messages=[{"role": "user", "content": "hello"}],
        stream_kwargs={},
        state=state,
    )
    chunks = asyncio.run(_collect(stream))
    return chunks, state, calls, auth_calls, session, primary, fallback


def test_auto_primary_stream_commits_requested_model(monkeypatch):
    expected_session = (
        "manual-model",
        "https://manual.invalid/v1/chat/completions",
        {"kept": "manual"},
        True,
    )
    chunks, state, calls, auth_calls, session, *_ = _run(
        monkeypatch,
        lambda model: ['data: {"delta": "hello"}\n\n', "data: [DONE]\n\n"],
    )

    assert chunks == ['data: {"delta": "hello"}\n\n', "data: [DONE]\n\n"]
    assert state.winner_model == "auto-model"
    assert auth_calls == ["auto_chat"]
    assert calls[0][1:3] == ("auto-model", {"Authorization": "Bearer primary"})
    assert (
        session.model,
        session.endpoint_url,
        session.headers,
        session.auto_route,
    ) == expected_session


@pytest.mark.parametrize("error", [
    ChatDispatchError(504, "secret", kind="timeout"),
    ChatDispatchError(502, "secret", kind="network"),
    ChatDispatchError(408, "secret", kind="upstream_status"),
    ChatDispatchError(429, "secret", kind="upstream_status"),
    ChatDispatchError(503, "secret", kind="upstream_status"),
])
def test_recoverable_precommit_failure_uses_jit_fallback(monkeypatch, error):
    chunks, state, calls, auth_calls, *_ = _run(
        monkeypatch,
        lambda model: error if model == "auto-model" else [
            'data: {"delta": "fallback"}\n\n',
            "data: [DONE]\n\n",
        ],
    )

    assert auth_calls == ["auto_chat", "manual_fallback"]
    assert [call[1] for call in calls] == ["auto-model", "manual-model"]
    assert calls[0][2] == {"Authorization": "Bearer primary"}
    assert calls[1][2] == {"Authorization": "Bearer fallback"}
    assert state.winner_model == "manual-model"
    fallback = next(json.loads(c[6:]) for c in chunks if '"fallback"' in c)
    assert fallback["selected_model"] == "auto-model"
    assert fallback["answered_by"] == "manual-model"


@pytest.mark.parametrize("error", [
    ChatDispatchError(400, "Bearer SECRET", kind="upstream_status"),
    ChatDispatchError(401, "Bearer SECRET", kind="upstream_status"),
    ChatDispatchError(403, "Bearer SECRET", kind="upstream_status"),
    ChatDispatchError(404, "Bearer SECRET", kind="upstream_status"),
    ChatDispatchError(502, "Bearer SECRET", kind="invalid_response"),
    RuntimeError("Bearer SECRET https://private.internal"),
])
def test_terminal_precommit_failure_is_sanitized_without_fallback(monkeypatch, error):
    chunks, state, calls, auth_calls, *_ = _run(monkeypatch, lambda model: error)

    assert len(calls) == 1
    assert auth_calls == ["auto_chat"]
    assert state.winner_model is None
    public = "".join(chunks)
    assert public.startswith("event: error")
    assert "SECRET" not in public
    assert "private.internal" not in public


@pytest.mark.parametrize("first", [
    'data: {"delta": "visible"}\n\n',
    'data: {"delta": "reasoning", "thinking": true}\n\n',
    'data: {"type": "tool_call_delta", "index": 0, "arg_delta": "{"}\n\n',
    'data: {"type": "tool_calls", "calls": [{"name": "lookup"}]}\n\n',
])
def test_postcommit_failure_never_falls_back(monkeypatch, first):
    error = ChatDispatchError(503, "secret", kind="upstream_status")

    def outcomes(model):
        async def impossible():
            yield None
        return []

    primary = _route()
    fallback = _fallback_route()
    session = Session("s1", "Auto", "https://manual", "manual-model", auto_route=True)
    calls = []

    def authorize(route, sess, auth):
        calls.append(("auth", route.reason))
        return _authorized(
            "auto-model" if route is primary else "manual-model",
            "primary" if route is primary else "fallback",
            auto=route is primary,
        )

    async def fake_stream(url, model, messages, headers=None, **kwargs):
        calls.append(("stream", model))
        yield first
        raise error

    monkeypatch.setattr(chat_routes, "authorize_chat_route", authorize)
    monkeypatch.setattr(chat_routes, "stream_llm", fake_stream)
    state = chat_routes._AutoStreamState(requested_model="auto-model")
    chunks = asyncio.run(_collect(chat_routes._stream_auto_chat_with_fallback(
        selected_primary=primary,
        context_route=primary,
        manual_fallback=fallback,
        sess=session,
        auth=SimpleNamespace(),
        messages=[],
        stream_kwargs={},
        state=state,
    )))

    assert calls == [("auth", "auto_chat"), ("stream", "auto-model")]
    assert chunks[0] == first
    assert chunks[-1].startswith("event: error")
    assert state.winner_model == "auto-model"


@pytest.mark.parametrize("primary_chunks", [[], ["data: [DONE]\n\n"]])
def test_empty_primary_stream_falls_back(monkeypatch, primary_chunks):
    chunks, state, calls, *_ = _run(
        monkeypatch,
        lambda model: primary_chunks if model == "auto-model" else [
            'data: {"delta": "fallback"}\n\n',
            "data: [DONE]\n\n",
        ],
    )

    assert [call[1] for call in calls] == ["auto-model", "manual-model"]
    assert state.winner_model == "manual-model"
    assert any('"fallback"' in chunk for chunk in chunks)


def test_precommit_metadata_from_discarded_primary_is_not_emitted(monkeypatch):
    primary_metadata = [
        'data: {"type": "model_actual", "requested_model": "auto-model", "model": "discarded"}\n\n',
        'data: {"type": "usage", "data": {"input_tokens": 1}}\n\n',
        'event: error\ndata: {"status": 503}\n\n',
    ]
    chunks, state, *_ = _run(
        monkeypatch,
        lambda model: primary_metadata if model == "auto-model" else [
            'data: {"delta": "fallback"}\n\n',
            "data: [DONE]\n\n",
        ],
    )

    public = "".join(chunks)
    assert "discarded" not in public
    assert '"input_tokens": 1' not in public
    assert state.winner_model == "manual-model"


def test_buffered_metadata_flushes_when_primary_commits(monkeypatch):
    chunks, state, *_ = _run(
        monkeypatch,
        lambda model: [
            'data: {"type": "model_actual", "requested_model": "auto-model", "model": "actual-model"}\n\n',
            'data: {"type": "usage", "data": {"input_tokens": 1}}\n\n',
            'data: {"delta": "answer"}\n\n',
            "data: [DONE]\n\n",
        ],
    )

    assert [json.loads(c[6:]).get("type") for c in chunks[:2]] == ["model_actual", "usage"]
    assert state.winner_model == "actual-model"


def test_cancellation_never_authorizes_fallback(monkeypatch):
    chunks = []
    primary = _route()
    fallback = _fallback_route()
    session = Session("s1", "Auto", "https://manual", "manual-model", auto_route=True)
    auth_calls = []

    def authorize(route, sess, auth):
        auth_calls.append(route.reason)
        return _authorized("auto-model", "primary")

    async def fake_stream(*args, **kwargs):
        raise asyncio.CancelledError()
        yield

    monkeypatch.setattr(chat_routes, "authorize_chat_route", authorize)
    monkeypatch.setattr(chat_routes, "stream_llm", fake_stream)
    state = chat_routes._AutoStreamState(requested_model="auto-model")

    async def run():
        async for chunk in chat_routes._stream_auto_chat_with_fallback(
            selected_primary=primary,
            context_route=primary,
            manual_fallback=fallback,
            sess=session,
            auth=SimpleNamespace(),
            messages=[],
            stream_kwargs={},
            state=state,
        ):
            chunks.append(chunk)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    assert auth_calls == ["auto_chat"]


def test_candidate_unavailable_at_jit_authorizes_fallback_only_then(monkeypatch):
    primary = _route()
    fallback = _fallback_route()
    attempts = []

    def authorize(route, sess, auth):
        attempts.append(route.reason)
        if route.reason == "auto_chat":
            raise ChatRouteAuthorizationError("endpoint_not_found")
        return _authorized("manual-model", "fallback", auto=False)

    chunks, state, calls, _, *_ = _run(
        monkeypatch,
        lambda model: ['data: {"delta": "fallback"}\n\n', "data: [DONE]\n\n"],
        authorize=authorize,
    )

    assert attempts == ["auto_chat", "manual_fallback"]
    assert [call[1] for call in calls] == ["manual-model"]
    assert state.winner_model == "manual-model"


@pytest.mark.parametrize(
    "failure_code",
    ["endpoint_not_found", "credentials_unavailable", "model_hidden"],
)
def test_precontext_candidate_unavailable_builds_with_manual_fallback(
    monkeypatch, failure_code
):
    session = Session(
        "s1",
        "Auto",
        "https://manual.invalid/v1/chat/completions",
        "manual-model",
        auto_route=True,
    )
    selected = _route()
    attempts = []
    monkeypatch.setattr(chat_routes, "resolve_chat_route", lambda *a, **k: selected)

    def authorize(route, sess, auth):
        attempts.append(route.reason)
        if route is selected:
            raise ChatRouteAuthorizationError(failure_code)
        return _authorized("manual-model", "fallback", auto=False)

    monkeypatch.setattr(chat_routes, "authorize_chat_route", authorize)

    result = chat_routes._select_auto_stream_context_candidate(
        session,
        owner="alice",
        auth=SimpleNamespace(),
    )

    assert result[1] == "auto-model"
    assert result[3].reason == "manual_fallback"
    assert result[4].model == "manual-model"
    assert attempts == ["auto_chat", "manual_fallback"]


def test_precontext_model_denial_does_not_authorize_fallback(monkeypatch):
    session = Session(
        "s1",
        "Auto",
        "https://manual.invalid/v1/chat/completions",
        "manual-model",
        auto_route=True,
    )
    selected = _route()
    attempts = []
    monkeypatch.setattr(chat_routes, "resolve_chat_route", lambda *a, **k: selected)

    def authorize(route, sess, auth):
        attempts.append(route.reason)
        raise ChatRouteAuthorizationError("model_not_allowed")

    monkeypatch.setattr(chat_routes, "authorize_chat_route", authorize)

    with pytest.raises(ChatRouteAuthorizationError) as exc:
        chat_routes._select_auto_stream_context_candidate(
            session,
            owner="alice",
            auth=SimpleNamespace(),
        )

    assert exc.value.code == "model_not_allowed"
    assert attempts == ["auto_chat"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"chat_mode": "agent"},
        {"att_ids": ["attachment"]},
        {"image_bypass": True},
        {"do_research": True},
        {"compare_mode": True},
    ],
)
def test_non_plain_stream_modes_bypass_auto(overrides):
    session = Session("s1", "Auto", "", "", auto_route=True)
    values = {
        "chat_mode": "chat",
        "att_ids": [],
        "image_bypass": False,
        "do_research": False,
        "compare_mode": False,
    }
    values.update(overrides)

    assert chat_routes._is_plain_auto_stream_chat(session, **values) is False


def test_plain_stream_chat_requires_auto_enabled():
    values = {
        "chat_mode": "chat",
        "att_ids": [],
        "image_bypass": False,
        "do_research": False,
        "compare_mode": False,
    }
    enabled = Session("enabled", "Auto", "", "", auto_route=True)
    disabled = Session("disabled", "Manual", "", "", auto_route=False)

    assert chat_routes._is_plain_auto_stream_chat(enabled, **values) is True
    assert chat_routes._is_plain_auto_stream_chat(disabled, **values) is False
