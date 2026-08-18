import asyncio
import json
from types import SimpleNamespace

import pytest

from routes import chat_routes
from src import agent_loop
from src.chat_model_router import ChatRoute, RouteTarget
from src.chat_route_authorizer import ChatRouteAuthorizationError
from src.llm_core import ChatDispatchError


def _collect(stream):
    async def run():
        return [chunk async for chunk in stream]

    return asyncio.run(run())


def _events(chunks):
    events = []
    for chunk in chunks:
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
            events.append(json.loads(chunk[6:]))
    return events


def _route(name, *, auto):
    return ChatRoute(
        auto=auto,
        lane="agent" if auto else "manual",
        target=RouteTarget(
            endpoint_id=name if auto else None,
            endpoint_url=None if auto else f"https://{name}.invalid/v1/chat/completions",
            model=f"{name}-model",
        ),
        reason="auto_agent" if auto else "manual_fallback",
    )


def _candidate(name):
    return SimpleNamespace(
        endpoint_id=name,
        endpoint_url=f"https://{name}.invalid/v1/chat/completions",
        model=f"{name}-model",
        headers={"Authorization": f"Bearer {name}"},
    )


def _state(authorize, *, active_primary=True):
    primary = _route("primary", auto=True)
    fallback = _route("fallback", auto=False)
    return agent_loop.AgentRouteState(
        requested_model="primary-model",
        selected_primary_route=primary,
        manual_fallback_route=fallback,
        active_route=primary if active_primary else fallback,
        authorize_route=authorize,
    )


def _patch_common(monkeypatch):
    monkeypatch.setattr(agent_loop, "get_setting", lambda key, default=None: default)
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None)
    monkeypatch.setattr(agent_loop, "estimate_tokens", lambda *args, **kwargs: 10)


def _run_auto(monkeypatch, state, stream, *, execute=None, max_rounds=2):
    _patch_common(monkeypatch)
    monkeypatch.setattr(agent_loop, "stream_llm", stream)
    if execute is not None:
        monkeypatch.setattr(agent_loop, "execute_tool_block", execute)
    return _collect(agent_loop.stream_agent_loop(
        "https://context.invalid/v1/chat/completions",
        "context-model",
        [{"role": "user", "content": "perform the requested task"}],
        headers={"Authorization": "Bearer context"},
        max_rounds=max_rounds,
        relevant_tools={"web_search"},
        route_state=state,
        _is_teacher_run=True,
    ))


def test_auto_agent_primary_commits_visible_output_and_preserves_requested(monkeypatch):
    auth_calls = []

    def authorize(route):
        auth_calls.append(route.reason)
        return _candidate("primary")

    async def stream(url, model, messages, headers=None, **kwargs):
        assert kwargs["typed_errors"] is True
        assert (url, model, headers) == (
            "https://primary.invalid/v1/chat/completions",
            "primary-model",
            {"Authorization": "Bearer primary"},
        )
        yield 'data: {"delta": "answer"}\n\n'
        yield "data: [DONE]\n\n"

    state = _state(authorize)
    chunks = _run_auto(monkeypatch, state, stream)
    metrics = next(e["data"] for e in _events(chunks) if e.get("type") == "metrics")

    assert auth_calls == ["auto_agent"]
    assert state.committed is True
    assert state.commit_reason == "visible_output"
    assert state.winner_model == "primary-model"
    assert metrics["requested_model"] == "primary-model"
    assert metrics["model"] == "primary-model"


@pytest.mark.parametrize("error", [
    ChatDispatchError(504, "secret", kind="timeout"),
    ChatDispatchError(502, "secret", kind="network"),
    ChatDispatchError(408, "secret", kind="upstream_status"),
    ChatDispatchError(429, "secret", kind="upstream_status"),
    ChatDispatchError(503, "secret", kind="upstream_status"),
])
def test_auto_agent_recoverable_precommit_failure_uses_jit_manual_fallback(
    monkeypatch, error
):
    auth_calls = []
    stream_calls = []

    def authorize(route):
        auth_calls.append(route.reason)
        return _candidate("primary" if route.auto else "fallback")

    async def stream(url, model, messages, headers=None, **kwargs):
        stream_calls.append((model, dict(headers or {})))
        if model == "primary-model":
            raise error
        yield 'data: {"delta": "fallback answer"}\n\n'
        yield "data: [DONE]\n\n"

    state = _state(authorize)
    chunks = _run_auto(monkeypatch, state, stream)

    assert auth_calls == ["auto_agent", "manual_fallback"]
    assert stream_calls == [
        ("primary-model", {"Authorization": "Bearer primary"}),
        ("fallback-model", {"Authorization": "Bearer fallback"}),
    ]
    assert state.winner_model == "fallback-model"
    assert state.fallback_available is False
    fallback = next(e for e in _events(chunks) if e.get("type") == "fallback")
    assert fallback["selected_model"] == "primary-model"
    assert fallback["answered_by"] == "fallback-model"


@pytest.mark.parametrize("error", [
    ChatDispatchError(400, "Bearer SECRET", kind="upstream_status"),
    ChatDispatchError(502, "Bearer SECRET", kind="invalid_response"),
    RuntimeError("Bearer SECRET https://private.invalid"),
])
def test_auto_agent_terminal_precommit_failure_is_sanitized_without_fallback(
    monkeypatch, error
):
    auth_calls = []

    def authorize(route):
        auth_calls.append(route.reason)
        return _candidate("primary")

    async def stream(*args, **kwargs):
        raise error
        yield

    state = _state(authorize)
    chunks = _run_auto(monkeypatch, state, stream)
    public = "".join(chunks)

    assert auth_calls == ["auto_agent"]
    assert state.winner_model is None
    assert "SECRET" not in public
    assert "private.invalid" not in public
    assert public.count("event: error") == 1


def test_auto_agent_candidate_unavailable_at_jit_uses_fallback(monkeypatch):
    calls = []

    def authorize(route):
        calls.append(route.reason)
        if route.auto:
            raise ChatRouteAuthorizationError("credentials_unavailable")
        return _candidate("fallback")

    async def stream(url, model, messages, **kwargs):
        yield 'data: {"delta": "ok"}\n\n'
        yield "data: [DONE]\n\n"

    state = _state(authorize)
    _run_auto(monkeypatch, state, stream)

    assert calls == ["auto_agent", "manual_fallback"]
    assert state.winner_model == "fallback-model"


def test_auto_agent_model_denial_never_authorizes_fallback(monkeypatch):
    calls = []

    def authorize(route):
        calls.append(route.reason)
        raise ChatRouteAuthorizationError("model_not_allowed")

    async def stream(*args, **kwargs):
        raise AssertionError("LLM dispatch must not occur")
        yield

    state = _state(authorize)
    chunks = _run_auto(monkeypatch, state, stream)

    assert calls == ["auto_agent"]
    assert state.winner_model is None
    assert "event: error" in "".join(chunks)


def test_auto_agent_model_hidden_without_fallback_is_sanitized_403(monkeypatch):
    primary = _route("primary", auto=True)
    state = agent_loop.AgentRouteState(
        requested_model="primary-model",
        selected_primary_route=primary,
        manual_fallback_route=None,
        active_route=primary,
        authorize_route=lambda route: (_ for _ in ()).throw(
            ChatRouteAuthorizationError("model_hidden")
        ),
    )

    async def stream(*args, **kwargs):
        raise AssertionError("LLM dispatch must not occur")
        yield

    chunks = _run_auto(monkeypatch, state, stream)
    error = next(chunk for chunk in chunks if chunk.startswith("event: error"))

    assert '"status": 403' in error
    assert state.winner_model is None


def test_auto_agent_tool_commits_before_start_and_execution(monkeypatch):
    observations = []

    def authorize(route):
        return _candidate("primary")

    async def stream(url, model, messages, **kwargs):
        yield 'data: {"type": "tool_calls", "calls": [{"id": "c1", "name": "web_search", "arguments": "{\\"query\\": \\"x\\"}"}]}\n\n'
        yield "data: [DONE]\n\n"

    state = _state(authorize)

    async def execute(block, *args, **kwargs):
        observations.append(("execute", state.committed, state.commit_reason))
        return "search", {"output": "result", "exit_code": 0}

    chunks = _run_auto(monkeypatch, state, stream, execute=execute, max_rounds=1)
    types = [e.get("type") for e in _events(chunks)]

    observations.insert(0, ("tool_start", state.committed, state.commit_reason))
    assert observations[1] == ("execute", True, "tool_execution")
    assert types.index("tool_start") < types.index("tool_output")
    assert state.winner_model == "primary-model"


def test_internal_tool_proposal_does_not_commit_and_can_fallback_before_acceptance(monkeypatch):
    streams = []

    def authorize(route):
        return _candidate("primary" if route.auto else "fallback")

    async def stream(url, model, messages, **kwargs):
        streams.append(model)
        if model == "primary-model":
            yield 'data: {"type": "tool_call_delta", "name": "web_search", "arg_delta": "{\\"query\\":"}\n\n'
            raise ChatDispatchError(503, "secret", kind="upstream_status")
        yield 'data: {"delta": "fallback answer"}\n\n'
        yield "data: [DONE]\n\n"

    state = _state(authorize)
    chunks = _run_auto(monkeypatch, state, stream)

    assert streams == ["primary-model", "fallback-model"]
    assert state.winner_model == "fallback-model"
    assert "tool_call_delta" not in "".join(chunks)


@pytest.mark.parametrize("tool_error", [RuntimeError("failed"), asyncio.TimeoutError()])
def test_tool_execution_failure_after_commit_never_falls_back(monkeypatch, tool_error):
    auth_calls = []

    def authorize(route):
        auth_calls.append(route.reason)
        return _candidate("primary" if route.auto else "fallback")

    async def stream(url, model, messages, **kwargs):
        yield 'data: {"type": "tool_calls", "calls": [{"id": "c1", "name": "web_search", "arguments": "{\\"query\\": \\"x\\"}"}]}\n\n'
        yield "data: [DONE]\n\n"

    state = _state(authorize)

    async def execute(*args, **kwargs):
        assert state.committed is True
        raise tool_error

    with pytest.raises(type(tool_error)):
        _run_auto(monkeypatch, state, stream, execute=execute)

    assert auth_calls == ["auto_agent"]
    assert state.winner_model == "primary-model"
    assert state.tool_execution_uncertain is True


def test_auto_agent_round_two_failure_after_tool_never_falls_back(monkeypatch):
    auth_calls = []
    streams = []

    def authorize(route):
        auth_calls.append(route.reason)
        return _candidate("primary" if route.auto else "fallback")

    async def stream(url, model, messages, **kwargs):
        streams.append(model)
        if len(streams) == 1:
            yield 'data: {"type": "tool_calls", "calls": [{"id": "c1", "name": "web_search", "arguments": "{\\"query\\": \\"x\\"}"}]}\n\n'
            yield "data: [DONE]\n\n"
            return
        raise ChatDispatchError(503, "secret", kind="upstream_status")
        yield

    async def execute(*args, **kwargs):
        return "search", {"output": "result", "exit_code": 0}

    state = _state(authorize)
    chunks = _run_auto(monkeypatch, state, stream, execute=execute)

    assert auth_calls == ["auto_agent", "auto_agent"]
    assert streams == ["primary-model", "primary-model"]
    assert state.winner_model == "primary-model"
    assert "secret" not in "".join(chunks)


def test_auto_agent_same_candidate_reauth_refreshes_headers(monkeypatch):
    generation = 0
    seen_headers = []

    def authorize(route):
        nonlocal generation
        generation += 1
        candidate = _candidate("primary")
        candidate.headers = {"Authorization": f"Bearer fresh-{generation}"}
        return candidate

    async def stream(url, model, messages, headers=None, **kwargs):
        seen_headers.append(dict(headers or {}))
        if len(seen_headers) == 1:
            yield 'data: {"type": "tool_calls", "calls": [{"id": "c1", "name": "web_search", "arguments": "{\\"query\\": \\"x\\"}"}]}\n\n'
            yield "data: [DONE]\n\n"
        else:
            yield 'data: {"delta": "done"}\n\n'
            yield "data: [DONE]\n\n"

    async def execute(*args, **kwargs):
        return "search", {"output": "result", "exit_code": 0}

    state = _state(authorize)
    _run_auto(monkeypatch, state, stream, execute=execute)

    assert seen_headers == [
        {"Authorization": "Bearer fresh-1"},
        {"Authorization": "Bearer fresh-2"},
    ]
    assert state.winner_model == "primary-model"


def test_auto_agent_pinned_identity_change_is_terminal(monkeypatch):
    generation = 0
    streams = []

    def authorize(route):
        nonlocal generation
        generation += 1
        return _candidate("primary" if generation == 1 else "changed")

    async def stream(url, model, messages, **kwargs):
        streams.append(model)
        yield 'data: {"type": "tool_calls", "calls": [{"id": "c1", "name": "web_search", "arguments": "{\\"query\\": \\"x\\"}"}]}\n\n'
        yield "data: [DONE]\n\n"

    async def execute(*args, **kwargs):
        return "search", {"output": "result", "exit_code": 0}

    state = _state(authorize)
    chunks = _run_auto(monkeypatch, state, stream, execute=execute)

    assert streams == ["primary-model"]
    assert state.winner_model == "primary-model"
    assert "event: error" in "".join(chunks)


def test_auto_agent_cancellation_never_authorizes_fallback(monkeypatch):
    calls = []

    def authorize(route):
        calls.append(route.reason)
        return _candidate("primary")

    async def stream(*args, **kwargs):
        raise asyncio.CancelledError()
        yield

    state = _state(authorize)
    _patch_common(monkeypatch)
    monkeypatch.setattr(agent_loop, "stream_llm", stream)

    with pytest.raises(asyncio.CancelledError):
        _collect(agent_loop.stream_agent_loop(
            "https://context.invalid/v1", "context-model",
            [{"role": "user", "content": "perform the task"}],
            relevant_tools={"web_search"}, route_state=state,
            _is_teacher_run=True,
        ))

    assert calls == ["auto_agent"]
    assert state.winner_model is None


def test_auto_agent_scope_excludes_incompatible_modes():
    session = SimpleNamespace(auto_route=True)
    base = {
        "chat_mode": "agent",
        "att_ids": [],
        "image_bypass": False,
        "do_research": False,
        "compare_mode": False,
    }
    assert chat_routes._is_plain_auto_agent(session, **base) is True
    for key, value in (
        ("chat_mode", "chat"),
        ("att_ids", ["a"]),
        ("image_bypass", True),
        ("do_research", True),
        ("compare_mode", True),
    ):
        values = dict(base)
        values[key] = value
        assert chat_routes._is_plain_auto_agent(session, **values) is False


def test_auto_agent_precontext_selects_agent_lane_and_hydrates_only_primary(monkeypatch):
    primary = _route("primary", auto=True)
    fallback = _route("fallback", auto=False)
    session = SimpleNamespace(
        model="manual-model",
        endpoint_url="https://manual.invalid/v1/chat/completions",
        headers={"Authorization": "Bearer manual"},
        auto_route=True,
    )
    before = dict(vars(session))
    route_calls = []
    auth_calls = []

    def resolve(sess, *, owner, agent_mode):
        route_calls.append((owner, agent_mode))
        return primary

    monkeypatch.setattr(chat_routes, "resolve_chat_route", resolve)
    monkeypatch.setattr(chat_routes, "_manual_fallback_route", lambda *args: fallback)
    monkeypatch.setattr(
        chat_routes,
        "authorize_chat_route",
        lambda route, sess, auth: auth_calls.append(route) or _candidate("primary"),
    )

    selected = chat_routes._select_auto_agent_context_candidate(
        session, owner="alice", auth=object()
    )

    assert route_calls == [("alice", True)]
    assert auth_calls == [primary]
    assert selected[:4] == (primary, "primary-model", fallback, primary)
    assert vars(session) == before


def test_auto_agent_precontext_candidate_unavailable_hydrates_fallback_lazily(monkeypatch):
    primary = _route("primary", auto=True)
    fallback = _route("fallback", auto=False)
    calls = []

    monkeypatch.setattr(chat_routes, "resolve_chat_route", lambda *args, **kwargs: primary)
    monkeypatch.setattr(chat_routes, "_manual_fallback_route", lambda *args: fallback)

    def authorize(route, sess, *, auth):
        calls.append(route)
        if route is primary:
            raise ChatRouteAuthorizationError("credentials_unavailable")
        return _candidate("fallback")

    monkeypatch.setattr(chat_routes, "authorize_chat_route", authorize)
    selected = chat_routes._select_auto_agent_context_candidate(
        SimpleNamespace(), owner="alice", auth=object()
    )

    assert calls == [primary, fallback]
    assert selected[1] == "primary-model"
    assert selected[3] is fallback
    assert selected[4].model == "fallback-model"


def test_auto_agent_precontext_model_denial_does_not_hydrate_fallback(monkeypatch):
    primary = _route("primary", auto=True)
    fallback = _route("fallback", auto=False)
    calls = []
    monkeypatch.setattr(chat_routes, "resolve_chat_route", lambda *args, **kwargs: primary)
    monkeypatch.setattr(chat_routes, "_manual_fallback_route", lambda *args: fallback)

    def authorize(route, sess, *, auth):
        calls.append(route)
        raise ChatRouteAuthorizationError("model_not_allowed")

    monkeypatch.setattr(chat_routes, "authorize_chat_route", authorize)
    with pytest.raises(ChatRouteAuthorizationError, match="model_not_allowed"):
        chat_routes._select_auto_agent_context_candidate(
            SimpleNamespace(), owner="alice", auth=object()
        )
    assert calls == [primary]


def test_auto_agent_tool_only_state_is_sanitized():
    state = _state(lambda route: _candidate("primary"))
    state.set_round_candidate(_candidate("primary"))
    assert state.mark_current_tool_execution(tool="write_file", round_num=2)

    serialized = json.dumps(state.safe_tool_events())
    assert "write_file" in serialized
    assert "Authorization" not in serialized
    assert "Bearer" not in serialized
    assert "invalid" not in serialized


def test_legacy_agent_still_uses_legacy_fallback(monkeypatch):
    _patch_common(monkeypatch)
    called = []

    async def legacy(candidates, messages, **kwargs):
        called.append(candidates)
        yield 'data: {"delta": "legacy"}\n\n'
        yield "data: [DONE]\n\n"

    async def auto(*args, **kwargs):
        raise AssertionError("Auto single-candidate stream must not be used")
        yield

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", legacy)
    monkeypatch.setattr(agent_loop, "stream_llm", auto)
    _collect(agent_loop.stream_agent_loop(
        "https://legacy.invalid/v1", "legacy-model",
        [{"role": "user", "content": "perform the task"}],
        headers={"legacy": "header"}, max_rounds=1,
        relevant_tools={"web_search"}, fallbacks=[("u2", "m2", {"h": "2"})],
        _is_teacher_run=True,
    ))

    assert called == [[
        ("https://legacy.invalid/v1", "legacy-model", {"legacy": "header"}),
        ("u2", "m2", {"h": "2"}),
    ]]


def test_teacher_escalation_is_disabled_only_for_auto_agent(monkeypatch):
    _patch_common(monkeypatch)
    teacher_calls = []

    async def teacher(**kwargs):
        teacher_calls.append(kwargs["student_reply"])
        yield 'data: {"type": "teacher"}\n\n'

    async def auto_stream(*args, **kwargs):
        yield 'data: {"delta": "auto answer"}\n\n'
        yield "data: [DONE]\n\n"

    async def legacy_stream(*args, **kwargs):
        yield 'data: {"delta": "legacy answer"}\n\n'
        yield "data: [DONE]\n\n"

    import src.teacher_escalation as teacher_escalation

    monkeypatch.setattr(teacher_escalation, "run_teacher_inline", teacher)
    monkeypatch.setattr(agent_loop, "stream_llm", auto_stream)
    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", legacy_stream)

    state = _state(lambda route: _candidate("primary"))
    auto_chunks = _collect(agent_loop.stream_agent_loop(
        "https://context.invalid/v1", "context-model",
        [{"role": "user", "content": "perform the task"}],
        relevant_tools={"web_search"}, route_state=state, max_rounds=1,
    ))
    assert teacher_calls == []
    assert not any(e.get("type") == "teacher" for e in _events(auto_chunks))

    legacy_chunks = _collect(agent_loop.stream_agent_loop(
        "https://legacy.invalid/v1", "legacy-model",
        [{"role": "user", "content": "perform the task"}],
        relevant_tools={"web_search"}, max_rounds=1,
    ))
    assert teacher_calls == ["legacy answer"]
    assert any(e.get("type") == "teacher" for e in _events(legacy_chunks))
