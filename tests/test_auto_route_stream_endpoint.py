import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.models import ChatMessage, Session
from routes import chat_routes
from src import agent_loop
from src.chat_model_router import ChatRoute, RouteTarget
from src.chat_route_authorizer import (
    AuthorizedChatRoute,
    ChatRouteAuthContext,
    ChatRouteAuthorizationError,
)
from src.llm_core import ChatDispatchError


class _Manager:
    def __init__(self, session):
        self.session = session
        self.save_calls = 0

    def get_session(self, session_id):
        assert session_id == self.session.id
        return self.session

    def save_sessions(self):
        self.save_calls += 1


class _Query:
    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        return None


class _Db:
    def query(self, *args, **kwargs):
        return _Query()

    def close(self):
        pass

    def rollback(self):
        pass

    def commit(self):
        pass

    def expunge(self, value):
        pass


class _Request:
    def __init__(self, mode):
        self.headers = {}
        self._form = {
            "message": "answer this request",
            "session": "s1",
            "mode": mode,
        }
        privileges = {
            "allowed_models": [],
            "allowed_models_restricted": False,
            "block_all_models": False,
            "max_messages_per_day": 0,
            "can_use_agent": True,
        }
        auth_manager = SimpleNamespace(
            get_privileges=lambda owner: dict(privileges),
            is_admin=lambda owner: False,
        )
        self.app = SimpleNamespace(state=SimpleNamespace(auth_manager=auth_manager))

    async def form(self):
        return dict(self._form)


def _route(lane):
    return ChatRoute(
        auto=True,
        lane=lane,
        target=RouteTarget(endpoint_id=f"{lane}-primary", model=f"{lane}-model"),
        reason=f"auto_{lane}",
    )


def _candidate(lane):
    return AuthorizedChatRoute(
        auto=True,
        lane=lane,
        reason=f"auto_{lane}",
        model=f"{lane}-model",
        endpoint_id=f"{lane}-primary",
        endpoint_url=f"https://{lane}.invalid/v1/chat/completions",
        headers={"Authorization": f"Bearer {lane}-secret"},
    )


def _context():
    return SimpleNamespace(
        messages=[{"role": "user", "content": "answer this request"}],
        preset=SimpleNamespace(temperature=None, max_tokens=None, character_name=None),
        preprocessed=SimpleNamespace(attachment_meta=[]),
        auto_opened_docs=[],
        rag_sources=[],
        web_sources=[],
        used_memories=[],
        uploaded_files=[],
        uprefs={},
        user="alice",
        was_compacted=False,
        context_trimmed=False,
        context_length=8192,
        context_messages_before_trim=1,
        context_messages_after_trim=1,
        context_tokens_before_trim=4,
        context_tokens_after_trim=4,
    )


def _policy():
    return SimpleNamespace(
        block_all_tool_calls=False,
        disable_mcp=False,
        mode="normal",
        blocks=lambda name: False,
        all_disabled_names=lambda: set(),
        reason_for=lambda name: "blocked",
    )


def _install_handler_harness(monkeypatch, *, mode, deny_code=None, manual_url=""):
    session = Session(
        "s1", "Auto", manual_url, "", auto_route=True,
        headers={}, owner="alice",
    )
    manager = _Manager(session)
    router = chat_routes.setup_chat_routes(
        manager,
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
    )
    endpoint = next(route.endpoint for route in router.routes if route.path == "/api/chat_stream")
    request = _Request(mode)
    lane = "agent" if mode == "agent" else "chat"
    selected = _route(lane)
    calls = {"owner": 0, "quota": 0, "route": [], "auth": [], "context": [], "post": 0}

    monkeypatch.setattr(chat_routes, "_set_user_time_from_request", lambda request: None)
    monkeypatch.setattr(chat_routes, "_resolve_request_workspace", lambda *args: (None, None))
    monkeypatch.setattr(chat_routes, "_classify_tool_intent", lambda message: None)
    monkeypatch.setattr(chat_routes, "_is_contextual_web_followup", lambda *args: False)
    monkeypatch.setattr(chat_routes, "_verify_session_owner", lambda *args: calls.__setitem__("owner", calls["owner"] + 1))
    monkeypatch.setattr(chat_routes, "effective_user", lambda request: "alice")
    monkeypatch.setattr(chat_routes, "get_current_user", lambda request: "alice")
    monkeypatch.setattr(chat_routes, "get_session_mode", lambda session_id: "chat")
    monkeypatch.setattr(chat_routes, "set_session_mode", lambda *args: None)
    monkeypatch.setattr(chat_routes, "_is_image_generation_session", lambda *args, **kwargs: False)
    monkeypatch.setattr(chat_routes, "_clear_orphaned_session_endpoint", lambda *args, **kwargs: pytest.fail("legacy orphan cleanup"))
    monkeypatch.setattr(chat_routes, "_recover_empty_session_model", lambda *args, **kwargs: pytest.fail("legacy model recovery"))
    monkeypatch.setattr(chat_routes, "resolve_session_auth", lambda *args, **kwargs: pytest.fail("legacy auth"))
    monkeypatch.setattr(chat_routes, "_enforce_chat_quota", lambda request: calls.__setitem__("quota", calls["quota"] + 1))
    monkeypatch.setattr(chat_routes, "build_effective_tool_policy", lambda **kwargs: _policy())
    monkeypatch.setattr(chat_routes, "SessionLocal", lambda: _Db())
    monkeypatch.setattr(chat_routes, "estimate_tokens", lambda messages: 4)

    auth_context = ChatRouteAuthContext(
        owner="alice",
        privileges={
            "allowed_models": [],
            "allowed_models_restricted": False,
            "block_all_models": False,
            "max_messages_per_day": 0,
            "can_use_agent": True,
        },
    )
    monkeypatch.setattr(chat_routes, "build_chat_route_auth_context", lambda request: auth_context)

    def resolve(sess, *, owner, agent_mode):
        calls["route"].append((owner, agent_mode))
        return selected

    def authorize(route, sess, *, auth):
        calls["auth"].append(route.reason)
        if deny_code:
            raise ChatRouteAuthorizationError(deny_code)
        return _candidate(lane)

    async def build_context(*args, **kwargs):
        calls["context"].append(kwargs)
        assert kwargs["runtime_model"] == f"{lane}-model"
        assert kwargs["runtime_endpoint_url"] == f"https://{lane}.invalid/v1/chat/completions"
        assert kwargs["runtime_headers"] == {"Authorization": f"Bearer {lane}-secret"}
        assert kwargs["model_event_override"] == f"{lane}-model"
        return _context()

    monkeypatch.setattr(chat_routes, "resolve_chat_route", resolve)
    monkeypatch.setattr(chat_routes, "authorize_chat_route", authorize)
    monkeypatch.setattr(chat_routes, "build_chat_context", build_context)
    monkeypatch.setattr(chat_routes, "run_post_response_tasks", lambda *args, **kwargs: calls.__setitem__("post", calls["post"] + 1))
    monkeypatch.setattr("core.database.update_session_last_accessed", lambda *args: None)

    run = {}
    monkeypatch.setattr(chat_routes.agent_runs, "start", lambda session_id, stream: run.setdefault("stream", stream))

    async def subscribe(session_id):
        async for chunk in run["stream"]:
            yield chunk

    monkeypatch.setattr(chat_routes.agent_runs, "subscribe", subscribe)
    return endpoint, request, session, manager, calls


async def _body(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return chunks


def _data_events(chunks):
    return [
        json.loads(chunk[6:])
        for chunk in chunks
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]")
    ]


@pytest.mark.asyncio
async def test_plain_auto_chat_stream_handler_integrates_route_auth_and_persistence(monkeypatch):
    endpoint, request, session, manager, calls = _install_handler_harness(monkeypatch, mode="chat")
    before = (session.model, session.endpoint_url, dict(session.headers), session.auto_route)

    async def stream(url, model, messages, headers=None, **kwargs):
        assert kwargs["typed_errors"] is True
        yield 'data: {"delta": "plain answer"}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(chat_routes, "stream_llm", stream)
    response = await endpoint(request)
    chunks = await _body(response)
    events = _data_events(chunks)

    assert calls["owner"] == 1
    assert calls["quota"] == 1
    assert calls["route"] == [("alice", False)]
    assert calls["auth"] == ["auto_chat", "auto_chat"]
    assert next(event for event in events if event.get("type") == "model_info")["model"] == "chat-model"
    assistants = [message for message in session.history if message.role == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].content == "plain answer"
    assert assistants[0].metadata["requested_model"] == "chat-model"
    assert assistants[0].metadata["model"] == "chat-model"
    assert calls["post"] == 1
    assert manager.save_calls == 1
    assert (session.model, session.endpoint_url, dict(session.headers), session.auto_route) == before


@pytest.mark.asyncio
async def test_agent_auto_chat_stream_handler_integrates_route_state_and_persistence(monkeypatch):
    endpoint, request, session, manager, calls = _install_handler_harness(monkeypatch, mode="agent")
    before = (session.model, session.endpoint_url, dict(session.headers), session.auto_route)
    original_loop = agent_loop.stream_agent_loop

    monkeypatch.setattr(agent_loop, "get_setting", lambda key, default=None: default)
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None)
    monkeypatch.setattr(agent_loop, "estimate_tokens", lambda messages: 4)

    async def stream(url, model, messages, headers=None, **kwargs):
        assert kwargs["typed_errors"] is True
        yield 'data: {"delta": "agent answer"}\n\n'
        yield "data: [DONE]\n\n"

    async def handler_loop(*args, **kwargs):
        assert isinstance(kwargs.get("route_state"), agent_loop.AgentRouteState)
        async for chunk in original_loop(*args, relevant_tools={"web_search"}, **kwargs):
            yield chunk

    monkeypatch.setattr(agent_loop, "stream_llm", stream)
    monkeypatch.setattr(chat_routes, "stream_agent_loop", handler_loop)

    import src.teacher_escalation as teacher_escalation
    async def forbidden_teacher(**kwargs):
        pytest.fail("Auto agent must not invoke teacher escalation")
        yield
    monkeypatch.setattr(teacher_escalation, "run_teacher_inline", forbidden_teacher)

    response = await endpoint(request)
    chunks = await _body(response)
    events = _data_events(chunks)

    assert calls["owner"] == 1
    assert calls["quota"] == 1
    assert calls["route"] == [("alice", True)]
    assert calls["auth"] == ["auto_agent", "auto_agent"]
    assert next(event for event in events if event.get("type") == "model_info")["model"] == "agent-model"
    assistants = [message for message in session.history if message.role == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].content == "agent answer"
    assert assistants[0].metadata["requested_model"] == "agent-model"
    assert assistants[0].metadata["model"] == "agent-model"
    assert calls["post"] == 1
    assert manager.save_calls == 1
    assert (session.model, session.endpoint_url, dict(session.headers), session.auto_route) == before


@pytest.mark.asyncio
async def test_auto_chat_stream_terminal_auth_denial_has_no_assistant_or_posttasks(monkeypatch):
    endpoint, request, session, manager, calls = _install_handler_harness(
        monkeypatch, mode="chat", deny_code="model_not_allowed"
    )

    with pytest.raises(HTTPException) as exc:
        await endpoint(request)

    assert getattr(exc.value, "status_code", None) == 403
    assert not [message for message in session.history if message.role == "assistant"]
    assert calls["post"] == 0
    assert manager.save_calls == 0


@pytest.mark.asyncio
async def test_plain_auto_handler_error_log_omits_persistent_target_url(monkeypatch, caplog):
    endpoint, request, session, manager, calls = _install_handler_harness(
        monkeypatch,
        mode="chat",
        manual_url="https://secret-user:secret-pass@private-host.invalid/v1/chat/completions",
    )

    async def stream(*args, **kwargs):
        raise ChatDispatchError(
            400,
            "Bearer TOP-SECRET-TOKEN raw provider body",
            kind="upstream_status",
        )
        yield

    monkeypatch.setattr(chat_routes, "stream_llm", stream)
    caplog.set_level("WARNING", logger=chat_routes.__name__)

    response = await endpoint(request)
    chunks = await _body(response)

    assert "event: error" in "".join(chunks)
    for secret in (
        "private-host.invalid", "secret-user", "secret-pass",
        "TOP-SECRET-TOKEN", "raw provider body",
    ):
        assert secret not in caplog.text
    assert not [message for message in session.history if message.role == "assistant"]
    assert calls["post"] == 0
    assert manager.save_calls == 0
