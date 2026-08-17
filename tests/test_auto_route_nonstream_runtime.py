from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.models import Session
from routes import chat_routes
from routes import chat_helpers
from src.chat_model_router import ChatRoute, RouteTarget
from src.chat_route_authorizer import (
    AuthorizedChatRoute,
    ChatRouteAuthorizationError,
)
from src.llm_core import ChatDispatchError
from src.request_models import ChatRequest


def _route(endpoint_id="auto", model="auto-model"):
    return ChatRoute(
        auto=True,
        lane="chat",
        target=RouteTarget(endpoint_id=endpoint_id, model=model),
        reason="auto_chat",
        manual_fallback=RouteTarget(
            model="manual-model",
            endpoint_url="http://manual.invalid/v1/chat/completions",
        ),
    )


def _authorized(model, marker, *, auto=True):
    return AuthorizedChatRoute(
        auto=auto,
        lane="chat" if auto else "manual",
        reason="auto_chat" if auto else "manual_fallback",
        model=model,
        endpoint_id=marker,
        endpoint_url=f"http://{marker}.invalid/v1/chat/completions",
        headers={"Authorization": f"Bearer {marker}"},
    )


class _SessionManager:
    def __init__(self, session):
        self.session = session

    def get_session(self, session_id):
        assert session_id == self.session.id
        return self.session


class _ChatHandler:
    def __init__(self, memory_response=None):
        self.memory_response = memory_response

    async def handle_memory_command(self, session, message):
        return self.memory_response


def _request():
    auth = SimpleNamespace(
        get_privileges=lambda owner: {
            "allowed_models": [],
            "allowed_models_restricted": False,
            "block_all_models": False,
            "max_messages_per_day": 0,
        },
        is_admin=lambda owner: False,
    )
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(auth_manager=auth)))


def _endpoint(monkeypatch, session, *, memory_response=None):
    manager = _SessionManager(session)
    router = chat_routes.setup_chat_routes(
        manager,
        _ChatHandler(memory_response),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
    )
    endpoint = next(r.endpoint for r in router.routes if r.path == "/api/chat")
    monkeypatch.setattr(chat_routes, "_verify_session_owner", lambda request, session_id: None)
    monkeypatch.setattr(chat_routes, "effective_user", lambda request: "alice")
    monkeypatch.setattr(chat_routes, "_is_image_generation_session", lambda session, owner=None: False)
    monkeypatch.setattr(chat_routes, "_enforce_chat_quota", lambda request: None)
    monkeypatch.setattr(chat_routes, "build_effective_tool_policy", lambda **kwargs: SimpleNamespace(
        block_all_tool_calls=False,
        blocks=lambda name: False,
    ))
    monkeypatch.setattr(
        chat_routes,
        "build_chat_route_auth_context",
        lambda request: SimpleNamespace(privileges={"resolved": True}),
    )
    return endpoint


def _context():
    return SimpleNamespace(
        messages=[{"role": "user", "content": "hello"}],
        preset=SimpleNamespace(temperature=None, max_tokens=None, character_name=None),
        preface=[],
        uprefs={},
        user="alice",
    )


@pytest.mark.asyncio
async def test_auto_empty_manual_uses_request_target_without_legacy_mutation(monkeypatch):
    session = Session("s1", "Auto", "", "", auto_route=True, headers={}, owner="alice")
    before = (session.model, session.endpoint_url, dict(session.headers), session.auto_route)
    endpoint = _endpoint(monkeypatch, session)
    selected = _route()
    order = []
    calls = []

    monkeypatch.setattr(chat_routes, "_clear_orphaned_session_endpoint", lambda *a, **k: pytest.fail("legacy cleanup"))
    monkeypatch.setattr(chat_routes, "_recover_empty_session_model", lambda *a, **k: pytest.fail("legacy recovery"))
    monkeypatch.setattr(chat_routes, "resolve_chat_route", lambda *a, **k: selected)

    def authorize(route, session, auth):
        order.append("authorize")
        return _authorized("auto-model", "auto")

    async def build(*args, **kwargs):
        order.append("context")
        assert kwargs["runtime_model"] == "auto-model"
        assert kwargs["model_event_override"] == "auto-model"
        return _context()

    async def dispatch(url, model, messages, headers=None, **kwargs):
        order.append("dispatch")
        calls.append((url, model, dict(headers)))
        return "answer"

    saved = []
    monkeypatch.setattr(chat_routes, "authorize_chat_route", authorize)
    monkeypatch.setattr(chat_routes, "build_chat_context", build)
    monkeypatch.setattr(chat_routes, "llm_call_async", dispatch)
    monkeypatch.setattr(chat_routes, "save_assistant_response", lambda *a, **k: saved.append(a[4]))
    monkeypatch.setattr(chat_routes, "run_post_response_tasks", lambda *a, **k: None)

    result = await endpoint(_request(), ChatRequest(message="hello", session="s1"))

    assert result == {"response": "answer"}
    assert order == ["authorize", "context", "authorize", "dispatch"]
    assert calls == [("http://auto.invalid/v1/chat/completions", "auto-model", {"Authorization": "Bearer auto"})]
    assert saved == [{"requested_model": "auto-model", "model": "auto-model"}]
    assert (session.model, session.endpoint_url, dict(session.headers), session.auto_route) == before


@pytest.mark.asyncio
async def test_memory_command_returns_before_route_or_credentials(monkeypatch):
    session = Session("s1", "Manual", "http://manual", "manual", owner="alice")
    endpoint = _endpoint(monkeypatch, session, memory_response="saved")
    quota = []
    monkeypatch.setattr(chat_routes, "_clear_orphaned_session_endpoint", lambda *a, **k: False)
    monkeypatch.setattr(chat_routes, "_recover_empty_session_model", lambda *a, **k: False)
    monkeypatch.setattr(chat_routes, "_enforce_chat_quota", lambda request: quota.append(True))
    monkeypatch.setattr(chat_routes, "resolve_chat_route", lambda *a, **k: pytest.fail("routing"))
    monkeypatch.setattr(chat_routes, "authorize_chat_route", lambda *a, **k: pytest.fail("hydration"))

    result = await endpoint(_request(), ChatRequest(message="remember: x", session="s1"))

    assert result == {"response": "saved"}
    assert quota == [True]


@pytest.mark.asyncio
async def test_session_ownership_failure_precedes_routing(monkeypatch):
    session = Session("s1", "Auto", "", "", auto_route=True, owner="bob")
    endpoint = _endpoint(monkeypatch, session)

    def deny(request, session_id):
        raise HTTPException(404, "not found")

    monkeypatch.setattr(chat_routes, "_verify_session_owner", deny)
    monkeypatch.setattr(chat_routes, "resolve_chat_route", lambda *a, **k: pytest.fail("routing"))

    with pytest.raises(HTTPException) as exc:
        await endpoint(_request(), ChatRequest(message="hello", session="s1"))

    assert exc.value.status_code == 404


@pytest.mark.parametrize("failing_lookup", ["get_privileges", "is_admin"])
@pytest.mark.asyncio
async def test_auth_lookup_failure_is_sanitized_before_memory_or_routing(
    monkeypatch, failing_lookup
):
    session = Session("s1", "Auto", "", "", auto_route=True, owner="alice")
    endpoint = _endpoint(monkeypatch, session)
    secret = "Bearer AUTH_LOOKUP_SECRET"

    class FailingAuth:
        def get_privileges(self, owner):
            if failing_lookup == "get_privileges":
                raise RuntimeError(secret)
            return {"allowed_models_restricted": False, "max_messages_per_day": 0}

        def is_admin(self, owner):
            if failing_lookup == "is_admin":
                raise RuntimeError(secret)
            return False

    request = _request()
    request.app.state.auth_manager = FailingAuth()
    monkeypatch.setattr(chat_helpers, "effective_user", lambda request: "alice")
    monkeypatch.setattr(chat_helpers, "_auth_disabled", lambda: False)
    monkeypatch.setattr(chat_routes, "_enforce_chat_quota", chat_helpers._enforce_chat_quota)
    monkeypatch.setattr(
        chat_routes,
        "build_chat_route_auth_context",
        chat_helpers.build_chat_route_auth_context,
    )
    monkeypatch.setattr(chat_routes, "resolve_chat_route", lambda *a, **k: pytest.fail("routing"))
    monkeypatch.setattr(chat_routes, "authorize_chat_route", lambda *a, **k: pytest.fail("hydration"))

    with pytest.raises(HTTPException) as exc:
        await endpoint(request, ChatRequest(message="hello", session="s1"))

    assert exc.value.status_code == 403
    assert secret not in str(exc.value.detail)


@pytest.mark.asyncio
async def test_dispatch_error_public_detail_does_not_expose_provider_secrets(monkeypatch):
    session = Session("s1", "Manual", "http://manual", "manual-model", owner="alice")
    endpoint = _endpoint(monkeypatch, session)
    monkeypatch.setattr(chat_routes, "_clear_orphaned_session_endpoint", lambda *a, **k: False)
    monkeypatch.setattr(chat_routes, "_recover_empty_session_model", lambda *a, **k: False)
    monkeypatch.setattr(
        chat_routes,
        "authorize_chat_route",
        lambda *a, **k: _authorized("manual-model", "manual", auto=False),
    )
    monkeypatch.setattr(chat_routes, "build_chat_context", lambda *a, **k: _async_value(_context()))
    private_detail = (
        "https://private-host.internal:8443 Authorization "
        "Bearer SUPER_SECRET_TOKEN provider raw payload"
    )

    async def dispatch(*args, **kwargs):
        raise ChatDispatchError(400, private_detail, kind="upstream_status")

    monkeypatch.setattr(chat_routes, "llm_call_async", dispatch)
    monkeypatch.setattr(chat_routes, "save_assistant_response", lambda *a, **k: pytest.fail("save"))
    monkeypatch.setattr(chat_routes, "run_post_response_tasks", lambda *a, **k: pytest.fail("post"))

    with pytest.raises(HTTPException) as exc:
        await endpoint(_request(), ChatRequest(message="hello", session="s1"))

    public = str(exc.value.detail)
    for secret in (
        "private-host.internal",
        "Authorization",
        "SUPER_SECRET_TOKEN",
        "provider raw payload",
    ):
        assert secret not in public


@pytest.mark.parametrize("bypass", ["attachments", "image"])
@pytest.mark.asyncio
async def test_attachment_or_image_bypasses_auto_without_image_dispatch(monkeypatch, bypass):
    session = Session("s1", "Auto", "http://manual", "manual-model", auto_route=True, owner="alice")
    endpoint = _endpoint(monkeypatch, session)
    monkeypatch.setattr(chat_routes, "_is_image_generation_session", lambda *a, **k: bypass == "image")
    monkeypatch.setattr(chat_routes, "_clear_orphaned_session_endpoint", lambda *a, **k: False)
    monkeypatch.setattr(chat_routes, "_recover_empty_session_model", lambda *a, **k: False)
    monkeypatch.setattr(chat_routes, "resolve_chat_route", lambda *a, **k: pytest.fail("Auto routing"))
    monkeypatch.setattr(chat_routes, "authorize_chat_route", lambda route, session, auth: _authorized("manual-model", "manual", auto=False))
    monkeypatch.setattr(chat_routes, "build_chat_context", lambda *a, **k: _async_value(_context()))
    monkeypatch.setattr(chat_routes, "llm_call_async", lambda *a, **k: _async_value("manual answer"))
    monkeypatch.setattr(chat_routes, "save_assistant_response", lambda *a, **k: None)
    monkeypatch.setattr(chat_routes, "run_post_response_tasks", lambda *a, **k: None)
    request = ChatRequest(
        message="hello",
        session="s1",
        attachments=["att-1"] if bypass == "attachments" else [],
    )

    result = await endpoint(_request(), request)

    assert result == {"response": "manual answer"}


@pytest.mark.asyncio
async def test_auto_off_does_not_add_image_session_classification(monkeypatch):
    session = Session("s1", "Manual", "http://manual", "manual-model", owner="alice")
    endpoint = _endpoint(monkeypatch, session)
    monkeypatch.setattr(
        chat_routes,
        "_is_image_generation_session",
        lambda *a, **k: pytest.fail("new image classification"),
    )
    monkeypatch.setattr(chat_routes, "_clear_orphaned_session_endpoint", lambda *a, **k: False)
    monkeypatch.setattr(chat_routes, "_recover_empty_session_model", lambda *a, **k: False)
    monkeypatch.setattr(
        chat_routes,
        "authorize_chat_route",
        lambda *a, **k: _authorized("manual-model", "manual", auto=False),
    )
    monkeypatch.setattr(chat_routes, "build_chat_context", lambda *a, **k: _async_value(_context()))
    monkeypatch.setattr(chat_routes, "llm_call_async", lambda *a, **k: _async_value("answer"))
    monkeypatch.setattr(chat_routes, "save_assistant_response", lambda *a, **k: None)
    monkeypatch.setattr(chat_routes, "run_post_response_tasks", lambda *a, **k: None)

    result = await endpoint(_request(), ChatRequest(message="hello", session="s1"))

    assert result == {"response": "answer"}


async def _async_value(value):
    return value


async def _record_async(target, item, result):
    target.append(item)
    return result


@pytest.mark.parametrize("failure_code", ["endpoint_not_found", "credentials_unavailable"])
@pytest.mark.asyncio
async def test_precontext_unavailable_primary_builds_with_manual_fallback(monkeypatch, failure_code):
    session = Session("s1", "Auto", "http://manual", "manual-model", auto_route=True, owner="alice")
    endpoint = _endpoint(monkeypatch, session)
    selected = _route()
    attempts = []
    contexts = []
    monkeypatch.setattr(chat_routes, "resolve_chat_route", lambda *a, **k: selected)

    def authorize(route, session, auth):
        attempts.append(route.reason)
        if route is selected:
            raise ChatRouteAuthorizationError(failure_code)
        return _authorized("manual-model", "manual", auto=False)

    async def build(*args, **kwargs):
        contexts.append(kwargs)
        return _context()

    monkeypatch.setattr(chat_routes, "authorize_chat_route", authorize)
    monkeypatch.setattr(chat_routes, "build_chat_context", build)
    monkeypatch.setattr(chat_routes, "llm_call_async", lambda *a, **k: _async_value("manual answer"))
    saved = []
    monkeypatch.setattr(chat_routes, "save_assistant_response", lambda *a, **k: saved.append(a[4]))
    monkeypatch.setattr(chat_routes, "run_post_response_tasks", lambda *a, **k: None)

    await endpoint(_request(), ChatRequest(message="hello", session="s1"))

    assert attempts == ["auto_chat", "manual_fallback", "manual_fallback"]
    assert contexts[0]["runtime_model"] == "manual-model"
    assert contexts[0]["model_event_override"] == "auto-model"
    assert saved == [{"requested_model": "auto-model", "model": "manual-model"}]


@pytest.mark.asyncio
async def test_precontext_model_denial_is_terminal_without_fallback(monkeypatch):
    session = Session("s1", "Auto", "http://manual", "manual-model", auto_route=True, owner="alice")
    endpoint = _endpoint(monkeypatch, session)
    selected = _route()
    attempts = []
    monkeypatch.setattr(chat_routes, "resolve_chat_route", lambda *a, **k: selected)

    def authorize(route, session, auth):
        attempts.append(route.reason)
        raise ChatRouteAuthorizationError("model_not_allowed")

    monkeypatch.setattr(chat_routes, "authorize_chat_route", authorize)
    monkeypatch.setattr(chat_routes, "build_chat_context", lambda *a, **k: pytest.fail("context"))

    with pytest.raises(HTTPException) as exc:
        await endpoint(_request(), ChatRequest(message="hello", session="s1"))

    assert exc.value.status_code == 403
    assert attempts == ["auto_chat"]


@pytest.mark.asyncio
async def test_primary_is_reauthorized_jit_and_late_disappearance_uses_fallback(monkeypatch):
    session = Session("s1", "Auto", "http://manual", "manual-model", auto_route=True, owner="alice")
    endpoint = _endpoint(monkeypatch, session)
    selected = _route()
    monkeypatch.setattr(chat_routes, "resolve_chat_route", lambda *a, **k: selected)
    attempts = []

    def authorize(route, session, auth):
        attempts.append(route.reason)
        if route is selected and attempts.count("auto_chat") == 2:
            raise ChatRouteAuthorizationError("endpoint_not_found")
        if route is selected:
            return _authorized("auto-model", "primary")
        return _authorized("manual-model", "fallback", auto=False)

    calls = []
    monkeypatch.setattr(chat_routes, "authorize_chat_route", authorize)
    monkeypatch.setattr(chat_routes, "build_chat_context", lambda *a, **k: _async_value(_context()))
    monkeypatch.setattr(
        chat_routes,
        "llm_call_async",
        lambda url, model, messages, headers=None, **kwargs: _record_async(
            calls, (url, model, dict(headers)), "fallback answer"
        ),
    )
    monkeypatch.setattr(chat_routes, "save_assistant_response", lambda *a, **k: None)
    monkeypatch.setattr(chat_routes, "run_post_response_tasks", lambda *a, **k: None)

    result = await endpoint(_request(), ChatRequest(message="hello", session="s1"))

    assert result == {"response": "fallback answer"}
    assert attempts == ["auto_chat", "auto_chat", "manual_fallback"]
    assert calls == [(
        "http://fallback.invalid/v1/chat/completions",
        "manual-model",
        {"Authorization": "Bearer fallback"},
    )]


@pytest.mark.parametrize(
    ("error", "fallback_expected"),
    [
        (ChatDispatchError(504, "timeout", kind="timeout"), True),
        (ChatDispatchError(502, "network", kind="network"), True),
        (ChatDispatchError(408, "upstream", kind="upstream_status"), True),
        (ChatDispatchError(429, "upstream", kind="upstream_status"), True),
        (ChatDispatchError(500, "upstream", kind="upstream_status"), True),
        (ChatDispatchError(502, "upstream", kind="upstream_status"), True),
        (ChatDispatchError(400, "bad", kind="upstream_status"), False),
        (ChatDispatchError(401, "bad", kind="upstream_status"), False),
        (ChatDispatchError(403, "bad", kind="upstream_status"), False),
        (ChatDispatchError(404, "bad", kind="upstream_status"), False),
        (ChatDispatchError(502, "schema", kind="invalid_response"), False),
        (ValueError("unknown timeout 503"), False),
    ],
)
@pytest.mark.asyncio
async def test_dispatch_fallback_policy_and_candidate_isolation(monkeypatch, error, fallback_expected):
    session = Session("s1", "Auto", "http://manual", "manual-model", auto_route=True, headers={"kept": "manual"}, owner="alice")
    endpoint = _endpoint(monkeypatch, session)
    quota_calls = []
    monkeypatch.setattr(chat_routes, "_enforce_chat_quota", lambda request: quota_calls.append(True))
    selected = _route()
    monkeypatch.setattr(chat_routes, "resolve_chat_route", lambda *a, **k: selected)
    authorizations = []

    def authorize(route, session, auth):
        authorizations.append(route.reason)
        if route is selected:
            return _authorized("auto-model", "primary")
        return _authorized("manual-model", "fallback", auto=False)

    calls = []

    async def dispatch(url, model, messages, headers=None, **kwargs):
        calls.append((url, model, dict(headers)))
        if model == "auto-model":
            raise error
        return "fallback answer"

    monkeypatch.setattr(chat_routes, "authorize_chat_route", authorize)
    monkeypatch.setattr(chat_routes, "build_chat_context", lambda *a, **k: _async_value(_context()))
    monkeypatch.setattr(chat_routes, "llm_call_async", dispatch)
    saved = []
    completed = []
    monkeypatch.setattr(chat_routes, "save_assistant_response", lambda *a, **k: saved.append(a[4]))
    monkeypatch.setattr(
        chat_routes,
        "run_post_response_tasks",
        lambda *a, **k: completed.append(k.get("response_model")),
    )

    if fallback_expected:
        result = await endpoint(_request(), ChatRequest(message="hello", session="s1"))
        assert result == {"response": "fallback answer"}
        assert calls[-1] == (
            "http://fallback.invalid/v1/chat/completions",
            "manual-model",
            {"Authorization": "Bearer fallback"},
        )
        assert authorizations[-1] == "manual_fallback"
        assert quota_calls == [True]
        assert saved == [{"requested_model": "auto-model", "model": "manual-model"}]
        assert completed == ["manual-model"]
        assert session.headers == {"kept": "manual"}
    else:
        with pytest.raises(HTTPException) as exc:
            await endpoint(_request(), ChatRequest(message="hello", session="s1"))
        assert len(calls) == 1
        assert "timeout" not in exc.value.detail
        assert quota_calls == [True]
        assert saved == []
        assert completed == []


@pytest.mark.asyncio
async def test_primary_and_fallback_failure_have_no_assistant_side_effects(monkeypatch):
    session = Session(
        "s1",
        "Auto",
        "http://manual",
        "manual-model",
        auto_route=True,
        headers={"Authorization": "Bearer manual"},
        owner="alice",
    )
    before = (session.model, session.endpoint_url, dict(session.headers), session.auto_route)
    endpoint = _endpoint(monkeypatch, session)
    selected = _route()
    monkeypatch.setattr(chat_routes, "resolve_chat_route", lambda *a, **k: selected)
    monkeypatch.setattr(
        chat_routes,
        "authorize_chat_route",
        lambda route, session, auth: (
            _authorized("auto-model", "primary")
            if route is selected
            else _authorized("manual-model", "fallback", auto=False)
        ),
    )
    monkeypatch.setattr(chat_routes, "build_chat_context", lambda *a, **k: _async_value(_context()))

    async def dispatch(url, model, messages, headers=None, **kwargs):
        raise ChatDispatchError(503, f"{model} unavailable", kind="upstream_status")

    monkeypatch.setattr(chat_routes, "llm_call_async", dispatch)
    monkeypatch.setattr(chat_routes, "save_assistant_response", lambda *a, **k: pytest.fail("save"))
    monkeypatch.setattr(chat_routes, "run_post_response_tasks", lambda *a, **k: pytest.fail("post"))

    with pytest.raises(HTTPException):
        await endpoint(_request(), ChatRequest(message="hello", session="s1"))

    assert (session.model, session.endpoint_url, dict(session.headers), session.auto_route) == before
