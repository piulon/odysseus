from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from routes import chat_helpers
from src.chat_model_router import ChatRoute
from src.request_models import ChatRequest


class _AuthManager:
    def __init__(self, privileges):
        self._privileges = privileges

    def get_privileges(self, _user):
        return dict(self._privileges)


def _request(privileges):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                auth_manager=_AuthManager(privileges)
            )
        )
    )


def test_privilege_gate_accepts_effective_routed_model(monkeypatch):
    monkeypatch.setattr(
        chat_helpers,
        "effective_user",
        lambda _request: "pau",
    )

    request = _request({
        "allowed_models": ["routed-model"],
        "allowed_models_restricted": True,
        "max_messages_per_day": 0,
    })

    sess = SimpleNamespace(model="persistent-fallback")

    chat_helpers._enforce_chat_privileges(
        request,
        sess,
        model_override="routed-model",
    )


def test_privilege_gate_rejects_disallowed_routed_model(monkeypatch):
    monkeypatch.setattr(
        chat_helpers,
        "effective_user",
        lambda _request: "pau",
    )

    request = _request({
        "allowed_models": ["persistent-fallback"],
        "allowed_models_restricted": True,
        "max_messages_per_day": 0,
    })

    sess = SimpleNamespace(model="persistent-fallback")

    with pytest.raises(HTTPException) as exc:
        chat_helpers._enforce_chat_privileges(
            request,
            sess,
            model_override="routed-model",
        )

    assert exc.value.status_code == 403
    assert "routed-model" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_context_uses_runtime_route_without_mutating_session(
    monkeypatch,
):
    sess = SimpleNamespace(
        id="s1",
        name="test",
        endpoint_url="http://persistent/chat",
        model="persistent-model",
        headers={"X-Persistent": "1"},
        owner="pau",
        history=[],
        get_context_messages=lambda: [],
    )

    preset = SimpleNamespace(
        system_prompt="",
        character_name=None,
    )

    preprocessed = SimpleNamespace(
        enhanced_message="hello",
        text_for_context="hello",
        youtube_transcripts=[],
    )

    class Processor:
        _last_used_memories = []

        def build_context_preface(self, **_kwargs):
            return [], [], []

    async def fake_preprocess(*_args, **_kwargs):
        return preprocessed

    captured = {}

    async def fake_compact(
        sess_arg,
        endpoint_url,
        model,
        messages,
        headers,
        owner=None,
    ):
        captured["sess"] = sess_arg
        captured["endpoint_url"] = endpoint_url
        captured["model"] = model
        captured["headers"] = headers
        captured["owner"] = owner
        return messages, 32768, False

    monkeypatch.setattr(
        chat_helpers,
        "extract_preset",
        lambda *_args, **_kwargs: preset,
    )
    monkeypatch.setattr(
        chat_helpers,
        "preprocess",
        fake_preprocess,
    )
    monkeypatch.setattr(
        chat_helpers,
        "add_user_message",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        chat_helpers,
        "effective_user",
        lambda _request: "pau",
    )
    monkeypatch.setattr(
        chat_helpers,
        "load_prefs_for_user",
        lambda _owner: {},
    )
    monkeypatch.setattr(
        chat_helpers,
        "build_uploaded_file_manifest",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        chat_helpers,
        "_is_casual_low_signal",
        lambda _message: False,
    )
    monkeypatch.setattr(
        chat_helpers,
        "_session_is_research_spinoff",
        lambda _sess: False,
    )
    monkeypatch.setattr(
        chat_helpers,
        "maybe_compact",
        fake_compact,
    )
    monkeypatch.setattr(
        chat_helpers,
        "trim_for_context",
        lambda messages, _limit: messages,
    )
    monkeypatch.setattr(
        chat_helpers,
        "estimate_tokens",
        lambda _messages: 0,
    )

    await chat_helpers.build_chat_context(
        sess,
        SimpleNamespace(),
        SimpleNamespace(upload_handler=None),
        Processor(),
        message="hello",
        session_id="s1",
        incognito=True,
        agent_mode=True,
        runtime_endpoint_url="http://tower/chat",
        runtime_model="routed-model",
        runtime_headers={"X-Routed": "1"},
    )

    assert captured["endpoint_url"] == "http://tower/chat"
    assert captured["model"] == "routed-model"
    assert captured["headers"] == {"X-Routed": "1"}

    # Critical invariant: request routing never becomes persistent selection.
    assert sess.endpoint_url == "http://persistent/chat"
    assert sess.model == "persistent-model"
    assert sess.headers == {"X-Persistent": "1"}


@pytest.mark.asyncio
async def test_non_streaming_auto_route_uses_allowed_fallback_chain(
    monkeypatch,
):
    from routes import chat_routes
    from src import chat_model_router
    from src import llm_core
    from core import database

    persistent_route = (
        "http://persistent/chat",
        "persistent-model",
        {"X-Persistent": "1"},
    )
    primary = (
        "http://primary/chat",
        "primary-model",
        {"X-Primary": "1"},
    )
    fallback = (
        "http://fallback/chat",
        "fallback-model",
        {"X-Fallback": "1", "Authorization": "Bearer fallback-secret"},
    )

    saved_messages = []
    sess = SimpleNamespace(
        id="s1",
        name="test",
        endpoint_url=persistent_route[0],
        model=persistent_route[1],
        headers=persistent_route[2],
        owner="pau",
        add_message=saved_messages.append,
    )

    class SessionManager:
        def get_session(self, session_id):
            assert session_id == "s1"
            return sess

        def save_sessions(self):
            return None

    class ChatHandler:
        async def handle_memory_command(self, _sess, _message):
            return None

    class ToolPolicy:
        block_all_tool_calls = False

        def blocks(self, _tool):
            return False

    async def fake_build_context(*_args, **_kwargs):
        return SimpleNamespace(
            messages=[{"role": "user", "content": "hello"}],
            preset=SimpleNamespace(
                temperature=0.2,
                max_tokens=128,
                character_name=None,
            ),
            uprefs={},
            user="pau",
        )

    calls = []

    async def fake_llm_call(url, model, messages, headers=None, **kwargs):
        calls.append((url, model, headers))
        if model == primary[1]:
            raise HTTPException(503, "primary unavailable")
        assert messages == [{"role": "user", "content": "hello"}]
        assert kwargs["session_id"] == "s1"
        return "fallback response"

    async def fail_direct_primary(*_args, **_kwargs):
        raise AssertionError(
            "non-streaming Auto called the primary directly instead of "
            "using the fallback helper"
        )

    monkeypatch.setattr(chat_routes, "_verify_session_owner", lambda *_args: None)
    monkeypatch.setattr(chat_routes, "effective_user", lambda _request: "pau")
    monkeypatch.setattr(chat_routes, "_clear_orphaned_session_endpoint", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_recover_empty_session_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_routes, "_is_image_generation_session", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(chat_routes, "_enforce_chat_privileges", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        chat_routes,
        "_filter_allowed_model_candidates",
        lambda _request, candidates: list(candidates),
    )
    monkeypatch.setattr(chat_routes, "build_effective_tool_policy", lambda **_kwargs: ToolPolicy())
    monkeypatch.setattr(chat_routes, "build_chat_context", fake_build_context)
    monkeypatch.setattr(
        chat_routes,
        "llm_call_async",
        fail_direct_primary,
        raising=False,
    )
    monkeypatch.setattr(llm_core, "llm_call_async", fake_llm_call)
    monkeypatch.setattr(
        chat_routes,
        "clean_thinking_for_save",
        lambda reply, metadata: (reply, dict(metadata)),
    )
    monkeypatch.setattr(chat_routes, "run_post_response_tasks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(database, "update_session_last_accessed", lambda _session_id: None)
    monkeypatch.setattr(
        chat_model_router,
        "resolve_chat_route",
        lambda *_args, **_kwargs: ChatRoute(
            endpoint_url=primary[0],
            model=primary[1],
            headers=primary[2],
            fallbacks=(fallback,),
            reason="adaptive_chat:test",
            lane="chat",
            auto=True,
        ),
    )

    router = chat_routes.setup_chat_routes(
        SessionManager(),
        ChatHandler(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
    )
    endpoint = next(
        route.endpoint
        for route in router.routes
        if route.path == "/api/chat"
    )

    response = await endpoint(
        SimpleNamespace(headers={}),
        ChatRequest(message="hello", session="s1"),
    )

    assert response == {"response": "fallback response"}
    assert calls == [primary, fallback]
    assert saved_messages[-1].content == "fallback response"
    assert saved_messages[-1].metadata == {
        "model": fallback[1],
        "requested_model": primary[1],
    }
    metadata_text = repr(saved_messages[-1].metadata)
    assert fallback[0] not in metadata_text
    assert "Authorization" not in metadata_text
    assert "fallback-secret" not in metadata_text

    assert sess.endpoint_url == persistent_route[0]
    assert sess.model == persistent_route[1]
    assert sess.headers == persistent_route[2]


def test_chat_routes_dispatch_uses_runtime_route():
    from pathlib import Path

    source = Path("routes/chat_routes.py").read_text()

    assert "runtime_route = resolve_chat_route(" in source
    assert "model_override=runtime_route.model" in source

    assert (
        "runtime_endpoint_url=("
        in source
    )
    assert (
        "runtime_route.endpoint_url,\n"
        "                            runtime_route.model,"
        in source
    )
    assert (
        "async for chunk in stream_agent_loop(\n"
        "                        runtime_route.endpoint_url,\n"
        "                        runtime_route.model,"
        in source
    )


def test_fallback_filter_removes_disallowed_model(monkeypatch):
    monkeypatch.setattr(
        chat_helpers,
        "effective_user",
        lambda _request: "pau",
    )

    request = _request({
        "allowed_models": ["primary-model", "allowed-fallback"],
        "allowed_models_restricted": True,
        "max_messages_per_day": 0,
    })

    candidates = [
        ("http://allowed/v1", "allowed-fallback", {}),
        ("http://blocked/v1", "blocked-fallback", {}),
    ]

    filtered = chat_helpers._filter_allowed_model_candidates(
        request,
        candidates,
    )

    assert filtered == [
        ("http://allowed/v1", "allowed-fallback", {}),
    ]


def test_fallback_filter_preserves_allowed_model(monkeypatch):
    monkeypatch.setattr(
        chat_helpers,
        "effective_user",
        lambda _request: "pau",
    )

    request = _request({
        "allowed_models": ["fallback-model"],
        "allowed_models_restricted": True,
        "max_messages_per_day": 0,
    })

    candidate = (
        "http://fallback/v1",
        "fallback-model",
        {"X-Test": "1"},
    )

    filtered = chat_helpers._filter_allowed_model_candidates(
        request,
        [candidate],
    )

    assert filtered == [candidate]


def test_fallback_filter_preserves_chain_when_unrestricted(monkeypatch):
    monkeypatch.setattr(
        chat_helpers,
        "effective_user",
        lambda _request: "pau",
    )

    request = _request({
        "allowed_models": [],
        "allowed_models_restricted": False,
        "max_messages_per_day": 0,
    })

    candidates = [
        ("http://one/v1", "one", {}),
        ("http://two/v1", "two", {}),
    ]

    filtered = chat_helpers._filter_allowed_model_candidates(
        request,
        candidates,
    )

    assert filtered == candidates


def test_stream_dispatch_uses_allowlist_filtered_fallbacks():
    from pathlib import Path

    source = Path("routes/chat_routes.py").read_text()

    assert (
        "allowed_runtime_fallbacks = "
        "_filter_allowed_model_candidates("
        in source
    )

    assert (
        "_fallback_candidates = list(allowed_runtime_fallbacks)"
        in source
    )


def test_capability_sensitive_requests_bypass_auto_route():
    from pathlib import Path

    source = Path("routes/chat_routes.py").read_text()

    assert "image_generation_session = _is_image_generation_session(" in source
    assert "_image_generation_session = _is_image_generation_session(" in source

    assert "bool(att_ids)" in source
    assert "bool(_has_atts)" in source

    assert source.count("allow_auto=not (") >= 2

    # The stream must reuse the same image-mode decision made before routing.
    assert "if _image_generation_session:" in source

    # Research clarification telemetry must report the model that will
    # actually answer that clarification request.
    assert (
        '"model": runtime_route.model, "suffix": "Research"'
        in source
    )
