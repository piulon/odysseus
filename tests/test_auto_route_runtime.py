from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from routes import chat_helpers


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
