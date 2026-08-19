from types import SimpleNamespace

import pytest

import routes.chat_helpers as chat_helpers
from routes.chat_helpers import PreprocessedMessage, PresetInfo


class _AuthManager:
    def __init__(self, privileges, *, admin=False):
        self.privileges = privileges
        self.admin = admin

    def get_privileges(self, owner):
        return self.privileges

    def is_admin(self, owner):
        return self.admin


def _request(privileges=None, *, admin=False):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                auth_manager=_AuthManager(privileges, admin=admin),
            )
        )
    )


def test_route_auth_context_uses_server_authority(monkeypatch):
    monkeypatch.setattr(chat_helpers, "effective_user", lambda request: "alice")
    monkeypatch.setattr(chat_helpers, "_auth_disabled", lambda: False)
    privileges = {"allowed_models": ["owned-model"]}

    auth = chat_helpers.build_chat_route_auth_context(
        _request(privileges, admin=True)
    )

    assert auth.owner == "alice"
    assert dict(auth.privileges) == privileges
    assert auth.is_admin is True
    assert auth.single_user is False


def test_route_auth_context_single_user_is_explicit_auth_disabled(monkeypatch):
    monkeypatch.setattr(chat_helpers, "effective_user", lambda request: None)
    monkeypatch.setattr(chat_helpers, "_auth_disabled", lambda: True)

    auth = chat_helpers.build_chat_route_auth_context(_request())

    assert auth.owner is None
    assert auth.privileges is None
    assert auth.is_admin is False
    assert auth.single_user is True


def test_route_auth_context_multiuser_without_privileges_fails_closed(monkeypatch):
    monkeypatch.setattr(chat_helpers, "effective_user", lambda request: "alice")
    monkeypatch.setattr(chat_helpers, "_auth_disabled", lambda: False)

    auth = chat_helpers.build_chat_route_auth_context(_request(None))

    assert auth.owner == "alice"
    assert auth.privileges is None
    assert auth.single_user is False


def test_route_auth_context_is_admin_lookup_failure_fails_closed(monkeypatch):
    monkeypatch.setattr(chat_helpers, "effective_user", lambda request: "alice")
    monkeypatch.setattr(chat_helpers, "_auth_disabled", lambda: False)
    request = _request({"allowed_models_restricted": False})
    request.app.state.auth_manager.is_admin = lambda owner: (_ for _ in ()).throw(
        RuntimeError("Bearer AUTH_LOOKUP_SECRET")
    )

    auth = chat_helpers.build_chat_route_auth_context(request)

    assert auth.owner == "alice"
    assert auth.privileges is None
    assert auth.is_admin is False


def test_legacy_privilege_gate_preserves_lookup_exception(monkeypatch):
    monkeypatch.setattr(chat_helpers, "effective_user", lambda request: "alice")
    request = _request({})
    request.app.state.auth_manager.get_privileges = lambda owner: (_ for _ in ()).throw(
        RuntimeError("legacy lookup failure")
    )

    with pytest.raises(RuntimeError, match="legacy lookup failure"):
        chat_helpers._enforce_chat_privileges(
            request,
            SimpleNamespace(model="manual-model"),
        )


def test_chat_message_model_override_is_request_scoped(monkeypatch):
    emitted = []
    webhook = SimpleNamespace(
        fire_and_forget=lambda name, payload: emitted.append((name, payload))
    )
    request = _request({})
    monkeypatch.setattr(chat_helpers, "effective_user", lambda request: "alice")
    monkeypatch.setattr("src.event_bus.fire_event", lambda name, owner: emitted.append((name, owner)))
    session = SimpleNamespace(model="manual-model")

    chat_helpers.fire_message_event(
        request,
        webhook,
        "session-1",
        session,
        "hello",
        model_override="auto-primary",
    )

    assert emitted[0] == (
        "chat.message",
        {"session_id": "session-1", "model": "auto-primary", "message": "hello"},
    )
    assert session.model == "manual-model"


@pytest.mark.asyncio
async def test_build_context_runtime_target_drives_compaction_without_mutating_session(monkeypatch):
    captured = {}
    session = SimpleNamespace(
        model="manual-model",
        endpoint_url="http://manual.invalid/v1/chat/completions",
        headers={"Authorization": "Bearer manual"},
        owner="alice",
        history=[],
        get_context_messages=lambda: [],
    )
    before = (session.model, session.endpoint_url, dict(session.headers))

    async def preprocess(*args, **kwargs):
        return PreprocessedMessage("hello", "hello", "hello", [], [])

    async def compact(sess, endpoint_url, model, messages, headers, owner=None):
        captured.update(
            endpoint_url=endpoint_url,
            model=model,
            headers=dict(headers),
            owner=owner,
        )
        return messages, 8192, False

    monkeypatch.setattr(chat_helpers, "preprocess", preprocess)
    monkeypatch.setattr(chat_helpers, "extract_preset", lambda handler, preset: PresetInfo(None, None, None, None))
    monkeypatch.setattr(chat_helpers, "add_user_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_helpers, "load_prefs_for_user", lambda owner: {})
    monkeypatch.setattr(chat_helpers, "effective_user", lambda request: "alice")
    monkeypatch.setattr(chat_helpers, "_normalize_model_id_from_cache", lambda sess: pytest.fail("legacy normalization"))
    monkeypatch.setattr(chat_helpers, "normalize_model_id", lambda *args, **kwargs: pytest.fail("live normalization"))
    monkeypatch.setattr(chat_helpers, "maybe_compact", compact)
    monkeypatch.setattr(chat_helpers, "trim_for_context", lambda messages, length: messages)

    await chat_helpers.build_chat_context(
        session,
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(build_context_preface=lambda **kwargs: ([], [], [])),
        message="hello",
        session_id="s1",
        incognito=True,
        runtime_model="auto-model",
        runtime_endpoint_url="http://auto.invalid/v1/chat/completions",
        runtime_headers={"Authorization": "Bearer auto"},
        model_event_override="auto-model",
    )

    assert captured == {
        "endpoint_url": "http://auto.invalid/v1/chat/completions",
        "model": "auto-model",
        "headers": {"Authorization": "Bearer auto"},
        "owner": "alice",
    }
    assert (session.model, session.endpoint_url, dict(session.headers)) == before


def test_post_response_default_uses_persistent_session_model(monkeypatch):
    emitted = []
    session = SimpleNamespace(
        model="manual-model",
        history=[],
        name="Already named",
    )
    webhook = SimpleNamespace(
        fire_and_forget=lambda name, payload: emitted.append((name, payload))
    )

    chat_helpers.run_post_response_tasks(
        session,
        SimpleNamespace(),
        "s1",
        "hello",
        "answer",
        None,
        {},
        SimpleNamespace(),
        None,
        webhook,
        allow_background_extraction=False,
    )

    assert emitted == [(
        "chat.completed",
        {
            "session_id": "s1",
            "model": "manual-model",
            "user_message": "hello",
            "response": "answer",
        },
    )]
