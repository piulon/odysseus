import json
import urllib.request
from types import SimpleNamespace

import pytest

from src import adaptive_routing_refresh as refresh
from src.adaptive_routing_snapshot import (
    clear_adaptive_routing_snapshot,
    get_adaptive_routing_snapshot,
    publish_adaptive_routing_snapshot,
)


def setup_function():
    clear_adaptive_routing_snapshot()


def teardown_function():
    clear_adaptive_routing_snapshot()


def _endpoint(endpoint_id="ep", *, kind="local", hidden=()):
    return refresh._ProbeEndpoint(
        endpoint_id=endpoint_id,
        scope="local" if kind == "local" else "cloud",
        hidden_models=frozenset(hidden),
        base_url="http://127.0.0.1:11434",
        models_url="http://127.0.0.1:11434/api/tags",
        show_url="http://127.0.0.1:11434/api/show",
        chat_url="http://127.0.0.1:11434/api/chat",
        headers={"Authorization": "recognizable-super-secret"},
    )


def _show(caps=("completion", "tools"), context=40960):
    return {"capabilities": list(caps), "details": {"family": "qwen3"}, "model_info": {"qwen3.context_length": context}}


def _run(monkeypatch, endpoint_ids=("ep",), endpoint=None, request=None, **kwargs):
    monkeypatch.setattr(refresh, "_configured_endpoint_ids", lambda owner: endpoint_ids)
    monkeypatch.setattr(refresh, "_load_ollama_endpoint", lambda endpoint_id, owner: endpoint or _endpoint(endpoint_id))
    return refresh.refresh_owner_adaptive_snapshot("alice", request_json=request, generated_at=100, **kwargs)


def test_configured_ids_are_owner_scoped_and_deduplicated(monkeypatch):
    from src import settings

    calls = []

    def get_user_setting(key, owner, default=None, *, inherit_global=True):
        calls.append((key, owner, default, inherit_global))
        return "shared"

    monkeypatch.setattr(settings, "get_user_setting", get_user_setting)

    assert refresh._configured_endpoint_ids(" alice ") == ("shared",)
    assert {owner for _, owner, _, _ in calls} == {"alice"}
    assert {default for _, _, default, _ in calls} == {""}
    assert {inherit for _, _, _, inherit in calls} == {False}


def test_explicit_owner_does_not_inherit_global_auto_endpoints(monkeypatch):
    from src import settings

    globals_by_key = {
        "auto_chat_endpoint_id": "global-chat",
        "auto_agent_endpoint_id": "global-agent",
    }
    calls = []

    def get_user_setting(key, owner, default=None, *, inherit_global=True):
        calls.append((key, owner, inherit_global))
        if owner and not inherit_global:
            return default
        return globals_by_key.get(key, default)

    monkeypatch.setattr(settings, "get_user_setting", get_user_setting)

    assert refresh._configured_endpoint_ids("alice") == ()
    assert calls == [
        ("auto_chat_endpoint_id", "alice", False),
        ("auto_agent_endpoint_id", "alice", False),
    ]


def test_ownerless_refresh_can_use_global_auto_endpoints(monkeypatch):
    from src import settings

    globals_by_key = {
        "auto_chat_endpoint_id": "global-chat",
        "auto_agent_endpoint_id": "global-agent",
    }
    calls = []

    def get_user_setting(key, owner, default=None, *, inherit_global=True):
        calls.append((key, owner, inherit_global))
        if not owner:
            return globals_by_key.get(key, default)
        return default

    monkeypatch.setattr(settings, "get_user_setting", get_user_setting)

    assert refresh._configured_endpoint_ids(None) == (
        "global-chat",
        "global-agent",
    )
    assert calls == [
        ("auto_chat_endpoint_id", "", False),
        ("auto_agent_endpoint_id", "", False),
    ]


class _Column:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return lambda row: getattr(row, self.name) == value


class _ModelEndpoint:
    id = _Column("id")
    is_enabled = _Column("is_enabled")


class _Query:
    def __init__(self, row):
        self.row = row

    def filter(self, *predicates):
        if self.row is not None and not all(predicate(self.row) for predicate in predicates):
            self.row = None
        return self

    def first(self):
        return self.row


class _DB:
    def __init__(self, row):
        self.row = row
        self.closed = False

    def query(self, _model):
        return _Query(self.row)

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("owner", "endpoint_owner", "enabled", "model_type", "expected"),
    [
        ("alice", "alice", True, "llm", True),
        ("alice", None, True, "llm", True),
        ("alice", "bob", True, "llm", False),
        ("alice", "alice", False, "llm", False),
        ("alice", "alice", True, "embedding", False),
    ],
)
def test_load_endpoint_owner_visibility_and_filters(monkeypatch, owner, endpoint_owner, enabled, model_type, expected):
    import core.database as database
    import src.auth_helpers as auth_helpers
    import src.endpoint_resolver as endpoint_resolver

    row = SimpleNamespace(
        id="ep",
        owner=endpoint_owner,
        is_enabled=enabled,
        model_type=model_type,
        base_url="http://127.0.0.1:11434",
        endpoint_kind="local",
        api_key=None,
        provider_auth_id=None,
        hidden_models=None,
    )
    db = _DB(row)
    monkeypatch.setattr(database, "ModelEndpoint", _ModelEndpoint)
    monkeypatch.setattr(database, "SessionLocal", lambda: db)
    monkeypatch.setattr(
        auth_helpers,
        "owner_filter",
        lambda query, _model, user: query.filter(lambda candidate: candidate.owner in (None, user)),
    )
    monkeypatch.setattr(endpoint_resolver, "resolve_endpoint_runtime", lambda endpoint, owner=None: (endpoint.base_url, None))
    monkeypatch.setattr(endpoint_resolver, "build_headers", lambda api_key, base: {})
    monkeypatch.setattr(endpoint_resolver, "_endpoint_hidden_models", lambda endpoint: set())

    loaded = refresh._load_ollama_endpoint("ep", owner)
    assert (loaded is not None) is expected
    assert db.closed is True


@pytest.mark.parametrize("url", [
    "file:///tmp/x",
    "ftp://host/x",
    "data:text/plain,x",
    "gopher://host/x",
    "http://user:pass@host:11434",
    "http://host name:11434",
    "http://host:11434/api tags",
    "http://host\\evil/api/tags",
    "http:\\\\host\\api\\tags",
    "http://host:11434/\tfoo",
    "http://host:11434/\nfoo",
    "http://host:11434/x?x=1",
    "http://host:11434/x#f",
])
def test_probe_url_rejected(url):
    with pytest.raises(ValueError):
        refresh._safe_probe_url(url)


def test_url_userinfo_rejected_before_probe(monkeypatch):
    monkeypatch.setattr(refresh, "_safe_probe_url", refresh._safe_probe_url)
    with pytest.raises(ValueError, match="userinfo"):
        refresh._safe_probe_url("https://user:recognizable-super-secret@host:11434")


def test_scope_derivation():
    assert refresh._scope_for_endpoint("local", "http://public.example") == "local"
    assert refresh._scope_for_endpoint("api", "http://127.0.0.1:11434") == "cloud"
    assert refresh._scope_for_endpoint("proxy", "http://10.0.0.2:11434") == "cloud"
    assert refresh._scope_for_endpoint("auto", "http://127.0.0.1:11434") == "local"
    assert refresh._scope_for_endpoint("auto", "http://100.64.1.2:11434") == "local"
    assert refresh._scope_for_endpoint("auto", "https://public.example") == "cloud"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "http://127.0.0.1:11434/api/tags",
        "https://ollama.example.com/api/tags",
    ],
)
def test_probe_url_valid(url):
    assert refresh._safe_probe_url(url) == url


def test_canonical_discovery_hidden_and_secret_not_persisted(monkeypatch):
    calls = []
    def request(url, *, payload=None, headers=None, timeout=3):
        calls.append((url, payload))
        if url.endswith("/tags"):
            return {"models": [{"name": "visible"}, {"name": "hidden"}]}
        assert payload == {"model": "visible"}
        return _show(("completion", "tools", "thinking"), 131072)
    snap = _run(monkeypatch, endpoint=_endpoint(hidden=("hidden",)), request=request)
    assert [candidate.model for candidate in snap.candidates] == ["visible"]
    assert len(calls) == 2
    assert "recognizable-super-secret" not in repr(snap)
    assert snap.candidates[0].context_tokens == 131072


def test_model_limit_is_deterministic(monkeypatch):
    old = refresh.MAX_MODELS_PER_ENDPOINT
    monkeypatch.setattr(refresh, "MAX_MODELS_PER_ENDPOINT", 2)
    calls = []
    def request(url, *, payload=None, headers=None, timeout=3):
        calls.append(payload["model"] if payload else "tags")
        return {"models": [{"name": f"m{i}"} for i in range(4)]} if payload is None else _show()
    snap = _run(monkeypatch, request=request)
    assert [c.model for c in snap.candidates] == ["m0", "m1"]
    assert calls == ["tags", "m0", "m1"]
    assert refresh.MAX_MODELS_PER_ENDPOINT == 2 or old != 2


def test_partial_success_and_owner_isolation(monkeypatch):
    endpoints = {"alice": _endpoint("alice"), "bob": _endpoint("bob")}
    monkeypatch.setattr(refresh, "_configured_endpoint_ids", lambda owner: (owner,))
    monkeypatch.setattr(refresh, "_load_ollama_endpoint", lambda eid, owner: endpoints[owner])
    def request(url, *, payload=None, headers=None, timeout=3):
        return {"models": [{"name": "m"}]} if payload is None else _show()
    alice = refresh.refresh_owner_adaptive_snapshot("alice", request_json=request, generated_at=100)
    bob = refresh.refresh_owner_adaptive_snapshot("bob", request_json=request, generated_at=100)
    assert alice.owner == "alice" and bob.owner == "bob"
    assert get_adaptive_routing_snapshot("alice", max_age_seconds=30, now=101) == alice
    assert get_adaptive_routing_snapshot("bob", max_age_seconds=30, now=101) == bob


def test_setup_failure_preserves_previous_snapshot(monkeypatch):
    previous = publish_adaptive_routing_snapshot("alice", (), generated_at=90)
    monkeypatch.setattr(refresh, "_configured_endpoint_ids", lambda owner: (_ for _ in ()).throw(RuntimeError("settings")))
    assert refresh.refresh_owner_adaptive_snapshot("alice", request_json=lambda *a, **k: {}) is None
    assert get_adaptive_routing_snapshot("alice", max_age_seconds=30, now=91) == previous


def test_no_configuration_publishes_empty_snapshot(monkeypatch):
    monkeypatch.setattr(refresh, "_configured_endpoint_ids", lambda owner: ())
    snap = refresh.refresh_owner_adaptive_snapshot("alice", request_json=lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    assert snap.candidates == ()


def test_invalid_configured_endpoints_preserve_snapshot(monkeypatch):
    previous = publish_adaptive_routing_snapshot("alice", (), generated_at=90)
    monkeypatch.setattr(refresh, "_configured_endpoint_ids", lambda owner: ("bad",))
    monkeypatch.setattr(refresh, "_load_ollama_endpoint", lambda eid, owner: None)
    assert refresh.refresh_owner_adaptive_snapshot("alice", request_json=lambda *a, **k: {}) is None
    assert get_adaptive_routing_snapshot("alice", max_age_seconds=30, now=91) == previous


def test_all_tags_fail_preserves_previous_snapshot(monkeypatch):
    previous = publish_adaptive_routing_snapshot("alice", (), generated_at=90)
    snap = _run(monkeypatch, request=lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    assert snap is None
    assert get_adaptive_routing_snapshot("alice", max_age_seconds=30, now=91) == previous


def test_tags_success_zero_models_publishes_empty(monkeypatch):
    snap = _run(monkeypatch, request=lambda url, **kwargs: {"models": []})
    assert snap.candidates == ()


def test_timeout_validation_and_deadline(monkeypatch):
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            refresh.refresh_owner_adaptive_snapshot("alice", request_json=lambda *a, **k: {}, timeout=value)
    clock = iter([0.0, 2.0, 2.0])
    monkeypatch.setattr(refresh.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(refresh, "_configured_endpoint_ids", lambda owner: ("ep",))
    monkeypatch.setattr(refresh, "_load_ollama_endpoint", lambda eid, owner: _endpoint(eid))
    assert refresh.refresh_owner_adaptive_snapshot("alice", request_json=lambda *a, **k: (_ for _ in ()).throw(OSError()), timeout=1) is None


def test_deadline_before_first_show_preserves_previous_snapshot(monkeypatch):
    monkeypatch.setattr(refresh, "_configured_endpoint_ids", lambda owner: ("ep",))
    monkeypatch.setattr(refresh, "_load_ollama_endpoint", lambda eid, owner: _endpoint(eid))
    previous = publish_adaptive_routing_snapshot(
        "alice", (SimpleNamespace(model="previous"),), generated_at=90
    )
    calls = []
    clock = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(refresh.time, "monotonic", lambda: next(clock))
    def request(url, **kwargs):
        calls.append(url)
        return {"models": [{"name": "m"}]}
    snapshot = refresh.refresh_owner_adaptive_snapshot("alice", request_json=request, timeout=1)
    assert snapshot is None
    assert get_adaptive_routing_snapshot("alice", max_age_seconds=30, now=91) == previous
    assert calls == ["http://127.0.0.1:11434/api/tags"]


def test_deadline_after_some_shows_preserves_previous_snapshot(monkeypatch):
    monkeypatch.setattr(refresh, "_configured_endpoint_ids", lambda owner: ("ep",))
    monkeypatch.setattr(refresh, "_load_ollama_endpoint", lambda eid, owner: _endpoint(eid))
    previous = publish_adaptive_routing_snapshot("alice", (), generated_at=90)
    calls = []
    clock = iter([0.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr(refresh.time, "monotonic", lambda: next(clock))

    def request(url, *, payload=None, **kwargs):
        calls.append(payload["model"] if payload else "tags")
        return {"models": [{"name": "m1"}, {"name": "m2"}, {"name": "m3"}]} if payload is None else _show()

    assert refresh.refresh_owner_adaptive_snapshot("alice", request_json=request, timeout=1) is None
    assert get_adaptive_routing_snapshot("alice", max_age_seconds=30, now=91) == previous
    assert calls == ["tags", "m1"]


def test_request_json_mapping_and_size_limit(monkeypatch):
    class Response:
        def __init__(self, body): self.body = body
        def read(self, limit): return self.body[:limit]
        def close(self): pass
    monkeypatch.setattr(refresh._PROBE_OPENER, "open", lambda request, timeout: Response(json.dumps({"ok": 1}).encode()))
    assert refresh._request_json("http://host/api/tags") == {"ok": 1}
    monkeypatch.setattr(refresh, "MAX_PROBE_RESPONSE_BYTES", 8)
    monkeypatch.setattr(refresh._PROBE_OPENER, "open", lambda request, timeout: Response(b"x" * 9))
    with pytest.raises(ValueError): refresh._request_json("http://host/api/tags")


def test_request_json_non_mapping_is_empty(monkeypatch):
    class Response:
        def read(self, limit): return b"[1, 2]"
        def close(self): pass
    monkeypatch.setattr(refresh._PROBE_OPENER, "open", lambda request, timeout: Response())
    assert refresh._request_json("http://host/api/tags") == {}


def test_redirect_handler_rejects_redirect():
    handler = refresh._NoRedirect()
    with pytest.raises(Exception):
        handler.redirect_request(SimpleNamespace(full_url="http://a"), None, 302, "redirect", {}, "http://b")


def test_redirect_handler_never_returns_request_with_auth_headers():
    handler = refresh._NoRedirect()
    request = SimpleNamespace(full_url="http://a", headers={"Authorization": "recognizable-super-secret"})
    with pytest.raises(Exception):
        handler.redirect_request(request, None, 307, "redirect", {}, "https://other")


def test_probe_opener_uses_only_no_redirect_handler():
    handlers = refresh._PROBE_OPENER.handlers
    assert any(isinstance(handler, refresh._NoRedirect) for handler in handlers)
    assert not any(type(handler) is urllib.request.HTTPRedirectHandler for handler in handlers)


def test_request_json_rejects_userinfo_before_opener(monkeypatch):
    monkeypatch.setattr(
        refresh._PROBE_OPENER,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("opener called")),
    )
    with pytest.raises(ValueError, match="userinfo"):
        refresh._request_json("http://user:recognizable-super-secret@host:11434/api/tags")


def test_show_failure_publishes_empty_after_successful_tags(monkeypatch):
    snap = _run(
        monkeypatch,
        request=lambda url, payload=None, **kwargs: (
            {"models": [{"name": "m1"}, {"name": "m2"}]}
            if url.endswith("/tags")
            else _show() if payload == {"model": "m1"}
            else (_ for _ in ()).throw(OSError("show"))
        ),
    )
    assert snap is not None
    assert [candidate.model for candidate in snap.candidates] == ["m1"]


def test_transient_probe_repr_redacts_urls_and_headers():
    endpoint = _endpoint()
    rendered = repr(endpoint)
    assert "recognizable-super-secret" not in rendered
    assert "127.0.0.1" not in rendered


def test_request_json_rejects_nonfinite_timeout():
    for value in (None, "abc", 0, -1, float("nan"), float("inf")):
        with pytest.raises((TypeError, ValueError)):
            refresh._request_json("http://host/api/tags", timeout=value)
