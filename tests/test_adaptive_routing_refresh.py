from src import model_capabilities as mc
from src.adaptive_routing import RoutingCandidate
from src.adaptive_routing_refresh import (
    _configured_endpoint_ids,
    refresh_owner_adaptive_snapshot,
)
from src.adaptive_routing_snapshot import (
    clear_adaptive_routing_snapshot,
    get_adaptive_routing_snapshot,
    publish_adaptive_routing_snapshot,
)


def setup_function():
    clear_adaptive_routing_snapshot()


def teardown_function():
    clear_adaptive_routing_snapshot()


def endpoint(
    endpoint_id,
    *,
    base_url,
    hidden_models=(),
):
    root = base_url.rstrip("/")
    return {
        "endpoint_id": endpoint_id,
        "base_url": root,
        "models_url": root + "/api/tags",
        "show_url": root + "/api/show",
        "chat_url": root + "/api/chat",
        "headers": {},
        "hidden_models": frozenset(hidden_models),
    }


def show_payload(
    capabilities,
    *,
    family="qwen3",
    context_tokens=40960,
):
    return {
        "capabilities": list(capabilities),
        "details": {
            "family": family,
        },
        "model_info": {
            f"{family}.context_length": context_tokens,
        },
    }


def test_configured_endpoint_ids_are_owner_scoped_and_deduplicated(monkeypatch):
    from src import settings

    values = {
        "auto_chat_endpoint_id": "ep-shared",
        "auto_agent_endpoint_id": "ep-shared",
    }
    calls = []

    monkeypatch.setattr(
        settings,
        "load_settings",
        lambda: dict(values),
    )

    def fake_get_user_setting(key, owner, default):
        calls.append((key, owner, default))
        return values.get(key, default)

    monkeypatch.setattr(
        settings,
        "get_user_setting",
        fake_get_user_setting,
    )

    assert _configured_endpoint_ids("alice") == (
        "ep-shared",
    )
    assert {call[1] for call in calls} == {"alice"}


def test_refresh_builds_snapshot_from_canonical_ollama_evidence(monkeypatch):
    import src.adaptive_routing_refresh as refresh

    monkeypatch.setattr(
        refresh,
        "_configured_endpoint_ids",
        lambda owner: ("msi",),
    )
    monkeypatch.setattr(
        refresh,
        "_load_ollama_endpoint",
        lambda endpoint_id, owner: endpoint(
            "msi",
            base_url="http://msi.example",
        ),
    )

    def request_json(url, *, payload=None, headers=None, timeout=3.0):
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {"name": "qwen3:4b"},
                    {"name": "gemma3:4b"},
                ],
            }

        model = payload["model"]

        if model == "qwen3:4b":
            return show_payload(
                ["completion", "tools", "thinking"],
                context_tokens=262144,
            )

        if model == "gemma3:4b":
            return show_payload(
                ["completion", "vision"],
                family="gemma3",
                context_tokens=131072,
            )

        raise AssertionError(model)

    snapshot = refresh_owner_adaptive_snapshot(
        "alice",
        request_json=request_json,
        generated_at=100,
    )

    assert snapshot is not None
    assert snapshot.owner == "alice"
    assert snapshot.generated_at == 100
    assert [item.model for item in snapshot.candidates] == [
        "qwen3:4b",
        "gemma3:4b",
    ]

    qwen, gemma = snapshot.candidates

    assert set(qwen.capabilities) == {
        mc.CAP_TOOL_CALL,
        mc.CAP_REASONING,
    }
    assert qwen.context_tokens == 262144
    assert qwen.endpoint_id == "msi"

    assert gemma.capabilities == (
        mc.CAP_VISION,
    )
    assert gemma.context_tokens == 131072


def test_hidden_models_are_not_published(monkeypatch):
    import src.adaptive_routing_refresh as refresh

    monkeypatch.setattr(
        refresh,
        "_configured_endpoint_ids",
        lambda owner: ("msi",),
    )
    monkeypatch.setattr(
        refresh,
        "_load_ollama_endpoint",
        lambda endpoint_id, owner: endpoint(
            "msi",
            base_url="http://msi.example",
            hidden_models=("hidden-model",),
        ),
    )

    def request_json(url, *, payload=None, headers=None, timeout=3.0):
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {"name": "visible-model"},
                    {"name": "hidden-model"},
                ],
            }

        assert payload["model"] == "visible-model"
        return show_payload(
            ["completion", "tools"],
        )

    snapshot = refresh_owner_adaptive_snapshot(
        "alice",
        request_json=request_json,
        generated_at=100,
    )

    assert snapshot is not None
    assert [item.model for item in snapshot.candidates] == [
        "visible-model",
    ]


def test_endpoint_failure_publishes_surviving_endpoint(monkeypatch):
    import src.adaptive_routing_refresh as refresh

    endpoints = {
        "msi": endpoint(
            "msi",
            base_url="http://msi.example",
        ),
        "tower": endpoint(
            "tower",
            base_url="http://tower.example",
        ),
    }

    monkeypatch.setattr(
        refresh,
        "_configured_endpoint_ids",
        lambda owner: ("msi", "tower"),
    )
    monkeypatch.setattr(
        refresh,
        "_load_ollama_endpoint",
        lambda endpoint_id, owner: endpoints[endpoint_id],
    )

    def request_json(url, *, payload=None, headers=None, timeout=3.0):
        if "tower.example" in url:
            raise OSError("tower offline")

        if url.endswith("/api/tags"):
            return {
                "models": [
                    {"name": "qwen3:4b"},
                ],
            }

        return show_payload(
            ["completion", "tools", "thinking"],
            context_tokens=262144,
        )

    snapshot = refresh_owner_adaptive_snapshot(
        "alice",
        request_json=request_json,
        generated_at=100,
    )

    assert snapshot is not None
    assert [
        (item.endpoint_id, item.model)
        for item in snapshot.candidates
    ] == [
        ("msi", "qwen3:4b"),
    ]


def test_all_endpoint_probe_failures_publish_empty_fresh_snapshot(monkeypatch):
    import src.adaptive_routing_refresh as refresh

    monkeypatch.setattr(
        refresh,
        "_configured_endpoint_ids",
        lambda owner: ("msi", "tower"),
    )
    monkeypatch.setattr(
        refresh,
        "_load_ollama_endpoint",
        lambda endpoint_id, owner: endpoint(
            endpoint_id,
            base_url=f"http://{endpoint_id}.example",
        ),
    )

    publish_adaptive_routing_snapshot(
        "alice",
        [
            RoutingCandidate(
                endpoint_id="old",
                endpoint_url="http://old.example/chat",
                model="old-model",
                node="old",
                scope="local",
            ),
        ],
        generated_at=90,
    )

    def request_json(url, *, payload=None, headers=None, timeout=3.0):
        raise OSError("offline")

    snapshot = refresh_owner_adaptive_snapshot(
        "alice",
        request_json=request_json,
        generated_at=100,
    )

    assert snapshot is not None
    assert snapshot.generated_at == 100
    assert snapshot.candidates == ()

    current = get_adaptive_routing_snapshot(
        "alice",
        max_age_seconds=30,
        now=101,
    )
    assert current == snapshot


def test_setup_failure_preserves_previous_snapshot(monkeypatch):
    import src.adaptive_routing_refresh as refresh

    previous = publish_adaptive_routing_snapshot(
        "alice",
        [
            RoutingCandidate(
                endpoint_id="msi",
                endpoint_url="http://msi.example/api/chat",
                model="qwen3:4b",
                node="msi",
                scope="local",
            ),
        ],
        generated_at=100,
    )

    def fail_settings(owner):
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(
        refresh,
        "_configured_endpoint_ids",
        fail_settings,
    )

    result = refresh_owner_adaptive_snapshot(
        "alice",
        request_json=lambda *args, **kwargs: {},
        generated_at=101,
    )

    assert result is None

    current = get_adaptive_routing_snapshot(
        "alice",
        max_age_seconds=30,
        now=102,
    )
    assert current == previous


def test_non_ollama_or_unavailable_target_does_not_trigger_network(monkeypatch):
    import src.adaptive_routing_refresh as refresh

    monkeypatch.setattr(
        refresh,
        "_configured_endpoint_ids",
        lambda owner: ("cloud",),
    )
    monkeypatch.setattr(
        refresh,
        "_load_ollama_endpoint",
        lambda endpoint_id, owner: None,
    )

    def forbidden_request(*args, **kwargs):
        raise AssertionError("network should not be called")

    snapshot = refresh_owner_adaptive_snapshot(
        "alice",
        request_json=forbidden_request,
        generated_at=100,
    )

    assert snapshot is not None
    assert snapshot.owner == "alice"
    assert snapshot.candidates == ()
