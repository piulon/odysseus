from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.adaptive_routing import RoutingCandidate
from src.adaptive_routing_dispatch import (
    resolve_adaptive_dispatch_candidate,
)


def candidate(**overrides):
    values = {
        "endpoint_id": "ep-local",
        "endpoint_url": "http://snapshot-host:11434/api/chat",
        "model": "model-a",
        "node": "local-node",
        "scope": "local",
        "capabilities": ("tool_call",),
        "reachable": True,
    }
    values.update(overrides)
    return RoutingCandidate(**values)


class FakeQuery:
    def __init__(self, row):
        self.row = row

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.row


class FakeDB:
    def __init__(self, row):
        self.row = row
        self.closed = False

    def query(self, model):
        return FakeQuery(self.row)

    def close(self):
        self.closed = True


def endpoint_row(**overrides):
    values = {
        "id": "ep-local",
        "is_enabled": True,
        "base_url": "http://db-host:11434/v1",
        "api_key": "secret",
        "hidden_models": "[]",
        "model_type": "llm",
        "endpoint_kind": "local",
        "provider_auth_id": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dispatch_uses_snapshot_url_and_local_headers_without_url_resolution():
    db = FakeDB(endpoint_row())

    with (
        patch(
            "src.adaptive_routing_dispatch.SessionLocal",
            return_value=db,
        ),
        patch(
            "src.adaptive_routing_dispatch.build_headers",
            return_value={"Authorization": "Bearer redacted"},
        ) as headers,
        patch(
            "src.auth_helpers.owner_filter",
            side_effect=lambda q, model, owner: q,
        ) as owner_filter,
    ):
        result = resolve_adaptive_dispatch_candidate(
            candidate(),
            owner="pau",
        )

    assert result == (
        "http://snapshot-host:11434/api/chat",
        "model-a",
        {"Authorization": "Bearer redacted"},
    )
    headers.assert_called_once_with(
        "secret",
        "http://db-host:11434/v1",
    )
    owner_filter.assert_called_once()
    assert db.closed is True


def test_dispatch_rejects_hidden_model():
    db = FakeDB(
        endpoint_row(
            hidden_models='["model-a"]',
        )
    )

    with patch(
        "src.adaptive_routing_dispatch.SessionLocal",
        return_value=db,
    ):
        assert (
            resolve_adaptive_dispatch_candidate(
                candidate(),
                owner=None,
            )
            is None
        )


def test_dispatch_rejects_missing_endpoint_and_oauth_endpoint():
    missing_db = FakeDB(None)

    with patch(
        "src.adaptive_routing_dispatch.SessionLocal",
        return_value=missing_db,
    ):
        assert (
            resolve_adaptive_dispatch_candidate(
                candidate(),
            )
            is None
        )

    oauth_db = FakeDB(
        endpoint_row(
            provider_auth_id="auth-row",
        )
    )

    with patch(
        "src.adaptive_routing_dispatch.SessionLocal",
        return_value=oauth_db,
    ):
        assert (
            resolve_adaptive_dispatch_candidate(
                candidate(),
            )
            is None
        )


def test_dispatch_is_local_only():
    with patch(
        "src.adaptive_routing_dispatch.SessionLocal",
        side_effect=AssertionError(
            "cloud candidate must be rejected before DB access"
        ),
    ):
        assert (
            resolve_adaptive_dispatch_candidate(
                candidate(scope="cloud"),
            )
            is None
        )


def test_dispatch_module_has_no_network_resolution_calls():
    source = Path(
        "src/adaptive_routing_dispatch.py"
    ).read_text(encoding="utf-8")

    assert "resolve_endpoint_runtime" not in source
    assert "resolve_endpoint_by_id" not in source
    assert "build_chat_url" not in source
    assert "resolve_url" not in source
    assert "httpx" not in source
    assert "requests" not in source
