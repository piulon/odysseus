import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import core.database as database
from core.database import ModelEndpoint
from src import endpoint_resolver
from src.endpoint_resolver import (
    StrictEndpointTarget,
    StrictEndpointResolutionError,
    resolve_endpoint_by_id_strict,
)


@pytest.fixture
def strict_store(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'strict-endpoints.db'}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(endpoint_resolver, "SessionLocal", factory)
    monkeypatch.setattr(
        endpoint_resolver,
        "resolve_endpoint_runtime",
        lambda endpoint, owner=None: (endpoint.base_url, "runtime-secret"),
    )

    def add_endpoint(
        endpoint_id="endpoint",
        *,
        owner="alice",
        enabled=True,
        hidden=None,
        cached=None,
    ):
        db = factory()
        try:
            db.add(ModelEndpoint(
                id=endpoint_id,
                name=endpoint_id,
                base_url=f"https://{endpoint_id}.example/v1",
                owner=owner,
                is_enabled=enabled,
                hidden_models=json.dumps(hidden or []),
                cached_models=json.dumps(cached or []),
            ))
            db.commit()
        finally:
            db.close()

    yield factory, add_endpoint
    engine.dispose()


def _resolve(endpoint_id="endpoint", *, owner="alice", ownerless_shared_only=False):
    return resolve_endpoint_by_id_strict(
        endpoint_id,
        "requested-model",
        owner=owner,
        ownerless_shared_only=ownerless_shared_only,
    )


@pytest.mark.parametrize("endpoint_owner", ["alice", None])
def test_strict_resolver_accepts_owned_or_shared_enabled_endpoint(
    strict_store,
    endpoint_owner,
):
    _factory, add_endpoint = strict_store
    add_endpoint(owner=endpoint_owner)

    resolved = _resolve()

    assert resolved.endpoint_id == "endpoint"
    assert resolved.model == "requested-model"
    assert resolved.endpoint_url == "https://endpoint.example/v1/chat/completions"
    assert resolved.headers == {"Authorization": "Bearer runtime-secret"}


def test_strict_resolver_accepts_shared_endpoint_in_ownerless_shared_only_mode(
    strict_store,
):
    _factory, add_endpoint = strict_store
    add_endpoint(owner=None)

    assert _resolve(owner=None, ownerless_shared_only=True).endpoint_id == "endpoint"


def test_strict_resolver_defaults_ownerless_resolution_to_shared(strict_store):
    _factory, add_endpoint = strict_store
    add_endpoint(owner=None)

    resolved = resolve_endpoint_by_id_strict(
        "endpoint",
        "requested-model",
        owner=None,
    )

    assert resolved.endpoint_id == "endpoint"


def test_strict_resolver_default_ownerless_resolution_denies_owned_endpoint(
    strict_store,
):
    _factory, add_endpoint = strict_store
    add_endpoint(owner="alice")

    with pytest.raises(StrictEndpointResolutionError) as exc:
        resolve_endpoint_by_id_strict(
            "endpoint",
            "requested-model",
            owner=None,
        )

    assert exc.value.code == "endpoint_not_found"


@pytest.mark.parametrize(
    ("endpoint_owner", "owner", "ownerless_shared_only", "code"),
    [
        ("alice", None, True, "endpoint_not_found"),
        ("bob", "alice", False, "endpoint_not_found"),
    ],
)
def test_strict_resolver_denies_out_of_scope_endpoint(
    strict_store,
    endpoint_owner,
    owner,
    ownerless_shared_only,
    code,
):
    _factory, add_endpoint = strict_store
    add_endpoint(owner=endpoint_owner)

    with pytest.raises(StrictEndpointResolutionError) as exc:
        _resolve(owner=owner, ownerless_shared_only=ownerless_shared_only)

    assert exc.value.code == code


def test_strict_resolver_denies_missing_endpoint(strict_store):
    with pytest.raises(StrictEndpointResolutionError) as exc:
        _resolve()
    assert exc.value.code == "endpoint_not_found"


def test_strict_resolver_denies_disabled_endpoint(strict_store):
    _factory, add_endpoint = strict_store
    add_endpoint(enabled=False)

    with pytest.raises(StrictEndpointResolutionError) as exc:
        _resolve()
    assert exc.value.code == "endpoint_not_found"


def test_strict_resolver_denies_hidden_model_without_substitution(strict_store):
    _factory, add_endpoint = strict_store
    add_endpoint(hidden=["requested-model"], cached=["requested-model", "other-model"])

    with pytest.raises(StrictEndpointResolutionError) as exc:
        _resolve()

    assert exc.value.code == "model_hidden"
    assert "other-model" not in str(exc.value)


def test_strict_resolver_requires_a_nonempty_exact_model(strict_store):
    _factory, add_endpoint = strict_store
    add_endpoint(cached=["other-model"])

    with pytest.raises(StrictEndpointResolutionError) as exc:
        resolve_endpoint_by_id_strict("endpoint", "", owner="alice")

    assert exc.value.code == "model_required"


def test_strict_resolver_does_not_require_cached_model_membership(strict_store):
    _factory, add_endpoint = strict_store
    add_endpoint(cached=["different-known-model"])

    assert _resolve().model == "requested-model"


def test_strict_resolver_sanitizes_credential_errors(strict_store, monkeypatch):
    _factory, add_endpoint = strict_store
    add_endpoint()
    def mismatched_credentials(_endpoint, owner=None):
        assert owner == "alice"
        raise RuntimeError("Bearer super-secret-token")

    monkeypatch.setattr(
        endpoint_resolver,
        "resolve_endpoint_runtime",
        mismatched_credentials,
    )

    with pytest.raises(StrictEndpointResolutionError) as exc:
        _resolve()

    assert exc.value.code == "credentials_unavailable"
    assert "super-secret-token" not in str(exc.value)
    assert "super-secret-token" not in repr(exc.value)


def test_strict_target_repr_hides_url_headers_and_secrets():
    target = StrictEndpointTarget(
        endpoint_id="safe-endpoint-id",
        model="safe-model",
        endpoint_url="http://internal-recognizable-host:8123/v1/chat/completions",
        headers={"Authorization": "Bearer recognizable-secret"},
    )

    rendered = repr(target)
    assert "safe-endpoint-id" in rendered
    assert "safe-model" in rendered
    assert "internal-recognizable-host" not in rendered
    assert "Authorization" not in rendered
    assert "recognizable-secret" not in rendered
    assert "headers" not in rendered


def test_strict_target_headers_are_defensively_copied_and_immutable():
    original = {"Authorization": "Bearer original"}
    first = StrictEndpointTarget("endpoint-1", "model", "http://internal", original)
    second = StrictEndpointTarget("endpoint-2", "model", "http://internal", original)

    original["Authorization"] = "Bearer changed"

    assert first.headers["Authorization"] == "Bearer original"
    assert second.headers["Authorization"] == "Bearer original"
    assert first.headers is not second.headers
    assert dict(first.headers) == {"Authorization": "Bearer original"}
    with pytest.raises(TypeError):
        first.headers["Authorization"] = "Bearer other"


def test_strict_resolver_sanitizes_database_errors(monkeypatch):
    monkeypatch.setattr(
        endpoint_resolver,
        "SessionLocal",
        lambda: (_ for _ in ()).throw(RuntimeError("database-secret")),
    )

    with pytest.raises(StrictEndpointResolutionError) as exc:
        _resolve()

    assert exc.value.code == "endpoint_unavailable"
    assert "database-secret" not in str(exc.value)


def test_strict_resolver_does_not_mutate_endpoint_or_database(strict_store):
    factory, add_endpoint = strict_store
    add_endpoint(hidden=[], cached=["requested-model", "other-model"])
    before_db = factory()
    try:
        before = before_db.query(ModelEndpoint).filter(ModelEndpoint.id == "endpoint").one()
        snapshot = (before.owner, before.is_enabled, before.hidden_models, before.cached_models)
    finally:
        before_db.close()

    _resolve()

    after_db = factory()
    try:
        after = after_db.query(ModelEndpoint).filter(ModelEndpoint.id == "endpoint").one()
        assert (after.owner, after.is_enabled, after.hidden_models, after.cached_models) == snapshot
    finally:
        after_db.close()


def test_strict_resolver_requeries_database_on_every_call(strict_store):
    factory, add_endpoint = strict_store
    add_endpoint()
    assert _resolve().endpoint_id == "endpoint"

    db = factory()
    try:
        endpoint = db.query(ModelEndpoint).filter(ModelEndpoint.id == "endpoint").one()
        endpoint.is_enabled = False
        db.commit()
    finally:
        db.close()

    with pytest.raises(StrictEndpointResolutionError) as exc:
        _resolve()
    assert exc.value.code == "endpoint_not_found"


def test_strict_resolver_builds_url_without_host_resolution(strict_store, monkeypatch):
    _factory, add_endpoint = strict_store
    add_endpoint()
    monkeypatch.setattr(
        endpoint_resolver,
        "resolve_url",
        lambda _url: (_ for _ in ()).throw(AssertionError("resolve_url called")),
    )

    assert _resolve().endpoint_url.endswith("/chat/completions")
