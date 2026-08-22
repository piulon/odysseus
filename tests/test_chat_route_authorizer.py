from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import core.database as database
from core.auth import ADMIN_PRIVILEGES
from core.database import ModelEndpoint
from src import chat_route_authorizer, endpoint_resolver
from src.chat_model_router import ChatRoute, RouteTarget
from src.chat_route_authorizer import (
    ChatRouteAuthContext,
    ChatRouteAuthorizationError,
    authorize_chat_route,
)


NORMAL_PRIVILEGES = {
    "allowed_models": [],
    "allowed_models_restricted": False,
    "block_all_models": False,
    "max_messages_per_day": 0,
}


def _auth(
    owner="alice",
    *,
    privileges=NORMAL_PRIVILEGES,
    is_admin=False,
    single_user=False,
):
    return ChatRouteAuthContext(
        owner=owner,
        privileges=privileges,
        is_admin=is_admin,
        single_user=single_user,
    )


def _session(*, auto=True, url="http://manual.internal/v1/chat/completions"):
    return SimpleNamespace(
        id="session-1",
        model="manual-model",
        endpoint_url=url,
        headers={"Authorization": "Bearer manual-secret", "X-Manual": "yes"},
        auto_route=auto,
        owner="alice",
        name="Manual session",
    )


def _snapshot(session):
    return (
        session.model,
        session.endpoint_url,
        dict(session.headers),
        session.auto_route,
        session.owner,
        session.name,
    )


def _auto_route(endpoint_id="auto-endpoint", model="auto-model", lane="chat"):
    return ChatRoute(
        auto=True,
        lane=lane,
        target=RouteTarget(model=model, endpoint_id=endpoint_id),
        reason=f"auto_{lane}",
        manual_fallback=RouteTarget(
            model="manual-model",
            endpoint_url="http://manual.internal/v1/chat/completions",
        ),
    )


def _manual_route(session):
    return ChatRoute(
        auto=False,
        lane="manual",
        target=RouteTarget(model=session.model, endpoint_url=session.endpoint_url),
        reason="manual",
    )


@pytest.fixture
def authorizer_store(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'authorized-routes.db'}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(endpoint_resolver, "SessionLocal", factory)
    monkeypatch.setattr(chat_route_authorizer, "SessionLocal", factory)

    def runtime(endpoint, owner=None):
        return endpoint.base_url, f"secret-{endpoint.id}"

    monkeypatch.setattr(endpoint_resolver, "resolve_endpoint_runtime", runtime)

    def add_endpoint(
        endpoint_id="auto-endpoint",
        *,
        owner="alice",
        enabled=True,
        base_url=None,
        hidden=None,
    ):
        db = factory()
        try:
            db.add(ModelEndpoint(
                id=endpoint_id,
                name=endpoint_id,
                base_url=base_url or f"http://{endpoint_id}.internal/v1",
                owner=owner,
                is_enabled=enabled,
                hidden_models=json.dumps(hidden or []),
            ))
            db.commit()
        finally:
            db.close()

    yield factory, add_endpoint
    engine.dispose()


@pytest.mark.parametrize("endpoint_owner", ["alice", None])
def test_auto_owned_or_explicit_shared_target_is_authorized(
    authorizer_store,
    endpoint_owner,
):
    _factory, add_endpoint = authorizer_store
    add_endpoint(owner=endpoint_owner)
    session = _session()
    before = _snapshot(session)

    authorized = authorize_chat_route(
        _auto_route(), session, auth=_auth()
    )

    assert authorized.auto is True
    assert authorized.lane == "chat"
    assert authorized.reason == "auto_chat"
    assert authorized.endpoint_id == "auto-endpoint"
    assert authorized.model == "auto-model"
    assert authorized.endpoint_url == "http://auto-endpoint.internal/v1/chat/completions"
    assert authorized.headers["Authorization"] == "Bearer secret-auto-endpoint"
    assert _snapshot(session) == before


def test_ownerless_auto_shared_target_is_authorized(authorizer_store):
    _factory, add_endpoint = authorizer_store
    add_endpoint(owner=None)

    authorized = authorize_chat_route(
        _auto_route(), _session(), auth=_auth(owner=None, privileges=None, single_user=True)
    )

    assert authorized.endpoint_id == "auto-endpoint"


@pytest.mark.parametrize("privileges", [None, {}])
def test_explicit_owner_without_resolved_privileges_fails_closed(
    authorizer_store,
    privileges,
):
    _factory, add_endpoint = authorizer_store
    add_endpoint()

    with pytest.raises(ChatRouteAuthorizationError) as exc:
        authorize_chat_route(
            _auto_route(),
            _session(),
            auth=_auth(privileges=privileges),
        )

    assert exc.value.code == "privileges_unavailable"


@pytest.mark.parametrize(
    "auth",
    [
        _auth(owner="alice", single_user=True),
        _auth(owner=None, privileges=None, single_user=False),
        _auth(owner=None, privileges=ADMIN_PRIVILEGES, is_admin=True),
    ],
)
def test_ambiguous_auth_context_fails_closed(authorizer_store, auth):
    with pytest.raises(ChatRouteAuthorizationError) as exc:
        authorize_chat_route(_manual_route(_session(auto=False)), _session(auto=False), auth=auth)

    assert exc.value.code == "invalid_auth_context"


def test_single_user_raw_manual_remains_compatible(authorizer_store):
    session = _session(auto=False, url="http://single-user.internal/chat/completions")

    authorized = authorize_chat_route(
        _manual_route(session),
        session,
        auth=_auth(owner=None, privileges=None, single_user=True),
    )

    assert authorized.endpoint_url == session.endpoint_url
    assert authorized.model == "manual-model"


def test_permissive_normal_privileges_do_not_grant_raw_manual(authorizer_store):
    session = _session(auto=False, url="http://normal-user.internal/chat/completions")
    permissive = dict(ADMIN_PRIVILEGES)

    with pytest.raises(ChatRouteAuthorizationError) as exc:
        authorize_chat_route(
            _manual_route(session),
            session,
            auth=_auth(privileges=permissive, is_admin=False),
        )

    assert exc.value.code == "manual_endpoint_not_allowed"


@pytest.mark.parametrize("change", ["delete", "disable", "owner", "hidden"])
def test_auto_target_is_revalidated_after_phase2(authorizer_store, change):
    factory, add_endpoint = authorizer_store
    add_endpoint()
    route = _auto_route()
    db = factory()
    try:
        endpoint = db.query(ModelEndpoint).filter(ModelEndpoint.id == "auto-endpoint").one()
        if change == "delete":
            db.delete(endpoint)
        elif change == "disable":
            endpoint.is_enabled = False
        elif change == "owner":
            endpoint.owner = "bob"
        else:
            endpoint.hidden_models = json.dumps(["auto-model"])
        db.commit()
    finally:
        db.close()

    with pytest.raises(ChatRouteAuthorizationError) as exc:
        authorize_chat_route(route, _session(), auth=_auth())

    assert exc.value.code == ("model_hidden" if change == "hidden" else "endpoint_not_found")


def test_auto_model_authorization_allow_deny_and_block(authorizer_store):
    _factory, add_endpoint = authorizer_store
    add_endpoint()
    allowed = {"allowed_models_restricted": True, "allowed_models": ["auto-model"]}
    denied = {"allowed_models_restricted": True, "allowed_models": ["other"]}

    assert authorize_chat_route(
        _auto_route(), _session(), auth=_auth(privileges=allowed)
    ).model == "auto-model"
    with pytest.raises(ChatRouteAuthorizationError) as denied_exc:
        authorize_chat_route(
            _auto_route(), _session(), auth=_auth(privileges=denied)
        )
    assert denied_exc.value.code == "model_not_allowed"
    with pytest.raises(ChatRouteAuthorizationError) as blocked_exc:
        authorize_chat_route(
            _auto_route(),
            _session(),
            auth=_auth(privileges={"block_all_models": True}),
        )
    assert blocked_exc.value.code == "block_all_models"


def test_auto_credential_error_is_sanitized(authorizer_store, monkeypatch):
    _factory, add_endpoint = authorizer_store
    add_endpoint()
    monkeypatch.setattr(
        endpoint_resolver,
        "resolve_endpoint_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("Bearer credential-super-secret at http://internal-secret")
        ),
    )

    with pytest.raises(ChatRouteAuthorizationError) as exc:
        authorize_chat_route(_auto_route(), _session(), auth=_auth())

    assert exc.value.code == "credentials_unavailable"
    assert "credential-super-secret" not in str(exc.value)
    assert "internal-secret" not in repr(exc.value)


def test_manual_raw_admin_compatibility_and_immutable_headers(authorizer_store):
    session = _session(auto=False, url="http://raw-admin.internal/v1/chat/completions")
    before = _snapshot(session)

    authorized = authorize_chat_route(
        _manual_route(session),
        session,
        auth=_auth(owner="admin", privileges=ADMIN_PRIVILEGES, is_admin=True),
    )

    assert authorized.model == "manual-model"
    assert authorized.endpoint_id is None
    assert authorized.endpoint_url == session.endpoint_url
    assert dict(authorized.headers) == before[2]
    session.headers["Authorization"] = "Bearer changed"
    assert authorized.headers["Authorization"] == "Bearer manual-secret"
    with pytest.raises(TypeError):
        authorized.headers["Authorization"] = "Bearer other"


def test_manual_raw_non_admin_is_denied(authorizer_store):
    session = _session(auto=False, url="http://raw-user.internal/chat/completions")

    with pytest.raises(ChatRouteAuthorizationError) as exc:
        authorize_chat_route(
            _manual_route(session), session, auth=_auth()
        )

    assert exc.value.code == "manual_endpoint_not_allowed"


@pytest.mark.parametrize(
    ("endpoint_owner", "enabled", "allowed"),
    [("alice", True, True), (None, True, True), ("bob", True, False), ("alice", False, False)],
)
def test_manual_registered_endpoint_revalidates_owner_and_enabled(
    authorizer_store,
    endpoint_owner,
    enabled,
    allowed,
):
    _factory, add_endpoint = authorizer_store
    base = "http://registered.internal/v1"
    add_endpoint(
        "registered",
        owner=endpoint_owner,
        enabled=enabled,
        base_url=base,
    )
    session = _session(auto=False, url=f"{base}/chat/completions")

    if allowed:
        authorized = authorize_chat_route(
            _manual_route(session), session, auth=_auth()
        )
        assert authorized.endpoint_id == "registered"
        assert authorized.headers["Authorization"] == "Bearer secret-registered"
    else:
        with pytest.raises(ChatRouteAuthorizationError) as exc:
            authorize_chat_route(
                _manual_route(session),
                session,
                auth=_auth(),
            )
        assert exc.value.code == "manual_endpoint_not_allowed"


def test_manual_model_is_authorized_and_session_is_immutable(authorizer_store):
    session = _session(auto=False, url="http://raw-admin.internal/chat/completions")
    before = _snapshot(session)

    with pytest.raises(ChatRouteAuthorizationError) as exc:
        authorize_chat_route(
            _manual_route(session),
            session,
            auth=_auth(
                owner="admin",
                privileges={"allowed_models_restricted": True, "allowed_models": ["other"]},
                is_admin=True,
            ),
        )

    assert exc.value.code == "model_not_allowed"
    assert _snapshot(session) == before


def test_authorized_route_repr_and_safe_metadata_exclude_runtime_secrets(authorizer_store):
    session = _session(auto=False, url="http://recognizable-internal:8123/chat")
    authorized = authorize_chat_route(
        _manual_route(session),
        session,
        auth=_auth(owner="admin", privileges=ADMIN_PRIVILEGES, is_admin=True),
    )

    rendered = repr(authorized)
    assert "recognizable-internal" not in rendered
    assert "manual-secret" not in rendered
    assert "Authorization" not in rendered

    unsafe = asdict(authorized)
    assert unsafe["endpoint_url"] == session.endpoint_url
    safe = authorized.to_safe_metadata()
    assert set(safe) == {"auto", "lane", "reason", "model", "endpoint_id"}
    assert safe == {
        "auto": False,
        "lane": "manual",
        "reason": "manual",
        "model": "manual-model",
        "endpoint_id": None,
    }
    assert "endpoint_url" not in safe
    assert "headers" not in safe
    assert "manual-secret" not in repr(safe)


def test_reauthorization_detects_change_and_candidates_do_not_share_runtime_state(
    authorizer_store,
):
    factory, add_endpoint = authorizer_store
    add_endpoint("candidate-a", base_url="http://candidate-a.internal/v1")
    add_endpoint("candidate-b", base_url="http://candidate-b.internal/v1")
    first = authorize_chat_route(
        _auto_route("candidate-a", "model-a"),
        _session(),
        auth=_auth(),
    )
    second = authorize_chat_route(
        _auto_route("candidate-b", "model-b"),
        _session(),
        auth=_auth(),
    )

    assert first.endpoint_url != second.endpoint_url
    assert first.headers["Authorization"] == "Bearer secret-candidate-a"
    assert second.headers["Authorization"] == "Bearer secret-candidate-b"
    assert first.headers is not second.headers

    db = factory()
    try:
        endpoint = db.query(ModelEndpoint).filter(ModelEndpoint.id == "candidate-a").one()
        endpoint.is_enabled = False
        db.commit()
    finally:
        db.close()
    with pytest.raises(ChatRouteAuthorizationError) as exc:
        authorize_chat_route(
            _auto_route("candidate-a", "model-a"),
            _session(),
            auth=_auth(),
        )
    assert exc.value.code == "endpoint_not_found"


@pytest.mark.parametrize(
    ("lane", "agent_mode"),
    [("chat", False), ("agent", True)],
)
def test_adaptive_runtime_candidate_uses_authoritative_endpoint(
    authorizer_store,
    monkeypatch,
    lane,
    agent_mode,
):
    from routes import chat_routes
    from src import adaptive_chat_router
    from src import settings as settings_module
    from src.adaptive_routing import RoutingCandidate
    from src.adaptive_routing_snapshot import (
        clear_adaptive_routing_snapshot,
        publish_adaptive_routing_snapshot,
    )

    _factory, add_endpoint = authorizer_store

    add_endpoint(
        endpoint_id="adaptive-endpoint",
        owner="alice",
        base_url="http://authoritative.internal/v1",
    )

    legacy = _auto_route(
        endpoint_id=f"legacy-{lane}",
        model=f"legacy-{lane}-model",
        lane=lane,
    )

    monkeypatch.setattr(
        adaptive_chat_router,
        "resolve_chat_route",
        lambda *a, **k: legacy,
    )
    monkeypatch.setattr(
        settings_module,
        "get_setting",
        lambda key, default=None: {
            "adaptive_routing_enabled": True,
            "adaptive_routing_snapshot_ttl_seconds": 60,
        }.get(key, default),
    )

    publish_adaptive_routing_snapshot(
        "alice",
        [
            RoutingCandidate(
                endpoint_id="adaptive-endpoint",
                endpoint_url="http://snapshot-must-not-win.invalid/v1/chat/completions",
                model="adaptive-model",
                node="tower",
                scope="local",
                reachable=True,
            )
        ],
    )

    session = _session()
    before = _snapshot(session)

    try:
        selector = (
            chat_routes._select_auto_agent_context_candidate
            if agent_mode
            else chat_routes._select_auto_stream_context_candidate
        )

        (
            selected_primary,
            requested_model,
            manual_fallback,
            context_route,
            context_candidate,
        ) = selector(
            session,
            owner="alice",
            auth=_auth(),
        )

        assert selected_primary.reason == f"adaptive_{lane}"
        assert selected_primary.target.endpoint_id == "adaptive-endpoint"
        assert selected_primary.target.model == "adaptive-model"
        assert selected_primary.target.endpoint_url is None

        assert requested_model == "adaptive-model"
        assert context_route is selected_primary

        assert context_candidate.endpoint_id == "adaptive-endpoint"
        assert context_candidate.model == "adaptive-model"
        assert (
            context_candidate.endpoint_url
            == "http://authoritative.internal/v1/chat/completions"
        )
        assert (
            context_candidate.headers["Authorization"]
            == "Bearer secret-adaptive-endpoint"
        )

        assert "snapshot-must-not-win" not in context_candidate.endpoint_url
        assert manual_fallback is not None
        assert _snapshot(session) == before
    finally:
        clear_adaptive_routing_snapshot("alice")


@pytest.mark.parametrize(
    ("lane", "agent_mode"),
    [("chat", False), ("agent", True)],
)
def test_adaptive_foreign_endpoint_falls_back_without_leaking_owner_data(
    authorizer_store,
    monkeypatch,
    lane,
    agent_mode,
):
    from routes import chat_routes
    from src import adaptive_chat_router
    from src import settings as settings_module
    from src.adaptive_routing import RoutingCandidate
    from src.adaptive_routing_snapshot import (
        clear_adaptive_routing_snapshot,
        publish_adaptive_routing_snapshot,
    )

    _factory, add_endpoint = authorizer_store

    # Endpoint que Alice pot utilitzar com a fallback manual.
    add_endpoint(
        endpoint_id="manual-endpoint",
        owner="alice",
        base_url="http://manual.internal/v1",
    )

    # Endpoint existent però propietat de Bob.
    add_endpoint(
        endpoint_id="foreign-endpoint",
        owner="bob",
        base_url="http://bob-secret.internal/v1",
    )

    legacy = _auto_route(
        endpoint_id=f"legacy-{lane}",
        model=f"legacy-{lane}-model",
        lane=lane,
    )

    monkeypatch.setattr(
        adaptive_chat_router,
        "resolve_chat_route",
        lambda *a, **k: legacy,
    )
    monkeypatch.setattr(
        settings_module,
        "get_setting",
        lambda key, default=None: {
            "adaptive_routing_enabled": True,
            "adaptive_routing_snapshot_ttl_seconds": 60,
        }.get(key, default),
    )

    publish_adaptive_routing_snapshot(
        "alice",
        [
            RoutingCandidate(
                endpoint_id="foreign-endpoint",
                endpoint_url="http://snapshot-bob.invalid/v1/chat/completions",
                model="foreign-model",
                node="foreign-node",
                scope="local",
                reachable=True,
            )
        ],
    )

    session = _session()
    before = _snapshot(session)

    try:
        selector = (
            chat_routes._select_auto_agent_context_candidate
            if agent_mode
            else chat_routes._select_auto_stream_context_candidate
        )

        (
            selected_primary,
            requested_model,
            manual_fallback,
            context_route,
            context_candidate,
        ) = selector(
            session,
            owner="alice",
            auth=_auth(),
        )

        # Adaptive pot proposar la identitat, però no autoritzar-la.
        assert selected_primary.reason == f"adaptive_{lane}"
        assert selected_primary.target.endpoint_id == "foreign-endpoint"
        assert selected_primary.target.model == "foreign-model"
        assert selected_primary.target.endpoint_url is None
        assert requested_model == "foreign-model"

        # endpoint_not_found és recoverable: es passa al fallback manual.
        assert manual_fallback is not None
        assert context_route is manual_fallback
        assert context_route.auto is False
        assert context_route.reason == "manual_fallback"

        # El fallback es rehidrata exclusivament amb dades autoritzades d'Alice.
        assert context_candidate.endpoint_id == "manual-endpoint"
        assert context_candidate.model == "manual-model"
        assert (
            context_candidate.endpoint_url
            == "http://manual.internal/v1/chat/completions"
        )
        assert (
            context_candidate.headers["Authorization"]
            == "Bearer secret-manual-endpoint"
        )

        # Cap dada executable de Bob ni del snapshot pot travessar la frontera.
        assert "bob-secret" not in context_candidate.endpoint_url
        assert "snapshot-bob" not in context_candidate.endpoint_url
        assert "foreign-endpoint" not in context_candidate.headers["Authorization"]

        assert _snapshot(session) == before
    finally:
        clear_adaptive_routing_snapshot("alice")


@pytest.mark.parametrize(
    ("lane", "agent_mode"),
    [("chat", False), ("agent", True)],
)
def test_adaptive_model_policy_denial_is_terminal(
    authorizer_store,
    monkeypatch,
    lane,
    agent_mode,
):
    from routes import chat_routes
    from src import adaptive_chat_router
    from src import settings as settings_module
    from src.adaptive_routing import RoutingCandidate
    from src.adaptive_routing_snapshot import (
        clear_adaptive_routing_snapshot,
        publish_adaptive_routing_snapshot,
    )

    _factory, add_endpoint = authorizer_store

    add_endpoint(
        endpoint_id="adaptive-endpoint",
        owner="alice",
        base_url="http://adaptive.internal/v1",
    )

    # També existeix un fallback manual vàlid.
    add_endpoint(
        endpoint_id="manual-endpoint",
        owner="alice",
        base_url="http://manual.internal/v1",
    )

    legacy = _auto_route(
        endpoint_id=f"legacy-{lane}",
        model=f"legacy-{lane}-model",
        lane=lane,
    )

    monkeypatch.setattr(
        adaptive_chat_router,
        "resolve_chat_route",
        lambda *a, **k: legacy,
    )
    monkeypatch.setattr(
        settings_module,
        "get_setting",
        lambda key, default=None: {
            "adaptive_routing_enabled": True,
            "adaptive_routing_snapshot_ttl_seconds": 60,
        }.get(key, default),
    )

    publish_adaptive_routing_snapshot(
        "alice",
        [
            RoutingCandidate(
                endpoint_id="adaptive-endpoint",
                endpoint_url="http://snapshot.invalid/v1/chat/completions",
                model="adaptive-model",
                node="tower",
                scope="local",
                reachable=True,
            )
        ],
    )

    session = _session()
    before = _snapshot(session)

    denied_privileges = {
        "allowed_models": ["some-other-model"],
        "allowed_models_restricted": True,
        "block_all_models": False,
        "max_messages_per_day": 0,
    }

    try:
        selector = (
            chat_routes._select_auto_agent_context_candidate
            if agent_mode
            else chat_routes._select_auto_stream_context_candidate
        )

        with pytest.raises(ChatRouteAuthorizationError) as exc:
            selector(
                session,
                owner="alice",
                auth=_auth(privileges=denied_privileges),
            )

        assert exc.value.code == "model_not_allowed"

        # La política és terminal: el fallback manual no s'ha d'utilitzar.
        assert _snapshot(session) == before
    finally:
        clear_adaptive_routing_snapshot("alice")
