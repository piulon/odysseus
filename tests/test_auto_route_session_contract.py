import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import core.database as database
import core.session_manager as session_manager_module
import routes.session_routes as session_routes
from core.database import ModelEndpoint
from core.database import Session as DbSession
from core.models import Session as RuntimeSession
from core.session_manager import SessionManager


def _request(user="alice", *, admin=True):
    return SimpleNamespace(
        state=SimpleNamespace(current_user=user),
        app=SimpleNamespace(
            state=SimpleNamespace(
                auth_manager=SimpleNamespace(
                    is_admin=lambda _username: admin,
                )
            )
        ),
    )


def _endpoint(router, path, method):
    return next(
        route.endpoint
        for route in reversed(router.routes)
        if getattr(route, "path", "") == path
        and method in getattr(route, "methods", set())
    )


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    original_route_count = len(session_routes.router.routes)
    db_path = tmp_path / "sessions.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    monkeypatch.setattr(session_manager_module, "SessionLocal", factory)
    monkeypatch.setattr(session_routes, "SessionLocal", factory)
    monkeypatch.setattr(session_routes, "effective_user", lambda request: request.state.current_user)

    from src import event_bus

    monkeypatch.setattr(event_bus, "fire_event", lambda *_args, **_kwargs: None)

    manager = object.__new__(SessionManager)
    manager.sessions = {}
    manager.upload_handler = None

    yield factory, manager

    del session_routes.router.routes[original_route_count:]
    engine.dispose()


def test_runtime_default_and_database_round_trip(session_store):
    factory, manager = session_store

    default_session = RuntimeSession(
        id="runtime-default",
        name="Default",
        endpoint_url="http://manual/v1/chat/completions",
        model="manual-model",
    )
    assert default_session.auto_route is False

    created = manager.create_session(
        session_id="auto-session",
        name="Auto",
        endpoint_url="http://manual/v1/chat/completions",
        model="manual-model",
        owner="alice",
        auto_route=True,
    )
    assert created.auto_route is True

    db = factory()
    try:
        db.query(DbSession).filter_by(id="auto-session").one().message_count = 1
        db.commit()
    finally:
        db.close()

    manager.sessions = {}
    manager.load_sessions()
    assert manager.sessions["auto-session"].auto_route is True

    db = factory()
    try:
        stored = db.query(DbSession).filter_by(id="auto-session").one()
        assert stored.auto_route is True
        assert stored.to_dict()["auto_route"] is True
        stored.auto_route = False
        db.commit()
    finally:
        db.close()

    assert manager.sync_session_metadata("auto-session") is True
    assert manager.sessions["auto-session"].auto_route is False


def test_auto_route_migration_is_idempotent_and_preserves_existing_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE sessions ("
            "id TEXT PRIMARY KEY, name TEXT NOT NULL, "
            "endpoint_url TEXT NOT NULL, model TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO sessions (id, name, endpoint_url, model) "
            "VALUES ('legacy', 'Legacy', 'http://manual', 'manual-model')"
        )

    monkeypatch.setattr(database, "DATABASE_URL", f"sqlite:///{db_path}")

    database._migrate_add_auto_route_column()
    database._migrate_add_auto_route_column()

    with sqlite3.connect(db_path) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(sessions)")]
        row = conn.execute(
            "SELECT endpoint_url, model, auto_route FROM sessions WHERE id = 'legacy'"
        ).fetchone()

    assert columns.count("auto_route") == 1
    assert row == ("http://manual", "manual-model", 0)


@pytest.mark.parametrize("requested, expected", [(None, False), ("true", True)])
def test_create_api_exposes_and_persists_auto_route(session_store, requested, expected):
    factory, manager = session_store
    router = session_routes.setup_session_routes(manager, {})
    create = _endpoint(router, "/api/session", "POST")

    response = create(
        request=_request(),
        name="Created",
        endpoint_url="http://manual/v1/chat/completions",
        model="manual-model",
        rag=None,
        skip_validation="true",
        api_key="",
        endpoint_id="",
        auto_route=requested,
    )

    assert response.auto_route is expected
    db = factory()
    try:
        stored = db.query(DbSession).filter_by(id=response.id).one()
        assert stored.auto_route is expected
    finally:
        db.close()


def test_list_api_exposes_auto_route(session_store):
    _factory, manager = session_store
    manager.create_session(
        session_id="listed",
        name="Listed",
        endpoint_url="http://manual/v1/chat/completions",
        model="manual-model",
        owner="alice",
        auto_route=True,
    )
    router = session_routes.setup_session_routes(manager, {})
    list_sessions = _endpoint(router, "/api/sessions", "GET")

    result = list_sessions(request=_request())

    listed = next(item for item in result if item["id"] == "listed")
    assert listed["auto_route"] is True


@pytest.mark.parametrize("requested", ["true", "false"])
def test_auto_only_patch_preserves_manual_selection(session_store, requested):
    factory, manager = session_store
    session = manager.create_session(
        session_id=f"patch-{requested}",
        name="Patch",
        endpoint_url="http://manual/v1/chat/completions",
        model="manual-model",
        owner="alice",
        auto_route=requested == "false",
    )
    session.headers = {"Authorization": "Bearer persistent"}
    db = factory()
    try:
        row = db.query(DbSession).filter_by(id=session.id).one()
        row.headers = dict(session.headers)
        db.commit()
    finally:
        db.close()

    router = session_routes.setup_session_routes(manager, {})
    patch = _endpoint(router, "/api/session/{sid}", "PATCH")
    response = patch(
        request=_request(),
        sid=session.id,
        name=None,
        folder=None,
        model=None,
        endpoint_url=None,
        endpoint_id=None,
        auto_route=requested,
    )

    expected = requested == "true"
    assert response["auto_route"] is expected
    assert (session.model, session.endpoint_url, session.headers) == (
        "manual-model",
        "http://manual/v1/chat/completions",
        {"Authorization": "Bearer persistent"},
    )
    db = factory()
    try:
        stored = db.query(DbSession).filter_by(id=session.id).one()
        assert stored.auto_route is expected
        assert (stored.model, stored.endpoint_url, stored.headers) == (
            "manual-model",
            "http://manual/v1/chat/completions",
            {"Authorization": "Bearer persistent"},
        )
    finally:
        db.close()


@pytest.mark.parametrize("explicit_auto, expected", [(None, False), ("true", True)])
def test_manual_selection_controls_auto_route(session_store, explicit_auto, expected):
    factory, manager = session_store
    session = manager.create_session(
        session_id=f"manual-{expected}",
        name="Manual selection",
        endpoint_url="http://old/v1/chat/completions",
        model="old-model",
        owner="alice",
        auto_route=True,
    )
    router = session_routes.setup_session_routes(manager, {})
    patch = _endpoint(router, "/api/session/{sid}", "PATCH")

    response = patch(
        request=_request(),
        sid=session.id,
        name=None,
        folder=None,
        model="new-model",
        endpoint_url="http://new/v1/chat/completions",
        endpoint_id=None,
        auto_route=explicit_auto,
    )

    assert response["auto_route"] is expected
    assert (session.model, session.endpoint_url, session.auto_route) == (
        "new-model",
        "http://new/v1/chat/completions",
        expected,
    )
    db = factory()
    try:
        stored = db.query(DbSession).filter_by(id=session.id).one()
        assert (stored.model, stored.endpoint_url, stored.auto_route) == (
            "new-model",
            "http://new/v1/chat/completions",
            expected,
        )
    finally:
        db.close()


def test_auto_route_patch_preserves_owner_isolation(session_store):
    factory, manager = session_store
    manager.create_session(
        session_id="bob-session",
        name="Bob",
        endpoint_url="http://manual/v1/chat/completions",
        model="manual-model",
        owner="bob",
        auto_route=False,
    )
    router = session_routes.setup_session_routes(manager, {})
    patch = _endpoint(router, "/api/session/{sid}", "PATCH")

    with pytest.raises(HTTPException) as exc:
        patch(
            request=_request("alice"),
            sid="bob-session",
            name=None,
            folder=None,
            model=None,
            endpoint_url=None,
            endpoint_id=None,
            auto_route="true",
        )

    assert exc.value.status_code == 404
    db = factory()
    try:
        assert db.query(DbSession).filter_by(id="bob-session").one().auto_route is False
    finally:
        db.close()


@pytest.mark.parametrize("field", ["name", "folder"])
def test_metadata_only_patch_preserves_auto_route(session_store, field):
    factory, manager = session_store
    session = manager.create_session(
        session_id=f"metadata-{field}",
        name="Original",
        endpoint_url="http://manual/v1/chat/completions",
        model="manual-model",
        owner="alice",
        auto_route=True,
    )
    router = session_routes.setup_session_routes(manager, {})
    patch = _endpoint(router, "/api/session/{sid}", "PATCH")
    values = {"name": None, "folder": None}
    values[field] = "Updated"

    patch(
        request=_request(),
        sid=session.id,
        model=None,
        endpoint_url=None,
        endpoint_id=None,
        auto_route=None,
        **values,
    )

    assert session.auto_route is True
    db = factory()
    try:
        assert db.query(DbSession).filter_by(id=session.id).one().auto_route is True
    finally:
        db.close()


@pytest.mark.parametrize(
    ("partial_field", "partial_value"),
    [
        ("model", "ignored-model"),
        ("endpoint_url", "http://ignored/v1/chat/completions"),
        ("endpoint_id", "ignored-endpoint"),
    ],
)
def test_partial_manual_selection_is_ignored(
    session_store,
    partial_field,
    partial_value,
):
    factory, manager = session_store
    session = manager.create_session(
        session_id=f"partial-{partial_field}",
        name="Partial",
        endpoint_url="http://manual/v1/chat/completions",
        model="manual-model",
        owner="alice",
        auto_route=True,
    )
    session.headers = {"Authorization": "Bearer persistent"}
    db = factory()
    try:
        row = db.query(DbSession).filter_by(id=session.id).one()
        row.headers = dict(session.headers)
        db.commit()
    finally:
        db.close()

    router = session_routes.setup_session_routes(manager, {})
    patch = _endpoint(router, "/api/session/{sid}", "PATCH")
    values = {"model": None, "endpoint_url": None, "endpoint_id": None}
    values[partial_field] = partial_value

    patch(
        request=_request(),
        sid=session.id,
        name=None,
        folder=None,
        auto_route=None,
        **values,
    )

    expected = (
        "manual-model",
        "http://manual/v1/chat/completions",
        {"Authorization": "Bearer persistent"},
        True,
    )
    assert (session.model, session.endpoint_url, session.headers, session.auto_route) == expected
    db = factory()
    try:
        stored = db.query(DbSession).filter_by(id=session.id).one()
        assert (stored.model, stored.endpoint_url, stored.headers, stored.auto_route) == expected
    finally:
        db.close()


def test_complete_owned_endpoint_selection_disables_auto(session_store):
    factory, manager = session_store
    session = manager.create_session(
        session_id="owned-endpoint-selection",
        name="Owned endpoint",
        endpoint_url="http://old/v1/chat/completions",
        model="old-model",
        owner="alice",
        auto_route=True,
    )
    db = factory()
    try:
        db.add(ModelEndpoint(
            id="alice-endpoint",
            name="Alice endpoint",
            base_url="http://alice-endpoint/v1",
            owner="alice",
            is_enabled=True,
        ))
        db.commit()
    finally:
        db.close()

    router = session_routes.setup_session_routes(manager, {})
    patch = _endpoint(router, "/api/session/{sid}", "PATCH")
    response = patch(
        request=_request(),
        sid=session.id,
        name=None,
        folder=None,
        model="new-model",
        endpoint_url="http://untrusted-placeholder",
        endpoint_id="alice-endpoint",
        auto_route=None,
    )

    assert response["auto_route"] is False
    assert session.model == "new-model"
    assert session.endpoint_url == "http://alice-endpoint/v1/chat/completions"
    db = factory()
    try:
        stored = db.query(DbSession).filter_by(id=session.id).one()
        assert stored.auto_route is False
        assert stored.endpoint_url == "http://alice-endpoint/v1/chat/completions"
    finally:
        db.close()


def test_complete_foreign_endpoint_selection_is_rejected(session_store):
    factory, manager = session_store
    session = manager.create_session(
        session_id="foreign-endpoint-selection",
        name="Foreign endpoint",
        endpoint_url="http://old/v1/chat/completions",
        model="old-model",
        owner="alice",
        auto_route=True,
    )
    db = factory()
    try:
        db.add(ModelEndpoint(
            id="bob-endpoint",
            name="Bob endpoint",
            base_url="http://bob-endpoint/v1",
            owner="bob",
            is_enabled=True,
        ))
        db.commit()
    finally:
        db.close()

    router = session_routes.setup_session_routes(manager, {})
    patch = _endpoint(router, "/api/session/{sid}", "PATCH")
    with pytest.raises(HTTPException) as exc:
        patch(
            request=_request(),
            sid=session.id,
            name=None,
            folder=None,
            model="new-model",
            endpoint_url="http://untrusted-placeholder",
            endpoint_id="bob-endpoint",
            auto_route=None,
        )

    assert exc.value.status_code == 400
    assert (session.model, session.endpoint_url, session.auto_route) == (
        "old-model",
        "http://old/v1/chat/completions",
        True,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("false", False),
        ("", False),
        ("1", False),
        ("0", False),
        ("definitely-not-a-bool", False),
        (None, True),
    ],
)
def test_auto_route_form_parsing_contract(session_store, value, expected):
    factory, manager = session_store
    session = manager.create_session(
        session_id=f"parsing-{value}",
        name="Parsing",
        endpoint_url="http://manual/v1/chat/completions",
        model="manual-model",
        owner="alice",
        auto_route=True,
    )
    router = session_routes.setup_session_routes(manager, {})
    patch = _endpoint(router, "/api/session/{sid}", "PATCH")

    response = patch(
        request=_request(),
        sid=session.id,
        name=None,
        folder=None,
        model=None,
        endpoint_url=None,
        endpoint_id=None,
        auto_route=value,
    )

    assert session.auto_route is expected
    if value is None:
        assert "auto_route" not in response
    else:
        assert response["auto_route"] is expected
    db = factory()
    try:
        assert db.query(DbSession).filter_by(id=session.id).one().auto_route is expected
    finally:
        db.close()


def test_history_route_fork_defaults_auto_route_to_false(session_store, monkeypatch):
    from routes.history import history_routes

    _factory, manager = session_store
    source = manager.create_session(
        session_id="history-fork-source",
        name="Source",
        endpoint_url="http://manual/v1/chat/completions",
        model="manual-model",
        owner="alice",
        auto_route=True,
    )
    source.history = []
    monkeypatch.setattr(history_routes, "effective_user", lambda _request: "alice")
    router = history_routes.setup_history_routes(manager)
    fork = _endpoint(router, "/api/session/{session_id}/fork", "POST")
    request = _request()

    async def request_json():
        return {"keep_count": 0}

    request.json = request_json
    result = asyncio.run(fork(request=request, session_id=source.id))

    forked = manager.sessions[result["id"]]
    assert source.auto_route is True
    assert forked.auto_route is False
    assert (forked.model, forked.endpoint_url) == (source.model, source.endpoint_url)


def test_agent_tool_fork_defaults_auto_route_to_false(session_store, monkeypatch):
    import src.database as database_compat
    from src.agent_tools import session_tools

    factory, manager = session_store
    source = manager.create_session(
        session_id="agent-fork-source",
        name="Source",
        endpoint_url="http://manual/v1/chat/completions",
        model="manual-model",
        owner="alice",
        auto_route=True,
    )
    source.history = []
    monkeypatch.setattr(database_compat, "SessionLocal", factory)
    monkeypatch.setattr(session_tools, "get_session_manager", lambda: manager)

    result = asyncio.run(session_tools.manage_session(
        '{"action": "fork", "session_id": "agent-fork-source"}',
        owner="alice",
    ))

    forked = manager.sessions[result["session_id"]]
    assert source.auto_route is True
    assert forked.auto_route is False
    assert (forked.model, forked.endpoint_url) == (source.model, source.endpoint_url)
