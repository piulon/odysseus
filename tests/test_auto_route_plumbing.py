import inspect
import sqlite3
from types import SimpleNamespace

import core.database as database
from core.database import Session as DbSession
from core.models import Session as RuntimeSession
from core.session_manager import SessionManager
from src.request_models import SessionResponse
from src.settings import DEFAULT_SETTINGS, _PER_USER_KEYS


def test_runtime_session_has_auto_route_default_false():
    sess = RuntimeSession(
        id="s1",
        name="test",
        endpoint_url="http://example/chat",
        model="model-a",
    )

    assert sess.auto_route is False


def test_runtime_session_can_enable_auto_route():
    sess = RuntimeSession(
        id="s1",
        name="test",
        endpoint_url="http://example/chat",
        model="model-a",
        auto_route=True,
    )

    assert sess.auto_route is True


def test_database_session_has_auto_route_column():
    assert "auto_route" in DbSession.__table__.columns


def test_database_session_to_dict_exposes_auto_route():
    row = DbSession(
        id="s1",
        name="test",
        endpoint_url="http://example/chat",
        model="model-a",
        auto_route=True,
    )

    assert row.to_dict()["auto_route"] is True


def test_session_response_exposes_auto_route():
    response = SessionResponse(
        id="s1",
        name="test",
        model="model-a",
        auto_route=True,
    )

    assert response.auto_route is True


def test_auto_route_settings_exist():
    expected = {
        "auto_chat_endpoint_id",
        "auto_chat_model",
        "auto_agent_endpoint_id",
        "auto_agent_model",
    }

    assert expected <= set(DEFAULT_SETTINGS)
    assert expected <= _PER_USER_KEYS


def test_session_manager_create_accepts_auto_route():
    param = inspect.signature(
        SessionManager.create_session
    ).parameters["auto_route"]

    assert param.default is False


def test_db_to_session_meta_carries_auto_route():
    manager = object.__new__(SessionManager)

    row = SimpleNamespace(
        id="s1",
        name="test",
        endpoint_url="http://example/chat",
        model="model-a",
        rag=False,
        archived=False,
        auto_route=True,
        headers={},
        owner="pau",
        is_important=False,
        message_count=0,
    )

    sess = manager._db_to_session_meta(row)

    assert sess.auto_route is True


def test_auto_route_migration_is_idempotent(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE sessions ("
            "id TEXT PRIMARY KEY, "
            "name TEXT, "
            "endpoint_url TEXT, "
            "model TEXT"
            ")"
        )
        conn.commit()

    monkeypatch.setattr(
        database,
        "DATABASE_URL",
        f"sqlite:///{db_path}",
    )

    database._migrate_add_auto_route_column()
    database._migrate_add_auto_route_column()

    with sqlite3.connect(db_path) as conn:
        columns = {
            row[1]: row
            for row in conn.execute(
                "PRAGMA table_info(sessions)"
            ).fetchall()
        }

    assert "auto_route" in columns


def test_session_routes_post_exposes_auto_route():
    from pathlib import Path

    source = Path("routes/session_routes.py").read_text()

    patch_marker = '    @router.patch("/session/{sid}")'
    post_source = source.split(patch_marker, 1)[0]

    assert "auto_route: str = Form(None)" in post_source
    assert "auto_route=auto_route_val" in post_source

    response_start = post_source.rfind("return SessionResponse(")
    assert response_start >= 0

    response_source = post_source[response_start:]
    assert "auto_route=auto_route_val" in response_source


def test_session_routes_patch_has_manual_auto_semantics():
    from pathlib import Path

    source = Path("routes/session_routes.py").read_text()

    patch_marker = '    @router.patch("/session/{sid}")'
    patch_source = source.split(patch_marker, 1)[1]

    assert "requested_auto_route = (" in patch_source
    assert "session.auto_route = False" in patch_source
    assert "session.auto_route = requested_auto_route" in patch_source
    assert "db_session.auto_route = bool(session.auto_route)" in patch_source


def test_session_routes_supports_auto_only_patch():
    from pathlib import Path

    source = Path("routes/session_routes.py").read_text()

    patch_marker = '    @router.patch("/session/{sid}")'
    patch_source = source.split(patch_marker, 1)[1]

    assert "elif requested_auto_route is not None:" in patch_source
    assert (
        "db_session.auto_route = bool(requested_auto_route)"
        in patch_source
    )
    assert (
        'result["auto_route"] = bool(requested_auto_route)'
        in patch_source
    )
