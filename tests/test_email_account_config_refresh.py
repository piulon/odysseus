import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

pytest.importorskip("mcp")

import mcp_servers.email_server as email_server
import routes.email_routes as email_routes
from core.database import EmailAccount


def _endpoint(router, path, method):
    return next(
        route.endpoint
        for route in router.routes
        if route.path == path and method in route.methods
    )


@pytest.fixture
def account_runtime(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    EmailAccount.__table__.create(engine)
    factory = sessionmaker(bind=engine)

    monkeypatch.setattr("core.database.SessionLocal", factory)
    monkeypatch.setattr("src.secret_storage.encrypt", lambda value: value)
    monkeypatch.setattr("src.secret_storage.decrypt", lambda value: value)
    monkeypatch.setattr(email_server, "APP_DB", str(db_path))
    monkeypatch.setattr(
        email_server,
        "_imap_connect",
        lambda *args, **kwargs: pytest.fail("config refresh must not connect to IMAP"),
    )
    monkeypatch.setattr(email_routes, "_start_poller", lambda: None)
    for key in email_server._OWNER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key in (
        "IMAP_HOST",
        "IMAP_PORT",
        "IMAP_USER",
        "IMAP_PASSWORD",
        "IMAP_SSL",
        "IMAP_STARTTLS",
    ):
        monkeypatch.delenv(key, raising=False)

    router = email_routes.setup_email_routes()
    token = email_server._CURRENT_OWNER.set("admin")
    try:
        yield {
            "create": _endpoint(router, "/api/email/accounts", "POST"),
            "update": _endpoint(router, "/api/email/accounts/{account_id}", "PUT"),
            "delete": _endpoint(router, "/api/email/accounts/{account_id}", "DELETE"),
            "set_default": _endpoint(
                router, "/api/email/accounts/{account_id}/set-default", "POST"
            ),
            "db_path": db_path,
        }
    finally:
        email_server._CURRENT_OWNER.reset(token)
        engine.dispose()


def _account_data(name, host, *, default=False):
    return {
        "name": name,
        "enabled": True,
        "is_default": default,
        "imap_host": host,
        "imap_port": 993,
        "imap_user": f"{name.lower()}@example.test",
        "imap_password": "dummy-test-credential",
        "imap_starttls": False,
    }


@pytest.mark.asyncio
async def test_fallback_then_account_create_refreshes_persistent_mcp_config(account_runtime):
    before = email_server._load_config()
    assert (before["imap_host"], before["imap_port"]) == ("localhost", 31143)

    result = await account_runtime["create"](
        _account_data("Gmail", "imap.gmail.com", default=True), owner="admin"
    )
    assert result["ok"] is True

    after = email_server._load_config()
    assert after["account_id"] == result["id"]
    assert (after["imap_host"], after["imap_port"]) == ("imap.gmail.com", 993)
    assert after["imap_ssl"] is True
    assert after["imap_password"] == "dummy-test-credential"
    assert (after["imap_host"], after["imap_port"]) != ("localhost", 31143)


@pytest.mark.asyncio
async def test_account_update_refreshes_persistent_mcp_config(account_runtime):
    created = await account_runtime["create"](
        _account_data("Work", "imap.old.example", default=True), owner="admin"
    )
    assert email_server._load_config()["imap_host"] == "imap.old.example"

    updated = await account_runtime["update"](
        created["id"], {"imap_host": "imap.new.example", "imap_port": 1143}, owner="admin"
    )
    assert updated["ok"] is True
    config = email_server._load_config()
    assert (config["imap_host"], config["imap_port"]) == ("imap.new.example", 1143)


@pytest.mark.asyncio
async def test_account_delete_removes_persistent_mcp_resolution(account_runtime):
    created = await account_runtime["create"](
        _account_data("Work", "imap.work.example", default=True), owner="admin"
    )
    assert email_server._load_config()["account_id"] == created["id"]

    deleted = await account_runtime["delete"](created["id"], owner="admin")
    assert deleted["ok"] is True
    config = email_server._load_config()
    assert config["account_id"] is None
    assert (config["imap_host"], config["imap_port"]) == ("localhost", 31143)


@pytest.mark.asyncio
async def test_default_account_change_refreshes_persistent_mcp_config(account_runtime):
    first = await account_runtime["create"](
        _account_data("First", "imap.first.example", default=True), owner="admin"
    )
    second = await account_runtime["create"](
        _account_data("Second", "imap.second.example"), owner="admin"
    )
    assert email_server._load_config()["account_id"] == first["id"]

    changed = await account_runtime["set_default"](second["id"], owner="admin")
    assert changed["ok"] is True
    config = email_server._load_config()
    assert config["account_id"] == second["id"]
    assert config["imap_host"] == "imap.second.example"


@pytest.mark.asyncio
async def test_owner_scoping_remains_isolated_after_account_changes(account_runtime):
    admin = await account_runtime["create"](
        _account_data("Admin", "imap.admin.example", default=True), owner="admin"
    )
    alice = await account_runtime["create"](
        _account_data("Alice", "imap.alice.example", default=True), owner="alice"
    )

    assert email_server._load_config()["account_id"] == admin["id"]
    token = email_server._CURRENT_OWNER.set("alice")
    try:
        config = email_server._load_config()
        assert config["account_id"] == alice["id"]
        assert config["imap_host"] == "imap.alice.example"
    finally:
        email_server._CURRENT_OWNER.reset(token)

    assert email_server._load_config()["account_id"] == admin["id"]


def test_environment_fallback_still_works_without_database_account(
    account_runtime, monkeypatch
):
    monkeypatch.setenv("IMAP_HOST", "imap.legacy.example")
    monkeypatch.setenv("IMAP_PORT", "1993")
    monkeypatch.setenv("IMAP_USER", "legacy@example.test")
    monkeypatch.setenv("IMAP_PASSWORD", "dummy-legacy-credential")
    monkeypatch.setenv("IMAP_SSL", "true")
    monkeypatch.setenv("IMAP_STARTTLS", "false")

    config = email_server._load_config()
    assert config["account_id"] is None
    assert (config["imap_host"], config["imap_port"]) == (
        "imap.legacy.example",
        1993,
    )
    assert config["imap_user"] == "legacy@example.test"
    assert config["imap_password"] == "dummy-legacy-credential"
    assert config["imap_ssl"] is True
    assert config["imap_starttls"] is False
