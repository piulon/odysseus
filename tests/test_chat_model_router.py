from dataclasses import asdict
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import core.database as database
from core.database import ModelEndpoint
from src import chat_model_router
from src import settings as settings_module
from src.settings import DEFAULT_SETTINGS, _PER_USER_KEYS


AUTO_SETTING_KEYS = {
    "auto_chat_endpoint_id",
    "auto_chat_model",
    "auto_agent_endpoint_id",
    "auto_agent_model",
}


def _session(*, auto_route, url="http://manual/v1/chat/completions", model="manual-model"):
    return SimpleNamespace(
        auto_route=auto_route,
        endpoint_url=url,
        model=model,
        headers={"Authorization": "Bearer persistent-secret"},
    )


def _snapshot(session):
    return (
        session.auto_route,
        session.endpoint_url,
        session.model,
        dict(session.headers),
    )


@pytest.fixture
def router_store(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'router.db'}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(chat_model_router, "SessionLocal", factory)

    settings = {key: "" for key in AUTO_SETTING_KEYS}
    monkeypatch.setattr(
        chat_model_router,
        "get_user_setting",
        lambda key, _owner="", default=None, *, inherit_global=True: settings.get(
            key, default
        ),
    )

    def add_endpoint(
        endpoint_id,
        *,
        owner="alice",
        enabled=True,
        base_url=None,
    ):
        db = factory()
        try:
            db.add(ModelEndpoint(
                id=endpoint_id,
                name=endpoint_id,
                base_url=base_url or f"http://{endpoint_id}/v1",
                owner=owner,
                is_enabled=enabled,
            ))
            db.commit()
        finally:
            db.close()

    yield settings, add_endpoint
    engine.dispose()


def test_auto_target_settings_are_defaulted_and_owner_scoped():
    assert AUTO_SETTING_KEYS <= set(DEFAULT_SETTINGS)
    assert AUTO_SETTING_KEYS <= _PER_USER_KEYS
    assert all(DEFAULT_SETTINGS[key] == "" for key in AUTO_SETTING_KEYS)


def test_get_user_setting_can_disable_global_inheritance_without_changing_default(
    monkeypatch,
):
    from routes import prefs_routes

    monkeypatch.setattr(
        settings_module,
        "load_settings",
        lambda: {"auto_chat_model": "global-model"},
    )
    monkeypatch.setattr(prefs_routes, "_load_for_user", lambda _owner: {})

    assert settings_module.get_user_setting(
        "auto_chat_model",
        "alice",
        "",
    ) == "global-model"
    assert settings_module.get_user_setting(
        "auto_chat_model",
        "alice",
        "",
        inherit_global=False,
    ) == ""
    assert settings_module.get_user_setting(
        "auto_chat_model",
        "",
        "",
        inherit_global=False,
    ) == "global-model"


def test_manual_route_preserves_session_without_credentials_in_result(router_store):
    session = _session(auto_route=False)
    before = _snapshot(session)

    route = chat_model_router.resolve_chat_route(session, owner="alice")

    assert route.auto is False
    assert route.lane == "manual"
    assert route.reason == "manual"
    assert route.target.endpoint_id is None
    assert route.target.endpoint_url == session.endpoint_url
    assert route.target.model == session.model
    assert route.manual_fallback is None
    assert _snapshot(session) == before
    assert "persistent-secret" not in repr(asdict(route))


@pytest.mark.parametrize(
    ("agent_mode", "prefix", "lane"),
    [(False, "auto_chat", "chat"), (True, "auto_agent", "agent")],
)
def test_configured_auto_lane_selects_owned_endpoint(
    router_store,
    agent_mode,
    prefix,
    lane,
):
    settings, add_endpoint = router_store
    add_endpoint(f"{lane}-endpoint", owner="alice")
    settings[f"{prefix}_endpoint_id"] = f"{lane}-endpoint"
    settings[f"{prefix}_model"] = f"{lane}-model"
    session = _session(auto_route=True)
    before = _snapshot(session)

    route = chat_model_router.resolve_chat_route(
        session,
        owner="alice",
        agent_mode=agent_mode,
    )

    assert route.auto is True
    assert route.lane == lane
    assert route.reason == f"auto_{lane}"
    assert route.target.endpoint_id == f"{lane}-endpoint"
    assert route.target.endpoint_url is None
    assert route.target.model == f"{lane}-model"
    assert route.manual_fallback.endpoint_url == session.endpoint_url
    assert route.manual_fallback.model == session.model
    assert _snapshot(session) == before
    assert "persistent-secret" not in repr(asdict(route))


@pytest.mark.parametrize(
    ("endpoint_id", "model", "reason"),
    [
        ("", "", "auto_chat_unconfigured"),
        ("configured-endpoint", "", "auto_chat_unconfigured"),
        ("missing-endpoint", "auto-model", "auto_chat_unavailable"),
    ],
)
def test_incomplete_or_missing_auto_target_degrades_to_manual(
    router_store,
    endpoint_id,
    model,
    reason,
):
    settings, _add_endpoint = router_store
    settings["auto_chat_endpoint_id"] = endpoint_id
    settings["auto_chat_model"] = model
    session = _session(auto_route=True)
    before = _snapshot(session)

    route = chat_model_router.resolve_chat_route(session, owner="alice")

    assert route.auto is True
    assert route.lane == "chat"
    assert route.reason == reason
    assert route.target.endpoint_id is None
    assert route.target.endpoint_url == session.endpoint_url
    assert route.target.model == session.model
    assert route.manual_fallback is None
    assert _snapshot(session) == before


@pytest.mark.parametrize(
    ("endpoint_owner", "enabled"),
    [("bob", True), ("alice", False)],
)
def test_foreign_or_disabled_endpoint_degrades_to_manual(
    router_store,
    endpoint_owner,
    enabled,
):
    settings, add_endpoint = router_store
    add_endpoint("unsafe-endpoint", owner=endpoint_owner, enabled=enabled)
    settings["auto_chat_endpoint_id"] = "unsafe-endpoint"
    settings["auto_chat_model"] = "auto-model"
    session = _session(auto_route=True)
    before = _snapshot(session)

    route = chat_model_router.resolve_chat_route(session, owner="alice")

    assert route.reason == "auto_chat_unavailable"
    assert route.target.endpoint_id is None
    assert route.target.endpoint_url == session.endpoint_url
    assert route.target.model == session.model
    assert route.manual_fallback is None
    assert _snapshot(session) == before


def test_ownerless_resolution_cannot_select_an_owned_endpoint(router_store):
    settings, add_endpoint = router_store
    add_endpoint("owned-endpoint", owner="alice")
    settings["auto_chat_endpoint_id"] = "owned-endpoint"
    settings["auto_chat_model"] = "auto-model"
    session = _session(auto_route=True)

    route = chat_model_router.resolve_chat_route(session, owner=None)

    assert route.reason == "auto_chat_unavailable"
    assert route.target.endpoint_id is None


@pytest.mark.parametrize(
    ("lane", "agent_mode"),
    [("chat", False), ("agent", True)],
)
@pytest.mark.parametrize("endpoint_owner", [None, "admin"])
def test_global_auto_target_does_not_configure_an_explicit_owner(
    router_store,
    monkeypatch,
    lane,
    agent_mode,
    endpoint_owner,
):
    from routes import prefs_routes

    _settings, add_endpoint = router_store
    add_endpoint("global-endpoint", owner=endpoint_owner)
    monkeypatch.setattr(
        settings_module,
        "load_settings",
        lambda: {
            f"auto_{lane}_endpoint_id": "global-endpoint",
            f"auto_{lane}_model": "global-model",
        },
    )
    monkeypatch.setattr(prefs_routes, "_load_for_user", lambda _owner: {})
    monkeypatch.setattr(
        chat_model_router,
        "get_user_setting",
        settings_module.get_user_setting,
    )
    session = _session(auto_route=True)

    route = chat_model_router.resolve_chat_route(
        session,
        owner="alice",
        agent_mode=agent_mode,
    )

    assert route.reason == f"auto_{lane}_unconfigured"
    assert route.target.endpoint_url == session.endpoint_url
    assert route.target.model == session.model


@pytest.mark.parametrize(
    ("lane", "agent_mode"),
    [("chat", False), ("agent", True)],
)
@pytest.mark.parametrize("own_key", ["endpoint_id", "model"])
def test_partial_user_auto_target_never_mixes_with_global_settings(
    router_store,
    monkeypatch,
    lane,
    agent_mode,
    own_key,
):
    from routes import prefs_routes

    _settings, add_endpoint = router_store
    add_endpoint("shared-endpoint", owner=None)
    global_values = {
        f"auto_{lane}_endpoint_id": "shared-endpoint",
        f"auto_{lane}_model": "global-model",
    }
    own_values = {
        f"auto_{lane}_{own_key}": (
            "shared-endpoint" if own_key == "endpoint_id" else "alice-model"
        )
    }
    monkeypatch.setattr(settings_module, "load_settings", lambda: global_values)
    monkeypatch.setattr(
        prefs_routes,
        "_load_for_user",
        lambda owner: own_values if owner == "alice" else {},
    )
    monkeypatch.setattr(
        chat_model_router,
        "get_user_setting",
        settings_module.get_user_setting,
    )

    route = chat_model_router.resolve_chat_route(
        _session(auto_route=True),
        owner="alice",
        agent_mode=agent_mode,
    )

    assert route.reason == f"auto_{lane}_unconfigured"
    assert route.target.endpoint_id is None


@pytest.mark.parametrize(
    ("lane", "agent_mode"),
    [("chat", False), ("agent", True)],
)
@pytest.mark.parametrize("endpoint_owner", [None, "alice"])
def test_complete_explicit_user_auto_target_can_select_shared_or_owned_endpoint(
    router_store,
    monkeypatch,
    lane,
    agent_mode,
    endpoint_owner,
):
    from routes import prefs_routes

    _settings, add_endpoint = router_store
    add_endpoint("selected-endpoint", owner=endpoint_owner)
    own_values = {
        f"auto_{lane}_endpoint_id": "selected-endpoint",
        f"auto_{lane}_model": "selected-model",
    }
    monkeypatch.setattr(settings_module, "load_settings", lambda: {})
    monkeypatch.setattr(
        prefs_routes,
        "_load_for_user",
        lambda owner: own_values if owner == "alice" else {},
    )
    monkeypatch.setattr(
        chat_model_router,
        "get_user_setting",
        settings_module.get_user_setting,
    )

    route = chat_model_router.resolve_chat_route(
        _session(auto_route=True),
        owner="alice",
        agent_mode=agent_mode,
    )

    assert route.reason == f"auto_{lane}"
    assert route.target.endpoint_id == "selected-endpoint"
    assert route.target.model == "selected-model"


def test_ownerless_auto_route_can_use_global_shared_target(router_store, monkeypatch):
    _settings, add_endpoint = router_store
    add_endpoint("global-shared", owner=None)
    monkeypatch.setattr(
        settings_module,
        "load_settings",
        lambda: {
            "auto_chat_endpoint_id": "global-shared",
            "auto_chat_model": "global-model",
        },
    )
    monkeypatch.setattr(
        chat_model_router,
        "get_user_setting",
        settings_module.get_user_setting,
    )

    route = chat_model_router.resolve_chat_route(
        _session(auto_route=True),
        owner=None,
    )

    assert route.auto is True
    assert route.lane == "chat"
    assert route.reason == "auto_chat"
    assert route.target.endpoint_id == "global-shared"
    assert route.target.model == "global-model"


def test_auto_settings_are_read_for_the_requested_owner(router_store, monkeypatch):
    _settings, add_endpoint = router_store
    add_endpoint("alice-endpoint", owner="alice")
    calls = []

    def read_setting(key, owner="", default=None, *, inherit_global=True):
        calls.append((key, owner, inherit_global))
        return {
            "auto_chat_endpoint_id": "alice-endpoint",
            "auto_chat_model": "alice-model",
        }.get(key, default)

    monkeypatch.setattr(chat_model_router, "get_user_setting", read_setting)
    route = chat_model_router.resolve_chat_route(
        _session(auto_route=True),
        owner="alice",
    )

    assert route.target.endpoint_id == "alice-endpoint"
    assert calls == [
        ("auto_chat_endpoint_id", "alice", False),
        ("auto_chat_model", "alice", False),
    ]


@pytest.mark.parametrize("failure", ["settings", "database"])
def test_unexpected_local_resolution_failure_degrades_to_manual(
    router_store,
    monkeypatch,
    failure,
):
    settings, _add_endpoint = router_store
    settings["auto_chat_endpoint_id"] = "configured-endpoint"
    settings["auto_chat_model"] = "configured-model"
    if failure == "settings":
        monkeypatch.setattr(
            chat_model_router,
            "get_user_setting",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("settings failed")),
        )
    else:
        monkeypatch.setattr(
            chat_model_router,
            "SessionLocal",
            lambda: (_ for _ in ()).throw(RuntimeError("database failed")),
        )

    session = _session(auto_route=True)
    before = _snapshot(session)
    route = chat_model_router.resolve_chat_route(session, owner="alice")

    assert route.reason == "auto_chat_unavailable"
    assert route.target.endpoint_url == session.endpoint_url
    assert route.target.model == session.model
    assert _snapshot(session) == before


def test_identical_auto_and_manual_targets_do_not_duplicate_fallback(router_store):
    settings, add_endpoint = router_store
    add_endpoint(
        "same-endpoint",
        owner="alice",
        base_url="http://same-endpoint/v1",
    )
    settings["auto_chat_endpoint_id"] = "same-endpoint"
    settings["auto_chat_model"] = "same-model"
    session = _session(
        auto_route=True,
        url="http://same-endpoint/v1/chat/completions",
        model="same-model",
    )

    route = chat_model_router.resolve_chat_route(session, owner="alice")

    assert route.reason == "auto_chat"
    assert route.target.endpoint_id == "same-endpoint"
    assert route.manual_fallback is None


def test_route_resolution_has_no_homelab_or_message_specific_branch(router_store):
    settings, add_endpoint = router_store
    add_endpoint("agent-endpoint", owner="alice")
    settings["auto_agent_endpoint_id"] = "agent-endpoint"
    settings["auto_agent_model"] = "agent-model"
    session = _session(auto_route=True)

    ordinary = chat_model_router.resolve_chat_route(
        session,
        owner="alice",
        agent_mode=True,
    )
    homelab_wording = chat_model_router.resolve_chat_route(
        session,
        owner="alice",
        agent_mode=True,
    )

    assert ordinary == homelab_wording
    assert ordinary.lane == "agent"
    assert ordinary.reason == "auto_agent"
