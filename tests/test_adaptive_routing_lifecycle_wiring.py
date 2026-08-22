import ast
from pathlib import Path
from types import SimpleNamespace


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _load_owner_provider():
    source = APP_PATH.read_text()
    tree = ast.parse(source)

    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_adaptive_routing_owner_provider"
    )

    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {}
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace["_adaptive_routing_owner_provider"]


def test_owner_provider_auth_disabled_is_ownerless():
    provider = _load_owner_provider()
    provider.__globals__.update(
        AUTH_ENABLED=False,
        auth_manager=SimpleNamespace(
            is_configured=True,
            list_users=lambda: [{"username": "alice"}],
        ),
    )

    assert provider() == (None,)


def test_owner_provider_auth_enabled_without_users_is_empty():
    provider = _load_owner_provider()
    provider.__globals__.update(
        AUTH_ENABLED=True,
        auth_manager=SimpleNamespace(
            is_configured=False,
            list_users=lambda: (_ for _ in ()).throw(
                AssertionError("list_users must not run before setup")
            ),
        ),
    )

    assert provider() == ()


def test_owner_provider_auth_enabled_returns_only_valid_usernames():
    provider = _load_owner_provider()
    provider.__globals__.update(
        AUTH_ENABLED=True,
        auth_manager=SimpleNamespace(
            is_configured=True,
            list_users=lambda: [
                {"username": " alice "},
                {"username": ""},
                {"username": None},
                {"username": "bob"},
                "malformed",
            ],
        ),
    )

    assert provider() == ("alice", "bob")


def test_lifecycle_wiring_starts_and_stops_worker():
    source = APP_PATH.read_text()

    assert "start_adaptive_routing_worker(_adaptive_routing_owner_provider)" in source
    assert "await stop_adaptive_routing_worker()" in source


def test_request_path_does_not_reference_worker():
    chat_routes = (
        APP_PATH.parent / "routes" / "chat_routes.py"
    ).read_text()

    assert "adaptive_routing_worker" not in chat_routes
    assert "refresh_owner_adaptive_snapshot" not in chat_routes
