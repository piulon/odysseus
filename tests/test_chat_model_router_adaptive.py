import time
from types import SimpleNamespace
from unittest.mock import patch

from src.adaptive_routing import RoutingCandidate
from src.adaptive_routing_snapshot import (
    clear_adaptive_routing_snapshot,
    publish_adaptive_routing_snapshot,
)
from src.chat_model_router import resolve_chat_route
from src.settings import DEFAULT_SETTINGS, _PER_USER_KEYS


def session(
    *,
    auto=True,
    url="http://session/v1/chat/completions",
    model="session-model",
):
    return SimpleNamespace(
        endpoint_url=url,
        model=model,
        headers={"X-Session": "1"},
        auto_route=auto,
    )


def candidate(
    endpoint_id,
    model,
    *,
    node,
    capabilities=(),
    reachable=True,
):
    return RoutingCandidate(
        endpoint_id=endpoint_id,
        endpoint_url=f"http://{endpoint_id}/api/chat",
        model=model,
        node=node,
        scope="local",
        capabilities=tuple(capabilities),
        reachable=reachable,
    )


def setting_reader(
    *,
    mode="adaptive",
    chat=("chat-ep", "chat-model"),
    agent=("agent-ep", "agent-model"),
):
    values = {
        "auto_routing_mode": mode,
        "auto_chat_endpoint_id": chat[0],
        "auto_chat_model": chat[1],
        "auto_agent_endpoint_id": agent[0],
        "auto_agent_model": agent[1],
    }

    def read(key, owner=None):
        return values.get(key, "")

    return read


def adaptive_dispatch(routing_candidate, owner=None):
    return (
        routing_candidate.endpoint_url,
        routing_candidate.model,
        {
            "X-Endpoint": routing_candidate.endpoint_id,
            "X-Owner": owner or "",
        },
    )


def setup_function():
    clear_adaptive_routing_snapshot()


def teardown_function():
    clear_adaptive_routing_snapshot()


def test_auto_routing_mode_defaults_to_legacy_and_is_owner_scoped():
    assert DEFAULT_SETTINGS["auto_routing_mode"] == "legacy"
    assert "auto_routing_mode" in _PER_USER_KEYS


def test_legacy_mode_preserves_existing_auto_chat_path():
    sess = session()
    legacy = (
        "http://legacy/api/chat",
        "legacy-model",
        {"X-Legacy": "1"},
    )

    with (
        patch(
            "src.chat_model_router._read_setting",
            side_effect=setting_reader(mode="legacy"),
        ),
        patch(
            "src.chat_model_router._configured_target",
            return_value=legacy,
        ) as configured,
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=False,
            message="hello",
        )

    configured.assert_called_once_with("auto_chat", "pau")
    assert route.reason == "auto_chat"
    assert route.endpoint_url == legacy[0]
    assert route.model == legacy[1]
    assert route.headers == legacy[2]


def test_adaptive_chat_uses_fresh_owner_snapshot_and_caps_fallbacks():
    sess = session()
    publish_adaptive_routing_snapshot(
        "pau",
        [
            candidate("other-a", "other-a", node="a"),
            candidate("chat-ep", "chat-model", node="msi", capabilities=("reasoning",)),
            candidate("other-b", "other-b", node="b"),
            candidate("other-c", "other-c", node="c"),
        ],
        generated_at=time.time(),
    )

    with (
        patch(
            "src.chat_model_router._read_setting",
            side_effect=setting_reader(),
        ),
        patch(
            "src.chat_model_router._configured_target",
            side_effect=AssertionError(
                "fresh adaptive route must not invoke legacy auto target"
            ),
        ),
        patch(
            "src.chat_model_router._resolve_adaptive_candidate",
            side_effect=adaptive_dispatch,
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=False,
            message="hello",
        )

    assert route.auto is True
    assert route.lane == "chat"
    assert route.reason.startswith("adaptive_chat:")
    assert route.endpoint_url == "http://chat-ep/api/chat"
    assert route.model == "chat-model"
    assert route.headers["X-Endpoint"] == "chat-ep"
    assert route.headers["X-Owner"] == "pau"
    assert len(route.fallbacks) == 2


def test_adaptive_agent_hard_tool_requirement_beats_target_preference():
    sess = session()
    publish_adaptive_routing_snapshot(
        "pau",
        [
            candidate(
                "agent-ep",
                "agent-model",
                node="tower",
                capabilities=("reasoning",),
            ),
            candidate(
                "chat-ep",
                "chat-model",
                node="msi",
                capabilities=("tool_call", "reasoning"),
            ),
        ],
        generated_at=time.time(),
    )

    with (
        patch(
            "src.chat_model_router._read_setting",
            side_effect=setting_reader(),
        ),
        patch(
            "src.chat_model_router._configured_target",
            side_effect=AssertionError(
                "viable adaptive agent route must not invoke legacy"
            ),
        ),
        patch(
            "src.chat_model_router._direct_homelab_fastpath",
            return_value=False,
        ),
        patch(
            "src.chat_model_router._resolve_adaptive_candidate",
            side_effect=adaptive_dispatch,
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=True,
            message="use a tool",
        )

    assert route.lane == "agent"
    assert route.reason.startswith("adaptive_agent:")
    assert route.model == "chat-model"
    assert route.endpoint_url == "http://chat-ep/api/chat"


def test_missing_adaptive_snapshot_degrades_to_legacy():
    sess = session()
    legacy = ("http://legacy/api/chat", "legacy-model", {})

    with (
        patch(
            "src.chat_model_router._read_setting",
            side_effect=setting_reader(),
        ),
        patch(
            "src.chat_model_router._configured_target",
            return_value=legacy,
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=False,
        )

    assert route.reason == "auto_chat"
    assert route.model == "legacy-model"


def test_stale_adaptive_snapshot_degrades_to_legacy():
    sess = session()
    publish_adaptive_routing_snapshot(
        "pau",
        [candidate("chat-ep", "chat-model", node="msi")],
        generated_at=time.time() - 1000,
    )
    legacy = ("http://legacy/api/chat", "legacy-model", {})

    with (
        patch(
            "src.chat_model_router._read_setting",
            side_effect=setting_reader(),
        ),
        patch(
            "src.chat_model_router._configured_target",
            return_value=legacy,
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=False,
        )

    assert route.reason == "auto_chat"
    assert route.model == "legacy-model"


def test_adaptive_candidate_model_mismatch_is_not_dispatched():
    sess = session()
    publish_adaptive_routing_snapshot(
        "pau",
        [
            candidate("chat-ep", "chat-model", node="a"),
            candidate("other-ep", "other-model", node="b"),
        ],
        generated_at=time.time(),
    )

    def resolve(routing_candidate, owner=None):
        if routing_candidate.endpoint_id == "chat-ep":
            return (
                routing_candidate.endpoint_url,
                "replacement-model",
                {},
            )
        return adaptive_dispatch(routing_candidate, owner)

    with (
        patch(
            "src.chat_model_router._read_setting",
            side_effect=setting_reader(),
        ),
        patch(
            "src.chat_model_router._configured_target",
            side_effect=AssertionError(
                "second adaptive candidate should be promoted"
            ),
        ),
        patch(
            "src.chat_model_router._resolve_adaptive_candidate",
            side_effect=resolve,
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=False,
        )

    assert route.reason.startswith("adaptive_chat:")
    assert route.model == "other-model"
    assert route.endpoint_url == "http://other-ep/api/chat"


def test_adaptive_mode_keeps_direct_homelab_fastpath_on_legacy_chat_target():
    sess = session()
    publish_adaptive_routing_snapshot(
        "pau",
        [candidate("agent-ep", "agent-model", node="tower", capabilities=("tool_call", "reasoning"))],
        generated_at=time.time(),
    )
    local = ("http://msi/api/chat", "local-fast", {})

    with (
        patch(
            "src.chat_model_router._read_setting",
            side_effect=setting_reader(),
        ),
        patch(
            "src.chat_model_router._direct_homelab_fastpath",
            return_value=True,
        ),
        patch(
            "src.chat_model_router._configured_target",
            return_value=local,
        ) as configured,
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=True,
            message="Estat del homelab",
        )

    configured.assert_called_once_with("auto_chat", "pau")
    assert route.lane == "chat"
    assert route.reason == "auto_homelab_fastpath"
    assert route.model == "local-fast"


def test_manual_and_capability_bypass_do_not_read_adaptive_mode():
    manual = session(auto=False)
    bypass = session(auto=True)

    with (
        patch(
            "src.chat_model_router._read_setting",
            side_effect=AssertionError(
                "manual/bypass must not read adaptive settings"
            ),
        ),
        patch(
            "src.chat_model_router._default_fallbacks",
            return_value=[],
        ),
    ):
        manual_route = resolve_chat_route(manual, owner="pau")
        bypass_route = resolve_chat_route(
            bypass,
            owner="pau",
            allow_auto=False,
        )

    assert manual_route.reason == "manual"
    assert bypass_route.reason == "auto_capability_bypass"
