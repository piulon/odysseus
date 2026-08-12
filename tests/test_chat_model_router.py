from types import SimpleNamespace
from unittest.mock import patch

from src.chat_model_router import resolve_chat_route


def session(
    *,
    auto=False,
    url="http://session/v1/chat/completions",
    model="session-model",
):
    return SimpleNamespace(
        endpoint_url=url,
        model=model,
        headers={"X-Session": "1"},
        auto_route=auto,
    )


def test_manual_route_preserves_session_model():
    sess = session(auto=False)

    with (
        patch(
            "src.chat_model_router._default_fallbacks",
            return_value=[
                (
                    "http://fallback/v1/chat/completions",
                    "fallback-model",
                    {},
                ),
            ],
        ),
        patch(
            "src.chat_model_router._configured_target",
            side_effect=AssertionError(
                "manual mode must not resolve auto target"
            ),
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=True,
            message="do something",
        )

    assert route.auto is False
    assert route.lane == "manual"
    assert route.reason == "manual"
    assert route.endpoint_url == sess.endpoint_url
    assert route.model == sess.model
    assert route.headers == sess.headers
    assert len(route.fallbacks) == 1


def test_auto_chat_uses_configured_chat_target():
    sess = session(auto=True)

    chat_target = (
        "http://msi/v1/chat/completions",
        "qwen-chat",
        {"X-Chat": "1"},
    )

    with (
        patch(
            "src.chat_model_router._configured_target",
            return_value=chat_target,
        ) as configured,
        patch(
            "src.chat_model_router._default_fallbacks",
            side_effect=AssertionError(
                "Auto chat must not read global fallbacks"
            ),
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=False,
            message="hello",
        )

    configured.assert_called_once_with(
        "auto_chat",
        "pau",
    )

    assert route.auto is True
    assert route.lane == "chat"
    assert route.reason == "auto_chat"
    assert route.model == "qwen-chat"
    assert route.endpoint_url == chat_target[0]
    assert route.fallbacks == ()


def test_auto_agent_uses_configured_agent_target():
    sess = session(auto=True)

    agent_target = (
        "http://tower/v1/chat/completions",
        "qwen-agent",
        {},
    )

    def configured_target(prefix, owner=None):
        if prefix == "auto_agent":
            return agent_target
        if prefix == "auto_chat":
            return None
        raise AssertionError(f"unexpected prefix: {prefix}")

    with (
        patch(
            "src.chat_model_router._configured_target",
            side_effect=configured_target,
        ) as configured,
        patch(
            "src.chat_model_router._default_fallbacks",
            side_effect=AssertionError(
                "Auto agent must not read global fallbacks"
            ),
        ),
        patch(
            "src.chat_model_router._direct_homelab_fastpath",
            return_value=False,
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=True,
            message="search the web",
        )

    assert [call.args for call in configured.call_args_list] == [
        ("auto_agent", "pau"),
        ("auto_chat", "pau"),
    ]

    assert route.auto is True
    assert route.lane == "agent"
    assert route.reason == "auto_agent"
    assert route.model == "qwen-agent"
    assert route.endpoint_url == agent_target[0]
    assert route.fallbacks == ()


def test_direct_homelab_fastpath_uses_chat_lane():
    sess = session(auto=True)

    chat_target = (
        "http://msi/v1/chat/completions",
        "qwen-light",
        {},
    )

    with (
        patch(
            "src.chat_model_router._configured_target",
            return_value=chat_target,
        ) as configured,
        patch(
            "src.chat_model_router._default_fallbacks",
            return_value=[],
        ),
        patch(
            "src.chat_model_router._direct_homelab_fastpath",
            return_value=True,
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=True,
            message="Fes un diagnòstic del meu homelab.",
        )

    configured.assert_called_once_with(
        "auto_chat",
        "pau",
    )

    assert route.lane == "chat"
    assert route.reason == "auto_homelab_fastpath"
    assert route.model == "qwen-light"


def test_unconfigured_auto_target_falls_back_to_session():
    sess = session(auto=True)

    with (
        patch(
            "src.chat_model_router._configured_target",
            return_value=None,
        ),
        patch(
            "src.chat_model_router._default_fallbacks",
            return_value=[],
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=False,
        )

    assert route.auto is True
    assert route.reason == "auto_chat_unconfigured"
    assert route.endpoint_url == sess.endpoint_url
    assert route.model == sess.model


def test_fallbacks_are_deduplicated():
    sess = session(auto=False)

    duplicate_session = (
        sess.endpoint_url + "/",
        sess.model,
        {},
    )

    other = (
        "http://other/v1/chat/completions",
        "other",
        {},
    )

    with patch(
        "src.chat_model_router._default_fallbacks",
        return_value=[
            duplicate_session,
            other,
            other,
        ],
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=False,
        )

    assert route.auto is False
    assert route.lane == "manual"

    assert [
        (url.rstrip("/"), model)
        for url, model, _headers in route.fallbacks
    ] == [
        (
            "http://other/v1/chat/completions",
            "other",
        ),
    ]


def test_auto_route_never_mutates_persistent_session():
    sess = session(auto=True)

    original = (
        sess.endpoint_url,
        sess.model,
        dict(sess.headers),
        sess.auto_route,
    )

    target = (
        "http://tower/v1/chat/completions",
        "routed-model",
        {"X-Routed": "1"},
    )

    with (
        patch(
            "src.chat_model_router._configured_target",
            return_value=target,
        ),
        patch(
            "src.chat_model_router._default_fallbacks",
            return_value=[],
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=False,
            message="hello",
        )

    assert route.model == "routed-model"

    assert (
        sess.endpoint_url,
        sess.model,
        sess.headers,
        sess.auto_route,
    ) == original




def test_auto_route_can_be_bypassed_for_capability_sensitive_request():
    sess = session(auto=True)

    original = (
        sess.endpoint_url,
        sess.model,
        dict(sess.headers),
        sess.auto_route,
    )

    with (
        patch(
            "src.chat_model_router._configured_target",
            side_effect=AssertionError(
                "capability bypass must not resolve an Auto target"
            ),
        ),
        patch(
            "src.chat_model_router._default_fallbacks",
            side_effect=AssertionError(
                "Auto capability bypass must not use global fallbacks"
            ),
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=False,
            message="describe this image",
            allow_auto=False,
        )

    assert route.auto is False
    assert route.lane == "manual"
    assert route.reason == "auto_capability_bypass"
    assert route.endpoint_url == sess.endpoint_url
    assert route.model == sess.model
    assert route.headers == sess.headers
    assert route.fallbacks == ()

    assert (
        sess.endpoint_url,
        sess.model,
        sess.headers,
        sess.auto_route,
    ) == original
