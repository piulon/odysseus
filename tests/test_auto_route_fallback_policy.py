from unittest.mock import patch

from src.chat_model_router import resolve_chat_route
from tests.test_chat_model_router import session


CHAT = (
    "http://msi/v1/chat/completions",
    "chat-model",
    {"X-Chat": "1"},
)

AGENT = (
    "http://tower/v1/chat/completions",
    "agent-model",
    {"X-Agent": "1"},
)

GLOBAL = (
    "https://external.example/v1/chat/completions",
    "external-model",
    {"Authorization": "secret"},
)


def test_manual_route_keeps_default_fallbacks():
    sess = session(auto=False)

    with patch(
        "src.chat_model_router._default_fallbacks",
        return_value=[GLOBAL],
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            message="hello",
        )

    assert route.auto is False
    assert route.fallbacks == (GLOBAL,)


def test_auto_chat_has_no_implicit_fallbacks():
    sess = session(auto=True)

    def configured(prefix, owner=None):
        assert prefix == "auto_chat"
        return CHAT

    with (
        patch(
            "src.chat_model_router._configured_target",
            side_effect=configured,
        ),
        patch(
            "src.chat_model_router._default_fallbacks",
            side_effect=AssertionError(
                "Auto must not read global chat fallbacks"
            ),
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=False,
            message="hello",
        )

    assert route.model == "chat-model"
    assert route.fallbacks == ()


def test_auto_agent_falls_back_only_to_auto_chat():
    sess = session(auto=True)

    def configured(prefix, owner=None):
        return {
            "auto_agent": AGENT,
            "auto_chat": CHAT,
        }.get(prefix)

    with (
        patch(
            "src.chat_model_router._configured_target",
            side_effect=configured,
        ),
        patch(
            "src.chat_model_router._default_fallbacks",
            side_effect=AssertionError(
                "Auto must not read global chat fallbacks"
            ),
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=True,
            message="use a tool",
        )

    assert route.model == "agent-model"
    assert route.fallbacks == (CHAT,)


def test_auto_agent_dedupes_identical_chat_fallback():
    sess = session(auto=True)

    with patch(
        "src.chat_model_router._configured_target",
        return_value=AGENT,
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            agent_mode=True,
            message="use a tool",
        )

    assert route.model == "agent-model"
    assert route.fallbacks == ()


def test_unconfigured_auto_uses_session_without_global_fallback():
    sess = session(auto=True)

    with (
        patch(
            "src.chat_model_router._configured_target",
            return_value=None,
        ),
        patch(
            "src.chat_model_router._default_fallbacks",
            side_effect=AssertionError(
                "Unconfigured Auto must not inherit global fallbacks"
            ),
        ),
    ):
        route = resolve_chat_route(
            sess,
            owner="pau",
            message="hello",
        )

    assert route.auto is True
    assert route.endpoint_url == sess.endpoint_url
    assert route.model == sess.model
    assert route.headers == sess.headers
    assert route.fallbacks == ()
