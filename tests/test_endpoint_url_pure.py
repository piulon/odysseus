import subprocess

import pytest

from src import endpoint_resolver


def test_build_chat_url_default_uses_legacy_host_resolution(monkeypatch):
    calls = []
    monkeypatch.setattr(
        endpoint_resolver,
        "resolve_url",
        lambda url: calls.append(url) or url,
    )

    assert endpoint_resolver.build_chat_url("https://example.test/v1") == (
        "https://example.test/v1/chat/completions"
    )
    assert calls == ["https://example.test/v1"]


def test_build_chat_url_pure_skips_resolve_url(monkeypatch):
    monkeypatch.setattr(
        endpoint_resolver,
        "resolve_url",
        lambda _url: (_ for _ in ()).throw(AssertionError("resolve_url called")),
    )

    assert endpoint_resolver.build_chat_url(
        "https://example.test/v1",
        resolve_host=False,
    ) == "https://example.test/v1/chat/completions"


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("https://api.anthropic.com/v1", "https://api.anthropic.com/v1/messages"),
        ("https://api.openai.com", "https://api.openai.com/v1/chat/completions"),
        ("https://ollama.com/api", "https://ollama.com/api/chat"),
        (
            "https://chatgpt.com/backend-api/codex",
            "https://chatgpt.com/backend-api/codex/responses",
        ),
    ],
)
def test_build_chat_url_pure_preserves_provider_path_rules(monkeypatch, base, expected):
    monkeypatch.setattr(endpoint_resolver, "resolve_url", lambda url: url)

    assert endpoint_resolver.build_chat_url(base) == expected
    assert endpoint_resolver.build_chat_url(base, resolve_host=False) == expected


def test_build_chat_url_pure_requires_no_socket_or_tailscale(monkeypatch):
    monkeypatch.setattr(
        endpoint_resolver.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("DNS called")),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("subprocess called")),
    )
    monkeypatch.setattr(
        endpoint_resolver,
        "resolve_url",
        lambda _url: (_ for _ in ()).throw(AssertionError("resolve_url called")),
    )

    assert endpoint_resolver.build_chat_url(
        "http://private-host:8000/v1",
        resolve_host=False,
    ) == "http://private-host:8000/v1/chat/completions"
