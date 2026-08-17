import json

import httpx
import pytest
from fastapi import HTTPException

from src import llm_core
from src.llm_core import ChatDispatchError, is_recoverable_chat_dispatch_error


@pytest.mark.parametrize(
    ("kind", "status"),
    [
        ("timeout", 504),
        ("network", 502),
        ("upstream_status", 408),
        ("upstream_status", 429),
        ("upstream_status", 500),
        ("upstream_status", 502),
        ("upstream_status", 503),
    ],
)
def test_recoverable_dispatch_errors_are_typed(kind, status):
    exc = ChatDispatchError(status, "internal detail", kind=kind)

    assert isinstance(exc, HTTPException)
    assert is_recoverable_chat_dispatch_error(exc) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_upstream_terminal_statuses_do_not_fallback(status):
    exc = ChatDispatchError(status, "same text for every status", kind="upstream_status")

    assert is_recoverable_chat_dispatch_error(exc) is False


@pytest.mark.parametrize("kind", ["invalid_response", "internal"])
def test_invalid_or_internal_errors_are_terminal_even_with_502(kind):
    exc = ChatDispatchError(502, "network timeout words must not matter", kind=kind)

    assert is_recoverable_chat_dispatch_error(exc) is False


def test_json_parse_and_unknown_exceptions_are_terminal():
    assert is_recoverable_chat_dispatch_error(ValueError("invalid json")) is False
    assert is_recoverable_chat_dispatch_error(RuntimeError("timeout network 503")) is False
    assert is_recoverable_chat_dispatch_error(HTTPException(503, "untyped")) is False


class _Response:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self.is_success = 200 <= status < 300
        self.text = "provider detail"
        self._payload = payload

    def json(self):
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload


@pytest.mark.asyncio
async def test_llm_call_tags_upstream_502_separately_from_invalid_schema(monkeypatch):
    responses = [_Response(502), _Response(200, {})]

    async def post(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async", post)
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda url: False)

    with pytest.raises(ChatDispatchError) as upstream:
        await llm_core.llm_call_async(
            "https://provider.invalid/v1/chat/completions",
            "model",
            [{"role": "user", "content": "hello"}],
            max_retries=1,
        )
    assert upstream.value.kind == "upstream_status"
    assert is_recoverable_chat_dispatch_error(upstream.value) is True

    with pytest.raises(ChatDispatchError) as schema:
        await llm_core.llm_call_async(
            "https://provider.invalid/v1/chat/completions",
            "model",
            [{"role": "user", "content": "hello again"}],
            max_retries=1,
        )
    assert schema.value.kind == "invalid_response"
    assert is_recoverable_chat_dispatch_error(schema.value) is False


@pytest.mark.asyncio
async def test_llm_call_tags_read_timeout_but_leaves_json_parse_terminal(monkeypatch):
    outcomes = [
        httpx.ReadTimeout("slow"),
        _Response(200, json.JSONDecodeError("bad", "x", 0)),
    ]

    async def post(*args, **kwargs):
        value = outcomes.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async", post)
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda url: False)

    with pytest.raises(ChatDispatchError) as timeout:
        await llm_core.llm_call_async(
            "https://timeout.invalid/v1/chat/completions",
            "model",
            [{"role": "user", "content": "timeout"}],
            max_retries=1,
        )
    assert timeout.value.kind == "timeout"
    assert is_recoverable_chat_dispatch_error(timeout.value) is True

    with pytest.raises(json.JSONDecodeError) as invalid_json:
        await llm_core.llm_call_async(
            "https://json.invalid/v1/chat/completions",
            "model",
            [{"role": "user", "content": "json"}],
            max_retries=1,
        )
    assert is_recoverable_chat_dispatch_error(invalid_json.value) is False
