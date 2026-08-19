import asyncio
import logging

import httpx
import pytest

from src import llm_core
from src.llm_core import ChatDispatchError


_PRIVATE_URL = "https://secret-user:secret-pass@private-host.invalid/v1/chat/completions"
_FORBIDDEN = (
    "private-host.invalid",
    "secret-user",
    "secret-pass",
    "TOP-SECRET-TOKEN",
    "SUPER-SECRET",
    "Authorization",
    "raw provider body",
)


class _Response:
    status_code = 503
    is_success = False
    text = "raw provider body Bearer TOP-SECRET-TOKEN api_key=SUPER-SECRET"


@pytest.mark.asyncio
async def test_auto_nonstream_safe_logs_omit_target_and_upstream_detail(monkeypatch, caplog):
    async def post(*args, **kwargs):
        return _Response()

    monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async", post)
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda url: False)
    caplog.set_level(logging.DEBUG, logger=llm_core.__name__)

    with pytest.raises(ChatDispatchError) as exc:
        await llm_core.llm_call_async(
            _PRIVATE_URL,
            "safe-model",
            [{"role": "user", "content": "hello"}],
            headers={"Authorization": "Bearer TOP-SECRET-TOKEN"},
            max_retries=1,
            safe_logs=True,
        )

    assert exc.value.kind == "upstream_status"
    rendered = caplog.text
    for secret in _FORBIDDEN:
        assert secret not in rendered
    assert "status=503" in rendered


class _StreamContext:
    async def __aenter__(self):
        raise httpx.ConnectError("api_key=SUPER-SECRET raw provider body")

    async def __aexit__(self, *args):
        return False


class _Client:
    def stream(self, *args, **kwargs):
        return _StreamContext()


def test_auto_typed_stream_safe_logs_omit_target_and_exception(monkeypatch, caplog):
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: _Client())
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda url: False)
    monkeypatch.setattr(llm_core, "_mark_host_dead", lambda url: False)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *args, **kwargs: None)
    caplog.set_level(logging.DEBUG, logger=llm_core.__name__)

    async def collect():
        return [chunk async for chunk in llm_core.stream_llm(
            _PRIVATE_URL,
            "safe-model",
            [{"role": "user", "content": "hello"}],
            headers={"Authorization": "Bearer TOP-SECRET-TOKEN"},
            typed_errors=True,
        )]

    with pytest.raises(ChatDispatchError) as exc:
        asyncio.run(collect())

    assert exc.value.kind == "network"
    rendered = caplog.text
    for secret in _FORBIDDEN:
        assert secret not in rendered
    assert "kind=network" in rendered
