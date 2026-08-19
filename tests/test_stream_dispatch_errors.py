import asyncio

import pytest

from src import llm_core
from src.llm_core import ChatDispatchError


class _Response:
    def __init__(self, status_code=200, lines=None, raw=b"provider payload"):
        self.status_code = status_code
        self._lines = list(lines or [])
        self._raw = raw

    async def aread(self):
        return self._raw

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _StreamContext:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def __aenter__(self):
        if self.error:
            raise self.error
        return self.response

    async def __aexit__(self, *args):
        return False


class _Client:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def stream(self, *args, **kwargs):
        return _StreamContext(self.response, self.error)


async def _collect(stream):
    return [chunk async for chunk in stream]


def _prepare(monkeypatch, *, response=None, error=None):
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: _Client(response, error))
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda url: False)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_core, "_clear_host_dead", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_core, "_mark_host_dead", lambda url: False)


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503])
def test_typed_stream_upstream_status_is_structured(monkeypatch, status):
    _prepare(monkeypatch, response=_Response(status_code=status))

    with pytest.raises(ChatDispatchError) as exc:
        asyncio.run(_collect(llm_core.stream_llm(
            "https://provider.invalid/v1/chat/completions",
            "model",
            [{"role": "user", "content": "hello"}],
            typed_errors=True,
        )))

    assert exc.value.kind == "upstream_status"
    assert exc.value.status_code == status


@pytest.mark.parametrize("error,kind", [
    (llm_core.httpx.ConnectError("connect"), "network"),
    (llm_core.httpx.ReadTimeout("read"), "timeout"),
])
def test_typed_stream_network_errors_are_structured(monkeypatch, error, kind):
    _prepare(monkeypatch, error=error)

    with pytest.raises(ChatDispatchError) as exc:
        asyncio.run(_collect(llm_core.stream_llm(
            "https://provider.invalid/v1/chat/completions",
            "model",
            [{"role": "user", "content": "hello"}],
            typed_errors=True,
        )))

    assert exc.value.kind == kind


def test_stream_default_preserves_legacy_sse_error(monkeypatch):
    _prepare(monkeypatch, response=_Response(status_code=503))

    chunks = asyncio.run(_collect(llm_core.stream_llm(
        "https://provider.invalid/v1/chat/completions",
        "model",
        [{"role": "user", "content": "hello"}],
    )))

    assert len(chunks) == 1
    assert chunks[0].startswith("event: error")
    assert '"status": 503' in chunks[0]


def test_typed_stream_malformed_provider_chunk_is_terminal(monkeypatch):
    _prepare(monkeypatch, response=_Response(lines=['data: {not-json']))

    with pytest.raises(ChatDispatchError) as exc:
        asyncio.run(_collect(llm_core.stream_llm(
            "https://provider.invalid/v1/chat/completions",
            "model",
            [{"role": "user", "content": "hello"}],
            typed_errors=True,
        )))

    assert exc.value.kind == "invalid_response"
    assert exc.value.status_code == 502


def test_legacy_stream_still_ignores_malformed_provider_chunk(monkeypatch):
    _prepare(monkeypatch, response=_Response(lines=['data: {not-json']))

    chunks = asyncio.run(_collect(llm_core.stream_llm(
        "https://provider.invalid/v1/chat/completions",
        "model",
        [{"role": "user", "content": "hello"}],
    )))

    assert chunks == ["data: [DONE]\n\n"]


def test_typed_stream_cancellation_is_not_wrapped(monkeypatch):
    _prepare(monkeypatch, error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_collect(llm_core.stream_llm(
            "https://provider.invalid/v1/chat/completions",
            "model",
            [{"role": "user", "content": "hello"}],
            typed_errors=True,
        )))
