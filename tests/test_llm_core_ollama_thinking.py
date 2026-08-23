"""Tests for Ollama /v1 thinking-suppression helpers.

Covers:
- _is_ollama_openai_compat_url: URL classification (local host + /v1 path)
- reasoning_effort: none is injected for Ollama /v1 thinking models
- Ollama-native think:false does not leak into the /v1 payload
- reasoning controls are not injected for unrelated endpoints/models
"""
import asyncio
import json

from src import llm_core


# ---------------------------------------------------------------------------
# Fake HTTP client — captures the outgoing payload without network I/O
# ---------------------------------------------------------------------------

class _FakeResp:
    status_code = 200

    async def aiter_lines(self):
        # Yield a minimal done event so stream_llm exits cleanly
        yield json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})
        yield "data: [DONE]"

    async def aread(self):
        return b""


class _FakeStreamCtx:
    def __init__(self, captured):
        self._captured = captured

    async def __aenter__(self):
        return _FakeResp()

    async def __aexit__(self, *a):
        return False


class _FakeClient:
    """Minimal stand-in for httpx.AsyncClient that captures request payload."""

    def __init__(self):
        self.captured_payload = {}

    def stream(self, method, url, **kw):
        self.captured_payload = kw.get("json") or {}
        return _FakeStreamCtx(self.captured_payload)


def _capture_payload(monkeypatch, url, model):
    """Run stream_llm, intercept the HTTP payload, and return it."""
    client = _FakeClient()
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: client)
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda u: False)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "_clear_host_dead", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "get_context_length", lambda u, m: 32768)

    async def run():
        return [c async for c in llm_core.stream_llm(
            url, model, [{"role": "user", "content": "hi"}],
        )]

    asyncio.run(run())
    return client.captured_payload


# ---------------------------------------------------------------------------
# _is_ollama_openai_compat_url — pure function, no I/O
# ---------------------------------------------------------------------------

class TestIsOllamaOpenAICompatUrl:
    """Unit tests for the URL classifier that gates think-suppression."""

    # Positive cases — should be True
    def test_default_port_v1_root(self):
        assert llm_core._is_ollama_openai_compat_url("http://127.0.0.1:11434/v1")

    def test_default_port_chat_completions(self):
        assert llm_core._is_ollama_openai_compat_url("http://127.0.0.1:11434/v1/chat/completions")

    def test_localhost_default_port(self):
        assert llm_core._is_ollama_openai_compat_url("http://localhost:11434/v1")

    def test_localhost_default_port_with_path(self):
        assert llm_core._is_ollama_openai_compat_url("http://localhost:11434/v1/chat/completions")

    def test_loopback_ipv6(self):
        # IPv6 addresses in URLs require square brackets per RFC 3986
        assert llm_core._is_ollama_openai_compat_url("http://[::1]:11434/v1")

    def test_any_local_non_default_port(self):
        """Localhost on a non-default port (custom OLLAMA_HOST) must also match."""
        assert llm_core._is_ollama_openai_compat_url("http://127.0.0.1:11435/v1")

    def test_localhost_non_default_port(self):
        assert llm_core._is_ollama_openai_compat_url("http://localhost:8080/v1/chat/completions")

    def test_zero_dot_zero_host(self):
        assert llm_core._is_ollama_openai_compat_url("http://0.0.0.0:11434/v1")

    # Negative cases — should be False
    def test_openai_api_v1(self):
        """Real OpenAI endpoint must never match, even though path is /v1."""
        assert not llm_core._is_ollama_openai_compat_url("https://api.openai.com/v1")

    def test_openai_chat_completions(self):
        assert not llm_core._is_ollama_openai_compat_url("https://api.openai.com/v1/chat/completions")

    def test_ollama_native_api_path(self):
        """The native /api path is a different surface and must not match /v1."""
        assert not llm_core._is_ollama_openai_compat_url("http://localhost:11434/api")

    def test_ollama_native_api_chat(self):
        assert not llm_core._is_ollama_openai_compat_url("http://localhost:11434/api/chat")

    def test_remote_openrouter(self):
        assert not llm_core._is_ollama_openai_compat_url("https://openrouter.ai/api/v1")

    def test_empty_string(self):
        assert not llm_core._is_ollama_openai_compat_url("")

    def test_none_like_empty(self):
        assert not llm_core._is_ollama_openai_compat_url(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Payload injection — correct OpenAI-compatible reasoning control
# ---------------------------------------------------------------------------

class TestThinkSuppression:
    """Assert Ollama /v1 receives reasoning_effort:none, never think:false."""

    def test_reasoning_none_for_ollama_v1_thinking_model(self, monkeypatch):
        payload = _capture_payload(
            monkeypatch,
            "http://127.0.0.1:11434/v1/chat/completions",
            "qwen3:14b",
        )
        assert payload.get("reasoning_effort") == "none"
        assert "think" not in payload

    def test_no_reasoning_control_for_ollama_v1_non_thinking_model(
        self, monkeypatch
    ):
        payload = _capture_payload(
            monkeypatch,
            "http://127.0.0.1:11434/v1/chat/completions",
            "llama3.2:3b",
        )
        assert "reasoning_effort" not in payload
        assert "think" not in payload

    def test_no_ollama_reasoning_control_for_real_openai(
        self, monkeypatch
    ):
        payload = _capture_payload(
            monkeypatch,
            "https://api.openai.com/v1/chat/completions",
            "qwen3:14b",
        )
        assert "reasoning_effort" not in payload
        assert "think" not in payload

    def test_reasoning_none_for_non_default_port_thinking_model(
        self, monkeypatch
    ):
        payload = _capture_payload(
            monkeypatch,
            "http://127.0.0.1:11435/v1/chat/completions",
            "qwen3:14b",
        )
        assert payload.get("reasoning_effort") == "none"
        assert "think" not in payload


def test_shared_reasoning_policy_is_narrow():
    payload = {"think": False}
    llm_core._apply_ollama_openai_compat_reasoning_policy(
        payload,
        "http://127.0.0.1:11434/v1/chat/completions",
        "qwen3:14b",
    )
    assert payload == {"reasoning_effort": "none"}

    unrelated = {}
    llm_core._apply_ollama_openai_compat_reasoning_policy(
        unrelated,
        "https://api.openai.com/v1/chat/completions",
        "qwen3:14b",
    )
    assert unrelated == {}

    non_thinking = {}
    llm_core._apply_ollama_openai_compat_reasoning_policy(
        non_thinking,
        "http://127.0.0.1:11434/v1/chat/completions",
        "llama3.2:3b",
    )
    assert non_thinking == {}



# ---------------------------------------------------------------------------
# Sync / async / native-path regression coverage
# ---------------------------------------------------------------------------

class _NonStreamResp:
    status_code = 200
    is_success = True
    text = ""

    def json(self):
        return {
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }]
        }


def test_sync_path_uses_openai_compat_reasoning_control(monkeypatch):
    captured = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs.get("json") or {})
        return _NonStreamResp()

    monkeypatch.setattr(
        llm_core,
        "httpx_post_kimi_aware",
        fake_post,
    )
    monkeypatch.setattr(
        llm_core,
        "note_model_activity",
        lambda *a, **k: None,
    )

    result = llm_core.llm_call(
        "http://127.0.0.1:11434/v1/chat/completions",
        "qwen3:14b",
        [{"role": "user", "content": "sync payload regression"}],
        max_tokens=64,
    )

    assert result == "ok"
    assert captured.get("reasoning_effort") == "none"
    assert "think" not in captured


def test_async_path_uses_openai_compat_reasoning_control(monkeypatch):
    captured = {}

    async def fake_post(*args, **kwargs):
        captured.update(kwargs.get("json") or {})
        return _NonStreamResp()

    monkeypatch.setattr(
        llm_core,
        "_get_http_client",
        lambda: object(),
    )
    monkeypatch.setattr(
        llm_core,
        "httpx_post_kimi_aware_async",
        fake_post,
    )
    monkeypatch.setattr(
        llm_core,
        "_is_host_dead",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        llm_core,
        "note_model_activity",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        llm_core,
        "_clear_host_dead",
        lambda *a, **k: None,
    )

    async def run():
        return await llm_core.llm_call_async(
            "http://127.0.0.1:11434/v1/chat/completions",
            "qwen3:14b",
            [{"role": "user", "content": "async payload regression"}],
            max_tokens=64,
        )

    result = asyncio.run(run())

    assert result == "ok"
    assert captured.get("reasoning_effort") == "none"
    assert "think" not in captured


def test_native_ollama_tool_payload_keeps_native_think_control():
    payload = llm_core._build_ollama_payload(
        "qwen3:14b",
        [{"role": "user", "content": "use a tool"}],
        temperature=0,
        max_tokens=64,
        stream=False,
        tools=[{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }],
    )

    assert payload.get("think") is False
    assert "reasoning_effort" not in payload
