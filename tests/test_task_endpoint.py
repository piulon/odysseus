import pytest

from src import llm_core, task_endpoint
from src.llm_core import LLMFallbackResult
from src.task_endpoint import task_llm_call_async_result


@pytest.mark.asyncio
async def test_task_llm_result_preserves_primary_identity_owner_gate_and_kwargs(monkeypatch):
    candidates = [("https://primary.example/v1", "primary-model", {"X-Test": "secret"})]
    calls = {"resolve": 0, "gate": 0, "fallback": 0}

    def fake_resolve(**kwargs):
        calls["resolve"] += 1
        assert kwargs == {
            "fallback_url": "https://caller.example/v1",
            "fallback_model": "caller-model",
            "fallback_headers": {"Authorization": "hidden"},
            "owner": "alice",
        }
        return candidates

    async def fake_gate(label):
        calls["gate"] += 1
        assert label == "background task LLM"

    async def fake_fallback(actual_candidates, messages, **kwargs):
        calls["fallback"] += 1
        assert actual_candidates == candidates
        assert messages == [{"role": "user", "content": "hello"}]
        assert kwargs == {"timeout": 37, "workload": "background"}
        return LLMFallbackResult(
            response="primary response",
            model="primary-model",
            endpoint_url="https://primary.example/v1",
            candidate_index=0,
        )

    monkeypatch.setattr(task_endpoint, "resolve_task_candidates", fake_resolve)
    monkeypatch.setattr(task_endpoint, "wait_for_interactive_quiet", fake_gate)
    monkeypatch.setattr(task_endpoint, "llm_call_async_with_fallback_result", fake_fallback)

    result = await task_llm_call_async_result(
        [{"role": "user", "content": "hello"}],
        fallback_url="https://caller.example/v1",
        fallback_model="caller-model",
        fallback_headers={"Authorization": "hidden"},
        owner="alice",
        timeout=37,
    )

    assert result == LLMFallbackResult(
        response="primary response",
        model="primary-model",
        endpoint_url="https://primary.example/v1",
        candidate_index=0,
    )
    assert calls == {"resolve": 1, "gate": 1, "fallback": 1}


@pytest.mark.asyncio
async def test_task_llm_result_identifies_successful_fallback(monkeypatch):
    candidates = [
        ("https://primary.example/v1", "primary-model", {}),
        ("https://fallback.example/v1", "fallback-model", {}),
    ]
    attempted = []

    async def fake_llm_call(url, model, messages, **kwargs):
        attempted.append((url, model))
        if model == "primary-model":
            raise RuntimeError("primary unavailable")
        return "fallback response"

    monkeypatch.setattr(task_endpoint, "resolve_task_candidates", lambda **kwargs: candidates)
    monkeypatch.setattr(task_endpoint, "wait_for_interactive_quiet", lambda label: _async_none())
    monkeypatch.setattr(llm_core, "llm_call_async", fake_llm_call)

    result = await task_llm_call_async_result([{"role": "user", "content": "hello"}])

    assert attempted == [
        ("https://primary.example/v1", "primary-model"),
        ("https://fallback.example/v1", "fallback-model"),
    ]
    assert result == LLMFallbackResult(
        response="fallback response",
        model="fallback-model",
        endpoint_url="https://fallback.example/v1",
        candidate_index=1,
    )


async def _async_none():
    return None


@pytest.mark.asyncio
async def test_task_llm_result_preserves_explicit_workload(monkeypatch):
    captured = {}

    async def fake_fallback(candidates, messages, **kwargs):
        captured.update(kwargs)
        return LLMFallbackResult("response", "model", "https://example.test/v1", 0)

    monkeypatch.setattr(
        task_endpoint,
        "resolve_task_candidates",
        lambda **kwargs: [("https://example.test/v1", "model", {})],
    )
    monkeypatch.setattr(task_endpoint, "wait_for_interactive_quiet", lambda label: _async_none())
    monkeypatch.setattr(task_endpoint, "llm_call_async_with_fallback_result", fake_fallback)

    await task_llm_call_async_result([], workload="custom", timeout=19)

    assert captured == {"workload": "custom", "timeout": 19}


@pytest.mark.asyncio
async def test_task_llm_result_keeps_no_candidates_error_and_skips_gate(monkeypatch):
    gate_calls = 0

    async def fake_gate(label):
        nonlocal gate_calls
        gate_calls += 1

    monkeypatch.setattr(task_endpoint, "resolve_task_candidates", lambda **kwargs: [])
    monkeypatch.setattr(task_endpoint, "wait_for_interactive_quiet", fake_gate)

    with pytest.raises(RuntimeError, match="^No LLM endpoint available for background task$"):
        await task_llm_call_async_result([])

    assert gate_calls == 0


@pytest.mark.asyncio
async def test_task_llm_legacy_wrapper_returns_str_without_double_execution(monkeypatch):
    calls = {"resolve": 0, "gate": 0, "fallback": 0}

    def fake_resolve(**kwargs):
        calls["resolve"] += 1
        return [("https://example.test/v1", "model", {})]

    async def fake_gate(label):
        calls["gate"] += 1

    async def fake_fallback(candidates, messages, **kwargs):
        calls["fallback"] += 1
        return LLMFallbackResult("legacy response", "model", "https://example.test/v1", 0)

    monkeypatch.setattr(task_endpoint, "resolve_task_candidates", fake_resolve)
    monkeypatch.setattr(task_endpoint, "wait_for_interactive_quiet", fake_gate)
    monkeypatch.setattr(task_endpoint, "llm_call_async_with_fallback_result", fake_fallback)

    response = await task_endpoint.task_llm_call_async(
        [{"role": "user", "content": "hello"}], owner="alice", timeout=23
    )

    assert response == "legacy response"
    assert isinstance(response, str)
    assert not isinstance(response, LLMFallbackResult)
    assert calls == {"resolve": 1, "gate": 1, "fallback": 1}
