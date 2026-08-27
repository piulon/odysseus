"""Regression: stream_agent_loop emits `rounds_exhausted` only when the round
cap is hit while still working, and NOT on a normal finish.

The decision is a `for/else` in the loop: the `else` runs only if no `break`
fired (break = done / budget / error). A refactor that adds a stray break or
return, or moves the done-break, could silently flip this. See PR #1999 / #1997.
"""

import asyncio
import json

import src.agent_loop as al


def _collect(gen):
    async def _run():
        return [c async for c in gen]
    return asyncio.run(_run())


def _types(chunks):
    out = []
    for c in chunks:
        if c.startswith("data: ") and not c.startswith("data: [DONE]"):
            try:
                out.append(json.loads(c[6:]))
            except Exception:
                pass
    return out


def _patch_common(monkeypatch):
    # Skip RAG/tool-index, MCP, and settings lookups; keep the real loop body,
    # _resolve_tool_blocks, and parse_tool_blocks.
    monkeypatch.setattr(al, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(al, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(al, "estimate_tokens", lambda *a, **k: 10, raising=False)

    async def _fake_exec(block, *a, **k):
        return ("bash", {"output": "ok", "exit_code": 0})
    monkeypatch.setattr(al, "execute_tool_block", _fake_exec, raising=False)


def _run_loop(monkeypatch, round_text, max_rounds=2):
    async def _fake_stream(_candidates, messages, **kwargs):
        yield f'data: {json.dumps({"delta": round_text})}\n\n'
        yield "data: [DONE]\n\n"
    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)

    gen = al.stream_agent_loop(
        "http://x/v1", "m",
        [{"role": "user", "content": "do a long multi-step task"}],
        max_rounds=max_rounds,
        relevant_tools={"bash"},
    )
    return _types(_collect(gen))


def test_emits_rounds_exhausted_when_cap_hit_mid_task(monkeypatch):
    _patch_common(monkeypatch)
    # Every round returns a tool block -> never "done" -> loop exhausts the cap.
    events = _run_loop(monkeypatch, "```bash\necho hi\n```", max_rounds=2)
    assert any(e.get("type") == "rounds_exhausted" for e in events), events


def test_no_rounds_exhausted_on_normal_finish(monkeypatch):
    _patch_common(monkeypatch)
    # A plain answer (no tool block) -> done-break on round 1 -> no event.
    events = _run_loop(monkeypatch, "All done, here is your answer.", max_rounds=2)
    assert not any(e.get("type") == "rounds_exhausted" for e in events), events


def test_emits_intent_nudge_exhausted_when_cap_is_exhausted(monkeypatch):
    _patch_common(monkeypatch)

    events = _run_loop(monkeypatch, "Let me check the logs", max_rounds=5)

    guard = next((e for e in events if e.get("type") == "intent_nudge_exhausted"), None)
    assert guard is not None, events
    assert guard["reason"] == "intent_without_action_nudge_cap"
    assert guard["nudges"] == 2


def test_emits_loop_breaker_triggered_when_loop_breaker_trips(monkeypatch):
    _patch_common(monkeypatch)

    events = _run_loop(monkeypatch, "```bash\necho hi\n```", max_rounds=6)

    guard = next((e for e in events if e.get("type") == "loop_breaker_triggered"), None)
    assert guard is not None, events
    assert guard["reason"] == "loop_breaker_stall"


def _ask_user_resume_messages():
    return [
        {"role": "user", "content": "Search the mailbox and create a Word document."},
        {
            "role": "assistant",
            "content": "Should I continue?",
            "metadata": {"tool_events": [{
                "tool": "ask_user",
                "ask_user": {
                    "question": "Should I continue?",
                    "options": [{"label": "Yes"}, {"label": "No"}],
                },
            }]},
        },
        {"role": "user", "content": "Yes"},
    ]


def test_long_pending_available_tool_plan_gets_action_round(monkeypatch):
    _patch_common(monkeypatch)
    model_calls = []
    executed = []
    planning = (
        "The email search completed successfully. "
        + ("I have reviewed the result metadata and date ordering. " * 12)
        + "The next step is to create the document, so the tool call would be "
          "create_document with the ordered content."
    )
    assert len(planning) > 400

    async def _fake_stream(_candidates, messages, **kwargs):
        model_calls.append((messages, kwargs.get("tools") or []))
        if len(model_calls) == 1:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{
                    "name": "search_emails",
                    "arguments": json.dumps({"query": "from:test@example.com"}),
                }],
            }) + "\n\n"
        elif len(model_calls) == 2:
            yield f'data: {json.dumps({"delta": planning})}\n\n'
        elif len(model_calls) == 3:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{
                    "name": "create_document",
                    "arguments": json.dumps({
                        "title": "Emails by date",
                        "language": "markdown",
                        "content": "# Emails",
                    }),
                }],
            }) + "\n\n"
        else:
            yield 'data: {"delta": "Done."}\n\n'
        yield "data: [DONE]\n\n"

    async def _fake_exec(block, *args, **kwargs):
        executed.append(block.tool_type)
        if block.tool_type == "create_document":
            return block.tool_type, {
                "action": "create", "doc_id": "doc-1", "title": "Emails by date",
                "language": "markdown", "content": "# Emails", "version": 1,
                "exit_code": 0,
            }
        return block.tool_type, {"output": "90 emails", "exit_code": 0}

    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)
    monkeypatch.setattr(al, "execute_tool_block", _fake_exec, raising=False)

    events = _types(_collect(al.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", _ask_user_resume_messages(),
        max_rounds=4,
        relevant_tools={"ask_user", "search_emails", "create_document"},
        owner="admin", _is_teacher_run=True,
    )))

    assert executed == ["mcp__email__search_emails", "create_document"]
    # A normal synthesis round follows successful document creation.
    assert len(model_calls) == 4
    tool_names_by_round = [
        {tool.get("function", {}).get("name") for tool in tools}
        for _messages, tools in model_calls
    ]
    assert tool_names_by_round[2] == {"create_document"}
    assert tool_names_by_round[3] == tool_names_by_round[1]
    assert "create_document" in tool_names_by_round[3]
    pending_instruction = "\n".join(
        str(message.get("content") or "")
        for message in model_calls[2][0]
        if message.get("role") == "system"
    )
    assert "explicitly identified `create_document` as a pending action" in pending_instruction
    assert "previous attempt ended without making a tool call" not in pending_instruction
    assert not any(event.get("type") == "rounds_exhausted" for event in events)


def test_missing_pending_tool_restriction_falls_back_to_normal_schemas(monkeypatch, caplog):
    _patch_common(monkeypatch)
    model_tools = []

    async def _fake_stream(_candidates, messages, **kwargs):
        model_tools.append(kwargs.get("tools") or [])
        if len(model_tools) == 1:
            yield f'data: {json.dumps({"delta": "I will call create_document now."})}\n\n'
        else:
            yield 'data: {"delta": "Done."}\n\n'
        yield "data: [DONE]\n\n"

    real_detector = al._pending_available_tool_action
    detector_calls = 0

    def _missing_detector(text, available_tools):
        nonlocal detector_calls
        detector_calls += 1
        if detector_calls == 1:
            return "missing_tool"
        return real_detector(text, available_tools)

    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)
    monkeypatch.setattr(al, "_pending_available_tool_action", _missing_detector)

    _collect(al.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o",
        [{"role": "user", "content": "Create a document"}],
        max_rounds=2, relevant_tools={"create_document"},
        _is_teacher_run=True,
    ))

    assert len(model_tools) == 2
    assert model_tools[1]
    assert {
        tool.get("function", {}).get("name") for tool in model_tools[1]
    } == {
        tool.get("function", {}).get("name") for tool in model_tools[0]
    }
    assert "pending tool restriction unavailable" in caplog.text


def test_negative_long_tool_mention_does_not_nudge(monkeypatch):
    _patch_common(monkeypatch)
    model_calls = []
    response = ("Context and explanation. " * 25) + (
        "I cannot call create_document because it is unavailable in this environment."
    )
    assert len(response) > 400

    async def _fake_stream(_candidates, messages, **kwargs):
        model_calls.append(messages)
        yield f'data: {json.dumps({"delta": response})}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)
    events = _types(_collect(al.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o",
        [{"role": "user", "content": "Create a document"}],
        max_rounds=3, relevant_tools={"create_document"},
        _is_teacher_run=True,
    )))

    assert len(model_calls) == 1
    assert not any(event.get("type") == "agent_step" for event in events)


def test_pending_tool_detector_supports_required_phrases_and_rejects_negatives():
    positives = [
        "The next step is to call create_document.",
        "The tool call would be create_document.",
        "I need to call create_document.",
        "I need to use create_document.",
        "I should call create_document.",
        "I should use create_document.",
        "I will call create_document.",
        "I will use create_document.",
        "Let me call create_document.",
        "Let me use create_document.",
    ]
    negatives = [
        "I cannot call create_document.",
        "I should not call create_document.",
        "I will not call create_document.",
        "Do not call create_document.",
        "create_document is unavailable.",
    ]
    for text in positives:
        assert al._pending_available_tool_action(text, {"create_document"}) == "create_document", text
    for text in negatives:
        assert al._pending_available_tool_action(text, {"create_document"}) is None, text
    assert al._pending_available_tool_action("I will use create_document.", {"other_tool"}) is None


def test_later_ask_user_after_substantive_progress_remains_legal(monkeypatch):
    _patch_common(monkeypatch)
    model_calls = []
    executed = []

    async def _fake_stream(_candidates, messages, **kwargs):
        model_calls.append(messages)
        if len(model_calls) == 1:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{"name": "list_email_accounts", "arguments": "{}"}],
            }) + "\n\n"
        elif len(model_calls) == 2:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{
                    "name": "ask_user",
                    "arguments": json.dumps({
                        "question": "Should I continue?",
                        "options": [{"label": "Yes"}, {"label": "No"}],
                    }),
                }],
            }) + "\n\n"
        yield "data: [DONE]\n\n"

    async def _fake_exec(block, *args, **kwargs):
        executed.append(block.tool_type)
        if block.tool_type == "ask_user":
            return block.tool_type, {
                "ask_user": {
                    "question": "Should I continue?",
                    "options": [{"label": "Yes"}, {"label": "No"}],
                },
                "exit_code": 0,
            }
        return block.tool_type, {"output": "accounts", "exit_code": 0}

    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)
    monkeypatch.setattr(al, "execute_tool_block", _fake_exec, raising=False)
    _collect(al.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", _ask_user_resume_messages(),
        max_rounds=3,
        relevant_tools={"ask_user", "list_email_accounts"},
        owner="admin", _is_teacher_run=True,
    ))

    assert executed == ["mcp__email__list_email_accounts", "ask_user"]
    assert len(model_calls) == 2


def test_duplicate_ask_user_retry_guard_is_bounded(monkeypatch):
    _patch_common(monkeypatch)
    model_calls = []
    executed = []

    async def _fake_stream(_candidates, messages, **kwargs):
        model_calls.append(messages)
        if len(model_calls) == 1:
            yield 'data: {"delta": "I will continue."}\n\n'
        else:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{
                    "name": "ask_user",
                    "arguments": json.dumps({
                        "question": "Should I continue?",
                        "options": [{"label": "Yes"}, {"label": "No"}],
                    }),
                }],
            }) + "\n\n"
        yield "data: [DONE]\n\n"

    async def _fake_exec(block, *args, **kwargs):
        executed.append(block.tool_type)
        return block.tool_type, {"output": "unexpected", "exit_code": 0}

    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)
    monkeypatch.setattr(al, "execute_tool_block", _fake_exec, raising=False)
    chunks = _collect(al.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", _ask_user_resume_messages(),
        max_rounds=20,
        relevant_tools={"ask_user", "list_email_accounts"},
        owner="admin", _is_teacher_run=True,
    ))

    assert executed == []
    assert len(model_calls) == 3
    assert any("repeated an already answered question" in chunk for chunk in chunks)
    assert not any("The model returned an empty response" in chunk for chunk in chunks)
