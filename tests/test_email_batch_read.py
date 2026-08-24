"""Bounded batch behavior for the existing read_email tool."""

import json

import pytest

pytest.importorskip("mcp")

import src.agent_tools  # Initialize schema re-exports in application order.
import src.agent_loop as agent_loop
import mcp_servers.email_server as es
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS


@pytest.fixture(autouse=True)
def _isolated_accounts(monkeypatch):
    """Keep these unit tests away from configured or fixture mailboxes."""
    monkeypatch.setattr(es, "_read_accounts_from_db", lambda: [])
    monkeypatch.setattr(es, "_list_accounts_raw", lambda: [{}])


def _message(uid, body, **overrides):
    result = {
        "uid": str(uid),
        "account": "mailbox",
        "account_email": "mailbox@example.test",
        "message_id": f"<{uid}@example.test>",
        "subject": f"Subject {uid}",
        "from": "Synthetic Sender",
        "from_address": "sender@example.test",
        "date": "Mon, 24 Aug 2026 12:00:00 +0200",
        "body": body,
        "attachments": [],
    }
    result.update(overrides)
    return result


async def _batch(arguments):
    response = await es.call_tool("read_email", arguments)
    assert len(response) == 1
    return json.loads(response[0].text)


@pytest.mark.asyncio
async def test_read_email_schema_exposes_one_ordered_bounded_batch_contract():
    mcp_tool = next(tool for tool in await es.list_tools() if tool.name == "read_email")
    targets = mcp_tool.inputSchema["properties"]["targets"]
    assert targets["minItems"] == 1
    assert targets["maxItems"] == es.BATCH_READ_MAX_TARGETS == 20
    assert {"uid", "message_id", "folder", "account"} <= set(targets["items"]["properties"])

    native = next(
        schema for schema in FUNCTION_TOOL_SCHEMAS
        if schema["function"]["name"] == "read_email"
    )["function"]["parameters"]["properties"]["targets"]
    assert native["minItems"] == 1
    assert native["maxItems"] == 20


@pytest.mark.asyncio
async def test_read_email_legacy_single_message_output_is_preserved(monkeypatch):
    calls = []

    def fake_read(**kwargs):
        calls.append(kwargs)
        return _message("7", "legacy body")

    monkeypatch.setattr(es, "_read_email", fake_read)
    response = await es.call_tool(
        "read_email",
        {"uid": "7", "folder": "Archive", "account": "personal"},
    )

    assert calls == [{
        "uid": "7", "message_id": None, "folder": "Archive", "account": "personal",
    }]
    assert response[0].text.startswith("**Subject:** Subject 7")
    assert response[0].text.endswith("legacy body")
    assert '"batch"' not in response[0].text


@pytest.mark.asyncio
async def test_batch_preserves_target_order_and_per_target_addressing(monkeypatch):
    calls = []

    def fake_read(**kwargs):
        calls.append(kwargs)
        identity = kwargs["uid"] or kwargs["message_id"]
        return _message(identity, f"body:{identity}")

    monkeypatch.setattr(es, "_read_email", fake_read)
    targets = [
        {"uid": "9", "folder": "Archive", "account": "work"},
        {"message_id": "<alpha@example.test>", "folder": "INBOX", "account": "personal"},
        {"uid": "2", "folder": "Receipts", "account": "work"},
    ]
    result = await _batch({"targets": targets})

    assert calls == [
        {"uid": "9", "message_id": None, "folder": "Archive", "account": "work"},
        {"uid": None, "message_id": "<alpha@example.test>", "folder": "INBOX", "account": "personal"},
        {"uid": "2", "message_id": None, "folder": "Receipts", "account": "work"},
    ]
    assert [item["index"] for item in result["items"]] == [0, 1, 2]
    assert [item["uid"] for item in result["items"]] == ["9", None, "2"]
    assert [item["body"] for item in result["items"]] == [
        "body:9", "body:<alpha@example.test>", "body:2",
    ]


@pytest.mark.asyncio
async def test_batch_keeps_sibling_results_when_one_read_fails(monkeypatch):
    def fake_read(**kwargs):
        if kwargs["uid"] == "raises":
            raise RuntimeError("synthetic connection failure")
        if kwargs["uid"] == "missing":
            return {"error": "synthetic not found"}
        return _message(kwargs["uid"], "available")

    monkeypatch.setattr(es, "_read_email", fake_read)
    result = await _batch({"targets": [
        {"uid": "ok", "account": "mailbox"},
        {"uid": "raises", "account": "mailbox"},
        {"uid": "missing", "account": "mailbox"},
        {"folder": "INBOX", "account": "mailbox"},
    ]})

    assert [item["status"] for item in result["items"]] == [
        "success", "error", "error", "error",
    ]
    assert result["items"][0]["body"] == "available"
    assert result["items"][1]["error"] == "synthetic connection failure"
    assert result["items"][2]["error"] == "synthetic not found"
    assert result["items"][3]["error"] == "uid or message_id is required"


@pytest.mark.asyncio
async def test_batch_body_cap_is_exact_and_marks_truncated_and_omitted(monkeypatch):
    monkeypatch.setattr(es, "BATCH_READ_MAX_BODY_CHARS", 10)
    bodies = {"a": "123456", "b": "abcdefgh", "c": "unread"}
    calls = []

    def fake_read(**kwargs):
        calls.append(kwargs["uid"])
        return _message(kwargs["uid"], bodies[kwargs["uid"]])

    monkeypatch.setattr(es, "_read_email", fake_read)
    result = await _batch({"targets": [
        {"uid": "a", "account": "mailbox"},
        {"uid": "b", "account": "mailbox"},
        {"uid": "c", "account": "mailbox"},
    ]})

    assert calls == ["a", "b"]
    assert result["body_chars"] == result["max_body_chars"] == 10
    assert result["truncated"] is True
    assert result["items"][0]["body"] == "123456"
    assert result["items"][0]["truncated"] is False
    assert result["items"][1]["body"] == "abcd"
    assert result["items"][1]["truncated"] is True
    assert result["items"][2]["status"] == "omitted"
    assert result["items"][2]["reason"] == "aggregate_body_cap_reached"


@pytest.mark.asyncio
async def test_batch_never_reads_more_than_twenty_targets(monkeypatch):
    calls = []

    def fake_read(**kwargs):
        calls.append(kwargs["uid"])
        return _message(kwargs["uid"], "x")

    monkeypatch.setattr(es, "_read_email", fake_read)
    result = await _batch({"targets": [
        {"uid": str(index), "account": "mailbox"} for index in range(22)
    ]})

    assert calls == [str(index) for index in range(20)]
    assert result["requested"] == 22
    assert result["processed"] == 20
    assert [item["reason"] for item in result["items"][20:]] == [
        "target_limit_exceeded", "target_limit_exceeded",
    ]


@pytest.mark.asyncio
async def test_batch_rejects_empty_or_mixed_single_and_batch_inputs():
    empty = await es.call_tool("read_email", {"targets": []})
    assert empty[0].text == "Error: targets must contain at least one message"

    mixed = await es.call_tool("read_email", {"uid": "1", "targets": [{"uid": "2"}]})
    assert "either targets" in mixed[0].text


def _schema_names(tools):
    return {
        tool.get("function", {}).get("name") or tool.get("name")
        for tool in (tools or [])
        if isinstance(tool, dict)
    }


def _patch_agent_loop_dependencies(monkeypatch, fake_stream, fake_execute):
    monkeypatch.setattr(
        agent_loop, "get_setting", lambda key, default=None: default, raising=False,
    )
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(
        agent_loop, "estimate_tokens", lambda *args, **kwargs: 10, raising=False,
    )
    monkeypatch.setattr(
        agent_loop, "stream_llm_with_fallback", fake_stream, raising=False,
    )
    monkeypatch.setattr(
        agent_loop, "execute_tool_block", fake_execute, raising=False,
    )


@pytest.mark.asyncio
async def test_production_shaped_batch_reaches_document_in_four_rounds(monkeypatch):
    targets = [
        {"uid": str(index), "folder": "Archive", "account": "work"}
        for index in range(10)
    ]
    model_rounds = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_rounds.append({
            "messages": json.loads(json.dumps(messages)),
            "schemas": _schema_names(kwargs.get("tools")),
        })
        round_number = len(model_rounds)
        if round_number == 1:
            call = {"name": "search_emails", "arguments": json.dumps({"query": "synthetic"})}
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        elif round_number == 2:
            assert "SYNTHETIC_SEARCH_RESULT" in json.dumps(messages)
            call = {"name": "read_email", "arguments": json.dumps({"targets": targets})}
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        elif round_number == 3:
            assert "SYNTHETIC_BATCH_RESULT" in json.dumps(messages)
            call = {
                "name": "create_document",
                "arguments": json.dumps({
                    "title": "Synthetic email report",
                    "language": "markdown",
                    "content": "# Synthetic report",
                }),
            }
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        else:
            assert "SYNTHETIC_DOCUMENT_RESULT" in json.dumps(messages)
            yield 'data: {"delta": "Synthetic report created."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        if block.tool_type == "create_document":
            title, language, content = block.content.split("\n", 2)
            payload = {"title": title, "language": language, "content": content}
        else:
            payload = json.loads(block.content)
        executed.append((block.tool_type, payload))
        if block.tool_type == "mcp__email__search_emails":
            return block.tool_type, {
                "output": "SYNTHETIC_SEARCH_RESULT " + json.dumps(targets),
                "exit_code": 0,
            }
        if block.tool_type == "mcp__email__read_email":
            assert payload == {"targets": targets}
            return block.tool_type, {
                "output": "SYNTHETIC_BATCH_RESULT " + json.dumps({
                    "batch": True,
                    "items": [{"uid": target["uid"], "body": "synthetic body"} for target in targets],
                }),
                "exit_code": 0,
            }
        assert block.tool_type == "create_document"
        return block.tool_type, {
            "output": "SYNTHETIC_DOCUMENT_RESULT",
            "action": "create",
            "doc_id": "synthetic-doc",
            "title": payload["title"],
            "language": payload["language"],
            "content": payload["content"],
            "version": 1,
            "exit_code": 0,
        }

    _patch_agent_loop_dependencies(monkeypatch, fake_stream, fake_execute)
    chunks = [chunk async for chunk in agent_loop.stream_agent_loop(
        "https://api.openai.com/v1",
        "gpt-4o",
        [{"role": "user", "content": "Create a document from ten synthetic emails."}],
        max_rounds=20,
        relevant_tools={"search_emails", "read_email", "create_document"},
        owner="admin",
        _is_teacher_run=True,
    )]

    tool_types = [tool_type for tool_type, _payload in executed]
    assert tool_types.count("mcp__email__search_emails") == 1
    assert tool_types.count("mcp__email__read_email") == 1
    assert tool_types.count("create_document") == 1
    batch_payload = next(
        payload for tool_type, payload in executed
        if tool_type == "mcp__email__read_email"
    )
    assert batch_payload == {"targets": targets}
    assert len(batch_payload["targets"]) == len(targets) == 10
    assert tool_types == [
        "mcp__email__search_emails", "mcp__email__read_email", "create_document",
    ]
    assert len(model_rounds) == 4
    assert len(model_rounds) < 20
    assert all(round_data["schemas"] == model_rounds[0]["schemas"] for round_data in model_rounds)
    assert any("Synthetic report created." in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_post_batch_pending_document_restriction_is_one_round_only(monkeypatch):
    targets = [
        {"uid": str(index), "folder": "INBOX", "account": "work"}
        for index in range(10)
    ]
    model_rounds = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_rounds.append({
            "messages": json.loads(json.dumps(messages)),
            "schemas": _schema_names(kwargs.get("tools")),
        })
        round_number = len(model_rounds)
        if round_number == 1:
            call = {"name": "search_emails", "arguments": json.dumps({"query": "synthetic"})}
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        elif round_number == 2:
            assert "SYNTHETIC_SEARCH_RESULT" in json.dumps(messages)
            call = {"name": "read_email", "arguments": json.dumps({"targets": targets})}
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        elif round_number == 3:
            assert "SYNTHETIC_BATCH_RESULT" in json.dumps(messages)
            yield 'data: {"delta": "The next step is to call create_document."}\n\n'
        elif round_number == 4:
            call = {
                "name": "create_document",
                "arguments": json.dumps({
                    "title": "Synthetic pending report",
                    "language": "markdown",
                    "content": "# Synthetic pending report",
                }),
            }
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        else:
            assert "SYNTHETIC_DOCUMENT_RESULT" in json.dumps(messages)
            yield 'data: {"delta": "Synthetic pending report created."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        if block.tool_type == "create_document":
            title, language, content = block.content.split("\n", 2)
            payload = {"title": title, "language": language, "content": content}
        else:
            payload = json.loads(block.content)
        executed.append((block.tool_type, payload))
        if block.tool_type == "mcp__email__search_emails":
            return block.tool_type, {
                "output": "SYNTHETIC_SEARCH_RESULT " + json.dumps(targets),
                "exit_code": 0,
            }
        if block.tool_type == "mcp__email__read_email":
            assert payload == {"targets": targets}
            return block.tool_type, {
                "output": "SYNTHETIC_BATCH_RESULT " + json.dumps({
                    "batch": True,
                    "items": [{"uid": target["uid"], "body": "synthetic body"} for target in targets],
                }),
                "exit_code": 0,
            }
        assert block.tool_type == "create_document"
        return block.tool_type, {
            "output": "SYNTHETIC_DOCUMENT_RESULT",
            "action": "create",
            "doc_id": "synthetic-pending-doc",
            "title": payload["title"],
            "language": payload["language"],
            "content": payload["content"],
            "version": 1,
            "exit_code": 0,
        }

    _patch_agent_loop_dependencies(monkeypatch, fake_stream, fake_execute)
    chunks = [chunk async for chunk in agent_loop.stream_agent_loop(
        "https://api.openai.com/v1",
        "gpt-4o",
        [{"role": "user", "content": "Create a document from ten synthetic emails."}],
        max_rounds=20,
        relevant_tools={"search_emails", "read_email", "create_document"},
        owner="admin",
        _is_teacher_run=True,
    )]

    tool_types = [tool_type for tool_type, _payload in executed]
    assert tool_types == [
        "mcp__email__search_emails", "mcp__email__read_email", "create_document",
    ]
    assert len(model_rounds) == 5
    assert len(model_rounds) < 20
    assert model_rounds[3]["schemas"] == {"create_document"}
    assert model_rounds[2]["schemas"] != {"create_document"}
    assert "create_document" in model_rounds[2]["schemas"]
    assert model_rounds[4]["schemas"] == model_rounds[2]["schemas"]
    pending_instruction = "\n".join(
        str(message.get("content") or "")
        for message in model_rounds[3]["messages"]
        if message.get("role") == "system"
    )
    assert "explicitly identified `create_document` as a pending action" in pending_instruction
    assert any("Synthetic pending report created." in chunk for chunk in chunks)
