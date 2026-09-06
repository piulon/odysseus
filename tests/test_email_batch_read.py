"""Bounded batch behavior for the existing read_email tool."""

import json

import pytest

pytest.importorskip("mcp")

import src.agent_tools  # Initialize schema re-exports in application order.
import src.agent_loop as agent_loop
import mcp_servers.email_server as es
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
from src.tool_schemas import function_call_to_tool_block


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
    assert "uids" not in next(
        schema for schema in FUNCTION_TOOL_SCHEMAS
        if schema["function"]["name"] == "read_email"
    )["function"]["parameters"]["properties"]

    search_tool = next(tool for tool in await es.list_tools() if tool.name == "search_emails")
    search_properties = search_tool.inputSchema["properties"]
    assert search_properties["max_results"]["maximum"] == 100
    assert "cursor" not in search_properties
    assert "offset" not in search_properties


def test_model_plural_uids_normalize_to_existing_batch_contract():
    block = function_call_to_tool_block(
        "read_email",
        json.dumps({
            "uids": ["10", 11],
            "folder": "Archive",
            "account": "work",
        }),
    )

    assert block is not None
    assert block.tool_type == "mcp__email__read_email"
    assert json.loads(block.content) == {
        "targets": [
            {"uid": "10", "folder": "Archive", "account": "work"},
            {"uid": "11", "folder": "Archive", "account": "work"},
        ],
    }


@pytest.mark.parametrize("user_text", [
    "Create a document with all emails from alex@example.test",
    "Archive every email about the project",
    "Document all messages involving Alex",
    "Crea un document amb tots els correus de l'Anna",
    "Documento con todos los emails de Ana",
    "Create the complete email history with Pat",
    "Create a full email history for this address",
])
def test_exhaustive_email_intent_is_narrowly_recognized(user_text):
    assert agent_loop._has_exhaustive_email_intent(user_text)


@pytest.mark.parametrize("user_text", [
    "Find emails from alex@example.test",
    "Summarize the latest 20 emails",
    "Search messages about the project",
])
def test_ordinary_email_search_is_not_exhaustive(user_text):
    assert not agent_loop._has_exhaustive_email_intent(user_text)


def test_exhaustive_search_uses_only_supported_bounded_limit():
    original = json.dumps({
        "query": "alex@example.test",
        "max_results": 20,
        "folders": ["INBOX", "Sent", "Archive"],
    })
    exhaustive = json.loads(agent_loop._email_search_arguments_for_intent(
        original, exhaustive=True,
    ))
    assert exhaustive == {
        "query": "alex@example.test",
        "max_results": agent_loop._EXHAUSTIVE_EMAIL_SEARCH_MAX_RESULTS,
        "folders": ["INBOX", "Sent", "Archive"],
    }
    assert "cursor" not in exhaustive
    assert "offset" not in exhaustive
    assert agent_loop._email_search_arguments_for_intent(
        original, exhaustive=False,
    ) == original


def test_canonical_message_id_deduplicates_provider_folder_overlap():
    inbox = {
        "resolved_message_id": "<same@example.test>",
        "folder": "INBOX",
        "uid": "2365",
    }
    all_mail = {
        "resolved_message_id": "<SAME@example.test>",
        "folder": "[Provider]/All Mail",
        "uid": "3697",
    }
    assert agent_loop._email_canonical_identity(inbox) == agent_loop._email_canonical_identity(all_mail)


def test_same_subject_and_date_with_distinct_message_ids_remain_distinct():
    common = {"subject": "Re: Euro", "date": "3 May 2023", "body": "same"}
    first = {**common, "resolved_message_id": "<first@example.test>"}
    second = {**common, "resolved_message_id": "<second@example.test>"}
    assert agent_loop._email_canonical_identity(first) != agent_loop._email_canonical_identity(second)


def test_missing_message_id_fallback_is_deterministic_and_conservative():
    first = {
        "subject": "No RFC identity", "from_address": "a@example.test",
        "to": "b@example.test", "date": "Tue, 1 Jan 2030 10:00:00 +0000",
        "body": "Exact retrieved body", "folder": "INBOX", "uid": "1",
    }
    overlap = {**first, "folder": "[Provider]/All Mail", "uid": "99"}
    distinct = {**overlap, "body": "A legitimately different body"}
    assert agent_loop._email_canonical_identity(first) == agent_loop._email_canonical_identity(overlap)
    assert agent_loop._email_canonical_identity(first) != agent_loop._email_canonical_identity(distinct)


def test_lossless_document_intent_excludes_non_artifact_summaries():
    assert agent_loop._requires_lossless_email_document(
        "Create a Word document with all emails involving Alex"
    )
    assert not agent_loop._requires_lossless_email_document(
        "Summarize all emails involving Alex"
    )
    assert not agent_loop._requires_lossless_email_document(
        "Create a document summarizing recent emails involving Alex"
    )


def test_lossless_renderer_handles_zero_one_and_malformed_dates():
    assert agent_loop._required_email_document_block(
        [], "Create a document with all emails", lossless=True,
    ) is None
    one = {
        "resolved_message_id": "<one@example.test>",
        "subject": "Only message",
        "date": "not an RFC date",
        "body": "body with trailing whitespace  \n",
    }
    rendered = agent_loop._render_lossless_email_document([one])
    assert rendered.count("<!-- odysseus-email-entry -->") == 1
    assert "body with trailing whitespace  \n" in rendered

    missing = {**one, "resolved_message_id": "<missing@example.test>", "date": ""}
    valid = {
        **one,
        "resolved_message_id": "<valid@example.test>",
        "date": "Mon, 24 Aug 2026 12:00:00 +0200",
    }
    ordered = agent_loop._render_lossless_email_document([missing, one, valid])
    assert ordered.index("<valid@example.test>") < ordered.index("<missing@example.test>")
    assert ordered.index("<valid@example.test>") < ordered.index("<one@example.test>")

    incomplete = agent_loop._render_lossless_email_document(
        [one], discovery_complete=False,
    )
    assert "not guaranteed to contain every mailbox match" in incomplete


def test_discovery_incomplete_marker_is_structural():
    assert agent_loop._email_search_result_is_incomplete({
        "stdout": "Found 100 email(s)\n[DISCOVERY INCOMPLETE: limit reached]",
    })
    assert not agent_loop._email_search_result_is_incomplete({
        "stdout": "Found 60 email(s)",
    })


def test_lossless_renderer_keeps_same_subject_messages_with_distinct_ids():
    messages = [
        {
            "resolved_message_id": f"<{index}@example.test>",
            "subject": "Same subject",
            "date": "Mon, 24 Aug 2026 12:00:00 +0200",
            "body": f"body {index}",
        }
        for index in (1, 2)
    ]
    rendered = agent_loop._render_lossless_email_document(messages)
    assert rendered.count("<!-- odysseus-email-entry -->") == 2
    assert "body 1" in rendered
    assert "body 2" in rendered


@pytest.mark.parametrize("uids", [[], "10", None, [{}], [True], [1.5]])
def test_model_plural_uids_reject_malformed_values(uids):
    assert function_call_to_tool_block(
        "read_email", json.dumps({"uids": uids}),
    ) is None


def test_model_plural_uids_leave_over_limit_handling_to_existing_batch_path():
    block = function_call_to_tool_block(
        "read_email", json.dumps({"uids": list(range(22))}),
    )

    assert block is not None
    assert len(json.loads(block.content)["targets"]) == 22


@pytest.mark.parametrize("conflict", [
    {"uid": "11"},
    {"message_id": "<message@example.test>"},
    {"targets": [{"uid": "11"}]},
])
def test_model_plural_uids_reject_ambiguous_identifier_forms(conflict):
    arguments = {"uids": ["10"], **conflict}
    assert function_call_to_tool_block(
        "read_email", json.dumps(arguments),
    ) is None


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


class _EmailSchemaManager:
    def get_all_openai_schemas(self, _disabled_map):
        return [
            schema for schema in FUNCTION_TOOL_SCHEMAS
            if schema.get("function", {}).get("name") in {"search_emails", "read_email"}
        ]

    def get_tool_descriptions_for_prompt(self, _disabled_map):
        return "**email:**\n- search_emails: Search mailbox emails.\n- read_email: Read email bodies."


def _patch_agent_loop_dependencies(monkeypatch, fake_stream, fake_execute):
    monkeypatch.setattr(
        agent_loop, "blocked_tools_for_owner", lambda _owner: set(), raising=False,
    )
    monkeypatch.setattr(
        agent_loop, "get_setting", lambda key, default=None: default, raising=False,
    )
    monkeypatch.setattr(
        agent_loop, "get_mcp_manager", lambda: _EmailSchemaManager(), raising=False,
    )
    monkeypatch.setattr(
        agent_loop, "estimate_tokens", lambda *args, **kwargs: 10, raising=False,
    )
    monkeypatch.setattr(
        agent_loop, "stream_llm_with_fallback", fake_stream, raising=False,
    )
    monkeypatch.setattr(
        agent_loop, "execute_tool_block", fake_execute, raising=False,
    )
    monkeypatch.setattr(
        agent_loop, "FUNCTION_TOOL_SCHEMAS", FUNCTION_TOOL_SCHEMAS, raising=False,
    )


def _synthetic_search_output(targets):
    lines = [f'Found {len(targets)} email(s) matching "synthetic":', ""]
    for index, target in enumerate(targets, 1):
        lines.extend([
            f"{index}. **Synthetic {target['uid']}**",
            "   From: Synthetic Sender (sender@example.test)",
            "   Date: Mon, 24 Aug 2026 12:00:00 +0200",
            f"   Folder: {target.get('folder', 'INBOX')}",
            f"   UID: {target['uid']}",
        ])
    return "\n".join(lines)


def _synthetic_batch_output(targets, *, failures=None):
    failures = failures or {}
    items = []
    for index, target in enumerate(targets):
        status = failures.get(target["uid"], "success")
        item = {
            "index": index,
            "uid": target["uid"],
            "message_id": None,
            "folder": target.get("folder", "INBOX"),
            "account": target.get("account"),
            "status": status,
        }
        if status == "success":
            item.update({"body": f"synthetic body {target['uid']}", "truncated": False})
        elif status == "omitted":
            item.update({"truncated": True, "reason": "aggregate_body_cap_reached"})
        else:
            item["error"] = "synthetic read failure"
        items.append(item)
    return json.dumps({"batch": True, "items": items})


def test_unattempted_omissions_do_not_mark_messages_terminal():
    targets = [{"uid": str(index), "folder": "INBOX"} for index in range(22)]
    items = [
        {
            "index": index,
            "status": "success",
            "body": f"body {index}",
        }
        for index in range(20)
    ]
    items.append({
        "index": 20,
        "status": "omitted",
        "reason": "target_limit_exceeded",
    })
    items.append({
        "index": 21,
        "status": "omitted",
        "reason": "aggregate_body_cap_reached",
    })

    terminal, usable = agent_loop._email_read_progress_from_result(
        json.dumps({"targets": targets}),
        {"output": json.dumps({"batch": True, "items": items})},
    )

    assert len(terminal) == 20
    assert len(usable) == 20
    assert agent_loop._email_retrieval_target_key(
        uid="20", folder="INBOX",
    ) not in terminal
    assert agent_loop._email_retrieval_target_key(
        uid="21", folder="INBOX",
    ) not in terminal


def test_production_mcp_stdout_batch_advances_retrieval_accounting():
    targets = [
        {"uid": "11", "folder": "INBOX"},
        {"uid": "12", "folder": "[Provider]/Sent", "account": "work"},
        {"uid": "13", "folder": "Archive"},
    ]
    payload = _synthetic_batch_output(
        targets,
        failures={"12": "error"},
    )

    terminal, usable = agent_loop._email_read_progress_from_result(
        json.dumps({"targets": targets}),
        {"stdout": payload, "stderr": "", "exit_code": 0},
    )

    expected = {
        agent_loop._email_retrieval_target_key(**target) for target in targets
    }
    assert terminal == expected
    assert usable == {
        agent_loop._email_retrieval_target_key(**targets[0]),
        agent_loop._email_retrieval_target_key(**targets[2]),
    }


def test_read_progress_keeps_output_compatibility_and_accepts_empty_stdout():
    targets = [{"uid": "21", "folder": "INBOX"}]
    payload = _synthetic_batch_output(targets)

    terminal, usable = agent_loop._email_read_progress_from_result(
        json.dumps({"targets": targets}),
        {"stdout": "", "output": payload, "exit_code": 0},
    )

    expected = {agent_loop._email_retrieval_target_key(**targets[0])}
    assert terminal == expected
    assert usable == expected


def test_malformed_production_stdout_fabricates_no_read_progress():
    targets = [{"uid": "31", "folder": "INBOX"}]

    terminal, usable = agent_loop._email_read_progress_from_result(
        json.dumps({"targets": targets}),
        {"stdout": "malformed batch output", "stderr": "", "exit_code": 0},
    )

    assert terminal == set()
    assert usable == set()


def test_read_progress_uses_full_execution_result_not_persisted_truncation():
    targets = [
        {"uid": str(index), "folder": "INBOX" if index % 2 else "Archive"}
        for index in range(20)
    ]
    full_stdout = _synthetic_batch_output(targets)
    persisted_tool_event = {
        "output": full_stdout[:100] + "\n... (truncated, persisted copy)",
    }

    terminal, usable = agent_loop._email_read_progress_from_result(
        json.dumps({"targets": targets}),
        {"stdout": full_stdout, "stderr": "", "exit_code": 0},
    )

    expected = {
        agent_loop._email_retrieval_target_key(**target) for target in targets
    }
    assert terminal == expected
    assert usable == expected
    assert "truncated" in persisted_tool_event["output"]


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
            "tool_choice_name": kwargs.get("tool_choice_name"),
        })
        round_number = len(model_rounds)
        if round_number == 1:
            call = {"name": "search_emails", "arguments": json.dumps({"query": "synthetic", "account": "work"})}
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        elif round_number == 2:
            assert "synthetic body 0" in json.dumps(messages)
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
                "output": "SYNTHETIC_SEARCH_RESULT\n" + _synthetic_search_output(targets),
                "exit_code": 0,
            }
        if block.tool_type == "mcp__email__read_email":
            return block.tool_type, {
                "output": _synthetic_batch_output(payload["targets"]),
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
    assert {tuple(sorted(target.items())) for target in batch_payload["targets"]} == {
        tuple(sorted(target.items())) for target in targets
    }
    assert len(batch_payload["targets"]) == len(targets) == 10
    assert tool_types == [
        "mcp__email__search_emails", "mcp__email__read_email", "create_document",
    ]
    assert len(model_rounds) == 3
    assert len(model_rounds) < 20
    assert "create_document" not in model_rounds[0]["schemas"]
    assert "read_email" in model_rounds[0]["schemas"]
    assert model_rounds[1]["schemas"] == {"create_document"}
    assert model_rounds[1]["tool_choice_name"] == "create_document"
    assert "create_document" in model_rounds[2]["schemas"]
    assert model_rounds[2]["tool_choice_name"] is None
    assert model_rounds[2]["schemas"] != {"create_document"}
    assert any("Synthetic report created." in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_email_document_gate_waits_for_all_fifty_targets_with_terminal_failures(monkeypatch):
    targets = [
        {
            "uid": str(index),
            "folder": "Archive" if index % 2 else "INBOX",
            "account": "work",
        }
        for index in range(50)
    ]
    model_rounds = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_rounds.append(_schema_names(kwargs.get("tools")))
        round_number = len(model_rounds)
        if round_number == 1:
            call = {"name": "search_emails", "arguments": json.dumps({
                "query": "synthetic", "account": "work", "max_results": 50,
            })}
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        elif round_number == 2:
            assert "synthetic read failure" in json.dumps(messages)
            assert "synthetic body 49" in json.dumps(messages)
            call = {"name": "create_document", "arguments": json.dumps({
                "title": "Fifty synthetic emails",
                "language": "markdown",
                "content": "# Fifty synthetic emails\n\nPopulated from retrieved bodies.",
            })}
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        else:
            yield 'data: {"delta": "Fifty-message report created."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        if block.tool_type == "create_document":
            title, language, content = block.content.split("\n", 2)
            payload = {"title": title, "language": language, "content": content}
        else:
            payload = json.loads(block.content)
        executed.append((block.tool_type, payload))
        if block.tool_type == "mcp__email__search_emails":
            return block.tool_type, {"output": _synthetic_search_output(targets), "exit_code": 0}
        if block.tool_type == "mcp__email__read_email":
            failures = (
                {"48": "error", "49": "omitted"}
                if len(payload["targets"]) > 1
                else {}
            )
            return block.tool_type, {
                "output": _synthetic_batch_output(payload["targets"], failures=failures),
                "exit_code": 0,
            }
        return block.tool_type, {
            "output": "SYNTHETIC_DOCUMENT_RESULT",
            "action": "create",
            "doc_id": "fifty-doc",
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
        [{"role": "user", "content": "Search fifty emails and create a document from them."}],
        max_rounds=20,
        relevant_tools={"search_emails", "read_email", "create_document"},
        owner="admin",
        _is_teacher_run=True,
    )]

    read_payloads = [
        payload for tool_type, payload in executed
        if tool_type == "mcp__email__read_email"
    ]
    assert [len(payload["targets"]) for payload in read_payloads] == [20, 20, 10, 1]
    assert {
        tuple(sorted(target.items()))
        for payload in read_payloads[:3]
        for target in payload["targets"]
    } == {tuple(sorted(target.items())) for target in targets}
    assert read_payloads[3]["targets"] == [
        {"folder": targets[49]["folder"], "uid": targets[49]["uid"], "account": "work"}
    ]
    assert "create_document" not in model_rounds[0]
    assert "create_document" in model_rounds[1]
    assert [tool_type for tool_type, _payload in executed].count("create_document") == 1
    assert len(model_rounds) == 3 < 20
    assert any("Fifty-message report created." in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_deterministic_read_no_progress_stops_without_another_model_round(monkeypatch, caplog):
    targets = [{"uid": "1", "folder": "INBOX", "account": "work"}]
    model_schemas = []
    log_messages = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_schemas.append(_schema_names(kwargs.get("tools")))
        round_number = len(model_schemas)
        if round_number == 1:
            call = {"name": "search_emails", "arguments": json.dumps({
                "query": "synthetic", "account": "work",
            })}
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        else:
            raise AssertionError("no-progress deterministic read must not call the model again")
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        if block.tool_type == "mcp__email__search_emails":
            return block.tool_type, {
                "stdout": _synthetic_search_output(targets),
                "stderr": "",
                "exit_code": 0,
            }
        assert block.tool_type == "mcp__email__read_email"
        return block.tool_type, {"output": "malformed batch output", "exit_code": 0}

    _patch_agent_loop_dependencies(monkeypatch, fake_stream, fake_execute)
    monkeypatch.setattr(
        agent_loop.logger,
        "info",
        lambda message, *args: log_messages.append(message % args if args else message),
    )
    chunks = [chunk async for chunk in agent_loop.stream_agent_loop(
        "https://api.openai.com/v1",
        "gpt-4o",
        [{"role": "user", "content": "Search an email and create a document from it."}],
        max_rounds=20,
        relevant_tools={"search_emails", "read_email", "create_document"},
        owner="admin",
        _is_teacher_run=True,
    )]

    assert len(model_schemas) == 1
    assert "create_document" not in model_schemas[0]
    assert any(
        "targeted email search auto-selected single candidate" in message
        for message in log_messages
    )
    assert "deterministic pending email retrieval made no progress" in caplog.text
    assert not chunks or all("create_document" not in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_deterministic_read_tool_error_is_preserved_and_stops(monkeypatch, caplog):
    targets = [{"uid": "7", "folder": "INBOX", "account": "work"}]
    model_calls = 0
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        nonlocal model_calls
        model_calls += 1
        if model_calls != 1:
            raise AssertionError("tool failure must not fall back to a model round")
        call = {"name": "search_emails", "arguments": json.dumps({
            "query": "synthetic", "account": "work",
        })}
        yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        if block.tool_type == "mcp__email__search_emails":
            return block.tool_type, {
                "stdout": _synthetic_search_output(targets),
                "stderr": "",
                "exit_code": 0,
            }
        assert block.tool_type == "mcp__email__read_email"
        return block.tool_type, {"error": "synthetic MCP read failure", "exit_code": 1}

    _patch_agent_loop_dependencies(monkeypatch, fake_stream, fake_execute)
    chunks = [chunk async for chunk in agent_loop.stream_agent_loop(
        "https://api.openai.com/v1",
        "gpt-4o",
        [{"role": "user", "content": "Create a document from one email."}],
        max_rounds=20,
        relevant_tools={"search_emails", "read_email", "create_document"},
        owner="admin",
        _is_teacher_run=True,
    )]

    assert model_calls == 1
    assert executed == ["mcp__email__search_emails", "mcp__email__read_email"]
    assert any("synthetic MCP read failure" in chunk for chunk in chunks)
    assert "deterministic pending email retrieval made no progress" in caplog.text


@pytest.mark.asyncio
async def test_sixty_pending_reads_execute_in_three_batches_without_selection_rounds(monkeypatch):
    targets = [
        {"uid": str(index), "folder": "INBOX", "account": "work"}
        for index in range(60)
    ]
    canonical = {
        agent_loop._email_retrieval_target_key(**target) for target in targets
    }
    first_keys = agent_loop._email_pending_read_batch(canonical, set())
    first_batch = agent_loop._email_read_batch_arguments(first_keys)["targets"]
    second_keys = agent_loop._email_pending_read_batch(canonical, set(first_keys))
    second_batch = agent_loop._email_read_batch_arguments(second_keys)["targets"]
    third_keys = agent_loop._email_pending_read_batch(
        canonical, set(first_keys) | set(second_keys),
    )
    third_batch = agent_loop._email_read_batch_arguments(third_keys)["targets"]
    model_rounds = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_rounds.append({
            "messages": json.loads(json.dumps(messages)),
            "schemas": _schema_names(kwargs.get("tools")),
        })
        round_number = len(model_rounds)
        if round_number == 1:
            call = {"name": "search_emails", "arguments": json.dumps({
                "query": "synthetic", "account": "work", "max_results": 60,
            })}
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        elif round_number == 2:
            assert "create_document" in model_rounds[-1]["schemas"]
            call = {"name": "create_document", "arguments": json.dumps({
                "title": "Synthetic report",
                "language": "markdown",
                "content": "# Synthetic report",
            })}
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        else:
            yield 'data: {"delta": "Done."}\n\n'
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
                "stdout": _synthetic_search_output(targets), "stderr": "", "exit_code": 0,
            }
        if block.tool_type == "mcp__email__read_email":
            return block.tool_type, {
                "output": _synthetic_batch_output(payload["targets"]), "exit_code": 0,
            }
        return block.tool_type, {
            "output": "SYNTHETIC_DOCUMENT_RESULT", "action": "create",
            "doc_id": "synthetic-doc", "title": payload["title"],
            "language": payload["language"], "content": payload["content"],
            "version": 1, "exit_code": 0,
        }

    _patch_agent_loop_dependencies(monkeypatch, fake_stream, fake_execute)
    chunks = [chunk async for chunk in agent_loop.stream_agent_loop(
        "https://api.openai.com/v1",
        "gpt-4o",
        [{"role": "user", "content": "Create a document from sixty emails."}],
        max_rounds=20,
        relevant_tools={"search_emails", "read_email", "create_document", "ask_user", "send_email"},
        owner="admin",
        _is_teacher_run=True,
    )]

    read_payloads = [
        payload for tool_type, payload in executed
        if tool_type == "mcp__email__read_email"
    ]
    assert read_payloads == [
        {"targets": first_batch},
        {"targets": second_batch},
        {"targets": third_batch},
    ]
    assert [len(payload["targets"]) for payload in read_payloads] == [20, 20, 20]
    assert len(model_rounds) == 3
    assert "create_document" not in model_rounds[0]["schemas"]
    assert "create_document" in model_rounds[1]["schemas"]
    assert any("Done." in chunk for chunk in chunks)


@pytest.mark.asyncio

async def test_post_batch_pending_document_is_structurally_completed_once(monkeypatch):
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
            "tool_choice_name": kwargs.get("tool_choice_name"),
        })

        round_number = len(model_rounds)

        if round_number == 1:
            call = {
                "name": "search_emails",
                "arguments": json.dumps({
                    "query": "synthetic",
                    "account": "work",
                }),
            }
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'

        elif round_number == 2:
            serialized = json.dumps(messages)

            assert "synthetic body 0" in serialized
            assert model_rounds[-1]["schemas"] == {"create_document"}
            assert model_rounds[-1]["tool_choice_name"] == "create_document"

            # Provider ignores the required tool once. The supervisor must
            # request exactly one corrective model-mediated creation round.
            yield 'data: {"delta": "The next step is to call create_document."}\n\n'

        elif round_number == 3:
            serialized = json.dumps(messages)

            assert "one corrective document-creation round" in serialized
            assert model_rounds[-1]["schemas"] == {"create_document"}
            assert model_rounds[-1]["tool_choice_name"] == "create_document"

            call = {
                "name": "create_document",
                "arguments": json.dumps({
                    "title": "Synthetic pending report",
                    "language": "markdown",
                    "content": (
                        "# Synthetic pending report\n\n"
                        "Model-mediated synthesis of the retrieved synthetic emails."
                    ),
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
            payload = {
                "title": title,
                "language": language,
                "content": content,
            }
        else:
            payload = json.loads(block.content)

        executed.append((block.tool_type, payload))

        if block.tool_type == "mcp__email__search_emails":
            return block.tool_type, {
                "output": (
                    "SYNTHETIC_SEARCH_RESULT\n"
                    + _synthetic_search_output(targets)
                ),
                "exit_code": 0,
            }

        if block.tool_type == "mcp__email__read_email":
            assert payload == {"targets": targets}

            return block.tool_type, {
                "output": _synthetic_batch_output(targets),
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

    _patch_agent_loop_dependencies(
        monkeypatch,
        fake_stream,
        fake_execute,
    )

    chunks = [
        chunk
        async for chunk in agent_loop.stream_agent_loop(
            "https://api.openai.com/v1",
            "gpt-4o",
            [{
                "role": "user",
                "content": "Create a document from ten synthetic emails.",
            }],
            max_rounds=20,
            relevant_tools={
                "search_emails",
                "read_email",
                "create_document",
            },
            owner="admin",
            _is_teacher_run=True,
        )
    ]

    tool_types = [
        tool_type
        for tool_type, _payload in executed
    ]

    assert tool_types == [
        "mcp__email__search_emails",
        "mcp__email__read_email",
        "create_document",
    ]

    assert len(model_rounds) == 4

    assert "create_document" not in model_rounds[0]["schemas"]

    assert model_rounds[1]["schemas"] == {"create_document"}
    assert model_rounds[1]["tool_choice_name"] == "create_document"

    assert model_rounds[2]["schemas"] == {"create_document"}
    assert model_rounds[2]["tool_choice_name"] == "create_document"

    assert model_rounds[3]["schemas"] != {"create_document"}

    document = next(
        payload
        for tool_type, payload in executed
        if tool_type == "create_document"
    )

    assert document["title"] == "Synthetic pending report"
    assert "Model-mediated synthesis" in document["content"]
    assert not document["content"].startswith("# Retrieved emails")

    assert any(
        "Synthetic pending report created." in chunk
        for chunk in chunks
    )

    assert not any(
        "required_artifact_creation_failed" in chunk
        for chunk in chunks
    )


@pytest.mark.asyncio

async def test_post_readiness_document_prose_is_structurally_completed(monkeypatch):
    targets = [
        {"uid": "2", "folder": "INBOX", "account": "work"},
        {"uid": "1", "folder": "INBOX", "account": "work"},
    ]

    model_rounds = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_rounds.append({
            "messages": json.loads(json.dumps(messages)),
            "schemas": _schema_names(kwargs.get("tools")),
            "tool_choice_name": kwargs.get("tool_choice_name"),
        })

        round_number = len(model_rounds)

        if round_number == 1:
            call = {
                "name": "search_emails",
                "arguments": json.dumps({
                    "query": "synthetic",
                    "account": "work",
                }),
            }

            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'

        elif round_number == 2:
            assert model_rounds[-1]["schemas"] == {"create_document"}
            assert model_rounds[-1]["tool_choice_name"] == "create_document"

            # First synthesis attempt ignores the forced tool.
            yield 'data: {"delta": "I have summarized the emails."}\n\n'

        elif round_number == 3:
            serialized = json.dumps(messages)

            assert "one corrective document-creation round" in serialized
            assert "synthetic body 1" in serialized
            assert "synthetic body 2" in serialized

            assert model_rounds[-1]["schemas"] == {"create_document"}
            assert model_rounds[-1]["tool_choice_name"] == "create_document"

            call = {
                "name": "create_document",
                "arguments": json.dumps({
                    "title": "Synthetic ordered report",
                    "language": "markdown",
                    "content": (
                        "# Synthetic ordered report\n\n"
                        "## Older message\n\n"
                        "synthetic body 1\n\n"
                        "## Newer message\n\n"
                        "synthetic body 2"
                    ),
                }),
            }

            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'

        else:
            assert "SYNTHETIC_DOCUMENT_RESULT" in json.dumps(messages)
            yield 'data: {"delta": "The document is ready."}\n\n'

        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        if block.tool_type == "create_document":
            title, language, content = block.content.split("\n", 2)

            executed.append(
                (
                    block.tool_type,
                    {
                        "title": title,
                        "language": language,
                        "content": content,
                    },
                )
            )

            return block.tool_type, {
                "output": "SYNTHETIC_DOCUMENT_RESULT",
                "action": "create",
                "doc_id": "ollama-supervised-doc",
                "title": title,
                "language": language,
                "content": content,
                "version": 1,
                "exit_code": 0,
            }

        payload = json.loads(block.content)
        executed.append((block.tool_type, payload))

        if block.tool_type == "mcp__email__search_emails":
            return block.tool_type, {
                "output": _synthetic_search_output(targets),
                "exit_code": 0,
            }

        assert block.tool_type == "mcp__email__read_email"

        batch = json.loads(
            _synthetic_batch_output(payload["targets"])
        )

        batch["items"][0]["date"] = (
            "Tue, 25 Aug 2026 12:00:00 +0200"
        )

        batch["items"][1]["date"] = (
            "Mon, 24 Aug 2026 12:00:00 +0200"
        )

        return block.tool_type, {
            "output": json.dumps(batch),
            "exit_code": 0,
        }

    _patch_agent_loop_dependencies(
        monkeypatch,
        fake_stream,
        fake_execute,
    )

    chunks = [
        chunk
        async for chunk in agent_loop.stream_agent_loop(
            "http://ollama:11434/v1/chat/completions",
            "qwen3:14b",
            [{
                "role": "user",
                "content": (
                    "Create a Word document from the synthetic emails, "
                    "ordered by date."
                ),
            }],
            max_rounds=20,
            relevant_tools={
                "search_emails",
                "read_email",
                "create_document",
                "send_email",
            },
            owner="admin",
            _is_teacher_run=True,
        )
    ]

    assert len(model_rounds) == 4

    assert model_rounds[1]["schemas"] == {"create_document"}
    assert model_rounds[1]["tool_choice_name"] == "create_document"

    assert model_rounds[2]["schemas"] == {"create_document"}
    assert model_rounds[2]["tool_choice_name"] == "create_document"

    assert model_rounds[3]["schemas"] != {"create_document"}

    tool_types = [
        tool_type
        for tool_type, _payload in executed
    ]

    assert tool_types == [
        "mcp__email__search_emails",
        "mcp__email__read_email",
        "create_document",
    ]

    document = next(
        payload
        for tool_type, payload in executed
        if tool_type == "create_document"
    )

    content = document["content"]

    assert "synthetic body 1" in content
    assert "synthetic body 2" in content

    assert (
        content.index("synthetic body 1")
        < content.index("synthetic body 2")
    )

    assert "placeholder" not in content.lower()
    assert not content.startswith("# Retrieved emails")

    assert not any(
        "required_artifact_creation_failed" in chunk
        for chunk in chunks
    )


@pytest.mark.asyncio
async def test_exhaustive_document_replaces_model_summary_with_exact_unique_corpus(monkeypatch):
    targets = [
        {
            "uid": str(index + 1),
            "folder": "INBOX" if index < 31 else "[Provider]/All Mail",
            "account": "work",
        }
        for index in range(47)
    ]
    executed = []
    model_rounds = 0

    def batch_payload(batch):
        items = []
        for index, target in enumerate(batch):
            global_index = int(target["uid"]) - 1
            canonical_index = global_index if global_index < 31 else global_index - 31
            day = canonical_index + 1
            items.append({
                "index": index,
                "uid": target["uid"],
                "folder": target["folder"],
                "account": target["account"],
                "status": "success",
                "truncated": False,
                "resolved_uid": target["uid"],
                "resolved_folder": target["folder"],
                "resolved_message_id": f"<message-{canonical_index:02d}@example.test>",
                "subject": "Same subject" if canonical_index in {4, 5} else f"Subject {canonical_index}",
                "from": "Synthetic Sender",
                "from_address": "sender@example.test",
                "to": "recipient@example.test",
                "date": f"{day:02d} Jan 2026 10:00:00 +0000",
                "body": f"LOSSLESS BODY {canonical_index:02d}\nsecond line",
                "attachments": [],
            })
        return json.dumps({"batch": True, "items": items})

    async def fake_stream(_candidates, messages, **kwargs):
        nonlocal model_rounds
        model_rounds += 1
        if model_rounds == 1:
            call = {"name": "search_emails", "arguments": json.dumps({
                "query": "synthetic", "account": "work", "max_results": 20,
                "folders": ["INBOX", "Sent", "Archive"],
            })}
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        elif model_rounds == 2:
            call = {"name": "create_document", "arguments": json.dumps({
                "title": "All synthetic emails",
                "language": "markdown",
                "content": "A model-written summary containing only three messages.",
            })}
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        else:
            yield 'data: {"delta": "Document created."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        if block.tool_type == "create_document":
            title, language, content = block.content.split("\n", 2)
            payload = {"title": title, "language": language, "content": content}
        else:
            payload = json.loads(block.content)
        executed.append((block.tool_type, payload))
        if block.tool_type == "mcp__email__search_emails":
            assert payload["max_results"] == agent_loop._EXHAUSTIVE_EMAIL_SEARCH_MAX_RESULTS
            assert "cursor" not in payload and "offset" not in payload
            return block.tool_type, {
                "stdout": _synthetic_search_output(targets), "stderr": "", "exit_code": 0,
            }
        if block.tool_type == "mcp__email__read_email":
            return block.tool_type, {
                "stdout": batch_payload(payload["targets"]), "stderr": "", "exit_code": 0,
            }
        return block.tool_type, {
            "output": "SYNTHETIC_DOCUMENT_RESULT", "action": "create",
            "doc_id": "lossless-doc", "title": payload["title"],
            "language": payload["language"], "content": payload["content"],
            "version": 1, "exit_code": 0,
        }

    _patch_agent_loop_dependencies(monkeypatch, fake_stream, fake_execute)
    chunks = [chunk async for chunk in agent_loop.stream_agent_loop(
        "https://api.openai.com/v1",
        "gpt-4o",
        [{"role": "user", "content": (
            "Create a Word document with all emails involving synthetic, "
            "ordered chronologically."
        )}],
        max_rounds=20,
        relevant_tools={"search_emails", "read_email", "create_document"},
        owner="admin",
        _is_teacher_run=True,
    )]

    read_payloads = [payload for tool, payload in executed if tool == "mcp__email__read_email"]
    assert [len(payload["targets"]) for payload in read_payloads] == [20, 20, 7]
    document = next(payload for tool, payload in executed if tool == "create_document")
    content = document["content"]
    assert document["title"] == "All synthetic emails"
    assert content.count("<!-- odysseus-email-entry -->") == 31
    assert "model-written summary" not in content
    for index in range(31):
        identity = f"<message-{index:02d}@example.test>"
        assert content.count(identity) == 1
        assert content.count(f"LOSSLESS BODY {index:02d}") == 1
    assert content.index("<message-00@example.test>") < content.index("<message-30@example.test>")
    assert any("Document created." in chunk for chunk in chunks)


def test_required_email_document_block_preserves_explicit_nonlossless_title(monkeypatch):
    """An explicit requested title must survive the forced email-document path."""
    import json

    calls = []

    def capture_tool_block(name, arguments):
        calls.append((name, arguments))
        return object()

    monkeypatch.setattr(
        agent_loop,
        "function_call_to_tool_block",
        capture_tool_block,
    )

    items = [{
        "subject": "Example subject",
        "from": "sender@example.invalid",
        "date": "2026-09-04",
        "body": "Example retrieved body.",
    }]

    requested_title = (
        "Gmail OAuth E2E - AutoDS - 2026-09-05"
    )

    result = agent_loop._required_email_document_block(
        items,
        "Create a brief document",
        title=requested_title,
        lossless=False,
        discovery_complete=True,
    )

    assert result is not None
    assert len(calls) == 1

    tool_name, raw_arguments = calls[0]

    assert tool_name == "create_document"

    payload = json.loads(raw_arguments)

    assert payload["title"] == requested_title

    calls.clear()

    result = agent_loop._required_email_document_block(
        items,
        "Create a document",
        title=None,
        lossless=False,
        discovery_complete=True,
    )

    assert result is not None
    assert len(calls) == 1

    tool_name, raw_arguments = calls[0]

    assert tool_name == "create_document"

    payload = json.loads(raw_arguments)

    assert payload["title"] == "Retrieved emails"



def test_required_email_document_block_extracts_exact_title_from_user_request(monkeypatch):
    """Supervisor fallback must preserve an exact title from the original request."""
    import json

    calls = []

    def capture_tool_block(name, arguments):
        calls.append((name, arguments))
        return object()

    monkeypatch.setattr(
        agent_loop,
        "function_call_to_tool_block",
        capture_tool_block,
    )

    requested_title = (
        "Gmail OAuth E2E - AutoDS - c7dfc76 - 2026-09-05"
    )

    user_request = f"""
Busca al meu Gmail el correu relacionat amb AutoDS.

El títol del document ha de ser exactament:

{requested_title}

No substitueixis aquest títol per "Retrieved emails".
"""

    items = [{
        "subject": "Has compartido algunos datos de tu cuenta de Google con AutoDS",
        "from": "Google",
        "date": "Fri, 04 Sep 2026 07:36:27 -0700",
        "body": "Example retrieved body.",
    }]

    result = agent_loop._required_email_document_block(
        items,
        user_request,
        title=None,
        lossless=False,
        discovery_complete=True,
    )

    assert result is not None
    assert len(calls) == 1

    tool_name, raw_arguments = calls[0]

    assert tool_name == "create_document"

    payload = json.loads(raw_arguments)

    assert payload["title"] == requested_title


def test_explicit_document_title_extractor_is_conservative():
    requested = "Exact document title"

    assert (
        agent_loop._explicit_document_title_from_request(
            "The document title must be exactly:\n\n"
            + requested
        )
        == requested
    )

    assert (
        agent_loop._explicit_document_title_from_request(
            'El título del documento debe ser exactamente: "Título exacto"'
        )
        == "Título exacto"
    )

    assert (
        agent_loop._explicit_document_title_from_request(
            "Create a document about AutoDS."
        )
        is None
    )



def test_targeted_email_document_search_hits_are_candidates_not_required():
    first = ("101", "", "INBOX", "gmail")
    second = ("102", "", "INBOX", "gmail")
    found = {first, second}

    assert agent_loop._email_document_search_targets_for_retrieval(
        found,
        corpus=False,
        discovery_complete=True,
    ) == set()

    assert agent_loop._email_document_search_targets_for_retrieval(
        found,
        corpus=True,
        discovery_complete=True,
    ) == found



def test_targeted_email_document_selected_read_defines_required_result_set():
    selected = ("101", "", "INBOX", "gmail")

    assert agent_loop._email_document_read_targets_for_retrieval(
        {selected},
        {selected},
        corpus=False,
    ) == {selected}

    assert agent_loop._email_document_read_targets_for_retrieval(
        {selected},
        {selected},
        corpus=True,
    ) == set()


def test_email_document_corpus_mode_is_separate_from_exhaustive_search():
    bounded = [
        "Create a document from ten synthetic emails.",
        "Search fifty emails and create a document from them.",
        "Create a Word document from the synthetic emails, ordered by date.",
        "Create a document summarizing all ten emails.",
    ]

    for request in bounded:
        assert agent_loop._requires_email_document_corpus(request)

    assert not agent_loop._has_exhaustive_email_intent(
        "Create a document from ten synthetic emails."
    )

    assert not agent_loop._requires_email_document_corpus(
        "Search 10 emails to identify the relevant one and create a brief document from it."
    )

    assert not agent_loop._requires_email_document_corpus(
        "Create a document from one email."
    )


def test_targeted_single_candidate_is_auto_selected_only_when_discovery_complete():
    selected = ("101", "", "INBOX", "gmail")

    assert agent_loop._email_document_search_targets_for_retrieval(
        {selected},
        corpus=False,
        discovery_complete=True,
    ) == {selected}

    assert agent_loop._email_document_search_targets_for_retrieval(
        {selected},
        corpus=False,
        discovery_complete=False,
    ) == set()


def test_lossless_email_document_excludes_explicit_summary_transformation():
    assert agent_loop._requires_email_document_corpus(
        "Create a document summarizing all emails involving Alex."
    )

    assert not agent_loop._requires_lossless_email_document(
        "Create a document summarizing all emails involving Alex."
    )

    assert agent_loop._requires_lossless_email_document(
        "Create a Word document with all emails involving Alex."
    )


@pytest.mark.asyncio
async def test_targeted_multi_result_selects_one_read_before_document_creation(monkeypatch):
    targets = [
        {"uid": "0", "folder": "INBOX", "account": "work"},
        {"uid": "1", "folder": "INBOX", "account": "work"},
        {"uid": "2", "folder": "INBOX", "account": "work"},
    ]

    selected = targets[1]

    model_rounds = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_rounds.append({
            "messages": json.loads(json.dumps(messages)),
            "schemas": _schema_names(kwargs.get("tools")),
            "tool_choice_name": kwargs.get("tool_choice_name"),
        })

        round_number = len(model_rounds)

        if round_number == 1:
            assert "create_document" not in model_rounds[-1]["schemas"]

            call = {
                "name": "search_emails",
                "arguments": json.dumps({
                    "query": "synthetic",
                    "account": "work",
                }),
            }

            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'

        elif round_number == 2:
            serialized = json.dumps(messages)

            assert "create_document" not in model_rounds[-1]["schemas"]
            assert "read_email" in model_rounds[-1]["schemas"]

            assert "synthetic body 0" not in serialized
            assert "synthetic body 1" not in serialized
            assert "synthetic body 2" not in serialized

            call = {
                "name": "read_email",
                "arguments": json.dumps({
                    "targets": [selected],
                }),
            }

            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'

        elif round_number == 3:
            serialized = json.dumps(messages)

            assert "synthetic body 1" in serialized
            assert "synthetic body 0" not in serialized
            assert "synthetic body 2" not in serialized

            assert model_rounds[-1]["schemas"] == {"create_document"}
            assert model_rounds[-1]["tool_choice_name"] == "create_document"

            call = {
                "name": "create_document",
                "arguments": json.dumps({
                    "title": "Selected synthetic email",
                    "language": "markdown",
                    "content": (
                        "# Selected synthetic email\n\n"
                        "Brief summary of synthetic body 1."
                    ),
                }),
            }

            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'

        else:
            assert "SYNTHETIC_DOCUMENT_RESULT" in json.dumps(messages)
            yield 'data: {"delta": "Selected document created."}\n\n'

        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        if block.tool_type == "create_document":
            title, language, content = block.content.split("\n", 2)

            payload = {
                "title": title,
                "language": language,
                "content": content,
            }
        else:
            payload = json.loads(block.content)

        executed.append((block.tool_type, payload))

        if block.tool_type == "mcp__email__search_emails":
            return block.tool_type, {
                "output": _synthetic_search_output(targets),
                "exit_code": 0,
            }

        if block.tool_type == "mcp__email__read_email":
            assert payload == {
                "targets": [selected],
            }

            return block.tool_type, {
                "output": _synthetic_batch_output([selected]),
                "exit_code": 0,
            }

        assert block.tool_type == "create_document"

        return block.tool_type, {
            "output": "SYNTHETIC_DOCUMENT_RESULT",
            "action": "create",
            "doc_id": "selected-synthetic-doc",
            "title": payload["title"],
            "language": payload["language"],
            "content": payload["content"],
            "version": 1,
            "exit_code": 0,
        }

    _patch_agent_loop_dependencies(
        monkeypatch,
        fake_stream,
        fake_execute,
    )

    request = (
        "Search synthetic email candidates to identify the relevant one. "
        "Read only that message and create a brief document summarizing it."
    )

    assert not agent_loop._has_exhaustive_email_intent(request)
    assert not agent_loop._requires_email_document_corpus(request)
    assert not agent_loop._requires_lossless_email_document(request)

    chunks = [
        chunk
        async for chunk in agent_loop.stream_agent_loop(
            "https://api.openai.com/v1",
            "gpt-4o",
            [{
                "role": "user",
                "content": request,
            }],
            max_rounds=20,
            relevant_tools={
                "search_emails",
                "read_email",
                "create_document",
            },
            owner="admin",
            _is_teacher_run=True,
        )
    ]

    tool_types = [
        tool_type
        for tool_type, _payload in executed
    ]

    assert tool_types == [
        "mcp__email__search_emails",
        "mcp__email__read_email",
        "create_document",
    ]

    read_payloads = [
        payload
        for tool_type, payload in executed
        if tool_type == "mcp__email__read_email"
    ]

    assert read_payloads == [
        {"targets": [selected]}
    ]

    document = next(
        payload
        for tool_type, payload in executed
        if tool_type == "create_document"
    )

    assert "synthetic body 1" in document["content"]
    assert "synthetic body 0" not in document["content"]
    assert "synthetic body 2" not in document["content"]

    assert len(model_rounds) == 4

    assert any(
        "Selected document created." in chunk
        for chunk in chunks
    )


_TARGETED_WORKFLOW = "Search email candidates, identify the relevant one, and create a brief document."
_LITERAL_WORKFLOW = "Create a document containing all emails."
_SUMMARY_WORKFLOW = "Summarize all emails in a document."


@pytest.mark.parametrize("prompt,active", [
    ("List all emails.", False), ("Archive all emails.", False),
    ("Search all emails for purchase history.", False), ("Show email history.", False),
    ("Create a list containing all emails.", True),
    ("Create an archive containing all emails.", True),
    ("Create a document with the email history.", True),
    ("Generate a listing of the selected emails.", True),
])
def test_last_findings_artifact_activation(prompt, active):
    assert agent_loop._email_document_workflow_semantics([
        {"role": "user", "content": prompt},
    ]).active is active
    # Explicit new searches must also supersede an unfinished artifact.
    if not active:
        assert not agent_loop._email_document_workflow_semantics([
            {"role": "user", "content": _LITERAL_WORKFLOW},
            {"role": "user", "content": prompt},
        ]).active


@pytest.mark.parametrize("prompt", [
    "Use one paragraph per email.", "Use only one heading per email.",
    "Use one sentence for each message.", "Make one section per email.",
])
@pytest.mark.parametrize("origin,source", [
    (_TARGETED_WORKFLOW, (False, False)), (_LITERAL_WORKFLOW, (True, True)),
])
def test_last_findings_singular_formatting(prompt, origin, source):
    assert agent_loop._email_source_set_override(prompt) is None
    state = agent_loop._email_document_workflow_semantics([
        {"role": "user", "content": origin}, {"role": "user", "content": prompt},
    ])
    assert state.active
    assert (state.corpus, state.exhaustive) == source


@pytest.mark.parametrize("prompt", [
    "use one email", "use one message", "use only one email", "use only that one",
    "use the relevant one", "use only the relevant one", "select one message", "read only UID 10595",
])
def test_last_findings_singular_selection(prompt):
    state = agent_loop._email_document_workflow_semantics([
        {"role": "user", "content": _LITERAL_WORKFLOW}, {"role": "user", "content": prompt},
    ])
    assert state == (True, False, False, False)


@pytest.mark.parametrize("items", [
    "ten unread emails", "ten new emails", "ten recent emails", "ten selected emails",
    "diez correos nuevos", "diez correos no leídos", "diez correos seleccionados",
    "deu correus nous", "deu correus no llegits", "deu correus seleccionats",
    "diez nuevos correos", "deu nous correus",
])
def test_last_findings_email_count_modifiers(items):
    assert agent_loop._email_source_set_override(items) == (True, False)
    state = agent_loop._email_document_workflow_semantics([
        {"role": "user", "content": _TARGETED_WORKFLOW}, {"role": "user", "content": items},
    ])
    assert state == (True, True, False, False)
    assert agent_loop._email_document_workflow_semantics([
        {"role": "user", "content": "Create an archive from " + items},
    ]).active


@pytest.mark.parametrize("items", [
    "ten paragraphs from emails", "ten new paragraphs about emails",
    "five selected headings from emails",
])
def test_last_findings_count_modifier_negatives(items):
    assert agent_loop._email_source_set_override(items) is None


@pytest.mark.parametrize("raw,success", [
    ({"exit_code": 0, "action": "create"}, False),
    ({"exit_code": 1, "doc_id": "abc"}, False),
    ({"exit_code": 0, "doc_id": "abc"}, True),
    ({"exit_code": 0, "doc_id": "abc", "error": "synthetic"}, False),
    ({"exit_code": 0, "doc_id": "abc", "blocked": True}, False),
])
@pytest.mark.asyncio
async def test_last_findings_exact_create_transport_and_persistence(monkeypatch, raw, success):
    rounds = 0
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        nonlocal rounds
        rounds += 1
        if rounds == 1:
            name, payload = "read_email", {"uid": "10595"}
        elif rounds == 2:
            name, payload = "create_document", {"title": "Synthetic", "language": "markdown", "content": "Body"}
        else:
            assert (kwargs.get("tool_choice_name") == "create_document") is not success
            if success and raw.get("action") == "create":
                tool_reply = messages[-1]["content"]
                assert "Error" not in tool_reply
                if "content" in raw:
                    assert raw["content"] in tool_reply
                else:
                    assert "Document created:" in tool_reply
            if not success:
                assert "Error" in json.dumps(messages)
                assert "Document created:" not in json.dumps(messages)
            yield 'data: {"delta": "Completion observed."}\n\n'
            yield "data: [DONE]\n\n"
            return
        call = {"name": name, "arguments": json.dumps(payload)}
        yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        if block.tool_type == "mcp__email__read_email":
            return block.tool_type, {"output": "Subject: Synthetic\n\nUsable body", "exit_code": 0}
        assert block.tool_type == "create_document"
        # Exact transport shape: no output key to bypass formatter branches.
        return block.tool_type, dict(raw)

    _patch_agent_loop_dependencies(monkeypatch, fake_stream, fake_execute)
    chunks = [chunk async for chunk in agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", [{"role": "user", "content": _TARGETED_WORKFLOW}],
        relevant_tools={"search_emails", "read_email", "create_document"},
        owner="admin", _is_teacher_run=True, max_rounds=3,
    )]
    events = [json.loads(line[6:]) for chunk in chunks for line in chunk.splitlines()
              if line.startswith("data: {")]
    assert rounds == 3
    assert executed.count("create_document") == 1
    updates = [event for event in events if event.get("type") == "doc_update"]
    assert bool(updates) is (success and "content" in raw and "version" in raw)
    outputs = [event for event in events if event.get("type") == "tool_output" and event.get("tool") == "create_document"]
    assert len(outputs) == 1
    if success and not ("content" in raw and "version" in raw):
        assert "doc_id" not in outputs[0]
        assert "document_content" not in outputs[0]
        assert "document_version" not in outputs[0]
    if not success:
        assert "Document created" not in outputs[0]["output"]
        assert "document_action" not in outputs[0]
        assert "doc_id" not in outputs[0]
    persisted = [entry for event in events if event.get("type") == "metrics"
                 for entry in event.get("data", {}).get("tool_events", [])
                 if entry.get("tool") == "create_document"]
    assert len(persisted) == 1
    event = persisted[0]
    assert event["blocked"] is bool(raw.get("blocked"))
    assert event["had_error"] is not success
    assert "error" not in event
    state = agent_loop._email_document_workflow_semantics([
        {"role": "user", "content": _TARGETED_WORKFLOW},
        {"role": "assistant", "metadata": {"tool_events": persisted}},
        {"role": "user", "content": "continue"},
    ])
    assert state.active is not success


@pytest.mark.parametrize("raw", [
    {"exit_code": 0, "action": "create"},
    {"exit_code": 1, "action": "create", "doc_id": "abc"},
    {"exit_code": 0, "action": "create", "doc_id": "abc", "blocked": True},
])
def test_last_findings_formatter_exact_invalid_create(raw):
    from src.tool_execution import format_tool_result
    formatted = format_tool_result("create_document", raw)
    assert "Error" in formatted
    assert "Document created:" not in formatted


def test_last_findings_formatter_preserves_other_tool_output():
    from src.tool_execution import format_tool_result
    formatted = format_tool_result("other", {"action": "create", "output": "Other tool result", "exit_code": 0}, tool="other")
    assert "Other tool result" in formatted
    assert "Document creation failed" not in formatted


@pytest.mark.parametrize("selection", [
    "Actually use only the relevant one.", "only the relevant one",
    "use the relevant one", "use that one", "use only that one",
    "read only that message", "read only UID 10595",
    "select the relevant message", "identify the relevant one",
])
def test_final_findings_source_narrowing(selection):
    history = [{"role": "user", "content": _LITERAL_WORKFLOW},
               {"role": "user", "content": selection}]
    assert agent_loop._email_source_set_override(selection) == (False, False)
    assert agent_loop._email_document_workflow_semantics(history) == (True, False, False, False)


@pytest.mark.parametrize("selection", [
    "all of them", "use all of them", "all ten", "use all ten",
    "these ten", "those ten", "ten of them", "use these ten emails",
    "use these ten messages", "use ten of them", "those ten emails", "the ten messages",
])
def test_final_findings_source_corpus(selection):
    history = [{"role": "user", "content": _TARGETED_WORKFLOW},
               {"role": "user", "content": selection}]
    assert agent_loop._email_source_set_override(selection) == (True, False)
    assert agent_loop._email_document_workflow_semantics(history) == (True, True, False, False)


@pytest.mark.parametrize("formatting", [
    "Write two bullet points from emails.", "Write three paragraphs about the emails.",
    "Create five headings from these emails.", "Summarize the emails in ten sentences.",
    "Make two sections from the messages.",
    "Write all ten paragraphs from emails.",
])
@pytest.mark.parametrize("origin,source", [
    (_TARGETED_WORKFLOW, (False, False)), (_LITERAL_WORKFLOW, (True, True)),
])
def test_final_findings_quantity_is_artifact_structure(formatting, origin, source):
    assert agent_loop._email_source_set_override(formatting) is None
    state = agent_loop._email_document_workflow_semantics([
        {"role": "user", "content": origin}, {"role": "user", "content": formatting},
    ])
    assert state.active
    assert (state.corpus, state.exhaustive) == source


@pytest.mark.parametrize("prompt,active", [
    ("Search all emails for invoices.", False),
    ("Create a document from all emails.", True),
    ("Create a document from these ten emails.", True),
    ("Summarize all emails in a document.", True),
    ("Search the candidates, read the relevant one, and create a brief document.", True),
    ("Create an archive from all emails.", True),
])
def test_final_findings_activation_requires_artifact(prompt, active):
    assert not agent_loop._email_document_workflow_semantics([
        {"role": "user", "content": "all emails"},
    ]).active
    for prefix in ([], [{"role": "user", "content": _LITERAL_WORKFLOW}]):
        state = agent_loop._email_document_workflow_semantics(
            prefix + [{"role": "user", "content": prompt}],
        )
        assert state.active is active


@pytest.mark.parametrize("prompt,document_active", [
    ("Search all emails for invoices.", False),
    ("Search all emails, read only UID 10595, and create the requested document.", True),
])
@pytest.mark.asyncio
async def test_last_findings_email_only_exhaustive_search_stream(monkeypatch, caplog, prompt, document_active):
    import logging

    caplog.set_level(logging.INFO, logger=agent_loop.__name__)
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        assert "EMAIL DOCUMENT RETRIEVAL GATE" not in json.dumps(messages)
        assert ("EMAIL DOCUMENT TARGETED RETRIEVAL GATE" in json.dumps(messages)) is document_active
        assert kwargs.get("tool_choice_name") != "create_document"
        if not executed:
            assert "search_emails" in _schema_names(kwargs.get("tools"))
            call = {"name": "search_emails", "arguments": json.dumps({"query": "invoices", "max_results": 20})}
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        else:
            yield 'data: {"delta": "Found matching invoices."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        assert block.tool_type == "mcp__email__search_emails"
        assert json.loads(block.content)["max_results"] == agent_loop._EXHAUSTIVE_EMAIL_SEARCH_MAX_RESULTS
        return block.tool_type, {"output": _synthetic_search_output([{"uid": "10595"}, {"uid": "10596"}]), "exit_code": 0}

    _patch_agent_loop_dependencies(monkeypatch, fake_stream, fake_execute)
    history = [{"role": "user", "content": _LITERAL_WORKFLOW},
               {"role": "user", "content": prompt}]
    chunks = [chunk async for chunk in agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", list(history), conversation_history=history,
        relevant_tools={"search_emails", "read_email", "create_document"},
        owner="admin", _is_teacher_run=True,
    )]
    assert executed == ["mcp__email__search_emails"]
    assert ("email document retrieval gate active" in caplog.text) is document_active
    assert "requiring post-readiness document creation" not in caplog.text
    assert any("Found matching invoices." in chunk for chunk in chunks)


@pytest.mark.parametrize("result,success", [
    ({"exit_code": 0, "doc_id": "abc"}, True),
    ({"exit_code": 0, "action": "create"}, False),
    ({"exit_code": 1, "doc_id": "abc"}, False),
    ({"exit_code": 0, "doc_id": ""}, False),
    ({"exit_code": 0, "doc_id": "abc", "blocked": True}, False),
    ({"exit_code": 0, "doc_id": "abc", "error": "synthetic failure"}, False),
    ({"exit_code": 0, "doc_id": "   "}, False),
    ({"exit_code": 0, "doc_id": {}}, False),
])
@pytest.mark.asyncio
async def test_final_findings_completion_live_and_history(monkeypatch, caplog, result, success):
    import logging

    caplog.set_level(logging.INFO, logger=agent_loop.__name__)
    assert agent_loop._email_document_creation_succeeded("create_document", result) is success
    for tool in ("update_document", "edit_document"):
        assert not agent_loop._email_document_creation_succeeded(tool, result)
    state = agent_loop._email_document_workflow_semantics([
        {"role": "user", "content": _TARGETED_WORKFLOW},
        {"role": "assistant", "metadata": {"tool_events": [{"tool": "create_document", **result}]}},
    ])
    assert state.active is not success
    rounds = 0
    created = 0

    async def fake_stream(_candidates, messages, **kwargs):
        nonlocal rounds
        rounds += 1
        if rounds == 1:
            name, payload = "read_email", {"uid": "10595"}
        elif rounds == 2:
            assert kwargs.get("tool_choice_name") == "create_document"
            name, payload = "create_document", {"title": "Synthetic", "language": "markdown", "content": "Body"}
        else:
            assert (kwargs.get("tool_choice_name") == "create_document") is not success
            # Stop at the first post-result round: this observes pending directly
            # through the production tool-choice obligation, before retry logic.
            yield 'data: {"delta": "Observed completion boundary."}\n\n'
            return
        call = {"name": name, "arguments": json.dumps(payload)}
        yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        nonlocal created
        if block.tool_type == "mcp__email__read_email":
            return block.tool_type, {"output": "Subject: Synthetic\n\nUsable body", "exit_code": 0}
        assert block.tool_type == "create_document"
        created += 1
        return block.tool_type, {"output": "Synthetic result", **result}

    _patch_agent_loop_dependencies(monkeypatch, fake_stream, fake_execute)
    stream = agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", [{"role": "user", "content": _TARGETED_WORKFLOW}],
        relevant_tools={"search_emails", "read_email", "create_document"},
        owner="admin", _is_teacher_run=True,
    )
    async for chunk in stream:
        if "Observed completion boundary." in chunk:
            break
    await stream.aclose()
    assert rounds == 3
    assert created == 1
    assert ("required post-readiness artifact created" in caplog.text) is success


@pytest.mark.parametrize("origin,followup,expected", [
    (_TARGETED_WORKFLOW, "Yes. Create the document from the emails discussed.", (True, False, False, False)),
    (_TARGETED_WORKFLOW, "yes, do that", (True, False, False, False)),
    ("Crea un documento con todos los correos.", "Continue.", (True, True, True, True)),
    ("Crea un documento con diez correos.", "Continue.", (True, True, False, False)),
    (_TARGETED_WORKFLOW, "continue", (True, False, False, False)),
    (_LITERAL_WORKFLOW, "yes", (True, True, True, True)),
    ("Create a document from ten emails.", "Use these two instead.", (True, True, False, False)),
    (_TARGETED_WORKFLOW, "read that one", (True, False, False, False)),
    (_TARGETED_WORKFLOW, "now make the document", (True, False, False, False)),
    (_TARGETED_WORKFLOW, "Create a document from these ten emails.", (True, True, False, False)),
    (_TARGETED_WORKFLOW, "Create a document containing all emails.", (True, True, True, True)),
    (_LITERAL_WORKFLOW, _TARGETED_WORKFLOW, (True, False, False, False)),
    (_LITERAL_WORKFLOW, "Instead, search email candidates, identify the relevant one, and create a brief document from that message.", (True, False, False, False)),
    (_LITERAL_WORKFLOW, "Create a document from these ten emails.", (True, True, False, False)),
    (_TARGETED_WORKFLOW, "Actually use all of them.", (True, True, False, False)),
    (_TARGETED_WORKFLOW, "No, use these ten instead.", (True, True, False, False)),
    (_TARGETED_WORKFLOW, "Use these ten emails instead.", (True, True, False, False)),
    (_TARGETED_WORKFLOW, "Use these 12 instead.", (True, True, False, False)),
    (_TARGETED_WORKFLOW, "Use all returned emails.", (True, True, False, False)),
    (_TARGETED_WORKFLOW, "Use every email.", (True, True, True, False)),
    (_LITERAL_WORKFLOW, "Use only that one.", (True, False, False, False)),
    (_LITERAL_WORKFLOW, "Read only that message.", (True, False, False, False)),
    (_LITERAL_WORKFLOW, "Read only UID 10595.", (True, False, False, False)),
    (_LITERAL_WORKFLOW, "Instead identify the relevant one.", (True, False, False, False)),
    (_LITERAL_WORKFLOW, "Select the relevant message.", (True, False, False, False)),
    (_LITERAL_WORKFLOW, "Yes. Make it a brief summary.", (True, True, True, False)),
    (_SUMMARY_WORKFLOW, "Yes. Copy them verbatim into a full archive.", (True, True, True, True)),
    (_LITERAL_WORKFLOW, "Make it a brief summary.", (True, True, True, False)),
    ("Summarize all emails.", "Copy them verbatim into a full archive.", (True, True, True, True)),
    (_LITERAL_WORKFLOW, "Forget that; search for invoices.", (False, False, False, False)),
    (_LITERAL_WORKFLOW, "What is the capital of France?", (False, False, False, False)),
    (_LITERAL_WORKFLOW, "Search email invoices.", (False, False, False, False)),
    (_LITERAL_WORKFLOW, "Write a document about gardening.", (False, False, False, False)),
    (_LITERAL_WORKFLOW, "Yes. Write a document about gardening.", (False, False, False, False)),
    (_LITERAL_WORKFLOW, "Yes. What is the capital of France?", (False, False, False, False)),
    (_SUMMARY_WORKFLOW, "Yes. Create a document containing all emails.", (True, True, True, True)),
])
def test_email_workflow_semantic_transitions(origin, followup, expected):
    history = [
        {"role": "user", "content": origin},
        {"role": "assistant", "content": "Which one should I read?"},
        {"role": "user", "content": followup},
    ]
    assert agent_loop._email_document_workflow_semantics(history) == expected


@pytest.mark.parametrize("event,expected", [
    ({"tool": "create_document", "exit_code": 0, "doc_id": "synthetic"}, (False, False, False, False)),
    ({"tool": "create_document", "exit_code": 1}, (True, True, True, False)),
    ({"tool": "create_document", "exit_code": 1, "doc_id": "synthetic"}, (True, True, True, False)),
    ({"tool": "create_document", "exit_code": 0}, (True, True, True, False)),
    ({"tool": "create_document", "doc_id": "synthetic"}, (True, True, True, False)),
    (None, (True, True, True, False)),
])
def test_email_workflow_completion_requires_successful_artifact(event, expected):
    history = [
        {"role": "user", "content": _LITERAL_WORKFLOW},
        {"role": "assistant", "content": "Document created.",
         "metadata": {"tool_events": [event] if event else []}},
        {"role": "user", "content": "Yes. Make it a brief summary."},
    ]
    assert agent_loop._email_document_workflow_semantics(history) == expected


@pytest.mark.parametrize("intervening", [
    [{"role": "assistant", "metadata": {"tool_events": [
        {"tool": "create_document", "exit_code": 0, "doc_id": "synthetic"},
    ]}}],
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Search email invoices."}],
    [{"role": "user", "content": "Write a document about gardening."}],
])
def test_email_workflow_boundaries_do_not_reuse_old_query(intervening):
    history = [{"role": "user", "content": _LITERAL_WORKFLOW}] + intervening + [
        {"role": "assistant", "content": "Which one should I read?"},
        {"role": "user", "content": "Yes. Read only UID 10595 and create the requested document."},
    ]
    assert agent_loop._has_exhaustive_email_intent(agent_loop._recent_context_for_retrieval(history))
    # Also prove the reset without a targeted override that could mask leakage.
    generic_followup = history[:-1] + [{"role": "user", "content": "Yes. Make the requested document."}]
    assert agent_loop._email_document_workflow_semantics(generic_followup) == (False, False, False, False)
    assert agent_loop._email_document_workflow_semantics(history) == (True, False, False, False)
    history.append({"role": "user", "content": _TARGETED_WORKFLOW})
    assert agent_loop._email_document_workflow_semantics(history) == (True, False, False, False)
    history.append({"role": "user", "content": _LITERAL_WORKFLOW})
    assert agent_loop._email_document_workflow_semantics(history) == (True, True, True, True)


@pytest.mark.parametrize("injected", [
    {"role": "user", "content": _LITERAL_WORKFLOW, "metadata": {"trusted": False}},
    {"role": "user", "content": [{"type": "text", "text": _LITERAL_WORKFLOW}], "metadata": {"trusted": False}},
    {"role": "tool", "content": _LITERAL_WORKFLOW},
    {"role": "user", "content": "[Tool execution results]\n" + _LITERAL_WORKFLOW},
    {"role": "user", "content": [{"type": "text", "text": "[Tool execution results]\n" + _LITERAL_WORKFLOW}]},
    {"role": "assistant", "content": _LITERAL_WORKFLOW},
])
def test_email_workflow_ignores_untrusted_and_tool_content(injected):
    history = [{"role": "user", "content": _TARGETED_WORKFLOW}, injected,
               {"role": "user", "content": "Yes. Create the requested document."}]
    assert agent_loop._email_document_workflow_semantics(history) == (True, False, False, False)
    assert agent_loop._email_document_workflow_semantics([injected]) == (False, False, False, False)


@pytest.mark.asyncio
async def test_targeted_two_turn_direct_uid_reentry_creates_synthesized_document(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger=agent_loop.__name__)
    targets = [{"uid": "10594"}, {"uid": "10595"}, {"uid": "10596"}]
    origin = (
        "Search synthetic email candidates and identify the relevant one. "
        "Read only that message and create a brief document summarizing it."
    )
    # This follow-up independently triggers the production corpus classifier.
    # Its length also removes the origin from the capped retrieval query.
    followup = (
        "Yes. Read only UID 10595. Do not read any other email. "
        "Then create the requested document from the emails discussed. "
        + "Keep the requested wording concise. " * 25
    )
    assert agent_loop._requires_email_document_corpus(followup)
    executed = []
    rounds = []
    reentered = False
    synthesized = "# Selected finding\n\nThe selected message describes a synthetic delivery update."

    async def fake_stream(_candidates, messages, **kwargs):
        if executed and not reentered:
            yield 'data: {"delta": "Which one should I read?"}\n\n'
            return
        rounds.append(_schema_names(kwargs.get("tools")))
        serialized = json.dumps(messages)
        if len(rounds) == 1:
            assert "create_document" not in rounds[-1]
            name, payload = "search_emails", {"query": "synthetic"}
        elif len(rounds) == 2:
            assert "create_document" not in rounds[-1]
            assert "EMAIL DOCUMENT TARGETED RETRIEVAL GATE" in serialized
            name, payload = "read_email", {"uid": "10595"}
        elif len(rounds) == 3:
            assert rounds[-1] == {"create_document"}
            assert kwargs.get("tool_choice_name") == "create_document"
            assert "SYNTHETIC SELECTED BODY" in serialized
            assert "UNRELATED BODY" not in serialized
            name, payload = "create_document", {
                "title": "Selected finding", "language": "markdown", "content": synthesized,
            }
        else:
            yield 'data: {"delta": "Document created."}\n\n'
            yield "data: [DONE]\n\n"
            return
        yield f'data: {json.dumps({"type": "tool_calls", "calls": [{"name": name, "arguments": json.dumps(payload)}]})}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        if block.tool_type == "create_document":
            title, language, content = block.content.split("\n", 2)
            payload = {"title": title, "language": language, "content": content}
        else:
            payload = json.loads(block.content)
        executed.append((block.tool_type, payload))
        if block.tool_type == "mcp__email__search_emails":
            return block.tool_type, {"output": _synthetic_search_output(targets), "exit_code": 0}
        if block.tool_type == "mcp__email__read_email":
            assert payload == {"uid": "10595"}
            return block.tool_type, {
                "output": "**Subject:** Synthetic update\n**From:** sender@example.test\n\nSYNTHETIC SELECTED BODY",
                "exit_code": 0,
            }
        assert block.tool_type == "create_document"
        assert payload["content"] == synthesized
        assert "Retrieved emails" not in payload["content"]
        return block.tool_type, {
            "output": "SYNTHETIC_DOCUMENT_RESULT", "action": "create",
            "doc_id": "synthetic-reentry", "version": 1, "exit_code": 0, **payload,
        }

    _patch_agent_loop_dependencies(monkeypatch, fake_stream, fake_execute)
    history = [{"role": "user", "content": origin}]
    options = dict(
        relevant_tools={"search_emails", "read_email", "create_document"},
        owner="admin", _is_teacher_run=True,
    )
    # Closing the stream after the completed search emulates an interrupted
    # invocation; the next invocation has only route-owned conversation history.
    first = agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", list(history), **options,
    )
    async for chunk in first:
        if "Which one should I read?" in chunk:
            break
    await first.aclose()
    assert [name for name, _ in executed] == ["mcp__email__search_emails"]
    assert "targeted email search candidates=3" in caplog.text
    reentered = True
    history.extend([
        {"role": "assistant", "content": (
            _synthetic_search_output(targets) + "\nWhich one should I read?"
        )},
        {"role": "user", "content": followup},
    ])
    chunks = [chunk async for chunk in agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", list(history),
        conversation_history=history, max_rounds=20, **options,
    )]
    assert [name for name, _ in executed] == [
        "mcp__email__search_emails", "mcp__email__read_email", "create_document",
    ]
    assert executed[1][1] == {"uid": "10595"}
    assert executed[2][1]["content"] == synthesized
    assert "email retrieval progress resolved=1/1 usable=1" in caplog.text
    assert "state=retrieval_ready targets=1 usable=1" in caplog.text
    assert len(rounds) == 4


@pytest.mark.parametrize("origin,followup,lossless", [
    (_LITERAL_WORKFLOW, "Yes. Make it a brief summary.", False),
    (_SUMMARY_WORKFLOW, "Yes. Copy them verbatim into a full archive.", True),
])
@pytest.mark.asyncio
async def test_email_workflow_transformation_reentry_controls_document_body(monkeypatch, origin, followup, lossless):
    targets = [{"uid": "10595"}, {"uid": "10596"}]
    executed = []
    model_rounds = 0
    synthesis = "A concise model-mediated synthesis."

    async def fake_stream(_candidates, messages, **kwargs):
        nonlocal model_rounds
        model_rounds += 1
        if model_rounds == 1:
            name, arguments = "search_emails", {"query": "synthetic", "max_results": 20}
        elif model_rounds == 2:
            assert _schema_names(kwargs.get("tools")) == {"create_document"}
            assert kwargs.get("tool_choice_name") == "create_document"
            assert "synthetic body 10595" in json.dumps(messages)
            assert "synthetic body 10596" in json.dumps(messages)
            name, arguments = "create_document", {
                "title": "Exact synthetic title", "language": "markdown", "content": synthesis,
            }
        else:
            yield 'data: {"delta": "Document created."}\n\n'
            yield "data: [DONE]\n\n"
            return
        yield f'data: {json.dumps({"type": "tool_calls", "calls": [{"name": name, "arguments": json.dumps(arguments)}]})}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        if block.tool_type == "create_document":
            title, language, content = block.content.split("\n", 2)
            payload = {"title": title, "language": language, "content": content}
        else:
            payload = json.loads(block.content)
        executed.append((block.tool_type, payload))
        if block.tool_type == "mcp__email__search_emails":
            assert payload["max_results"] == agent_loop._EXHAUSTIVE_EMAIL_SEARCH_MAX_RESULTS
            return block.tool_type, {"output": _synthetic_search_output(targets), "exit_code": 0}
        if block.tool_type == "mcp__email__read_email":
            assert payload["targets"] == [{"uid": t["uid"], "folder": "INBOX"} for t in targets]
            return block.tool_type, {"output": _synthetic_batch_output(targets), "exit_code": 0}
        assert block.tool_type == "create_document"
        return block.tool_type, {"output": "SYNTHETIC_DOCUMENT_RESULT", "exit_code": 0,
                                 "action": "create", "doc_id": "synthetic", "version": 1, **payload}

    _patch_agent_loop_dependencies(monkeypatch, fake_stream, fake_execute)
    history = [{"role": "user", "content": origin},
               {"role": "assistant", "content": "What should the document contain?"},
               {"role": "user", "content": followup}]
    chunks = [chunk async for chunk in agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", list(history), conversation_history=history,
        max_rounds=20, relevant_tools={"search_emails", "read_email", "create_document"},
        owner="admin", _is_teacher_run=True,
    )]
    assert [tool for tool, _ in executed] == [
        "mcp__email__search_emails", "mcp__email__read_email", "create_document",
    ]
    document = executed[-1][1]
    assert document["title"] == "Exact synthetic title"
    if lossless:
        assert "synthetic body 10595" in document["content"]
        assert "synthetic body 10596" in document["content"]
        assert synthesis not in document["content"]
    else:
        assert document["content"] == synthesis
        assert "Retrieved emails" not in document["content"]
    assert any("Document created." in chunk for chunk in chunks)



def test_second_review_active_state_distinguishes_targeted_from_inactive():
    inactive = agent_loop._email_document_workflow_semantics([])
    targeted = agent_loop._email_document_workflow_semantics([
        {"role": "user", "content": _TARGETED_WORKFLOW},
    ])
    assert inactive != targeted
    assert not inactive.active
    assert targeted.active
    assert not targeted.corpus and not targeted.exhaustive and not targeted.lossless


@pytest.mark.parametrize("answer", [
    "work", "yes", "Sí", "continue", "Només els del 2025", "only those from 2025", "search again",
])
@pytest.mark.parametrize("structured", [False, True])
def test_second_review_contextual_clarification_preserves_active_state(answer, structured):
    assistant = {"role": "assistant", "content": "Which mailbox should I use?"}
    if structured:
        assistant["metadata"] = {"tool_events": [{
            "tool": "ask_user", "ask_user": {"question": "Which mailbox should I use?"},
        }]}
    history = [{"role": "user", "content": _LITERAL_WORKFLOW}, assistant,
               {"role": "user", "content": answer}]
    assert agent_loop._email_document_workflow_semantics(history) == (True, True, True, True)
    # Another clarification uses only the latest assistant adjacency.
    history.extend([assistant, {"role": "user", "content": "work"}])
    assert agent_loop._email_document_workflow_semantics(history) == (True, True, True, True)


@pytest.mark.parametrize("followup", [
    "Now write a summary of Macbeth.", "Also summarize this report.",
    "And then write a summary of Macbeth.", "Now make a document about gardening.",
    "What is the capital of France?", "Search email invoices.", "start over",
])
def test_second_review_new_subject_overrides_clarification_context(followup):
    history = [{"role": "user", "content": _LITERAL_WORKFLOW},
               {"role": "assistant", "content": "What should the document contain?",
                "metadata": {"tool_events": [{"tool": "ask_user", "ask_user": {"question": "What next?"}}]}},
               {"role": "user", "content": followup}]
    assert agent_loop._email_document_workflow_semantics(history) == (False, False, False, False)


@pytest.mark.parametrize("followup", [
    "Haz una síntesis.", "Fes-ne una síntesi.", "Haz una sintesis.",
    "Fes-ne una sintesi.", "Make it a summary.",
])
def test_second_review_diacritics_share_lossless_normalization(followup):
    history = [{"role": "user", "content": _LITERAL_WORKFLOW},
               {"role": "user", "content": followup}]
    assert agent_loop._email_document_workflow_semantics(history) == (True, True, True, False)
    assert not agent_loop._requires_lossless_email_document(_LITERAL_WORKFLOW + " " + followup)


@pytest.mark.parametrize("selection", [
    "Actually use all ten.", "Use these ten.", "Use those ten.", "Use ten of them.",
    "all ten", "these ten", "those ten", "ten of them", "No, use these ten instead.",
    "Use all 10.", "Use these 10.", "Use 10 of them.",
])
def test_second_review_contextual_email_counts(selection):
    assert agent_loop._email_source_set_override(selection) == (True, False)
    history = [{"role": "user", "content": _TARGETED_WORKFLOW},
               {"role": "user", "content": selection}]
    assert agent_loop._email_document_workflow_semantics(history) == (True, True, False, False)


@pytest.mark.parametrize("formatting", [
    "Use these two bullet points.", "Write three paragraphs.", "Use five headings.",
    "Use these ten sentences.",
])
@pytest.mark.parametrize("origin,expected", [
    (_TARGETED_WORKFLOW, (True, False, False, False)),
    (_LITERAL_WORKFLOW, (True, True, True, True)),
])
def test_second_review_non_email_counts_do_not_select_sources(formatting, origin, expected):
    assert agent_loop._email_source_set_override(formatting) is None
    history = [{"role": "user", "content": origin}, {"role": "user", "content": formatting}]
    assert agent_loop._email_document_workflow_semantics(history) == expected


@pytest.mark.parametrize("followup,email_workflow", [
    ("work", True), ("search again", True),
    ("Now write a summary of Macbeth.", False), ("Also summarize this report.", False),
])
@pytest.mark.asyncio
async def test_second_review_stream_clarification_and_new_subject(monkeypatch, followup, email_workflow):
    targets = [{"uid": "10595"}, {"uid": "10596"}]
    executed = []
    rounds = 0
    model_content = "Synthetic requested document, written by the model."

    async def fake_stream(_candidates, messages, **kwargs):
        nonlocal rounds
        rounds += 1
        schemas = _schema_names(kwargs.get("tools"))
        if rounds == 1 and email_workflow:
            assert "EMAIL DOCUMENT RETRIEVAL GATE" in json.dumps(messages)
            assert "create_document" not in schemas
            name, payload = "search_emails", {"query": "synthetic", "max_results": 20}
        elif not any(tool == "create_document" for tool, _ in executed):
            assert "create_document" in schemas
            if email_workflow:
                assert schemas == {"create_document"}
                assert kwargs.get("tool_choice_name") == "create_document"
            else:
                assert "EMAIL DOCUMENT RETRIEVAL GATE" not in json.dumps(messages)
                assert "EMAIL DOCUMENT TARGETED RETRIEVAL GATE" not in json.dumps(messages)
            name, payload = "create_document", {
                "title": "Synthetic result", "language": "markdown", "content": model_content,
            }
        else:
            yield 'data: {"delta": "Document created."}\n\n'
            yield "data: [DONE]\n\n"
            return
        yield f'data: {json.dumps({"type": "tool_calls", "calls": [{"name": name, "arguments": json.dumps(payload)}]})}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        if block.tool_type == "create_document":
            title, language, content = block.content.split("\n", 2)
            payload = {"title": title, "language": language, "content": content}
        else:
            payload = json.loads(block.content)
        executed.append((block.tool_type, payload))
        if block.tool_type == "mcp__email__search_emails":
            assert email_workflow
            assert payload["max_results"] == agent_loop._EXHAUSTIVE_EMAIL_SEARCH_MAX_RESULTS
            return block.tool_type, {"output": _synthetic_search_output(targets), "exit_code": 0}
        if block.tool_type == "mcp__email__read_email":
            assert email_workflow
            assert [target["uid"] for target in payload["targets"]] == ["10595", "10596"]
            return block.tool_type, {"output": _synthetic_batch_output(targets), "exit_code": 0}
        assert block.tool_type == "create_document"
        return block.tool_type, {"output": "SYNTHETIC_DOCUMENT_RESULT", "exit_code": 0,
                                 "action": "create", "doc_id": "synthetic", "version": 1, **payload}

    _patch_agent_loop_dependencies(monkeypatch, fake_stream, fake_execute)
    history = [{"role": "user", "content": _LITERAL_WORKFLOW},
               {"role": "assistant", "content": "What should the document contain?",
                "metadata": {"tool_events": [{"tool": "ask_user", "ask_user": {"question": "Which mailbox?"}}]}},
               {"role": "user", "content": followup}]
    chunks = [chunk async for chunk in agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", list(history), conversation_history=history,
        max_rounds=20, relevant_tools={"search_emails", "read_email", "create_document"},
        owner="admin", _is_teacher_run=True,
    )]
    assert [tool for tool, _ in executed] == (
        ["mcp__email__search_emails", "mcp__email__read_email", "create_document"]
        if email_workflow else ["create_document"]
    )
    if email_workflow:
        assert "synthetic body 10595" in executed[-1][1]["content"]
        assert "synthetic body 10596" in executed[-1][1]["content"]
    else:
        assert executed[-1][1]["content"] == model_content
    assert any("Document created." in chunk for chunk in chunks)


@pytest.mark.parametrize("selection", [
    "use one selected message", "use only one selected message",
    "use one selected email", "use only one selected email",
    "use the selected message", "use only the selected message",
    "use the selected email", "use only the selected email",
    "use selected message", "use selected email",
])
def test_medium_closure_selected_source(selection):
    assert agent_loop._email_document_workflow_semantics([
        {"role": "user", "content": _LITERAL_WORKFLOW},
        {"role": "user", "content": selection},
    ]) == (True, False, False, False)


@pytest.mark.parametrize("formatting", [
    "use one selected paragraph per email", "use only one selected heading per email",
    "use one selected sentence per message", "make one selected section per email",
])
def test_medium_closure_selected_formatting(formatting):
    assert agent_loop._email_source_set_override(formatting) is None
    assert agent_loop._email_document_workflow_semantics([
        {"role": "user", "content": _LITERAL_WORKFLOW},
        {"role": "user", "content": formatting},
    ]) == (True, True, True, True)


@pytest.mark.parametrize("cancellation", [
    "cancel", "forget that", "discard that", "stop this", "Cancel the email document.",
])
def test_medium_closure_cancellation(cancellation):
    history = [{"role": "user", "content": _LITERAL_WORKFLOW},
               {"role": "assistant", "metadata": {"tool_events": [
                   {"tool": "create_document", "exit_code": 0, "doc_id": "abc", "had_error": True},
               ]}}]
    history.append({"role": "user", "content": cancellation})
    assert agent_loop._email_document_workflow_semantics(history) == (False, False, False, False)
    history.append({"role": "user", "content": _TARGETED_WORKFLOW})
    assert agent_loop._email_document_workflow_semantics(history) == (True, False, False, False)


def test_medium_closure_negated_cancellation():
    assert agent_loop._email_document_workflow_semantics([
        {"role": "user", "content": _LITERAL_WORKFLOW},
        {"role": "user", "content": "Do not cancel; continue."},
    ]) == (True, True, True, True)


@pytest.mark.parametrize("prompt,active", [
    ("Search all emails for the document.", False),
    ("Find emails mentioning the document.", False),
    ("Search emails about the report document.", False),
    ("Look for messages about an archived document.", False),
    ("Create a document from all emails.", True),
    ("Generate a document from the selected message.", True),
    ("Create a list containing all emails.", True),
    ("Create an archive containing all emails.", True),
    ("Search all emails and then create a document summarizing them.", True),
    ("Search candidates, read the relevant one, and create a brief document.", True),
])
def test_medium_closure_creation_intent(prompt, active):
    for prefix in ([], [{"role": "user", "content": _LITERAL_WORKFLOW}]):
        assert agent_loop._email_document_workflow_semantics(
            prefix + [{"role": "user", "content": prompt}],
        ).active is active


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt,active", [
    ("Search all emails for the document.", False),
    ("Search all emails and use only one selected message for the document.", True),
])
async def test_medium_closure_search_scope_stream(monkeypatch, caplog, prompt, active):
    state = agent_loop._email_document_workflow_semantics([
        {"role": "user", "content": _LITERAL_WORKFLOW}, {"role": "user", "content": prompt},
    ])
    assert state == (active, False, False, False)
    await test_last_findings_email_only_exhaustive_search_stream(monkeypatch, caplog, prompt, active)


@pytest.mark.asyncio
@pytest.mark.parametrize("presentation", [{}, {"content": "Body"}, {"version": 7}])
async def test_medium_closure_success_presentation_stream(monkeypatch, presentation):
    raw = {"exit_code": 0, "doc_id": "abc", "action": "create", **presentation}
    assert agent_loop._email_document_creation_succeeded("create_document", raw)
    await test_last_findings_exact_create_transport_and_persistence(monkeypatch, raw, True)
