"""Sender-address retrieval must not be confused with mailbox discovery."""

import asyncio
import json

import pytest

import src.agent_loop as agent_loop


def _collect(gen):
    async def _run():
        return [chunk async for chunk in gen]

    return asyncio.run(_run())


def _schema_names(tools):
    return {
        tool.get("function", {}).get("name") or tool.get("name")
        for tool in (tools or [])
        if isinstance(tool, dict)
    }


def _patch_agent(monkeypatch):
    monkeypatch.setattr(
        agent_loop, "get_setting", lambda key, default=None: default,
        raising=False,
    )
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(agent_loop, "estimate_tokens", lambda *a, **k: 10, raising=False)
    monkeypatch.setattr(agent_loop, "blocked_tools_for_owner", lambda owner: set(), raising=False)


_EMAIL_DOCUMENT_TOOLS = {
    "ask_user",
    "create_document",
    "list_email_accounts",
    "list_emails",
    "read_email",
    "search_emails",
    "send_email",
    "web_fetch",
    "web_search",
}


@pytest.mark.parametrize("user_text", [
    "Busca el correu més recent de niniprimer@gmail.com i digues-me només la data i l'assumpte.",
    "Busca el correo más reciente de sender@example.com y dime solo la fecha y el asunto.",
    "Find the most recent email from sender@example.com and tell me only the date and subject.",
])
def test_sender_specific_lookup_requires_search_as_first_retrieval_action(
    monkeypatch, user_text,
):
    _patch_agent(monkeypatch)
    model_schemas = []
    model_messages = []
    executed = []
    expected_address = (
        "niniprimer@gmail.com" if "niniprimer" in user_text else "sender@example.com"
    )

    async def fake_stream(_candidates, messages, **kwargs):
        model_schemas.append(_schema_names(kwargs.get("tools")))
        model_messages.append(messages)
        if len(model_schemas) == 1:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{
                    "name": "search_emails",
                    "arguments": json.dumps({"query": expected_address}),
                }],
            }) + "\n\n"
        else:
            yield 'data: {"delta": "No matches."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append((block.tool_type, json.loads(block.content)))
        return block.tool_type, {"output": "No emails matched.", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)

    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        [{"role": "user", "content": user_text}],
        max_rounds=2,
        relevant_tools=set(_EMAIL_DOCUMENT_TOOLS),
        owner="admin",
    ))

    assert agent_loop._explicit_email_sender_address(user_text) == expected_address
    assert model_schemas[0] == {"search_emails"}
    assert executed == [
        ("mcp__email__search_emails", {"query": expected_address}),
    ]
    system_text = "\n".join(
        str(message.get("content") or "")
        for message in model_messages[0]
        if message.get("role") == "system"
    )
    assert f"`{expected_address}` identifies the sender" in system_text
    assert "exact address as `query`" in system_text


def test_generic_recent_email_browsing_keeps_list_emails_available(monkeypatch):
    _patch_agent(monkeypatch)
    model_schemas = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_schemas.append(_schema_names(kwargs.get("tools")))
        yield "data: " + json.dumps({
            "type": "tool_calls",
            "calls": [{"name": "list_emails", "arguments": '{"max_results": 5}'}],
        }) + "\n\n"
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        return block.tool_type, {"output": "Found 5 email(s).", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)

    user_text = "Mostra els últims correus."
    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        [{"role": "user", "content": user_text}],
        max_rounds=1,
        relevant_tools=set(_EMAIL_DOCUMENT_TOOLS),
        owner="admin",
    ))

    assert agent_loop._explicit_email_sender_address(user_text) is None
    assert "list_emails" in model_schemas[0]
    assert executed == ["mcp__email__list_emails"]


def test_sender_search_is_seeded_and_clamped_for_mcp_schemas(monkeypatch):
    _patch_agent(monkeypatch)
    schemas = []
    executed = []

    class _FakeMcp:
        pass

    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: _FakeMcp(), raising=False)
    monkeypatch.setattr(agent_loop, "_load_mcp_disabled_map", lambda: {}, raising=False)
    monkeypatch.setattr(agent_loop, "FUNCTION_TOOL_SCHEMAS", [], raising=False)
    monkeypatch.setattr(
        agent_loop,
        "_build_system_prompt",
        lambda messages, *args, **kwargs: (
            messages,
            [
                {"type": "function", "function": {"name": "mcp__email__search_emails"}},
                {"type": "function", "function": {"name": "mcp__email__list_emails"}},
                {"type": "function", "function": {"name": "mcp__email__list_email_accounts"}},
            ],
        ),
        raising=False,
    )

    async def fake_stream(_candidates, messages, **kwargs):
        schemas.append(_schema_names(kwargs.get("tools")))
        if len(schemas) == 1:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{
                    "name": "mcp__email__search_emails",
                    "arguments": json.dumps({"query": "niniprimer@gmail.com"}),
                }],
            }) + "\n\n"
        else:
            yield 'data: {"delta": "Done."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append((block.tool_type, json.loads(block.content)))
        return block.tool_type, {"output": "Found 1 email.", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)
    user_text = "Busca el correu més recent de niniprimer@gmail.com i digues-me només la data i l'assumpte."

    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1/chat/completions", "qwen3:14b",
        [{"role": "user", "content": user_text}],
        max_rounds=2,
        relevant_tools={"list_email_accounts"},
        owner="admin",
    ))

    assert agent_loop._explicit_email_sender_address(user_text) == "niniprimer@gmail.com"
    assert schemas[0] == {"mcp__email__search_emails"}
    assert schemas[1] == {
        "mcp__email__search_emails",
        "mcp__email__list_emails",
        "mcp__email__list_email_accounts",
    }
    assert executed == [
        ("mcp__email__search_emails", {"query": "niniprimer@gmail.com"}),
    ]


def test_real_smoke_d_mcp_metadata_only_completion(monkeypatch):
    _patch_agent(monkeypatch)
    schemas, messages_by_round, executed = [], [], []
    registered = [
        {"type": "function", "function": {"name": "mcp__email__search_emails"}},
        {"type": "function", "function": {"name": "mcp__email__read_email"}},
        {"type": "function", "function": {"name": "mcp__email__list_emails"}},
        {"type": "function", "function": {"name": "mcp__email__list_email_accounts"}},
    ]

    class _FakeMcp:
        pass

    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: _FakeMcp(), raising=False)
    monkeypatch.setattr(agent_loop, "_load_mcp_disabled_map", lambda: {}, raising=False)
    monkeypatch.setattr(agent_loop, "FUNCTION_TOOL_SCHEMAS", [], raising=False)
    monkeypatch.setattr(
        agent_loop, "_build_system_prompt",
        lambda messages, *args, **kwargs: (messages, list(registered)), raising=False,
    )
    prompt = (
        "Busca el correu més recent de niniprimer@gmail.com i digues-me "
        "només la data i l'assumpte."
    )
    body_sentinel = "PRIVATE BODY SENTINEL — must never appear"

    async def fake_stream(_candidates, messages, **kwargs):
        schemas.append(_schema_names(kwargs.get("tools")))
        messages_by_round.append(messages)
        if len(schemas) == 1:
            yield "data: " + json.dumps({"type": "tool_calls", "calls": [{
                "name": "mcp__email__search_emails",
                "arguments": json.dumps({"query": "niniprimer@gmail.com", "max_results": 1}),
            }]}) + "\n\n"
        else:
            yield 'data: {"delta": "Wed, 3 May 2023 16:38:36 +0700 — Re: Euro"}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append((block.tool_type, json.loads(block.content)))
        return block.tool_type, {"output": "\n".join([
            "Found 2 email(s):",
            "1. **Re: Euro**",
            "From: nini primer (niniprimer@gmail.com)",
            "Date: Wed, 3 May 2023 16:38:36 +0700",
            "Folder: INBOX",
            "UID: 2365",
            "2. **Re: Euro**",
            "From: nini primer (niniprimer@gmail.com)",
            "Date: Wed, 3 May 2023 16:38:36 +0700",
            "Folder: Archive",
            "UID: 2365",
            body_sentinel,
        ]), "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)
    chunks = _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1/chat/completions", "qwen3:14b",
        [{"role": "user", "content": prompt}],
        max_rounds=2, relevant_tools=None, owner="admin",
    ))

    assert schemas[0] == {"mcp__email__search_emails"}
    assert executed == [
        ("mcp__email__search_emails", {"query": "niniprimer@gmail.com", "max_results": 1}),
    ]
    assert schemas[1] == set()
    reminder = "\n".join(
        str(message.get("content") or "") for message in messages_by_round[1]
    )
    assert "Answer only with the requested email metadata" in reminder
    assert "date" in reminder and "subject" in reminder
    output = "".join(chunks)
    assert "Wed, 3 May 2023 16:38:36 +0700" in output
    assert "Re: Euro" in output
    assert body_sentinel not in output


def _metadata_search_output(include_date=True, include_subject=True):
    lines = ["Found 1 email(s):", "1. **Re: Euro**", "From: nini primer (sender@example.com)"]
    if include_date:
        lines.append("Date: Wed, 3 May 2023 16:38:36 +0700")
    if include_subject:
        lines.append("Subject: Re: Euro")
    lines.extend(["Folder: INBOX", "UID: 2365"])
    return "\n".join(lines)


def test_metadata_only_search_suppresses_read_and_preserves_shape(monkeypatch):
    _patch_agent(monkeypatch)
    schemas, executed, messages_by_round = [], [], []

    async def fake_stream(_candidates, messages, **kwargs):
        schemas.append(_schema_names(kwargs.get("tools")))
        messages_by_round.append(messages)
        if len(schemas) == 1:
            yield "data: " + json.dumps({"type": "tool_calls", "calls": [{
                "name": "search_emails",
                "arguments": '{"query":"sender@example.com","max_results":1}',
            }]}) + "\n\n"
        else:
            yield 'data: {"delta": "Wed, 3 May 2023 16:38:36 +0700 — Re: Euro"}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append((block.tool_type, json.loads(block.content)))
        return block.tool_type, {"output": _metadata_search_output(), "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)
    chunks = _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        [{"role": "user", "content": "Busca el correo más reciente de sender@example.com y dime solo la fecha y el asunto."}],
        max_rounds=2, relevant_tools=set(_EMAIL_DOCUMENT_TOOLS), owner="admin",
    ))

    assert schemas[0] == {"search_emails"}
    assert schemas[1] == set()
    assert executed == [("mcp__email__search_emails", {"query": "sender@example.com", "max_results": 1})]
    output = "".join(chunks)
    assert "Wed, 3 May 2023 16:38:36 +0700" in output and "Re: Euro" in output
    assert "unsolicited offer" not in output
    reminder = "\n".join(str(m.get("content") or "") for m in messages_by_round[1])
    assert "Answer only with the requested email metadata" in reminder


def test_metadata_only_incomplete_result_keeps_read_email_available(monkeypatch):
    _patch_agent(monkeypatch)
    schemas, executed = [], []

    async def fake_stream(_candidates, messages, **kwargs):
        schemas.append(_schema_names(kwargs.get("tools")))
        if len(schemas) == 1:
            call = {"name": "search_emails", "arguments": '{"query":"sender@example.com"}'}
        elif len(schemas) == 2:
            call = {"name": "read_email", "arguments": '{"uid":"2365","folder":"INBOX"}'}
        else:
            yield 'data: {"delta": "Date unavailable."}\n\n'
            yield "data: [DONE]\n\n"
            return
        yield "data: " + json.dumps({"type": "tool_calls", "calls": [call]}) + "\n\n"
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        output = _metadata_search_output(include_date=False) if "search_emails" in block.tool_type else "Body"
        return block.tool_type, {"output": output, "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)
    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        [{"role": "user", "content": "Dime solo la fecha y el asunto del correo de sender@example.com."}],
        max_rounds=3, relevant_tools=set(_EMAIL_DOCUMENT_TOOLS), owner="admin",
    ))
    assert executed == ["mcp__email__search_emails", "mcp__email__read_email"]
    assert "read_email" in schemas[1]


def test_body_request_keeps_read_email_progression(monkeypatch):
    _patch_agent(monkeypatch)
    executed, schemas = [], []

    async def fake_stream(_candidates, messages, **kwargs):
        schemas.append(_schema_names(kwargs.get("tools")))
        if len(schemas) == 1:
            call = {"name": "search_emails", "arguments": '{"query":"sender@example.com"}'}
            yield "data: " + json.dumps({"type": "tool_calls", "calls": [call]}) + "\n\n"
        elif len(schemas) == 2:
            call = {"name": "read_email", "arguments": '{"uid":"2365","folder":"INBOX"}'}
            yield "data: " + json.dumps({"type": "tool_calls", "calls": [call]}) + "\n\n"
        else:
            yield 'data: {"delta": "Resumen del contenido."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        output = _metadata_search_output() if "search_emails" in block.tool_type else "Full body text"
        return block.tool_type, {"output": output, "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)
    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        [{"role": "user", "content": "Llegeix el correu més recent de sender@example.com i resumeix-me què diu."}],
        max_rounds=3, relevant_tools=set(_EMAIL_DOCUMENT_TOOLS), owner="admin",
    ))
    assert executed == ["mcp__email__search_emails", "mcp__email__read_email"]
    assert "read_email" in schemas[1]


@pytest.mark.parametrize("user_text", [
    "Crea'm un document Word amb tots els correus de sender@example.com, ordenats per data.",
    "Crea un documento Word con todos los correos de sender@example.com, ordenados por fecha.",
    "Create a Word document with all emails from sender@example.com, ordered by date.",
])
def test_explicit_sender_requires_search_as_first_retrieval_action(monkeypatch, user_text):
    _patch_agent(monkeypatch)
    model_schemas = []
    model_messages = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_schemas.append(_schema_names(kwargs.get("tools")))
        model_messages.append(messages)
        if len(model_schemas) == 1:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{
                    "name": "search_emails",
                    "arguments": json.dumps({"query": "sender@example.com"}),
                }],
            }) + "\n\n"
        else:
            yield 'data: {"delta": "No matches."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append((block.tool_type, json.loads(block.content)))
        return block.tool_type, {"output": "[]", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)

    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        [{"role": "user", "content": user_text}],
        max_rounds=2,
        relevant_tools=set(_EMAIL_DOCUMENT_TOOLS),
        owner="admin",
    ))

    assert agent_loop._explicit_email_sender_address(user_text) == "sender@example.com"
    assert model_schemas[0] == {"search_emails"}
    assert "list_email_accounts" not in model_schemas[0]
    assert "send_email" not in model_schemas[0]
    assert "create_document" not in model_schemas[0]
    assert executed == [
        ("mcp__email__search_emails", {
            "query": "sender@example.com",
            "max_results": agent_loop._EXHAUSTIVE_EMAIL_SEARCH_MAX_RESULTS,
        }),
    ]
    system_text = "\n".join(
        str(message.get("content") or "")
        for message in model_messages[0]
        if message.get("role") == "system"
    )
    assert "`sender@example.com` identifies the sender" in system_text
    assert "exact address as `query`" in system_text
    assert "web_search" not in model_schemas[1]
    assert "web_fetch" not in model_schemas[1]
    assert "create_document" not in model_schemas[1]


def test_explicit_account_discovery_still_exposes_list_email_accounts(monkeypatch):
    _patch_agent(monkeypatch)
    schemas = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        schemas.append(_schema_names(kwargs.get("tools")))
        yield "data: " + json.dumps({
            "type": "tool_calls",
            "calls": [{"name": "list_email_accounts", "arguments": "{}"}],
        }) + "\n\n"
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        return block.tool_type, {"output": "work (default)", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)

    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        [{"role": "user", "content": "Quins comptes de correu tinc configurats?"}],
        max_rounds=1,
        relevant_tools={"ask_user", "list_email_accounts", "send_email"},
        owner="admin",
    ))

    assert agent_loop._explicit_email_sender_address(
        "Quins comptes de correu tinc configurats?"
    ) is None
    assert "list_email_accounts" in schemas[0]
    assert executed == ["mcp__email__list_email_accounts"]


def test_account_discovery_result_keeps_document_task_on_retrieval(monkeypatch):
    _patch_agent(monkeypatch)
    schemas = []
    messages_by_round = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        schemas.append(_schema_names(kwargs.get("tools")))
        messages_by_round.append(messages)
        if len(schemas) == 1:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{"name": "list_email_accounts", "arguments": "{}"}],
            }) + "\n\n"
        else:
            yield 'data: {"delta": "Continuing retrieval."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        return block.tool_type, {"output": "work (default)", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)

    user_text = "Create a Word document from all emails in my work mailbox."
    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        [{"role": "user", "content": user_text}],
        max_rounds=2,
        relevant_tools=set(_EMAIL_DOCUMENT_TOOLS),
        owner="admin",
    ))

    assert "list_email_accounts" in schemas[0]
    assert schemas[1] == {"search_emails", "list_emails"}
    assert "send_email" not in schemas[1]
    assert "create_document" not in schemas[1]
    assert executed == ["mcp__email__list_email_accounts"]
    round_two_text = "\n".join(
        str(message.get("content") or "") for message in messages_by_round[1]
    )
    assert user_text in round_two_text
    assert "Continue the user's original email retrieval and document task" in round_two_text
    assert "Do not reinterpret it as a request to list accounts or send an email" in round_two_text


def test_sender_email_plus_web_preserves_web_tools_after_required_search(monkeypatch):
    _patch_agent(monkeypatch)
    schemas = []

    async def fake_stream(_candidates, messages, **kwargs):
        schemas.append(_schema_names(kwargs.get("tools")))
        if len(schemas) == 1:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{
                    "name": "search_emails",
                    "arguments": json.dumps({"query": "sender@example.com"}),
                }],
            }) + "\n\n"
        else:
            yield 'data: {"delta": "No matches."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        return block.tool_type, {"output": "[]", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)

    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        [{
            "role": "user",
            "content": (
                "Create a Word document with emails from sender@example.com "
                "and search the web for current company news."
            ),
        }],
        max_rounds=2,
        relevant_tools=set(_EMAIL_DOCUMENT_TOOLS),
        owner="admin",
    ))

    assert schemas[0] == {"search_emails"}
    assert {"web_search", "web_fetch"} <= schemas[1]
    assert "create_document" not in schemas[1]


def test_required_sender_search_retries_long_prose_once_then_executes(monkeypatch):
    _patch_agent(monkeypatch)
    schemas = []
    messages_by_round = []
    executed = []
    prose = (
        "Action: search_emails with query sender@example.com. "
        + ("I will describe the remaining processing steps. " * 12)
        + "Next action: Call search_emails to retrieve the emails."
    )
    assert len(prose) > 400

    async def fake_stream(_candidates, messages, **kwargs):
        schemas.append(_schema_names(kwargs.get("tools")))
        messages_by_round.append(messages)
        if len(schemas) == 1:
            yield "data: " + json.dumps({"delta": prose}) + "\n\n"
        elif len(schemas) == 2:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{
                    "name": "search_emails",
                    "arguments": json.dumps({"query": "sender@example.com"}),
                }],
            }) + "\n\n"
        else:
            yield 'data: {"delta": "No matches."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append((block.tool_type, json.loads(block.content)))
        return block.tool_type, {"output": "[]", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)

    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        [{
            "role": "user",
            "content": (
                "Crea'm un document Word amb tots els correus de "
                "sender@example.com, ordenats per data."
            ),
        }],
        max_rounds=3,
        relevant_tools=set(_EMAIL_DOCUMENT_TOOLS),
        owner="admin",
    ))

    assert schemas[:2] == [{"search_emails"}, {"search_emails"}]
    assert executed == [
        ("mcp__email__search_emails", {
            "query": "sender@example.com",
            "max_results": agent_loop._EXHAUSTIVE_EMAIL_SEARCH_MAX_RESULTS,
        }),
    ]
    assert len(schemas) == 3
    corrective_text = "\n".join(
        str(message.get("content") or "")
        for message in messages_by_round[1]
        if message.get("role") == "system"
    )
    assert "Call the available `search_emails` native tool now" in corrective_text
    assert "Do not describe or announce the action in prose" in corrective_text
    for forbidden in {
        "list_email_accounts", "send_email", "create_document",
        "web_search", "web_fetch",
    }:
        assert forbidden not in schemas[1]


def test_required_sender_search_prose_retry_is_bounded(monkeypatch, caplog):
    _patch_agent(monkeypatch)
    schemas = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        schemas.append(_schema_names(kwargs.get("tools")))
        yield "data: " + json.dumps({
            "delta": "Action: search_emails. Next action: Call search_emails.",
        }) + "\n\n"
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        return block.tool_type, {"output": "[]", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)

    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        [{
            "role": "user",
            "content": "Create a Word document with emails from sender@example.com.",
        }],
        max_rounds=5,
        relevant_tools=set(_EMAIL_DOCUMENT_TOOLS),
        owner="admin",
    ))

    assert schemas == [{"search_emails"}, {"search_emails"}]
    assert executed == []
    assert "required first retrieval tool still not executed" in caplog.text


def test_round_one_required_search_execution_does_not_correctively_retry(monkeypatch):
    _patch_agent(monkeypatch)
    schemas = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        schemas.append(_schema_names(kwargs.get("tools")))
        if len(schemas) == 1:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{
                    "name": "search_emails",
                    "arguments": json.dumps({"query": "sender@example.com"}),
                }],
            }) + "\n\n"
        else:
            yield 'data: {"delta": "No matches."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        return block.tool_type, {"output": "[]", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)

    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        [{
            "role": "user",
            "content": "Create a Word document with emails from sender@example.com.",
        }],
        max_rounds=2,
        relevant_tools=set(_EMAIL_DOCUMENT_TOOLS),
        owner="admin",
    ))

    assert schemas[0] == {"search_emails"}
    assert schemas[1] != {"search_emails"}
    assert executed == ["mcp__email__search_emails"]


def test_ordinary_prose_without_required_retrieval_still_completes(monkeypatch):
    _patch_agent(monkeypatch)
    schemas = []

    async def fake_stream(_candidates, messages, **kwargs):
        schemas.append(_schema_names(kwargs.get("tools")))
        yield 'data: {"delta": "A normal prose answer."}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)

    chunks = _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        [{"role": "user", "content": "What is chronological order?"}],
        max_rounds=4,
        relevant_tools=set(_EMAIL_DOCUMENT_TOOLS),
        owner="admin",
    ))

    assert len(schemas) == 1
    assert any("A normal prose answer." in chunk for chunk in chunks)


def _structured_ask_user_message():
    return {
        "role": "assistant",
        "content": "Vols que continuï?",
        "metadata": {
            "tool_events": [{
                "round": 1,
                "tool": "ask_user",
                "ask_user": {
                    "question": "Vols que continuï?",
                    "options": [{"label": "Sí"}, {"label": "No"}],
                    "multi": False,
                },
            }],
        },
    }


def test_new_session_foreign_ask_context_uses_required_search_retry(monkeypatch):
    _patch_agent(monkeypatch)
    schemas = []
    messages_by_round = []
    executed = []
    current = {
        "role": "user",
        "content": (
            "Crea'm un document Word amb tots els correus de "
            "sender@example.com, ordenats per data."
        ),
    }

    async def fake_stream(_candidates, messages, **kwargs):
        schemas.append(_schema_names(kwargs.get("tools")))
        messages_by_round.append(messages)
        if len(schemas) == 1:
            yield 'data: {"delta": "Action: search_emails. Next action: Call search_emails."}\n\n'
        elif len(schemas) == 2:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{
                    "name": "search_emails",
                    "arguments": json.dumps({"query": "sender@example.com"}),
                }],
            }) + "\n\n"
        else:
            yield 'data: {"delta": "No matches."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append((block.tool_type, json.loads(block.content)))
        return block.tool_type, {"output": "[]", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)

    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        [_structured_ask_user_message(), current],
        conversation_history=[current],
        max_rounds=3,
        relevant_tools=set(_EMAIL_DOCUMENT_TOOLS),
        owner="admin",
    ))

    assert schemas[:2] == [{"search_emails"}, {"search_emails"}]
    assert executed == [
        ("mcp__email__search_emails", {
            "query": "sender@example.com",
            "max_results": agent_loop._EXHAUSTIVE_EMAIL_SEARCH_MAX_RESULTS,
        }),
    ]
    corrective_system = "\n".join(
        str(message.get("content") or "")
        for message in messages_by_round[1]
        if message.get("role") == "system"
    )
    assert "Call the available `search_emails` native tool now" in corrective_system
    assert "resuming immediately after an `ask_user`" not in corrective_system


def test_required_search_wins_over_contradictory_ask_followup_state(monkeypatch):
    _patch_agent(monkeypatch)
    schemas = []
    messages_by_round = []
    current = {
        "role": "user",
        "content": "Create a document with emails from sender@example.com.",
    }
    history = [
        {"role": "user", "content": "Please prepare an email document."},
        _structured_ask_user_message(),
        current,
    ]

    async def fake_stream(_candidates, messages, **kwargs):
        schemas.append(_schema_names(kwargs.get("tools")))
        messages_by_round.append(messages)
        yield 'data: {"delta": "Action: search_emails. Next action: Call search_emails."}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)

    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "qwen3:14b",
        history,
        conversation_history=history,
        max_rounds=4,
        relevant_tools=set(_EMAIL_DOCUMENT_TOOLS),
        owner="admin",
    ))

    assert schemas == [{"search_emails"}, {"search_emails"}]
    for forbidden in {
        "list_email_accounts", "send_email", "create_document",
        "web_search", "web_fetch",
    }:
        assert forbidden not in schemas[1]
    corrective_system = "\n".join(
        str(message.get("content") or "")
        for message in messages_by_round[1]
        if message.get("role") == "system"
    )
    assert "Call the available `search_emails` native tool now" in corrective_system
    assert "resuming immediately after an `ask_user`" not in corrective_system
