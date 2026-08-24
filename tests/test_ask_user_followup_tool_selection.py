"""Regression: an answer to ask_user must preserve task tool selection.

The production failure was:

    email/document task
      -> ask_user(...)
      -> user: "Sí"
      -> classified low_signal
      -> Tool-RAG queried only "Sí"
      -> unrelated tools reached the model
      -> model replied "Sí." instead of continuing the task

This test drives the real stream_agent_loop selection path.  Semantic retrieval
is deliberately made wrong (returns only ``ls``); deterministic continuation
classification + domain seeding must still advertise the email/document tools.
"""

import asyncio
import json
from types import SimpleNamespace

import src.agent_loop as agent_loop
import src.tool_index as tool_index


def test_ask_user_function_call_conversion_preserves_json_payload():
    for question in ("Should I continue?", "Vols que continuï?"):
        block = agent_loop.function_call_to_tool_block(
            "ask_user",
            json.dumps({
                "question": question,
                "options": [{"label": "Yes"}, {"label": "No"}],
            }, ensure_ascii=False),
        )
        assert block.tool_type == "ask_user"
        payload = json.loads(block.content)
        assert payload["question"] == question


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


def _messages(answer="Sí"):
    return [
        {
            "role": "user",
            "content": (
                "Busca els correus rebuts de odysseus-regression-fixture@gmail.com, "
                "ordena'ls per data i crea un document Word."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Vols que busqui els correus electrònics rebuts de "
                "odysseus-regression-fixture@gmail.com i els ordeni per data?"
            ),
            "metadata": {
                "tool_events": [
                    {
                        "round": 1,
                        "tool": "ask_user",
                        "ask_user": {
                            "question": (
                                "Vols que busqui els correus electrònics rebuts de "
                                "odysseus-regression-fixture@gmail.com i els ordeni per data?"
                            ),
                            "options": [
                                {"label": "Sí"},
                                {"label": "No"},
                            ],
                            "multi": False,
                        },
                    }
                ]
            },
        },
        {"role": "user", "content": answer},
    ]


class _FakeEmailMcpManager:
    """Mirror production: built-in Python MCP schemas are not returned here."""

    def get_all_openai_schemas(self, _disabled_map):
        return []

    def get_tool_descriptions_for_prompt(self, _disabled_map):
        return "**email:**\n- search_emails: Search mailbox emails."


def test_ask_user_yes_survives_bad_tool_rag_and_sends_email_document_tools(monkeypatch):
    monkeypatch.setattr(
        agent_loop,
        "get_setting",
        lambda key, default=None: default,
        raising=False,
    )
    monkeypatch.setattr(agent_loop, "get_mcp_manager", _FakeEmailMcpManager, raising=False)
    monkeypatch.setattr(
        agent_loop,
        "estimate_tokens",
        lambda *args, **kwargs: 10,
        raising=False,
    )
    monkeypatch.setattr(
        agent_loop,
        "blocked_tools_for_owner",
        lambda owner: set(),
        raising=False,
    )

    retrieval_queries = []

    class BadToolIndex:
        def index_mcp_tools(self, *_args, **_kwargs):
            return None

        def get_tools_for_query(self, query, k=8):
            retrieval_queries.append(query)
            # Reproduce the important part of the production failure:
            # semantic retrieval returns unrelated web tools for a mailbox
            # request. Deterministic routing must rescue email tools and
            # remove the competing web tools late in selection.
            return {"ls", "web_search", "web_fetch"}

    monkeypatch.setattr(
        tool_index,
        "get_tool_index",
        lambda: BadToolIndex(),
    )

    sent_tools = []
    sent_messages = []

    async def fake_stream(_candidates, messages, **kwargs):
        sent_tools.append(kwargs.get("tools"))
        sent_messages.append(messages)
        yield "data: " + json.dumps({"delta": "ok"}) + "\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(
        agent_loop,
        "stream_llm_with_fallback",
        fake_stream,
        raising=False,
    )

    _collect(
        agent_loop.stream_agent_loop(
            "https://api.openai.com/v1",
            "qwen3:14b",
            _messages("Sí"),
            max_rounds=1,
            relevant_tools=None,
            owner="admin",
        )
    )

    assert retrieval_queries
    retrieval_query = retrieval_queries[0]

    # The retrieval query must be contextual, not the bare "Sí".
    assert "odysseus-regression-fixture@gmail.com" in retrieval_query
    assert "Sí" in retrieval_query
    assert retrieval_query.strip() != "Sí"

    assert sent_tools
    names = _schema_names(sent_tools[0])

    # Bad semantic retrieval returned only `ls`, but deterministic domain
    # seeding must rescue the tools required by the original task.
    assert "list_emails" in names
    assert "read_email" in names
    assert sum(
        tool.get("function", {}).get("name") == "search_emails"
        for tool in sent_tools[0]
    ) == 1
    assert "mcp__email__search_emails" not in names
    assert "create_document" in names
    assert "web_search" not in names
    assert "web_fetch" not in names

    search_schema = next(
        tool for tool in sent_tools[0]
        if tool.get("function", {}).get("name") == "search_emails"
    )
    assert search_schema["function"]["parameters"]["required"] == ["query"]
    block = agent_loop.function_call_to_tool_block(
        "search_emails",
        json.dumps({"query": "odysseus-regression-fixture@gmail.com"}),
    )
    assert block.tool_type == "mcp__email__search_emails"

    # The model must also be told that this is the answer to the pending
    # ask_user interaction. Tool availability alone is insufficient: the
    # production model otherwise treats "Sí" as standalone chat and echoes it.
    assert sent_messages
    system_text = "\n".join(
        str(message.get("content") or "")
        for message in sent_messages[0]
        if message.get("role") == "system"
    )
    assert "answer to the immediately preceding `ask_user` interaction" in system_text
    assert "continue the underlying task" in system_text
    assert "Do not merely echo or acknowledge" in system_text


def test_active_email_draft_prunes_search_fetch_tools(monkeypatch):
    monkeypatch.setattr(agent_loop, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(agent_loop, "get_mcp_manager", _FakeEmailMcpManager, raising=False)
    monkeypatch.setattr(agent_loop, "estimate_tokens", lambda *args, **kwargs: 10, raising=False)
    monkeypatch.setattr(agent_loop, "blocked_tools_for_owner", lambda owner: set(), raising=False)

    sent_tools = []

    async def fake_stream(_candidates, messages, **kwargs):
        sent_tools.append(kwargs.get("tools"))
        yield "data: " + json.dumps({"delta": "ok"}) + "\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    active_email = SimpleNamespace(
        id="email-draft",
        language="email",
        title="Reply draft",
        current_content="To: sender@example.com\nSubject: Re: Hello\n---\nQuoted text",
    )

    _collect(
        agent_loop.stream_agent_loop(
            "https://api.openai.com/v1",
            "qwen3:14b",
            [{"role": "user", "content": "reply to this email"}],
            max_rounds=1,
            relevant_tools={
                "list_email_accounts", "list_emails", "read_email", "search_emails",
                "mcp__email__list_emails", "mcp__email__read_email", "mcp__email__search_emails",
                "ui_control",
            },
            owner="admin",
            active_document=active_email,
        )
    )

    assert sent_tools
    names = _schema_names(sent_tools[0])
    assert "search_emails" not in names
    assert "mcp__email__search_emails" not in names
    assert "list_emails" not in names
    assert "mcp__email__list_emails" not in names
    assert "read_email" not in names
    assert "mcp__email__read_email" not in names


def _capture_model_messages(monkeypatch, messages):
    monkeypatch.setattr(
        agent_loop,
        "get_setting",
        lambda key, default=None: default,
        raising=False,
    )
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(
        agent_loop,
        "estimate_tokens",
        lambda *args, **kwargs: 10,
        raising=False,
    )
    monkeypatch.setattr(
        agent_loop,
        "blocked_tools_for_owner",
        lambda owner: set(),
        raising=False,
    )

    captured = []

    async def fake_stream(_candidates, llm_messages, **kwargs):
        captured.append(llm_messages)
        yield "data: " + json.dumps({"delta": "ok"}) + "\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(
        agent_loop,
        "stream_llm_with_fallback",
        fake_stream,
        raising=False,
    )

    _collect(
        agent_loop.stream_agent_loop(
            "https://api.openai.com/v1",
            "qwen3:14b",
            messages,
            max_rounds=1,
            relevant_tools={"ask_user", "list_emails", "create_document"},
            owner="admin",
        )
    )

    assert captured
    return "\n".join(
        str(message.get("content") or "")
        for message in captured[0]
        if message.get("role") == "system"
    )


def test_ask_user_freeform_answer_gets_resume_directive(monkeypatch):
    system_text = _capture_model_messages(
        monkeypatch,
        _messages("Només els del 2025"),
    )

    assert "answer to the immediately preceding `ask_user` interaction" in system_text
    assert "continue the underlying task" in system_text


def test_normal_turn_does_not_get_ask_user_resume_directive(monkeypatch):
    system_text = _capture_model_messages(
        monkeypatch,
        [
            {
                "role": "user",
                "content": (
                    "Busca els correus de "
                    "odysseus-regression-fixture@gmail.com."
                ),
            }
        ],
    )

    assert "answer to the immediately preceding `ask_user` interaction" not in system_text
    assert "Do not merely echo or acknowledge" not in system_text



def _patch_basic_agent_dependencies(monkeypatch):
    monkeypatch.setattr(
        agent_loop,
        "get_setting",
        lambda key, default=None: default,
        raising=False,
    )
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(
        agent_loop,
        "estimate_tokens",
        lambda *args, **kwargs: 10,
        raising=False,
    )
    monkeypatch.setattr(
        agent_loop,
        "blocked_tools_for_owner",
        lambda owner: set(),
        raising=False,
    )


def test_ask_user_text_only_attempt_gets_one_structural_retry_and_then_tools(monkeypatch):
    _patch_basic_agent_dependencies(monkeypatch)

    model_calls = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_calls.append([
            dict(message)
            for message in messages
        ])

        if len(model_calls) == 1:
            # Reproduce production exactly: Catalan promise, no tool call.
            yield "data: " + json.dumps({
                "delta": (
                    "Sí. Buscaré els correus de "
                    "odysseus-regression-fixture@gmail.com i crearé "
                    "un document Word ordenat per data."
                )
            }) + "\n\n"
        else:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [
                    {
                        "name": "list_emails",
                        "arguments": "{}",
                    }
                ],
            }) + "\n\n"

        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        return (
            block.tool_type,
            {
                "output": "[]",
                "exit_code": 0,
            },
        )

    monkeypatch.setattr(
        agent_loop,
        "stream_llm_with_fallback",
        fake_stream,
        raising=False,
    )
    monkeypatch.setattr(
        agent_loop,
        "execute_tool_block",
        fake_execute,
        raising=False,
    )

    _collect(
        agent_loop.stream_agent_loop(
            "https://api.openai.com/v1",
            "gpt-4o",
            _messages("Sí"),
            max_rounds=2,
            relevant_tools={"ask_user", "list_emails", "create_document"},
            owner="admin",
            _is_teacher_run=True,
        )
    )

    assert len(model_calls) == 2
    retry_system = "\n".join(
        str(message.get("content") or "")
        for message in model_calls[1]
        if message.get("role") == "system"
    )
    assert "previous attempt ended without making a tool call" in retry_system
    assert "execute the appropriate available tool now" in retry_system
    assert "If the answer rejects, cancels" in retry_system
    # Native email calls are converted to their executable MCP tool name
    # before execute_tool_block receives them.
    assert executed == ["mcp__email__list_emails"]


def test_ask_user_rejection_retry_does_not_force_action_tool(monkeypatch):
    _patch_basic_agent_dependencies(monkeypatch)

    model_calls = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_calls.append([
            dict(message)
            for message in messages
        ])

        if len(model_calls) == 1:
            yield "data: " + json.dumps({
                "delta": "No, no continuïs."
            }) + "\n\n"
        else:
            yield "data: " + json.dumps({
                "delta": "Entesos. No continuaré amb aquesta acció."
            }) + "\n\n"

        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        return (
            block.tool_type,
            {
                "output": "unexpected",
                "exit_code": 0,
            },
        )

    monkeypatch.setattr(
        agent_loop,
        "stream_llm_with_fallback",
        fake_stream,
        raising=False,
    )
    monkeypatch.setattr(
        agent_loop,
        "execute_tool_block",
        fake_execute,
        raising=False,
    )

    _collect(
        agent_loop.stream_agent_loop(
            "https://api.openai.com/v1",
            "gpt-4o",
            _messages("No"),
            max_rounds=3,
            relevant_tools={"ask_user", "list_emails", "create_document"},
            owner="admin",
            _is_teacher_run=True,
        )
    )

    # Exactly one structural reconsideration, then the rejection is accepted.
    assert len(model_calls) == 2
    assert executed == []


def test_immediate_duplicate_ask_user_is_suppressed_then_action_runs(monkeypatch):
    _patch_basic_agent_dependencies(monkeypatch)
    model_calls = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_calls.append(messages)
        if len(model_calls) == 1:
            yield "data: " + json.dumps({"delta": "Entesos, continuaré."}) + "\n\n"
        elif len(model_calls) == 2:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{
                    "name": "ask_user",
                    "arguments": json.dumps({
                        "question": (
                            "Vols que busqui els correus electrònics rebuts de "
                            "odysseus-regression-fixture@gmail.com, i els ordeni per data ara?"
                        ),
                        "options": [{"label": "Sí"}, {"label": "No"}],
                    }),
                }],
            }) + "\n\n"
        elif len(model_calls) == 3:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{"name": "list_emails", "arguments": "{}"}],
            }) + "\n\n"
        else:
            yield "data: " + json.dumps({"delta": "Fet."}) + "\n\n"
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        return block.tool_type, {"output": "[]", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)

    chunks = _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", _messages("Sí"),
        max_rounds=4,
        relevant_tools={"ask_user", "list_emails", "create_document"},
        owner="admin", _is_teacher_run=True,
    ))

    assert len(model_calls) == 4
    assert executed == ["mcp__email__list_emails"]
    events = [
        json.loads(chunk[6:]) for chunk in chunks
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]")
    ]
    assert not any(event.get("type") == "ask_user" for event in events)
    duplicate_retry_text = "\n".join(
        str(message.get("content") or "")
        for message in model_calls[2]
        if message.get("role") == "system"
    )
    assert "already been answered" in duplicate_retry_text


def test_materially_different_ask_user_clarification_remains_legal(monkeypatch):
    _patch_basic_agent_dependencies(monkeypatch)
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        yield "data: " + json.dumps({
            "type": "tool_calls",
            "calls": [{
                "name": "ask_user",
                "arguments": json.dumps({
                    "question": "Quin interval de dates vols incloure?",
                    "options": [{"label": "2025"}, {"label": "Tot"}],
                }),
            }],
        }) + "\n\n"
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        return block.tool_type, {
            "ask_user": {
                "question": "Quin interval de dates vols incloure?",
                "options": [{"label": "2025"}, {"label": "Tot"}],
            },
            "exit_code": 0,
        }

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)

    chunks = _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", _messages("Sí"),
        max_rounds=2,
        relevant_tools={"ask_user", "list_emails", "create_document"},
        owner="admin", _is_teacher_run=True,
    ))

    assert executed == ["ask_user"]
    assert any('"type": "ask_user"' in chunk for chunk in chunks)


def _answered_question_messages(question, answer="Yes"):
    return [
        {"role": "user", "content": "Complete the underlying task."},
        {
            "role": "assistant",
            "content": question,
            "metadata": {"tool_events": [{
                "tool": "ask_user",
                "ask_user": {
                    "question": question,
                    "options": [{"label": "Yes"}, {"label": "No"}],
                },
            }]},
        },
        {"role": "user", "content": answer},
    ]


def test_live_textual_ask_user_reconfirmation_continues_to_email_tool(monkeypatch):
    _patch_basic_agent_dependencies(monkeypatch)
    question = (
        "Vols que continui buscant els correus de niniprimer@gmail.com i creant "
        "un document Word ordenat per data?"
    )
    responses = [
        "Sí, vols que continui buscant els correus de niniprimer@gmail.com i "
        "creant un document Word ordenat per data?",
        "Sí. Estic preparat per continuar. Vols que busqui els correus de "
        "niniprimer@gmail.com i crei un document Word ordenat per data?",
    ]
    model_calls = []
    executed = []
    log_messages = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_calls.append(messages)
        call_number = len(model_calls)
        if call_number <= 2:
            yield "data: " + json.dumps({"delta": responses[call_number - 1]}) + "\n\n"
        elif call_number == 3:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{"name": "list_email_accounts", "arguments": "{}"}],
            }) + "\n\n"
        else:
            yield 'data: {"delta": "Fet."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        return block.tool_type, {"output": "accounts", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)
    monkeypatch.setattr(
        agent_loop.logger,
        "info",
        lambda message, *args: log_messages.append(message % args if args else message),
    )
    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o",
        _answered_question_messages(question, "Sí"),
        max_rounds=4,
        relevant_tools={"ask_user", "list_email_accounts", "search_emails", "create_document"},
        owner="admin", _is_teacher_run=True,
    ))

    assert len(model_calls) == 4
    assert executed == ["mcp__email__list_email_accounts"]
    structural_instruction = "\n".join(
        str(message.get("content") or "") for message in model_calls[1]
        if message.get("role") == "system"
    )
    textual_instruction = "\n".join(
        str(message.get("content") or "") for message in model_calls[2]
        if message.get("role") == "system"
    )
    assert "previous attempt ended without making a tool call" in structural_instruction
    assert "response text repeated or re-asked" in textual_instruction
    assert any(
        "textual ask_user reconfirmation detected on round 2" in message
        for message in log_messages
    )
    assert not any(tool == "ask_user" for tool in executed)


def test_generic_textual_reconfirmation_with_prefix_is_detected(monkeypatch):
    _patch_basic_agent_dependencies(monkeypatch)
    question = "Should I search the archived project messages and prepare the requested report?"
    model_calls = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_calls.append(messages)
        if len(model_calls) == 1:
            yield 'data: {"delta": "Understood."}\n\n'
        elif len(model_calls) == 2:
            yield "data: " + json.dumps({
                "delta": "I am ready now. Should I search the archived project messages and prepare the requested report?"
            }) + "\n\n"
        elif len(model_calls) == 3:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{"name": "list_email_accounts", "arguments": "{}"}],
            }) + "\n\n"
        else:
            yield 'data: {"delta": "Done."}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        return block.tool_type, {"output": "ok", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)
    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", _answered_question_messages(question),
        max_rounds=4, relevant_tools={"ask_user", "list_email_accounts"},
        owner="admin", _is_teacher_run=True,
    ))

    assert len(model_calls) == 4
    assert executed == ["mcp__email__list_email_accounts"]


def test_materially_different_textual_clarification_does_not_retry(monkeypatch):
    _patch_basic_agent_dependencies(monkeypatch)
    model_calls = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_calls.append(messages)
        text = "Understood." if len(model_calls) == 1 else "Which date range should the report include?"
        yield "data: " + json.dumps({"delta": text}) + "\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o",
        _answered_question_messages("Should I search the mailbox and create a report?"),
        max_rounds=5, relevant_tools={"ask_user", "search_emails", "create_document"},
        owner="admin", _is_teacher_run=True,
    ))

    assert len(model_calls) == 2


def test_textual_reconfirmation_retry_is_bounded(monkeypatch):
    _patch_basic_agent_dependencies(monkeypatch)
    question = "Should I search the mailbox and create the requested document?"
    model_calls = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_calls.append(messages)
        yield "data: " + json.dumps({"delta": question}) + "\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", _answered_question_messages(question),
        max_rounds=20, relevant_tools={"ask_user", "search_emails", "create_document"},
        owner="admin", _is_teacher_run=True,
    ))

    # One structural retry, two independently bounded textual retries, then stop.
    assert len(model_calls) == 4


def test_textual_reconfirmation_guard_stops_after_substantive_progress(monkeypatch):
    _patch_basic_agent_dependencies(monkeypatch)
    question = "Should I search the mailbox and create the requested document?"
    model_calls = []
    executed = []

    async def fake_stream(_candidates, messages, **kwargs):
        model_calls.append(messages)
        if len(model_calls) == 1:
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{"name": "list_email_accounts", "arguments": "{}"}],
            }) + "\n\n"
        else:
            yield "data: " + json.dumps({"delta": question}) + "\n\n"
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        executed.append(block.tool_type)
        return block.tool_type, {"output": "ok", "exit_code": 0}

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)
    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o", _answered_question_messages(question),
        max_rounds=6, relevant_tools={"ask_user", "list_email_accounts"},
        owner="admin", _is_teacher_run=True,
    ))

    assert executed == ["mcp__email__list_email_accounts"]
    assert len(model_calls) == 2
