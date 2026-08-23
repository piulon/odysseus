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

import src.agent_loop as agent_loop
import src.tool_index as tool_index


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


def test_ask_user_yes_survives_bad_tool_rag_and_sends_email_document_tools(monkeypatch):
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

    retrieval_queries = []

    class BadToolIndex:
        def get_tools_for_query(self, query, k=8):
            retrieval_queries.append(query)
            # Reproduce the important part of the production failure:
            # semantic retrieval returns an unrelated but truthy tool set.
            return {"ls"}

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
    assert "create_document" in names

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
