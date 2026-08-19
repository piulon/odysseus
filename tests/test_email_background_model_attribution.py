import asyncio
import json
import sqlite3

import pytest

from routes import email_helpers, email_pollers
from src import llm_core, task_endpoint


class _FakeImap:
    def __init__(self, raw_message):
        self.raw_message = raw_message
        self.logout_calls = 0

    def select(self, *_args, **_kwargs):
        return "OK", []

    def uid(self, command, *_args):
        if command == "SEARCH":
            return "OK", [b"1"]
        if command == "FETCH":
            return "OK", [(b"1 (RFC822)", self.raw_message)]
        raise AssertionError(f"unexpected IMAP command: {command}")

    def logout(self):
        self.logout_calls += 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "settings_key", "table", "content_column", "expected_content"),
    [
        ("summary", "email_auto_summarize", "email_summaries", "summary", "- Fallback summary"),
        ("reply", "email_auto_reply", "email_ai_replies", "reply", "Fallback reply"),
        ("tagging", "email_auto_tag", "email_tags", "tags", ["work", "action-needed"]),
    ],
)
async def test_email_poller_persists_actual_fallback_model(
    operation,
    settings_key,
    table,
    content_column,
    expected_content,
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "scheduled_emails.db"
    monkeypatch.setattr(email_helpers, "SCHEDULED_DB", db_path)
    monkeypatch.setattr(email_pollers, "SCHEDULED_DB", db_path)
    email_helpers._init_scheduled_db()

    settings = {
        "email_auto_summarize": False,
        "email_auto_reply": False,
        "email_auto_tag": False,
        "email_auto_spam": False,
        "email_auto_calendar": False,
        settings_key: True,
    }
    raw_message = (
        b"From: Sender <sender@example.com>\r\n"
        b"To: Alice <alice@example.com>\r\n"
        b"Subject: Background attribution\r\n"
        b"Message-ID: <background-attribution@example.com>\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n"
        b"\r\n"
        b"Please review this sufficiently long background email and reply with the requested action. "
        b"It contains enough text for summary and classification processing."
    )
    imap = _FakeImap(raw_message)
    candidates = [
        ("https://primary.example/v1", "primary-model", {"Authorization": "primary-secret"}),
        ("https://fallback.example/v1", "fallback-model", {"Authorization": "fallback-secret"}),
    ]
    attempted = []

    async def fake_llm_call(url, model, messages, **kwargs):
        attempted.append(model)
        if model == "primary-model":
            raise RuntimeError("primary unavailable")
        if operation == "summary":
            return "<<<SUMMARY>>>\n- Fallback summary\n<<<END>>>"
        if operation == "reply":
            return "<<<REPLY>>>\nFallback reply\n<<<END>>>"
        return json.dumps({"tags": ["work", "action-needed"], "spam": False, "reason": "Needs response"})

    async def fake_gate(label):
        assert label == "background task LLM"

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(email_pollers, "_load_settings", lambda: settings)
    monkeypatch.setattr(email_pollers, "_owner_for_email_account", lambda account_id: "alice")
    monkeypatch.setattr(email_pollers, "_imap_connect", lambda account_id, owner="": imap)
    monkeypatch.setattr(email_pollers, "resolve_task_candidates", lambda owner=None: candidates)
    monkeypatch.setattr(task_endpoint, "resolve_task_candidates", lambda **kwargs: candidates)
    monkeypatch.setattr(task_endpoint, "wait_for_interactive_quiet", fake_gate)
    monkeypatch.setattr(llm_core, "llm_call_async", fake_llm_call)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = await email_pollers._auto_summarize_pass_single(account_id="account-alice", max_process=1)

    assert not result.startswith("Error:")
    assert attempted == ["primary-model", "fallback-model"]
    assert imap.logout_calls == 1
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            f"SELECT owner, {content_column}, model_used FROM {table} WHERE message_id = ?",
            ("<background-attribution@example.com>",),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    owner, content, model_used = row
    if operation == "tagging":
        content = json.loads(content)
    assert owner == "alice"
    assert content == expected_content
    assert model_used == "fallback-model"
