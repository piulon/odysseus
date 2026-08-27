import asyncio
from email.message import EmailMessage

import pytest

pytest.importorskip("mcp")

import mcp_servers.email_server as es


class FakeSearchConn:
    def __init__(self, *, matches=None, select_failures=None, search_failures=None):
        self.matches = matches or {}
        self.select_failures = set(select_failures or ())
        self.search_failures = set(search_failures or ())
        self.selected = None
        self.calls = []

    def list(self):
        return "OK", [
            b'(\\HasNoChildren) "/" "INBOX"',
            b'(\\HasNoChildren \\Sent) "/" "Localized Sent"',
            b'(\\HasNoChildren \\All) "/" "Localized All"',
        ]

    def select(self, folder, readonly=False):
        physical = folder[1:-1]
        self.calls.append(("select", physical, readonly))
        self.selected = physical
        if physical in self.select_failures:
            return "NO", [b"missing"]
        return "OK", [b"1"]

    def uid(self, command, *args):
        self.calls.append(("uid", self.selected, command, *args))
        if command == "SEARCH":
            if self.selected in self.search_failures:
                return "NO", [b"search rejected"]
            return "OK", [" ".join(self.matches.get(self.selected, [])).encode()]
        if command == "FETCH":
            uid = args[0].decode()
            msg = EmailMessage()
            msg["Subject"] = f"Message {uid}"
            msg["From"] = "Sender <sender@example.com>"
            msg["Date"] = "Tue, 25 Aug 2026 10:00:00 +0200"
            msg["Message-ID"] = f"<{uid}@example.com>"
            return "OK", [(b"header", msg.as_bytes())]
        raise AssertionError(command)

    def logout(self):
        self.calls.append(("logout",))


def run_search(monkeypatch, conn, folders=None):
    monkeypatch.setattr(es, "_imap_connect", lambda account=None: conn)
    monkeypatch.setattr(es, "_get_cached_summaries", lambda: {})
    return es._search_emails("sender@example.com", folders=folders, max_results=20)


def run_tool_search(monkeypatch, conn, folders):
    monkeypatch.setattr(es, "_imap_connect", lambda account=None: conn)
    monkeypatch.setattr(es, "_get_cached_summaries", lambda: {})
    monkeypatch.setattr(es, "_read_accounts_from_db", lambda: [])
    return asyncio.run(es.call_tool("search_emails", {
        "query": "sender@example.com",
        "folders": folders,
    }))


def test_bare_sender_criterion_and_provider_folders_are_used(monkeypatch):
    conn = FakeSearchConn()
    results = run_search(monkeypatch, conn)

    searches = [call for call in conn.calls if call[0] == "uid" and call[2] == "SEARCH"]
    assert [call[1] for call in searches] == ["INBOX", "Localized Sent", "Localized All"]
    assert all('FROM "sender@example.com"' in call[4] for call in searches)
    assert all('FROM "from:sender@example.com"' not in call[4] for call in searches)
    assert [(d["logical_folder"], d["resolved_folder"]) for d in results.folder_diagnostics] == [
        ("INBOX", "INBOX"),
        ("Sent", "Localized Sent"),
        ("Archive", "Localized All"),
    ]


def test_all_successful_zero_is_genuine_zero(monkeypatch):
    results = run_search(monkeypatch, FakeSearchConn())

    assert results == []
    assert all(
        d["status"] == "ok" and d["match_count"] == 0 and d["error"] is None
        for d in results.folder_diagnostics
    )


def test_matches_survive_partial_folder_failure(monkeypatch):
    conn = FakeSearchConn(matches={"INBOX": ["42"]}, select_failures={"Localized Sent"})
    output = run_tool_search(monkeypatch, conn, ["INBOX", "Sent"])
    assert "Found 1 email(s)" in output[0].text
    assert "Folder: INBOX" in output[0].text
    assert "UID: 42" in output[0].text
    assert "FOLDER SEARCH ERRORS" in output[0].text
    assert "Sent -> Localized Sent: SELECT failed" in output[0].text


def test_optional_folder_failure_is_distinct_from_genuine_zero(monkeypatch):
    conn = FakeSearchConn(select_failures={"Localized Sent"})
    output = run_tool_search(monkeypatch, conn, ["INBOX", "Sent"])
    assert output[0].text.startswith('No emails matched "sender@example.com".')
    assert "FOLDER SEARCH ERRORS" in output[0].text


def test_all_requested_folders_fail_surfaces_failure(monkeypatch):
    conn = FakeSearchConn(select_failures={"Localized Sent"}, search_failures={"Localized All"})
    output = run_tool_search(monkeypatch, conn, ["Sent", "Archive"])
    assert output[0].text.startswith("Search failed in all requested folders:")
    assert "No emails matched" not in output[0].text
    assert "SELECT failed" in output[0].text
    assert "SEARCH failed" in output[0].text


def test_inbox_match_survives_missing_optional_folders(monkeypatch):
    conn = FakeSearchConn(
        matches={"INBOX": ["7"]},
        select_failures={"Localized Sent", "Localized All"},
    )
    results = run_search(monkeypatch, conn)

    assert [(item["uid"], item["_folder"], item["_resolved_folder"]) for item in results] == [
        ("7", "INBOX", "INBOX")
    ]
    assert results.folder_diagnostics[0]["status"] == "ok"
    assert [d["status"] for d in results.folder_diagnostics[1:]] == ["error", "error"]
