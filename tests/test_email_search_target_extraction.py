import json

from src import agent_loop
from src.tool_utils import _truncate


def _search_output(targets):
    lines = [f'Found {len(targets)} email(s) matching "sender@example.com":', ""]
    for index, target in enumerate(targets, 1):
        lines.extend([
            f"{index}. **Message {index}**",
            "   From: Sender (sender@example.com)",
            "   Date: Tue, 25 Aug 2026 10:00:00 +0200",
            f"   Folder: {target.get('logical_folder', target['folder'])}",
            f"   Resolved Folder: {target['folder']}",
            f"   UID: {target['uid']}",
        ])
    return "\n".join(lines)


def _key(uid, folder, account=""):
    return agent_loop._email_retrieval_target_key(
        uid=uid,
        folder=folder,
        account=account,
    )


def test_production_mcp_stdout_extracts_resolved_folder_targets():
    targets = [
        {"uid": "101", "folder": "INBOX"},
        {"uid": "202", "logical_folder": "Sent", "folder": "[Gmail]/Enviados"},
        {"uid": "303", "logical_folder": "Archive", "folder": "[Gmail]/Todos"},
    ]

    extracted = agent_loop._email_search_targets_from_result(
        json.dumps({"query": "sender@example.com", "account": "work"}),
        {"stdout": _search_output(targets), "stderr": "", "exit_code": 0},
    )

    assert extracted == {
        _key("101", "INBOX", "work"),
        _key("202", "[Gmail]/Enviados", "work"),
        _key("303", "[Gmail]/Todos", "work"),
    }


def test_same_uid_in_different_folders_remains_two_targets():
    targets = [
        {"uid": "1234", "folder": "INBOX"},
        {"uid": "1234", "logical_folder": "Archive", "folder": "[Gmail]/Todos"},
    ]

    extracted = agent_loop._email_search_targets_from_result(
        "{}",
        {"stdout": _search_output(targets), "stderr": "", "exit_code": 0},
    )

    assert extracted == {
        _key("1234", "INBOX"),
        _key("1234", "[Gmail]/Todos"),
    }


def test_existing_output_shape_remains_supported():
    output = _search_output([{"uid": "55", "folder": "INBOX"}])

    assert agent_loop._email_search_targets_from_result(
        "{}", {"output": output, "exit_code": 0}
    ) == {_key("55", "INBOX")}


def test_list_emails_without_folder_uses_requested_folder():
    output = "\n".join([
        "Found 1 email(s):",
        "",
        "1. **A message**",
        "   From: Sender (sender@example.com)",
        "   Date: Tue, 25 Aug 2026 10:00:00 +0200",
        "   UID: 77",
    ])

    assert agent_loop._email_search_targets_from_result(
        json.dumps({"folder": "Archive", "account": "work"}),
        {"stdout": output, "stderr": "", "exit_code": 0},
    ) == {_key("77", "Archive", "work")}


def test_malformed_stdout_does_not_fabricate_targets():
    malformed = "UID 1234 may appear in prose, but this is not a numbered result row."

    assert agent_loop._email_search_targets_from_result(
        "{}", {"stdout": malformed, "stderr": "", "exit_code": 0}
    ) == set()


def test_diagnostic_rows_are_not_targets():
    output = "\n".join([
        "No emails matched sender@example.com.",
        "[FOLDER SEARCH ERRORS: Sent -> Sent: SELECT failed; UID: 9999]",
        "Resolved Folder: [Gmail]/Todos",
    ])

    assert agent_loop._email_search_targets_from_result(
        "{}", {"stdout": output, "stderr": "", "exit_code": 0}
    ) == set()


def test_full_execution_result_is_independent_of_persisted_truncation():
    targets = [
        {"uid": str(index), "folder": "INBOX"}
        for index in range(150)
    ]
    full_stdout = _search_output(targets)
    persisted_output = _truncate(full_stdout)

    assert len(persisted_output) < len(full_stdout)
    extracted = agent_loop._email_search_targets_from_result(
        "{}",
        {"stdout": full_stdout, "stderr": "", "exit_code": 0},
    )
    assert len(extracted) == 150
    assert _key("149", "INBOX") in extracted
