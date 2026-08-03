import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.services.homelab.palworld_restart as restart_module
from src.services.homelab.action_client import (
    HomelabActionClient,
)
from src.services.homelab.palworld_lifecycle import (
    PalworldLifecycleBlockedError,
    classify_palworld_lifecycle_turn,
    execute_confirmed_palworld_stop,
    format_start_result,
    format_stop_confirmation,
    format_stop_result,
    prepare_palworld_stop_confirmation,
    start_palworld_verified,
)
from src.services.homelab.palworld_restart import (
    PalworldRestartConfirmationError,
    consume_pending_restart,
    consume_pending_stop,
    issue_pending_restart,
    issue_pending_stop,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        return False

    def read(self):
        return json.dumps(
            self.payload
        ).encode("utf-8")


class _ReadClient:
    def __init__(
        self,
        *,
        statuses,
        backups=None,
        calls=None,
    ):
        self.statuses = list(statuses)
        self.backups = list(
            backups or []
        )
        self.calls = (
            calls
            if calls is not None
            else []
        )

    def palworld_status(self):
        self.calls.append("status")
        return self.statuses.pop(0)

    def palworld_backups(self):
        self.calls.append("backups")
        return self.backups.pop(0)


class _ActionClient:
    def __init__(
        self,
        *,
        calls,
        start_response=None,
        stop_response=None,
        backup_response=None,
    ):
        self.calls = calls
        self.start_response = start_response
        self.stop_response = stop_response
        self.backup_response = backup_response

    def start_palworld(self):
        self.calls.append("start")
        return self.start_response

    def stop_palworld(self):
        self.calls.append("stop")
        return self.stop_response

    def create_palworld_backup(self):
        self.calls.append(
            "create_backup"
        )
        return self.backup_response


class PalworldLifecycleActionTests(
    unittest.TestCase
):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        base = Path(self.tempdir.name)

        self.file_patch = patch.object(
            restart_module,
            "PENDING_ACTIONS_FILE",
            base / "pending.json",
        )

        self.lock_patch = patch.object(
            restart_module,
            "PENDING_ACTIONS_LOCK_FILE",
            base / ".pending.lock",
        )

        self.file_patch.start()
        self.lock_patch.start()

    def tearDown(self):
        self.lock_patch.stop()
        self.file_patch.stop()
        self.tempdir.cleanup()

    def test_accepts_explicit_start_and_stop(self):
        cases = {
            "Inicia Palworld": {
                "action": "start",
                "kind": "request",
            },
            "Arranca el servidor de Palworld ahora": {
                "action": "start",
                "kind": "request",
            },
            "Detén Palworld": {
                "action": "stop",
                "kind": "request",
            },
            "Para el servidor de Palworld ahora": {
                "action": "stop",
                "kind": "request",
            },
            "Start the Palworld server": {
                "action": "start",
                "kind": "request",
            },
            "Stop the Palworld server": {
                "action": "stop",
                "kind": "request",
            },
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(
                    classify_palworld_lifecycle_turn(
                        text
                    ),
                    expected,
                )

    def test_rejects_ambiguous_or_other_actions(self):
        cases = (
            "¿Cómo inicio Palworld?",
            "No detengas Palworld",
            "Quizá deberíamos parar Palworld",
            "Inicia Palworld y Grafana",
            "Reinicia Palworld",
            "Crea una copia de Palworld",
        )

        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(
                    classify_palworld_lifecycle_turn(
                        text
                    )
                )

    def test_stop_confirmation_parser_is_strict(self):
        code = "PST-ABCDEFGH2345"

        self.assertEqual(
            classify_palworld_lifecycle_turn(
                "CONFIRMAR PARADA PALWORLD "
                + code,
                continuation=True,
            ),
            {
                "action": "stop",
                "kind": "confirmation",
                "code": code,
            },
        )

        self.assertIsNone(
            classify_palworld_lifecycle_turn(
                "sí, deténlo"
            )
        )

    def test_stop_and_restart_authorizations_are_independent(self):
        restart = issue_pending_restart(
            owner="pau",
            session_id="session-a",
            now=1000,
        )

        stop = issue_pending_stop(
            owner="pau",
            session_id="session-a",
            now=1000,
        )

        with self.assertRaises(
            PalworldRestartConfirmationError
        ):
            consume_pending_stop(
                owner="pau",
                session_id="session-a",
                code=restart["code"],
                now=1001,
            )

        with self.assertRaises(
            PalworldRestartConfirmationError
        ):
            consume_pending_restart(
                owner="pau",
                session_id="session-a",
                code=stop["code"],
                now=1001,
            )

        self.assertEqual(
            consume_pending_stop(
                owner="pau",
                session_id="session-a",
                code=stop["code"],
                now=1001,
            )["action"],
            "palworld.stop",
        )

        self.assertEqual(
            consume_pending_restart(
                owner="pau",
                session_id="session-a",
                code=restart["code"],
                now=1001,
            )["action"],
            "palworld.restart",
        )

    def test_players_block_stop_confirmation(self):
        reader = _ReadClient(
            statuses=[{
                "ok": True,
                "status": "running",
                "players": 2,
            }],
        )

        with self.assertRaises(
            PalworldLifecycleBlockedError
        ):
            prepare_palworld_stop_confirmation(
                owner="pau",
                session_id="session-a",
                read_client=reader,
            )

    def test_start_rejects_running_server(self):
        reader = _ReadClient(
            statuses=[{
                "ok": True,
                "status": "running",
                "players": 0,
            }],
        )

        actor = _ActionClient(
            calls=[],
        )

        with self.assertRaises(
            PalworldLifecycleBlockedError
        ):
            start_palworld_verified(
                read_client=reader,
                action_client=actor,
            )

        self.assertEqual(
            actor.calls,
            [],
        )

    def test_action_client_uses_dedicated_endpoints(self):
        captured = []

        def fake_urlopen(
            request,
            timeout,
        ):
            captured.append({
                "method":
                    request.get_method(),
                "url":
                    request.full_url,
                "timeout":
                    timeout,
                "headers": {
                    key.casefold(): value
                    for key, value
                    in request.header_items()
                },
            })

            return _Response({
                "ok": True,
                "result": {
                    "ok": True,
                    "status": "completed",
                    "result": "success",
                },
            })

        client = HomelabActionClient(
            base_url="http://operator.test:8765",
            action_token="action-secret",
            timeout=240,
        )

        with patch(
            "urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            client.start_palworld()
            client.stop_palworld()

        self.assertEqual(
            [
                item["url"]
                for item in captured
            ],
            [
                "http://operator.test:8765"
                "/v1/palworld/start",
                "http://operator.test:8765"
                "/v1/palworld/stop",
            ],
        )

        for item in captured:
            self.assertEqual(
                item["method"],
                "POST",
            )

            self.assertEqual(
                item["headers"][
                    "x-homelab-action-token"
                ],
                "action-secret",
            )

            self.assertNotIn(
                "x-homelab-read-token",
                item["headers"],
            )

    def test_verified_start_checks_before_and_after(self):
        calls = []

        reader = _ReadClient(
            statuses=[
                {
                    "ok": False,
                    "status": "inactive",
                    "players": 0,
                },
                {
                    "ok": True,
                    "status": "running",
                    "players": 0,
                },
            ],
            calls=calls,
        )

        actor = _ActionClient(
            calls=calls,
            start_response={
                "ok": True,
                "result": {
                    "ok": True,
                    "action": "palworld.start",
                    "status": "completed",
                    "result": "success",
                },
            },
        )

        result = start_palworld_verified(
            read_client=reader,
            action_client=actor,
        )

        self.assertEqual(
            calls,
            [
                "status",
                "start",
                "status",
            ],
        )

        self.assertEqual(
            result["after"]["status"],
            "running",
        )

    def test_confirmed_stop_orders_backup_first(self):
        issued = issue_pending_stop(
            owner="pau",
            session_id="session-a",
        )

        calls = []

        reader = _ReadClient(
            statuses=[
                {
                    "ok": True,
                    "status": "running",
                    "players": 0,
                },
                {
                    "ok": False,
                    "status": "inactive",
                    "players": 0,
                },
            ],
            backups=[
                {
                    "ok": True,
                    "count": 15,
                    "latest_backup":
                        "old.tar.gz",
                    "integrity": "correcta",
                },
                {
                    "ok": True,
                    "count": 16,
                    "latest_backup":
                        "new.tar.gz",
                    "size": "5.4 MiB",
                    "integrity": "correcta",
                },
            ],
            calls=calls,
        )

        actor = _ActionClient(
            calls=calls,
            backup_response={
                "ok": True,
                "result": {
                    "ok": True,
                    "status": "completed",
                    "result": "success",
                },
            },
            stop_response={
                "ok": True,
                "result": {
                    "ok": True,
                    "action": "palworld.stop",
                    "status": "completed",
                    "result": "success",
                },
            },
        )

        result = (
            execute_confirmed_palworld_stop(
                owner="pau",
                session_id="session-a",
                code=issued["code"],
                read_client=reader,
                action_client=actor,
            )
        )

        self.assertEqual(
            calls,
            [
                "status",
                "backups",
                "create_backup",
                "backups",
                "stop",
                "status",
            ],
        )

        self.assertEqual(
            result["after"]["status"],
            "inactive",
        )

    def test_formatters_report_verified_results(self):
        confirmation = (
            format_stop_confirmation(
                {
                    "authorization": {
                        "code":
                            "PST-ABCDEFGH2345",
                        "expires_in_seconds":
                            300,
                    },
                },
                "Detén Palworld",
            )
        )

        self.assertIn(
            "CONFIRMAR PARADA PALWORLD "
            "PST-ABCDEFGH2345",
            confirmation,
        )

        start_text = format_start_result(
            {
                "after": {
                    "status": "running",
                    "players": 0,
                },
            },
            "Inicia Palworld",
        )

        self.assertIn(
            "iniciado correctamente",
            start_text,
        )

        stop_text = format_stop_result(
            {
                "backup": {
                    "after": {
                        "latest_backup":
                            "new.tar.gz",
                        "size": "5.4 MiB",
                    },
                },
                "after": {
                    "status": "inactive",
                    "players": 0,
                },
            },
            "CONFIRMAR PARADA PALWORLD "
            "PST-ABCDEFGH2345",
        )

        self.assertIn(
            "new.tar.gz",
            stop_text,
        )

        self.assertIn(
            "inactive",
            stop_text,
        )


if __name__ == "__main__":
    unittest.main()
