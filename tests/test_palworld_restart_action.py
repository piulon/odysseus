import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.services.homelab.palworld_restart as restart_module
from src.services.homelab.action_client import (
    HomelabActionClient,
)
from src.services.homelab.palworld_restart import (
    PalworldRestartBlockedError,
    PalworldRestartConfirmationError,
    classify_palworld_restart_turn,
    consume_pending_restart,
    execute_confirmed_palworld_restart,
    format_restart_confirmation,
    format_verified_palworld_restart,
    issue_pending_restart,
    prepare_palworld_restart_confirmation,
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
        statuses=None,
        backups=None,
        calls=None,
    ):
        self.statuses = list(
            statuses or []
        )

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
        backup_response,
        restart_response,
        calls,
    ):
        self.backup_response = (
            backup_response
        )

        self.restart_response = (
            restart_response
        )

        self.calls = calls

    def create_palworld_backup(self):
        self.calls.append(
            "create_backup"
        )

        return self.backup_response

    def restart_palworld(self):
        self.calls.append("restart")

        return self.restart_response


class PalworldRestartActionTests(
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

    def test_accepts_explicit_restart_requests(self):
        cases = (
            "Reinicia Palworld",
            "Reinicia el servidor de Palworld ahora",
            "Por favor reiniciar Palworld",
            "Reinicia el servidor de Palworld ara",
            "Restart Palworld",
            "Please restart the Palworld server now",
        )

        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    classify_palworld_restart_turn(
                        text
                    ),
                    {"kind": "request"},
                )

    def test_rejects_ambiguous_or_other_actions(self):
        cases = (
            "¿Cómo reinicio Palworld?",
            "No reinicies Palworld",
            "Quizá deberíamos reiniciar Palworld",
            "Reinicia Palworld y Caddy",
            "Detén Palworld",
            "Inicia Palworld",
            "Crea una copia de Palworld",
            "Reinicia Grafana",
        )

        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(
                    classify_palworld_restart_turn(
                        text
                    )
                )

    def test_confirmation_parser_is_strict(self):
        code = "PRW-ABCDEFGH2345"

        self.assertEqual(
            classify_palworld_restart_turn(
                "CONFIRMAR REINICIO PALWORLD "
                + code,
                continuation=True,
            ),
            {
                "kind": "confirmation",
                "code": code,
            },
        )

        self.assertIsNone(
            classify_palworld_restart_turn(
                "sí, confirma el reinicio"
            )
        )

    def test_confirmation_is_one_time_and_session_bound(self):
        issued = issue_pending_restart(
            owner="pau",
            session_id="session-a",
            now=1000,
        )

        code = issued["code"]

        with self.assertRaises(
            PalworldRestartConfirmationError
        ):
            consume_pending_restart(
                owner="pau",
                session_id="session-b",
                code=code,
                now=1001,
            )

        consumed = consume_pending_restart(
            owner="pau",
            session_id="session-a",
            code=code,
            now=1001,
        )

        self.assertEqual(
            consumed["action"],
            "palworld.restart",
        )

        with self.assertRaises(
            PalworldRestartConfirmationError
        ):
            consume_pending_restart(
                owner="pau",
                session_id="session-a",
                code=code,
                now=1002,
            )

    def test_confirmation_is_owner_bound(self):
        issued = issue_pending_restart(
            owner="pau",
            session_id="session-a",
            now=1000,
        )

        with self.assertRaises(
            PalworldRestartConfirmationError
        ):
            consume_pending_restart(
                owner="other",
                session_id="session-a",
                code=issued["code"],
                now=1001,
            )

    def test_confirmation_expires(self):
        issued = issue_pending_restart(
            owner="pau",
            session_id="session-a",
            ttl=30,
            now=1000,
        )

        with self.assertRaises(
            PalworldRestartConfirmationError
        ):
            consume_pending_restart(
                owner="pau",
                session_id="session-a",
                code=issued["code"],
                now=1031,
            )

    def test_players_block_confirmation(self):
        reader = _ReadClient(
            statuses=[{
                "ok": True,
                "status": "running",
                "players": 2,
            }],
        )

        with self.assertRaises(
            PalworldRestartBlockedError
        ):
            prepare_palworld_restart_confirmation(
                owner="pau",
                session_id="session-a",
                read_client=reader,
            )

        self.assertFalse(
            restart_module
            .PENDING_ACTIONS_FILE
            .exists()
        )

    def test_action_client_uses_dedicated_endpoint(self):
        captured = {}

        def fake_urlopen(
            request,
            timeout,
        ):
            captured["method"] = (
                request.get_method()
            )

            captured["url"] = (
                request.full_url
            )

            captured["timeout"] = timeout

            captured["headers"] = {
                key.casefold(): value
                for key, value
                in request.header_items()
            }

            return _Response({
                "ok": True,
                "result": {
                    "ok": True,
                    "status": "completed",
                    "result": "success",
                },
                "server": {
                    "ok": True,
                    "status": "running",
                    "players": 0,
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
            result = (
                client.restart_palworld()
            )

        self.assertTrue(result["ok"])

        self.assertEqual(
            captured["method"],
            "POST",
        )

        self.assertEqual(
            captured["url"],
            "http://operator.test:8765"
            "/v1/palworld/restart",
        )

        self.assertEqual(
            captured["headers"][
                "x-homelab-action-token"
            ],
            "action-secret",
        )

        self.assertNotIn(
            "x-homelab-read-token",
            captured["headers"],
        )

    def test_confirmed_restart_orders_backup_first(self):
        issued = issue_pending_restart(
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
                    "ok": True,
                    "status": "running",
                    "players": 0,
                },
            ],
            backups=[
                {
                    "ok": True,
                    "count": 14,
                    "latest_backup":
                        "old.tar.gz",
                    "integrity": "correcta",
                },
                {
                    "ok": True,
                    "count": 15,
                    "latest_backup":
                        "new.tar.gz",
                    "size": "5.4 MiB",
                    "integrity": "correcta",
                },
            ],
            calls=calls,
        )

        actor = _ActionClient(
            backup_response={
                "ok": True,
                "result": {
                    "ok": True,
                    "status": "completed",
                    "result": "success",
                },
            },
            restart_response={
                "ok": True,
                "result": {
                    "ok": True,
                    "status": "completed",
                    "result": "success",
                },
                "server": {
                    "ok": True,
                    "status": "running",
                    "players": 0,
                },
            },
            calls=calls,
        )

        result = (
            execute_confirmed_palworld_restart(
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
                "restart",
                "status",
            ],
        )

        self.assertEqual(
            result["backup"]["after"][
                "latest_backup"
            ],
            "new.tar.gz",
        )

    def test_formatters_do_not_expose_internal_state(self):
        confirmation = (
            format_restart_confirmation(
                {
                    "authorization": {
                        "code":
                            "PRW-ABCDEFGH2345",
                        "expires_in_seconds":
                            300,
                    },
                },
                "Reinicia Palworld",
            )
        )

        self.assertIn(
            "CONFIRMAR REINICIO PALWORLD "
            "PRW-ABCDEFGH2345",
            confirmation,
        )

        result = {
            "backup": {
                "after": {
                    "latest_backup":
                        "new.tar.gz",
                    "size": "5.4 MiB",
                },
            },
            "after": {
                "ok": True,
                "status": "running",
                "players": 0,
            },
        }

        success = (
            format_verified_palworld_restart(
                result,
                "CONFIRMAR REINICIO PALWORLD "
                "PRW-ABCDEFGH2345",
            )
        )

        self.assertIn(
            "new.tar.gz",
            success,
        )

        self.assertIn(
            "running",
            success,
        )


if __name__ == "__main__":
    unittest.main()
