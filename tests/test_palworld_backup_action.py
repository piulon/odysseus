import asyncio
import json
import unittest
from unittest.mock import patch

from src.services.homelab.action_client import (
    HomelabActionClient,
)
from src.services.homelab.palworld_backup import (
    PalworldBackupVerificationError,
    classify_explicit_palworld_backup_request,
    create_verified_palworld_backup,
    format_verified_palworld_backup,
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


class _Reader:
    def __init__(self, responses):
        self.responses = list(responses)

    def palworld_backups(self):
        return self.responses.pop(0)


class _Actor:
    def __init__(self, response):
        self.response = response

    def create_palworld_backup(self):
        return self.response


class PalworldBackupActionTests(
    unittest.TestCase
):
    def test_accepts_explicit_commands(self):
        cases = (
            "Crea una copia de Palworld",
            "Haz un backup de Palworld ahora",
            "Créame una copia de seguridad "
            "de Palworld",
            "Fes una còpia de seguretat "
            "de Palworld ara",
            "Create a Palworld backup now",
            "Please make a backup of Palworld",
        )

        for text in cases:
            with self.subTest(text=text):
                self.assertTrue(
                    classify_explicit_palworld_backup_request(
                        text
                    )
                )

    def test_rejects_ambiguous_or_unrelated_text(self):
        cases = (
            "¿Cómo creo un backup de Palworld?",
            "¿Deberíamos crear una copia "
            "de Palworld?",
            "No crees una copia de Palworld",
            "Estado de las copias de Palworld",
            "Reinicia Palworld",
            "Palworld backup",
            "Puede que necesitemos un backup "
            "de Palworld",
            "Crea una copia de Grafana",
        )

        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(
                    classify_explicit_palworld_backup_request(
                        text
                    )
                )

    def test_rejects_continuations(self):
        self.assertFalse(
            classify_explicit_palworld_backup_request(
                "Crea una copia de Palworld",
                continuation=True,
            )
        )

    def test_action_client_uses_restricted_post(self):
        captured = {}

        def fake_urlopen(
            request,
            timeout,
        ):
            captured["method"] = (
                request.get_method()
            )
            captured["url"] = request.full_url
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
            })

        client = HomelabActionClient(
            base_url="http://operator.test:8765",
            action_token="action-secret",
            timeout=123,
        )

        with patch(
            "urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            result = (
                client.create_palworld_backup()
            )

        self.assertTrue(result["ok"])
        self.assertEqual(
            captured["method"],
            "POST",
        )
        self.assertEqual(
            captured["url"],
            "http://operator.test:8765"
            "/v1/palworld/backups/create",
        )
        self.assertEqual(
            captured["timeout"],
            123,
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

    def test_verified_backup_requires_change(self):
        before = {
            "ok": True,
            "count": 12,
            "latest_backup": "old.tar.gz",
            "integrity": "correcta",
        }

        after = {
            "ok": True,
            "count": 13,
            "latest_backup": "new.tar.gz",
            "size": "5.2 MiB",
            "integrity": "correcta",
        }

        action = {
            "ok": True,
            "result": {
                "ok": True,
                "status": "completed",
                "result": "success",
            },
        }

        result = create_verified_palworld_backup(
            read_client=_Reader([
                before,
                after,
            ]),
            action_client=_Actor(action),
        )

        self.assertEqual(
            result["after"]["latest_backup"],
            "new.tar.gz",
        )

    def test_unchanged_inventory_is_rejected(self):
        unchanged = {
            "ok": True,
            "count": 12,
            "latest_backup": "old.tar.gz",
            "integrity": "correcta",
        }

        action = {
            "ok": True,
            "result": {
                "ok": True,
                "status": "completed",
                "result": "success",
            },
        }

        with self.assertRaises(
            PalworldBackupVerificationError
        ):
            create_verified_palworld_backup(
                read_client=_Reader([
                    unchanged,
                    unchanged,
                ]),
                action_client=_Actor(action),
            )

    def test_formatter_reports_verified_file(self):
        result = {
            "after": {
                "ok": True,
                "count": 13,
                "latest_backup": "new.tar.gz",
                "size": "5.2 MiB",
                "integrity": "correcta",
            },
        }

        text = format_verified_palworld_backup(
            result,
            "Crea una copia de Palworld",
        )

        self.assertIn("new.tar.gz", text)
        self.assertIn("5.2 MiB", text)
        self.assertIn(
            "integridad correcta",
            text,
        )
        self.assertIn("13 copias", text)


if __name__ == "__main__":
    unittest.main()
