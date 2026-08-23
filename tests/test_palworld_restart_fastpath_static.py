import unittest
from pathlib import Path


class PalworldRestartFastPathStaticTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls):
        cls.agent_source = Path(
            "src/agent_loop.py"
        ).read_text(
            encoding="utf-8",
        )

        cls.schemas = Path(
            "src/tool_schemas.py"
        ).read_text(
            encoding="utf-8",
        )

    def restart_block(self):
        start = self.agent_source.index(
            "# Explicit confirmed Palworld "
            "restart requests"
        )

        end = self.agent_source.index(
            "# Explicit Palworld backup "
            "actions",
            start,
        )

        return self.agent_source[
            start:end
        ]

    def test_restart_precedes_backup_fastpath(self):
        restart_position = (
            self.agent_source.index(
                "# Explicit confirmed Palworld "
                "restart requests"
            )
        )

        backup_position = (
            self.agent_source.index(
                "# Explicit Palworld backup "
                "actions"
            )
        )

        self.assertLess(
            restart_position,
            backup_position,
        )

    def test_restart_is_admin_session_and_plan_gated(self):
        block = self.restart_block()

        for required in (
            "owner_is_admin_or_single_user",
            "session_id",
            "and not plan_mode",
            "and not approved_plan",
            "and not uploaded_files",
            "tool_policy.blocks(\"homelab\")",
        ):
            with self.subTest(
                required=required
            ):
                self.assertIn(
                    required,
                    block,
                )

    def test_restart_is_not_in_llm_schema(self):
        self.assertNotIn(
            "palworld_restart_confirmed",
            self.schemas,
        )

        self.assertNotIn(
            "/v1/palworld/restart",
            self.schemas,
        )

    def test_tool_command_contains_no_code(self):
        block = self.restart_block()

        self.assertIn(
            '''{
                "action":
                    "palworld_restart_confirmed",
            }''',
            block,
        )

        self.assertNotIn(
            '"code":',
            block,
        )

    def test_restart_failures_are_terminal(self):
        block = self.restart_block()

        self.assertIn(
            "failed closed",
            block,
        )

        self.assertIn(
            'yield "data: [DONE]',
            block,
        )

        self.assertIn(
            "return",
            block,
        )


if __name__ == "__main__":
    unittest.main()
