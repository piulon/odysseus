import unittest
from pathlib import Path


class PalworldLifecycleFastPathStaticTests(
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

    def lifecycle_block(self):
        start = self.agent_source.index(
            "# Explicit Palworld start and "
            "confirmed stop requests"
        )

        end = self.agent_source.index(
            "# Explicit confirmed Palworld "
            "restart requests",
            start,
        )

        return self.agent_source[
            start:end
        ]

    def test_lifecycle_precedes_restart(self):
        lifecycle = self.agent_source.index(
            "# Explicit Palworld start and "
            "confirmed stop requests"
        )

        restart = self.agent_source.index(
            "# Explicit confirmed Palworld "
            "restart requests"
        )

        self.assertLess(
            lifecycle,
            restart,
        )

    def test_lifecycle_is_admin_and_context_gated(self):
        block = self.lifecycle_block()

        for required in (
            "owner_is_admin_or_single_user",
            "session_id",
            "and not plan_mode",
            "and not approved_plan",
            "and not uploaded_files",
            'tool_policy.blocks("homelab")',
        ):
            with self.subTest(
                required=required
            ):
                self.assertIn(
                    required,
                    block,
                )

    def test_actions_are_not_in_llm_schema(self):
        for forbidden in (
            "palworld_start",
            "palworld_stop_confirmed",
            "/v1/palworld/start",
            "/v1/palworld/stop",
        ):
            with self.subTest(
                forbidden=forbidden
            ):
                self.assertNotIn(
                    forbidden,
                    self.schemas,
                )

    def test_tool_commands_contain_no_confirmation_code(self):
        block = self.lifecycle_block()

        self.assertIn(
            '"palworld_start"',
            block,
        )

        self.assertIn(
            '"palworld_stop_confirmed"',
            block,
        )

        self.assertNotIn(
            '"code":',
            block,
        )

    def test_failures_are_terminal(self):
        block = self.lifecycle_block()

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
