import unittest
from pathlib import Path


class PalworldBackupFastPathStaticTests(
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

    def test_action_precedes_read_only_fastpath(self):
        action_position = (
            self.agent_source.index(
                "# Explicit Palworld backup "
                "actions are deterministic"
            )
        )

        read_position = (
            self.agent_source.index(
                "# Deterministic homelab "
                "read-only requests"
            )
        )

        self.assertLess(
            action_position,
            read_position,
        )

    def test_action_is_admin_and_context_gated(self):
        start = self.agent_source.index(
            "# Explicit Palworld backup "
            "actions are deterministic"
        )

        end = self.agent_source.index(
            "# Deterministic homelab "
            "read-only requests",
            start,
        )

        block = self.agent_source[
            start:end
        ]

        for required in (
            "owner_is_admin_or_single_user",
            "and not plan_mode",
            "and not approved_plan",
            "and not uploaded_files",
            "and not _active_document_relevant",
            "tool_policy.blocks(\"homelab\")",
        ):
            with self.subTest(
                required=required
            ):
                self.assertIn(
                    required,
                    block,
                )

    def test_failure_is_terminal(self):
        start = self.agent_source.index(
            "# Explicit Palworld backup "
            "actions are deterministic"
        )

        end = self.agent_source.index(
            "# Deterministic homelab "
            "read-only requests",
            start,
        )

        block = self.agent_source[
            start:end
        ]

        self.assertIn(
            "failed closed",
            block,
        )

        done_lines = [
            line
            for line in block.splitlines()
            if 'yield "data: [DONE]' in line
        ]

        return_lines = [
            line
            for line in block.splitlines()
            if line.strip() == "return"
        ]

        self.assertGreaterEqual(
            len(done_lines),
            2,
        )

        self.assertGreaterEqual(
            len(return_lines),
            2,
        )

    def test_action_is_not_in_llm_schema(self):
        self.assertNotIn(
            "palworld_backup_create",
            self.schemas,
        )


if __name__ == "__main__":
    unittest.main()
