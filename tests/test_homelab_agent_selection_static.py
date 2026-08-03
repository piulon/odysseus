import unittest
from pathlib import Path


class HomelabAgentSelectionStaticTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(
            "src/agent_loop.py"
        ).read_text(
            encoding="utf-8",
        )

    def test_agent_imports_read_only_hint(self):
        self.assertIn(
            "should_include_homelab_tool",
            self.source,
        )

    def test_agent_adds_homelab_to_tools(self):
        self.assertIn(
            '_relevant_tools.add("homelab")',
            self.source,
        )

    def test_hint_runs_after_fast_path(self):
        fast_position = self.source.index(
            "if _direct_homelab_request:"
        )

        selection_position = self.source.index(
            '_relevant_tools.add("homelab")'
        )

        self.assertLess(
            fast_position,
            selection_position,
        )


if __name__ == "__main__":
    unittest.main()
