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

    def test_diagnostic_forces_homelab(self):
        marker = (
            "forced read-only "
            '"\n'
            '            "homelab diagnostic tool=%s'
        )

        self.assertIn(
            marker,
            self.source,
        )

        marker_position = self.source.index(
            marker
        )

        preceding = self.source[
            max(0, marker_position - 700):
            marker_position
        ]

        self.assertIn(
            'exclusive_tools = {"homelab"}',
            preceding,
        )

    def test_force_occurs_after_direct_fastpath(self):
        fastpath_position = self.source.index(
            "if _direct_homelab_request:"
        )

        diagnostic_position = self.source.index(
            "forced read-only "
        )

        self.assertLess(
            fastpath_position,
            diagnostic_position,
        )

    def test_diagnostic_synthesis_is_conservative(self):
        source = Path(
            "src/agent_loop.py"
        ).read_text(
            encoding="utf-8",
        )

        helper_position = source.index(
            "def _healthy_homelab_diagnostic_response("
        )

        loop_position = source.index(
            "async def stream_agent_loop("
        )

        branch_position = source.index(
            "# Healthy diagnostic results are factual"
        )

        budget_position = source.index(
            "# If budget was hit, stop the loop",
            branch_position,
        )

        self.assertLess(
            helper_position,
            loop_position,
        )

        self.assertLess(
            branch_position,
            budget_position,
        )

        branch = source[
            branch_position:
            budget_position
        ]

        self.assertIn(
            "_healthy_homelab_diagnostic_response(",
            branch,
        )

        self.assertIn(
            "_ody_homelab_status_completed = True",
            branch,
        )

        self.assertIn(
            "full_response = (",
            branch,
        )

        self.assertIn(
            "authoritative operator state",
            branch,
        )

        self.assertIn(
            "- estado: **ok**",
            source,
        )

        self.assertIn(
            "- runtime: `running`",
            source,
        )

        self.assertIn(
            "no indica un fallo",
            source,
        )

    def test_diagnostic_result_is_not_terminal(self):
        homelab_result_position = self.source.index(
            'block.tool_type == "homelab"'
        )

        bypass_position = self.source.index(
            "and not "
            "_homelab_agent_tool_required",
            homelab_result_position,
        )

        feedback_position = self.source.index(
            "_append_tool_results(",
            homelab_result_position,
        )

        self.assertLess(
            bypass_position,
            feedback_position,
        )


if __name__ == "__main__":
    unittest.main()
