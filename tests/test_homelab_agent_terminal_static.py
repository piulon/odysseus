from pathlib import Path
import unittest


class HomelabTerminalFlowTests(unittest.TestCase):
    def test_terminal_branch_precedes_llm_feedback(
        self,
    ) -> None:
        source = Path(
            "src/agent_loop.py"
        ).read_text(encoding="utf-8")

        terminal = source.index(
            "if _ody_homelab_status_completed:"
        )

        feedback = source.index(
            "# Feed results back to LLM for next round"
        )

        self.assertLess(
            terminal,
            feedback,
        )

    def test_status_tool_marks_terminal_response(
        self,
    ) -> None:
        source = Path(
            "src/agent_tools/homelab_tools.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            '"terminal_response": True',
            source,
        )

        self.assertIn(
            '"direct_response": direct_response',
            source,
        )


if __name__ == "__main__":
    unittest.main()
