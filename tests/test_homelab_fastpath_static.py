import unittest
from pathlib import Path


class HomelabFastPathStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(
            "/app/src/agent_loop.py"
        ).read_text(encoding="utf-8")

    def test_fastpath_precedes_tool_retrieval(self):
        fastpath = self.source.index(
            "# Simple whole-homelab status "
            "requests do not need an LLM round."
        )

        retrieval = self.source.index(
            "# RAG-based tool selection"
        )

        model_loop = self.source.index(
            "for round_num in range("
        )

        self.assertLess(fastpath, retrieval)
        self.assertLess(fastpath, model_loop)

    def test_fastpath_preserves_sse_contract(self):
        start = self.source.index(
            "# Simple whole-homelab status "
            "requests do not need an LLM round."
        )

        end = self.source.index(
            "# RAG-based tool selection",
            start,
        )

        block = self.source[start:end]

        for required in (
            '"type": "tool_start"',
            '"type": "tool_output"',
            '"type": "metrics"',
            '"action": "status"',
            'yield "data: [DONE]',
            "return",
        ):
            self.assertIn(required, block)


    def test_passive_ui_context_does_not_block_fastpath(self):
        start = self.source.index(
            "# Simple whole-homelab status "
            "requests do not need an LLM round."
        )

        end = self.source.index(
            "# RAG-based tool selection",
            start,
        )

        block = self.source[start:end]
        compact = " ".join(block.split())

        self.assertNotIn(
            "and not workspace",
            compact,
        )

        self.assertNotIn(
            "and not active_email",
            compact,
        )

        self.assertIn(
            'set(forced_tools) == {"homelab"}',
            compact,
        )

        self.assertIn(
            'set(relevant_tools) == {"homelab"}',
            compact,
        )

        self.assertIn(
            "[agent-fastpath] eligibility=",
            block,
        )


if __name__ == "__main__":
    unittest.main()
