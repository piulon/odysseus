import unittest
from pathlib import Path


class HomelabFastPathStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(
            "/app/src/agent_loop.py"
        ).read_text(encoding="utf-8")

        cls.marker = (
            "# Deterministic homelab read-only "
            "requests do not need an LLM round."
        )

    def fastpath_block(self):
        start = self.source.index(
            self.marker
        )

        end = self.source.index(
            "# RAG-based tool selection",
            start,
        )

        return self.source[start:end]

    def test_fastpath_precedes_tool_retrieval(self):
        fastpath = self.source.index(
            self.marker
        )

        retrieval = self.source.index(
            "# RAG-based tool selection"
        )

        model_loop = self.source.index(
            "for round_num in range("
        )

        self.assertLess(fastpath, retrieval)
        self.assertLess(fastpath, model_loop)

    def test_classifier_returns_dynamic_command(self):
        block = self.fastpath_block()

        self.assertIn(
            "classify_direct_homelab_request",
            block,
        )

        self.assertIn(
            "_fast_command_args",
            block,
        )

        self.assertIn(
            "json.dumps(",
            block,
        )

        self.assertNotIn(
            '{"action": "status"}',
            block,
        )

    def test_fastpath_preserves_sse_contract(self):
        block = self.fastpath_block()

        for required in (
            '"type": "tool_start"',
            '"type": "tool_output"',
            '"type": "metrics"',
            'yield "data: [DONE]',
            "direct_homelab_request",
            "direct_homelab_action",
            "return",
        ):
            self.assertIn(required, block)

    def test_passive_ui_context_is_allowed(self):
        block = self.fastpath_block()
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
            "set(forced_tools).issubset(",
            compact,
        )

        self.assertIn(
            '"web_fetch"',
            block,
        )

        self.assertIn(
            '"web_search"',
            block,
        )

        self.assertIn(
            "[agent-fastpath] eligibility=",
            block,
        )


if __name__ == "__main__":
    unittest.main()
