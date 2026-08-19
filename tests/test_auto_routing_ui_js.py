import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
HAS_NODE = shutil.which("node") is not None


@pytest.mark.skipif(not HAS_NODE, reason="node binary not on PATH")
def test_auto_routing_ui_node_suite():
    tests = sorted((ROOT / "tests" / "auto-routing-ui").glob("*.test.mjs"))
    assert tests
    result = subprocess.run(
        ["node", "--test", *(str(path) for path in tests)],
        cwd=ROOT,
        capture_output=True,
        timeout=60,
        text=True,
    )
    if result.returncode:
        raise AssertionError(
            f"node --test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
