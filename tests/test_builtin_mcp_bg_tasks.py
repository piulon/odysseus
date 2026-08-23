"""Built-in MCP background startup tasks must retain strong references."""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_builtin_mcp(monkeypatch):
    core = types.ModuleType("core")
    core.__path__ = []

    platform_compat = types.ModuleType("core.platform_compat")
    platform_compat.which_tool = lambda name: None

    monkeypatch.setitem(sys.modules, "core", core)
    monkeypatch.setitem(
        sys.modules,
        "core.platform_compat",
        platform_compat,
    )

    spec = importlib.util.spec_from_file_location(
        "builtin_mcp_under_test",
        ROOT / "src" / "builtin_mcp.py",
    )
    module = importlib.util.module_from_spec(spec)

    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module


async def test_spawn_bg_holds_strong_ref_until_task_finishes(monkeypatch):
    builtin_mcp = _load_builtin_mcp(monkeypatch)

    started = asyncio.Event()
    release = asyncio.Event()

    async def work():
        started.set()
        await release.wait()

    task = builtin_mcp._spawn_bg(work())
    await started.wait()

    assert task in builtin_mcp._BG_TASKS

    release.set()
    await task
    await asyncio.sleep(0)

    assert task not in builtin_mcp._BG_TASKS
