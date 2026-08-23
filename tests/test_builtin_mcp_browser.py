import importlib.util
from pathlib import Path
import sys
import types


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


def test_builtin_browser_uses_direct_offline_binary(monkeypatch):
    builtin_mcp = _load_builtin_mcp(monkeypatch)

    cfg = builtin_mcp._BUILTIN_COMMAND_SERVERS[
        "builtin_browser"
    ]

    assert cfg["command"] == "playwright-mcp"

    args = cfg["args"]

    # Runtime must never use npm/npx package resolution.
    assert "@playwright/mcp@latest" not in args
    assert "@playwright/mcp@0.0.78" not in args
    assert "-y" not in args
    assert "--no-install" not in args

    assert "--headless" in args
    assert "--no-sandbox" in args

    browser_i = args.index("--browser")
    assert args[browser_i + 1] == "chromium"

    executable_i = args.index("--executable-path")
    assert (
        args[executable_i + 1]
        == "/usr/local/bin/odysseus-chromium"
    )

    profile_i = args.index("--user-data-dir")
    assert (
        args[profile_i + 1]
        == "/tmp/odysseus-browser-profile"
    )
