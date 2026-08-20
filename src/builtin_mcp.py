"""
builtin_mcp.py

Auto-registration of built-in MCP servers on startup.
Each server runs as a stdio subprocess managed by McpManager.
"""

import asyncio
import logging
import os
import shutil
import sys

from core.platform_compat import which_tool
from src.runtime_paths import get_app_root

logger = logging.getLogger(__name__)


_BUILTIN_SERVERS = {
    "image_gen":  ("mcp_servers/image_gen_server.py",  "Built-in: Image Generation"),
    "memory":     ("mcp_servers/memory_server.py",     "Built-in: Memory"),
    "rag":        ("mcp_servers/rag_server.py",        "Built-in: RAG"),
    "email":      ("mcp_servers/email_server.py",      "Built-in: Email"),
}

# NPX-based built-in servers (run via npx, not Python)
# External-command built-in servers.
#
# Browser MCP is installed into the Docker image at build time and is
# deliberately executed directly rather than through npx, so runtime never
# resolves packages or requires access to the npm registry.
_BUILTIN_COMMAND_SERVERS = {
    "builtin_browser": {
        "name": "Built-in: Browser",
        "command": "playwright-mcp",
        "args": [
            "--headless",
            "--browser",
            "chromium",
            "--executable-path",
            "/usr/local/bin/odysseus-chromium",
            "--user-data-dir",
            "/tmp/odysseus-browser-profile",
            "--no-sandbox",
            "--caps",
            "vision",
        ],
    }
}

# Global flag to disable MCP if there are compatibility issues
MCP_DISABLED = os.environ.get("ODYSSEUS_DISABLE_MCP", "").lower() in ("1", "true", "yes")


# Strong references to the fire-and-forget startup tasks scheduled below.
# asyncio only keeps weak references to tasks created via create_task, so
# without this the GC can collect a task mid-execution and the server
# registration silently never runs. Mirrors _spawn_bg in routes/chat_helpers.py.
_BG_TASKS: set[asyncio.Task] = set()


def _spawn_bg(coro) -> asyncio.Task:
    """Schedule a background task and hold a strong reference until it finishes."""
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)
    return task


def builtin_python_env(base_dir: str) -> dict[str, str]:
    """Environment for built-in Python MCP subprocesses.

    The app root must be importable so mcp_servers can import local modules, but
    replacing PYTHONPATH entirely hides site-packages in container/dev launches
    that rely on PYTHONPATH for their active environment.
    """
    existing = os.environ.get("PYTHONPATH", "")
    parts = [base_dir]
    for item in existing.split(os.pathsep):
        if item and item not in parts:
            parts.append(item)
    return {"PYTHONPATH": os.pathsep.join(parts)}


async def register_builtin_servers(mcp_manager):
    """Connect all built-in MCP servers to the manager."""
    if MCP_DISABLED:
        logger.info("Built-in MCP servers disabled via ODYSSEUS_DISABLE_MCP")
        return

    base_dir = get_app_root()
    python = sys.executable

    async def _connect_python_server(server_id: str, script_path: str, name: str):
        try:
            ok = await mcp_manager.connect_server(
                server_id=server_id,
                name=name,
                transport="stdio",
                command=python,
                args=[script_path],
                env=builtin_python_env(base_dir),
            )
            if ok:
                logger.info(f"Built-in MCP server registered: {name}")
            else:
                logger.warning(f"Built-in MCP server failed to connect: {name}")
        except asyncio.CancelledError:
            logger.warning(f"Built-in MCP server {name} cancelled")
            raise
        except BaseException as e:
            logger.warning(f"Built-in MCP server {name} error: {type(e).__name__}: {e}")

    for server_id, (script, name) in _BUILTIN_SERVERS.items():
        script_path = os.path.join(base_dir, script)
        if not os.path.exists(script_path):
            logger.warning(f"Built-in MCP server script not found: {script_path}")
            continue
        _spawn_bg(_connect_python_server(server_id, script_path, name))

    # Register external-command built-ins after the Python MCP servers.
    async def _start_command_servers():
        await asyncio.sleep(3)

        for server_id, cfg in _BUILTIN_COMMAND_SERVERS.items():
            command = (
                which_tool(cfg["command"])
                or shutil.which(cfg["command"])
            )

            if not command:
                logger.warning(
                    f"{cfg['name']} is not available.\n"
                    f"  Reason: executable {cfg['command']!r} "
                    f"was not found.\n"
                    f"  Impact: tools provided by this MCP server "
                    f"will be unavailable."
                )
                continue

            args = cfg["args"]

            logger.info(
                f"Starting external MCP server: "
                f"{cfg['name']} "
                f"({command} {' '.join(args)})"
            )

            try:
                ok = await mcp_manager.connect_server(
                    server_id=server_id,
                    name=cfg["name"],
                    transport="stdio",
                    command=command,
                    args=args,
                )

                if ok:
                    logger.info(
                        f"Built-in external MCP server registered: "
                        f"{cfg['name']}"
                    )
                else:
                    logger.warning(
                        f"Built-in external MCP server failed to connect: "
                        f"{cfg['name']}"
                    )

            except asyncio.CancelledError:
                raise
            except BaseException as e:
                logger.warning(
                    f"Built-in external MCP server "
                    f"{cfg['name']} error: "
                    f"{type(e).__name__}: {e}"
                )

    _spawn_bg(_start_command_servers())
