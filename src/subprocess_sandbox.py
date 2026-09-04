"""Security boundary for model-controlled Bash/Python subprocesses.

The Odysseus application itself needs access to persistent data and secrets.
Arbitrary model-controlled shell/Python must not inherit that authority.

This module prepares:
- a minimal environment allowlist;
- an explicit per-turn workspace;
- a private temporary directory;
- a trusted launcher that applies Landlock + no_new_privs before exec.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_SAFE_ENV_EXACT = frozenset({
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "TERM",
    "COLUMNS",
    "LINES",
    "TZ",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
})

_SAFE_ENV_PREFIXES = (
    "LC_",
)

_SAFE_PATH = (
    "/usr/local/sbin:"
    "/usr/local/bin:"
    "/usr/sbin:"
    "/usr/bin:"
    "/sbin:"
    "/bin"
)


@dataclass(frozen=True)
class SandboxSpec:
    argv: list[str]
    env: dict[str, str]
    cwd: str
    tmpdir: str


def _safe_component(value: object, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return (text[:80] or fallback)


def _ensure_private_dir(path: str) -> str:
    os.makedirs(path, mode=0o700, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return os.path.realpath(path)


def _sandbox_workspace(ctx: dict) -> str:
    # Import lazily to avoid the agent_tools/tool_execution import cycle.
    from src.tool_execution import get_active_workspace, vet_workspace

    active = get_active_workspace()

    if active:
        vetted = vet_workspace(active)
        if not vetted:
            raise RuntimeError(
                f"Refusing unsafe subprocess workspace: {active!r}"
            )

        vetted = os.path.realpath(vetted)

        # Never authorize a workspace that is the persistent data directory
        # itself or one of its ancestors. Such a rule would expose app.db,
        # .app_key, auth.json and every other persistent application secret.
        #
        # A descendant workspace remains safe: Landlock authorizes only that
        # exact subtree, not DATA_DIR siblings.
        from src.constants import DATA_DIR

        data_dir = os.path.realpath(DATA_DIR)

        try:
            workspace_contains_data = (
                os.path.commonpath([vetted, data_dir]) == vetted
            )
        except ValueError:
            workspace_contains_data = False

        if workspace_contains_data:
            raise RuntimeError(
                "Refusing subprocess workspace that contains "
                "Odysseus persistent data"
            )

        return vetted

    # No workspace was explicitly bound for this turn. Never fall back to
    # DATA_DIR itself: it contains app.db, .app_key, auth.json, etc.
    from src.constants import DATA_DIR

    owner = _safe_component(ctx.get("owner"), "owner")
    session = _safe_component(ctx.get("session_id"), "session")

    root = _ensure_private_dir(
        os.path.join(DATA_DIR, "agent_workspaces")
    )

    return _ensure_private_dir(
        os.path.join(root, f"{owner}-{session}")
    )


def _sandbox_tmpdir(ctx: dict) -> str:
    owner = _safe_component(ctx.get("owner"), "owner")
    session = _safe_component(ctx.get("session_id"), "session")
    nonce = secrets.token_hex(8)

    root = _ensure_private_dir(
        os.path.join(tempfile.gettempdir(), "odysseus-agent")
    )

    return _ensure_private_dir(
        os.path.join(root, f"{owner}-{session}-{nonce}")
    )


def build_sandbox_env(
    source: Mapping[str, str] | None,
    *,
    workspace: str,
    tmpdir: str,
) -> dict[str, str]:
    """Return a fail-closed environment for arbitrary subprocesses.

    New parent-process variables are excluded by default. In particular API
    keys, OAuth credentials and Homelab Operator tokens never cross this
    boundary merely because they were added to Odysseus' environment.
    """
    source = source or {}

    env: dict[str, str] = {}

    for key, value in source.items():
        if (
            key in _SAFE_ENV_EXACT
            or any(key.startswith(prefix) for prefix in _SAFE_ENV_PREFIXES)
        ):
            env[key] = str(value)

    # Never inherit executable search paths from the privileged app process.
    env["PATH"] = _SAFE_PATH

    # Treat the selected workspace as the subprocess user's home.
    env["HOME"] = workspace
    env["TMPDIR"] = tmpdir

    env["TERM"] = str(source.get("TERM") or "xterm-256color")
    env["COLUMNS"] = str(source.get("COLUMNS") or "120")
    env["LINES"] = str(source.get("LINES") or "40")

    return env


def prepare_subprocess_sandbox(
    tool: str,
    content: str,
    ctx: dict,
) -> SandboxSpec:
    if tool not in {"bash", "python"}:
        raise ValueError(f"Unsupported sandbox tool: {tool}")

    workspace = _sandbox_workspace(ctx)
    tmpdir = _sandbox_tmpdir(ctx)

    source_env = ctx.get("subproc_env")
    if not isinstance(source_env, Mapping):
        source_env = os.environ

    env = build_sandbox_env(
        source_env,
        workspace=workspace,
        tmpdir=tmpdir,
    )

    launcher = Path(__file__).with_name(
        "subprocess_sandbox_launcher.py"
    )

    argv = [
        sys.executable,
        "-I",
        str(launcher),
        "--tool",
        tool,
        "--workspace",
        workspace,
        "--tmpdir",
        tmpdir,
        "--",
        content,
    ]

    return SandboxSpec(
        argv=argv,
        env=env,
        cwd=workspace,
        tmpdir=tmpdir,
    )


def cleanup_subprocess_sandbox(spec: SandboxSpec) -> None:
    # The workspace is persistent/explicit. Only per-invocation TMP is removed.
    shutil.rmtree(spec.tmpdir, ignore_errors=True)
