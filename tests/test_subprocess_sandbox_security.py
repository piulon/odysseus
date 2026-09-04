import asyncio
import os
import shlex



async def _call(tool: str, content: str, workspace):
    # Resolve tool_execution dynamically. Other tests intentionally reload
    # this module; retaining its old ContextVar would bind a stale workspace
    # that the current subprocess implementation cannot see.
    import src.tool_execution as tool_execution

    token = tool_execution._active_workspace.set(str(workspace))
    try:
        return await tool_execution._direct_fallback(
            tool,
            content,
            session_id="subprocess-sandbox-test",
            owner="admin",
        )
    finally:
        tool_execution._active_workspace.reset(token)


def _run(tool: str, content: str, workspace):
    return asyncio.run(_call(tool, content, workspace))


def _output(result):
    if not isinstance(result, dict):
        return str(result)
    return str(
        result.get("output")
        or result.get("stdout")
        or result.get("error")
        or ""
    )


def test_bash_does_not_inherit_parent_secret(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()

    secret = "ODYSSEUS_TEST_SECRET_DO_NOT_LEAK_74192"
    monkeypatch.setenv("ODYSSEUS_TEST_SECRET", secret)

    result = _run(
        "bash",
        'printf "%s" "${ODYSSEUS_TEST_SECRET:-ABSENT}"',
        ws,
    )

    assert secret not in _output(result)


def test_python_does_not_inherit_parent_secret(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()

    secret = "ODYSSEUS_TEST_SECRET_DO_NOT_LEAK_85317"
    monkeypatch.setenv("ODYSSEUS_TEST_SECRET", secret)

    result = _run(
        "python",
        (
            "import os; "
            "print(os.environ.get("
            "'ODYSSEUS_TEST_SECRET', 'ABSENT'))"
        ),
        ws,
    )

    assert secret not in _output(result)


def test_bash_cannot_read_absolute_path_outside_workspace(tmp_path):
    ws = tmp_path / "workspace"
    private = tmp_path / "private"

    ws.mkdir()
    private.mkdir()

    secret = "BASH_OUTSIDE_WORKSPACE_SECRET_59231"
    secret_file = private / "secret.txt"
    secret_file.write_text(secret)

    result = _run(
        "bash",
        "cat " + shlex.quote(str(secret_file)),
        ws,
    )

    assert secret not in _output(result)
    assert result.get("exit_code") != 0


def test_python_cannot_read_absolute_path_outside_workspace(tmp_path):
    ws = tmp_path / "workspace"
    private = tmp_path / "private"

    ws.mkdir()
    private.mkdir()

    secret = "PYTHON_OUTSIDE_WORKSPACE_SECRET_18473"
    secret_file = private / "secret.txt"
    secret_file.write_text(secret)

    result = _run(
        "python",
        f"print(open({str(secret_file)!r}).read())",
        ws,
    )

    assert secret not in _output(result)
    assert result.get("exit_code") != 0


def test_bash_can_write_inside_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()

    result = _run(
        "bash",
        "printf 'BASH_WORKSPACE_OK' > proof.txt && cat proof.txt",
        ws,
    )

    assert result.get("exit_code") == 0
    assert "BASH_WORKSPACE_OK" in _output(result)
    assert (ws / "proof.txt").read_text() == "BASH_WORKSPACE_OK"


def test_python_can_write_inside_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()

    result = _run(
        "python",
        (
            "from pathlib import Path; "
            "Path('proof-python.txt').write_text('PYTHON_WORKSPACE_OK'); "
            "print(Path('proof-python.txt').read_text())"
        ),
        ws,
    )

    assert result.get("exit_code") == 0
    assert "PYTHON_WORKSPACE_OK" in _output(result)
    assert (
        ws / "proof-python.txt"
    ).read_text() == "PYTHON_WORKSPACE_OK"


# ADVERSARIAL_SANDBOX_TESTS_V1

def _run_without_workspace(tool: str, content: str):
    async def call():
        import src.tool_execution as tool_execution

        return await tool_execution._direct_fallback(
            tool,
            content,
            session_id="default-workspace-security-test",
            owner="admin",
        )

    return asyncio.run(call())


def test_bash_named_sensitive_environment_is_removed(
    monkeypatch,
    tmp_path,
):
    ws = tmp_path / "workspace"
    ws.mkdir()

    action_secret = "ACTION_TOKEN_SECRET_41723"
    read_secret = "READ_TOKEN_SECRET_52834"
    oauth_secret = "GOOGLE_OAUTH_SECRET_63945"

    monkeypatch.setenv(
        "HOMELAB_OPERATOR_ACTION_TOKEN",
        action_secret,
    )
    monkeypatch.setenv(
        "HOMELAB_OPERATOR_READ_TOKEN",
        read_secret,
    )
    monkeypatch.setenv(
        "GOOGLE_OAUTH_CLIENT_SECRET",
        oauth_secret,
    )

    result = _run(
        "bash",
        (
            'printf "%s\\n" '
            '"${HOMELAB_OPERATOR_ACTION_TOKEN:-ABSENT}" '
            '"${HOMELAB_OPERATOR_READ_TOKEN:-ABSENT}" '
            '"${GOOGLE_OAUTH_CLIENT_SECRET:-ABSENT}"'
        ),
        ws,
    )

    output = _output(result)

    assert action_secret not in output
    assert read_secret not in output
    assert oauth_secret not in output
    assert output.count("ABSENT") == 3


def test_python_named_sensitive_environment_is_removed(
    monkeypatch,
    tmp_path,
):
    ws = tmp_path / "workspace"
    ws.mkdir()

    values = {
        "OPENAI_API_KEY": "OPENAI_SECRET_10482",
        "GOOGLE_API_KEY": "GOOGLE_SECRET_21593",
        "HF_TOKEN": "HF_SECRET_32604",
    }

    for key, value in values.items():
        monkeypatch.setenv(key, value)

    result = _run(
        "python",
        (
            "import os; "
            "print('|'.join("
            "os.environ.get(k, 'ABSENT') "
            "for k in "
            "('OPENAI_API_KEY','GOOGLE_API_KEY','HF_TOKEN')"
            "))"
        ),
        ws,
    )

    output = _output(result)

    for value in values.values():
        assert value not in output

    assert output.count("ABSENT") == 3


def test_bash_symlink_cannot_escape_workspace(tmp_path):
    ws = tmp_path / "workspace"
    private = tmp_path / "private"

    ws.mkdir()
    private.mkdir()

    secret = "SYMLINK_ESCAPE_SECRET_43715"

    outside = private / "secret.txt"
    outside.write_text(secret)

    link = ws / "escape-link"
    link.symlink_to(outside)

    result = _run(
        "bash",
        "cat escape-link",
        ws,
    )

    assert secret not in _output(result)
    assert result.get("exit_code") != 0


def test_python_symlink_cannot_escape_workspace(tmp_path):
    ws = tmp_path / "workspace"
    private = tmp_path / "private"

    ws.mkdir()
    private.mkdir()

    secret = "PY_SYMLINK_ESCAPE_SECRET_54826"

    outside = private / "secret.txt"
    outside.write_text(secret)

    link = ws / "escape-link"
    link.symlink_to(outside)

    result = _run(
        "python",
        "print(open('escape-link').read())",
        ws,
    )

    assert secret not in _output(result)
    assert result.get("exit_code") != 0


def test_bash_cannot_read_parent_process_environment(
    monkeypatch,
    tmp_path,
):
    ws = tmp_path / "workspace"
    ws.mkdir()

    secret = "PROC_ENV_SECRET_65937"

    monkeypatch.setenv(
        "HOMELAB_OPERATOR_ACTION_TOKEN",
        secret,
    )

    result = _run(
        "bash",
        "cat /proc/1/environ",
        ws,
    )

    assert secret not in _output(result)
    assert result.get("exit_code") != 0


def test_python_has_no_new_privs_enabled(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()

    result = _run(
        "python",
        (
            "import ctypes; "
            "libc=ctypes.CDLL(None); "
            "print(libc.prctl(39,0,0,0,0))"
        ),
        ws,
    )

    assert result.get("exit_code") == 0
    assert _output(result).strip() == "1"


def test_default_workspace_does_not_expose_data_siblings():
    from pathlib import Path
    from src.constants import DATA_DIR

    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    secret = "DEFAULT_WORKSPACE_SIBLING_SECRET_76048"
    sibling = data_dir / "sandbox-sibling-secret.txt"
    sibling.write_text(secret)

    try:
        result = _run_without_workspace(
            "bash",
            (
                "pwd; "
                "cat "
                + shlex.quote(str(sibling))
            ),
        )

        output = _output(result)

        assert secret not in output
        assert "agent_workspaces" in output
        assert result.get("exit_code") != 0

    finally:
        sibling.unlink(missing_ok=True)


def test_default_workspace_remains_writable():
    result = _run_without_workspace(
        "bash",
        (
            "printf 'DEFAULT_WORKSPACE_OK' > proof.txt "
            "&& pwd "
            "&& cat proof.txt"
        ),
    )

    output = _output(result)

    assert result.get("exit_code") == 0
    assert "DEFAULT_WORKSPACE_OK" in output
    assert "agent_workspaces" in output


def test_basic_cli_execution_survives_sandbox(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()

    result = _run(
        "bash",
        (
            "printf 'UID='; id -u; "
            "printf 'KERNEL='; uname -s; "
            "printf 'PWD='; pwd"
        ),
        ws,
    )

    output = _output(result)

    assert result.get("exit_code") == 0
    assert "UID=" in output
    assert "KERNEL=Linux" in output
    assert str(ws) in output


def test_persistent_data_directory_cannot_be_explicit_workspace():
    from src.constants import DATA_DIR

    result = _run(
        "bash",
        "echo SHOULD_NOT_EXECUTE",
        DATA_DIR,
    )

    output = _output(result)

    assert "SHOULD_NOT_EXECUTE" not in output
    assert result.get("exit_code") != 0
    assert "contains Odysseus persistent data" in output


def test_parent_of_persistent_data_cannot_be_explicit_workspace():
    import pytest
    from pathlib import Path
    from src.constants import DATA_DIR

    parent = Path(DATA_DIR).resolve().parent

    if parent == parent.parent:
        pytest.skip("DATA_DIR parent is filesystem root")

    result = _run(
        "bash",
        "echo SHOULD_NOT_EXECUTE",
        parent,
    )

    output = _output(result)

    assert "SHOULD_NOT_EXECUTE" not in output
    assert result.get("exit_code") != 0
    assert "contains Odysseus persistent data" in output


# --- subprocess network egress hardening v2 ---


def test_launcher_seccomp_denies_non_unix_sockets_and_preserves_unix(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    import src.subprocess_sandbox_launcher as launcher

    workspace = tmp_path / "workspace"
    tmpdir = tmp_path / "tmp"

    workspace.mkdir()
    tmpdir.mkdir()

    payload = r"""
import ctypes
import errno
import socket

def blocked(domain, kind):
    try:
        s = socket.socket(domain, kind)
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno == errno.EPERM
    else:
        s.close()
        return False

assert blocked(socket.AF_INET, socket.SOCK_STREAM)
assert blocked(socket.AF_INET, socket.SOCK_DGRAM)
assert blocked(socket.AF_INET6, socket.SOCK_STREAM)

if hasattr(socket, "AF_PACKET"):
    assert blocked(socket.AF_PACKET, socket.SOCK_RAW)

a, b = socket.socketpair()
a.sendall(b"x")
assert b.recv(1) == b"x"
a.close()
b.close()

libc = ctypes.CDLL(None, use_errno=True)
ctypes.set_errno(0)

rc = libc.syscall(
    425,  # io_uring_setup on supported sandbox architectures
    1,
    ctypes.c_void_p(0),
)

err = ctypes.get_errno()

assert rc == -1
assert err == errno.EPERM

print("SECCOMP_NETWORK_POLICY=PASS")
"""

    proc = subprocess.run(
        [
            sys.executable,
            str(Path(launcher.__file__).resolve()),
            "--tool",
            "python",
            "--workspace",
            str(workspace),
            "--tmpdir",
            str(tmpdir),
            "--",
            payload,
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr
    assert "SECCOMP_NETWORK_POLICY=PASS" in proc.stdout


def test_landlock_network_denies_tcp_connect_on_precreated_socket(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    import pytest
    import src.subprocess_sandbox_launcher as launcher

    if launcher._landlock_abi() < 4:
        pytest.skip(
            "Landlock network mediation requires ABI >= 4"
        )

    workspace = tmp_path / "workspace"
    tmpdir = tmp_path / "tmp"

    workspace.mkdir()
    tmpdir.mkdir()

    repo = Path(__file__).resolve().parents[1]

    code = f"""
import socket
from src import subprocess_sandbox_launcher as launcher

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

launcher._install_landlock(
    {str(workspace)!r},
    {str(tmpdir)!r},
)

try:
    s.connect(("127.0.0.1", 9))
except PermissionError:
    print("LANDLOCK_TCP_CONNECT=BLOCKED")
else:
    raise SystemExit("Landlock unexpectedly allowed TCP connect")
finally:
    s.close()
"""

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr
    assert "LANDLOCK_TCP_CONNECT=BLOCKED" in proc.stdout
