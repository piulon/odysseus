"""Trusted launcher for model-controlled Bash/Python.

This process starts with the already-sanitized environment supplied by
subprocess_sandbox.py, installs a Landlock filesystem allowlist, enables
no_new_privs, then execs the requested interpreter.

Fail closed: if the sandbox cannot be installed, the model-controlled command
is never executed.
"""

from __future__ import annotations

import ctypes
import os
import sys


SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446

LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1

PR_SET_NO_NEW_PRIVS = 38

EXECUTE = 1 << 0
WRITE_FILE = 1 << 1
READ_FILE = 1 << 2
READ_DIR = 1 << 3
REMOVE_DIR = 1 << 4
REMOVE_FILE = 1 << 5
MAKE_CHAR = 1 << 6
MAKE_DIR = 1 << 7
MAKE_REG = 1 << 8
MAKE_SOCK = 1 << 9
MAKE_FIFO = 1 << 10
MAKE_BLOCK = 1 << 11
MAKE_SYM = 1 << 12

# Added in later Landlock ABIs.
REFER = 1 << 13
TRUNCATE = 1 << 14


class RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
    ]


class PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


_LIBC = ctypes.CDLL(None, use_errno=True)


def _syscall_error(prefix: str) -> RuntimeError:
    err = ctypes.get_errno()
    return RuntimeError(
        f"{prefix}: errno={err} {os.strerror(err)}"
    )


def _landlock_abi() -> int:
    rc = _LIBC.syscall(
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.c_void_p(0),
        ctypes.c_size_t(0),
        ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION),
    )
    if rc < 0:
        raise _syscall_error("Landlock ABI query failed")
    return int(rc)


def _handled_rights(abi: int) -> int:
    rights = (
        EXECUTE
        | WRITE_FILE
        | READ_FILE
        | READ_DIR
        | REMOVE_DIR
        | REMOVE_FILE
        | MAKE_CHAR
        | MAKE_DIR
        | MAKE_REG
        | MAKE_SOCK
        | MAKE_FIFO
        | MAKE_BLOCK
        | MAKE_SYM
    )

    if abi >= 2:
        rights |= REFER

    if abi >= 3:
        rights |= TRUNCATE

    return rights


def _add_rule(
    ruleset_fd: int,
    path: str,
    allowed_access: int,
) -> None:
    if not os.path.exists(path):
        return

    o_path = getattr(os, "O_PATH", 0o10000000)

    fd = os.open(
        path,
        o_path | os.O_CLOEXEC,
    )

    try:
        attr = PathBeneathAttr(
            allowed_access,
            fd,
            0,
        )

        rc = _LIBC.syscall(
            SYS_LANDLOCK_ADD_RULE,
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(attr),
            0,
        )

        if rc != 0:
            raise _syscall_error(
                f"Landlock add rule failed for {path}"
            )
    finally:
        os.close(fd)


def _install_landlock(
    workspace: str,
    tmpdir: str,
) -> None:
    abi = _landlock_abi()

    if abi < 1:
        raise RuntimeError(
            f"Unsupported Landlock ABI: {abi}"
        )

    handled = _handled_rights(abi)

    attr = RulesetAttr(handled)

    ruleset_fd = _LIBC.syscall(
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(attr),
        ctypes.sizeof(attr),
        0,
    )

    if ruleset_fd < 0:
        raise _syscall_error(
            "Landlock ruleset creation failed"
        )

    try:
        ro_exec = EXECUTE | READ_FILE | READ_DIR
        rw = handled

        # Runtime executables, dynamic libraries and Python stdlib/site-packages.
        for path in (
            "/usr",
            "/bin",
            "/lib",
            "/lib64",
        ):
            _add_rule(
                ruleset_fd,
                path,
                ro_exec,
            )

        # Minimal configuration needed by normal TLS/DNS-capable programs.
        for path in (
            "/etc/ssl",
        ):
            _add_rule(
                ruleset_fd,
                path,
                READ_FILE | READ_DIR,
            )

        for path in (
            "/etc/ld.so.cache",
            "/etc/resolv.conf",
            "/etc/hosts",
            "/etc/nsswitch.conf",
            "/etc/gai.conf",
            "/etc/localtime",
        ):
            _add_rule(
                ruleset_fd,
                path,
                READ_FILE,
            )

        # Minimal harmless devices commonly expected by CLI/Python programs.
        for path in (
            "/dev/null",
            "/dev/zero",
            "/dev/urandom",
        ):
            _add_rule(
                ruleset_fd,
                path,
                READ_FILE | WRITE_FILE,
            )

        # The only mutable filesystem areas granted to model-controlled code.
        _add_rule(
            ruleset_fd,
            workspace,
            rw,
        )
        _add_rule(
            ruleset_fd,
            tmpdir,
            rw,
        )

        rc = _LIBC.prctl(
            PR_SET_NO_NEW_PRIVS,
            1,
            0,
            0,
            0,
        )

        if rc != 0:
            raise _syscall_error(
                "PR_SET_NO_NEW_PRIVS failed"
            )

        rc = _LIBC.syscall(
            SYS_LANDLOCK_RESTRICT_SELF,
            ruleset_fd,
            0,
        )

        if rc != 0:
            raise _syscall_error(
                "Landlock restrict_self failed"
            )

    finally:
        os.close(ruleset_fd)


def _parse(argv: list[str]) -> tuple[str, str, str, str]:
    try:
        separator = argv.index("--")
    except ValueError as exc:
        raise RuntimeError(
            "Sandbox launcher missing argument separator"
        ) from exc

    opts = argv[1:separator]
    rest = argv[separator + 1:]

    values: dict[str, str] = {}

    i = 0
    while i < len(opts):
        key = opts[i]
        if key not in {
            "--tool",
            "--workspace",
            "--tmpdir",
        }:
            raise RuntimeError(
                f"Unknown sandbox option: {key}"
            )
        if i + 1 >= len(opts):
            raise RuntimeError(
                f"Missing value for sandbox option: {key}"
            )
        values[key] = opts[i + 1]
        i += 2

    if len(rest) != 1:
        raise RuntimeError(
            "Sandbox launcher requires exactly one command payload"
        )

    tool = values.get("--tool", "")
    workspace = values.get("--workspace", "")
    tmpdir = values.get("--tmpdir", "")
    content = rest[0]

    if tool not in {"bash", "python"}:
        raise RuntimeError(
            f"Unsupported sandbox tool: {tool!r}"
        )

    for name, path in (
        ("workspace", workspace),
        ("tmpdir", tmpdir),
    ):
        if not path or not os.path.isdir(path):
            raise RuntimeError(
                f"Invalid sandbox {name}: {path!r}"
            )

    return tool, workspace, tmpdir, content


def main() -> int:
    try:
        tool, workspace, tmpdir, content = _parse(sys.argv)

        _install_landlock(
            os.path.realpath(workspace),
            os.path.realpath(tmpdir),
        )

        env = dict(os.environ)

        if tool == "bash":
            os.execve(
                "/bin/sh",
                ["/bin/sh", "-c", content],
                env,
            )

        os.execve(
            sys.executable,
            [
                sys.executable,
                "-I",
                "-c",
                content,
            ],
            env,
        )

    except Exception as exc:
        print(
            f"subprocess sandbox refused execution: {exc}",
            file=sys.stderr,
        )
        return 126

    return 126


if __name__ == "__main__":
    raise SystemExit(main())
