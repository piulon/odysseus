"""Trusted launcher for model-controlled Bash/Python.

This process starts with the already-sanitized environment supplied by
subprocess_sandbox.py, installs a Landlock filesystem allowlist, enables
no_new_privs, then execs the requested interpreter.

Fail closed: if the sandbox cannot be installed, the model-controlled command
is never executed.
"""

from __future__ import annotations

import ctypes
import errno
import os
import sys


SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446

LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1

PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2

# Landlock ABI >= 4 network rights.  With these rights handled and no
# LANDLOCK_RULE_NET_PORT allow rules installed, TCP bind/connect fail closed.
LANDLOCK_ACCESS_NET_BIND_TCP = 1 << 0
LANDLOCK_ACCESS_NET_CONNECT_TCP = 1 << 1

# Classic-BPF/seccomp constants.  The additional process-local filter permits
# AF_UNIX sockets but refuses creation of IP/packet sockets.  This closes the
# UDP/raw-socket gap left by Landlock's TCP-only network mediation.
BPF_LD = 0x00
BPF_W = 0x00
BPF_ABS = 0x20
BPF_JMP = 0x05
BPF_JEQ = 0x10
BPF_JGE = 0x30
BPF_K = 0x00
BPF_RET = 0x06

SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000

SECCOMP_DATA_NR_OFFSET = 0
SECCOMP_DATA_ARCH_OFFSET = 4
SECCOMP_DATA_ARG0_OFFSET = 16

AUDIT_ARCH_X86_64 = 0xC000003E
AUDIT_ARCH_AARCH64 = 0xC00000B7
X32_SYSCALL_BIT = 0x40000000

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
        ("handled_access_net", ctypes.c_uint64),
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

    # Network mediation arrived in Landlock ABI 4.  On older kernels retain
    # the filesystem sandbox and let the mandatory seccomp layer below provide
    # the network fail-closed boundary.
    handled_net = 0
    if abi >= 4:
        handled_net = (
            LANDLOCK_ACCESS_NET_BIND_TCP
            | LANDLOCK_ACCESS_NET_CONNECT_TCP
        )

    attr = RulesetAttr(
        handled,
        handled_net,
    )

    # Older Landlock ABIs only know the first u64 field.  Supplying the v1
    # structure size preserves compatibility while ABI >= 4 receives both
    # filesystem and network handled-access masks.
    attr_size = (
        ctypes.sizeof(attr)
        if abi >= 4
        else ctypes.sizeof(ctypes.c_uint64)
    )

    ruleset_fd = _LIBC.syscall(
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(attr),
        attr_size,
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



class SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class SockFprog(ctypes.Structure):
    _fields_ = [
        ("len", ctypes.c_ushort),
        ("filter", ctypes.POINTER(SockFilter)),
    ]


def _seccomp_syscalls() -> tuple[int, int, int, int, bool]:
    """Return audit arch + socket syscalls for supported native ABIs.

    Unknown architectures fail closed instead of silently running generated
    code without the intended network boundary.
    """
    machine = os.uname().machine.strip().lower()

    if machine in {"x86_64", "amd64"}:
        return (
            AUDIT_ARCH_X86_64,
            41,   # socket
            53,   # socketpair
            425,  # io_uring_setup
            True,
        )

    if machine in {"aarch64", "arm64"}:
        return (
            AUDIT_ARCH_AARCH64,
            198,  # socket
            199,  # socketpair
            425,  # io_uring_setup
            False,
        )

    raise RuntimeError(
        f"Unsupported architecture for subprocess seccomp sandbox: {machine}"
    )


def _install_network_seccomp() -> None:
    """Allow AF_UNIX IPC but deny model-controlled IP/network sockets.

    The filter is additive to Docker's existing seccomp policy and is inherited
    across execve().  It denies:
      - socket()/socketpair() for every domain except AF_UNIX;
      - io_uring_setup(), preventing IORING_OP_SOCKET style bypasses;
      - x32 ABI syscalls on x86_64.

    Landlock independently denies TCP bind/connect on ABI >= 4.  Seccomp closes
    the remaining UDP/raw/packet-socket gap.
    """
    (
        audit_arch,
        sys_socket,
        sys_socketpair,
        sys_io_uring_setup,
        deny_x32,
    ) = _seccomp_syscalls()

    instructions: list[SockFilter] = []
    labels: dict[str, int] = {}
    fixups: list[tuple[int, str, str]] = []

    def label(name: str) -> None:
        if name in labels:
            raise RuntimeError(
                f"Duplicate seccomp BPF label: {name}"
            )
        labels[name] = len(instructions)

    def stmt(code: int, k: int) -> None:
        instructions.append(
            SockFilter(code, 0, 0, k)
        )

    def jump(
        code: int,
        k: int,
        true_label: str,
        false_label: str,
    ) -> None:
        idx = len(instructions)
        instructions.append(
            SockFilter(code, 0, 0, k)
        )
        fixups.append(
            (idx, "jt", true_label)
        )
        fixups.append(
            (idx, "jf", false_label)
        )

    stmt(
        BPF_LD | BPF_W | BPF_ABS,
        SECCOMP_DATA_ARCH_OFFSET,
    )
    jump(
        BPF_JMP | BPF_JEQ | BPF_K,
        audit_arch,
        "load_nr",
        "deny",
    )

    label("load_nr")
    stmt(
        BPF_LD | BPF_W | BPF_ABS,
        SECCOMP_DATA_NR_OFFSET,
    )

    if deny_x32:
        jump(
            BPF_JMP | BPF_JGE | BPF_K,
            X32_SYSCALL_BIT,
            "deny",
            "check_io_uring",
        )
    else:
        label("check_io_uring")

    if deny_x32:
        label("check_io_uring")

    jump(
        BPF_JMP | BPF_JEQ | BPF_K,
        sys_io_uring_setup,
        "deny",
        "check_socket",
    )

    label("check_socket")
    jump(
        BPF_JMP | BPF_JEQ | BPF_K,
        sys_socket,
        "load_domain",
        "check_socketpair",
    )

    label("check_socketpair")
    jump(
        BPF_JMP | BPF_JEQ | BPF_K,
        sys_socketpair,
        "load_domain",
        "allow",
    )

    label("load_domain")
    stmt(
        BPF_LD | BPF_W | BPF_ABS,
        SECCOMP_DATA_ARG0_OFFSET,
    )
    jump(
        BPF_JMP | BPF_JEQ | BPF_K,
        1,  # AF_UNIX / AF_LOCAL
        "allow",
        "deny",
    )

    label("deny")
    stmt(
        BPF_RET | BPF_K,
        SECCOMP_RET_ERRNO | errno.EPERM,
    )

    label("allow")
    stmt(
        BPF_RET | BPF_K,
        SECCOMP_RET_ALLOW,
    )

    for idx, field_name, target in fixups:
        if target not in labels:
            raise RuntimeError(
                f"Unknown seccomp BPF label: {target}"
            )

        offset = labels[target] - idx - 1

        if not 0 <= offset <= 255:
            raise RuntimeError(
                "Invalid seccomp BPF jump offset"
            )

        setattr(
            instructions[idx],
            field_name,
            offset,
        )

    array_type = SockFilter * len(instructions)
    array = array_type(*instructions)

    prog = SockFprog(
        len(instructions),
        ctypes.cast(
            array,
            ctypes.POINTER(SockFilter),
        ),
    )

    # Landlock already sets no_new_privs; repeat defensively so this helper also
    # fails closed if its call order is ever changed.
    rc = _LIBC.prctl(
        PR_SET_NO_NEW_PRIVS,
        1,
        0,
        0,
        0,
    )

    if rc != 0:
        raise _syscall_error(
            "PR_SET_NO_NEW_PRIVS before seccomp failed"
        )

    rc = _LIBC.prctl(
        PR_SET_SECCOMP,
        SECCOMP_MODE_FILTER,
        ctypes.byref(prog),
    )

    if rc != 0:
        raise _syscall_error(
            "Subprocess network seccomp install failed"
        )

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
        _install_network_seccomp()

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
