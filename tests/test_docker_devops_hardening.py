"""Static regressions for Docker/devops hardening contracts."""

import ast
import re
from pathlib import Path

import yaml
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILES = [
    ROOT / "docker-compose.yml",
    ROOT / "docker-compose.gpu-nvidia.yml",
    ROOT / "docker-compose.gpu-amd.yml",
]
HOST_DOCKER_OVERLAY = ROOT / "docker" / "host-docker.yml"
TEST_DOCS = [
    ROOT / "tests" / "README.md",
    ROOT / "tests" / "TESTING_STANDARD.md",
    ROOT / "tests" / "LAYOUT_INVENTORY.md",
]


def _compose_env_names(path: Path) -> set[str]:
    compose = yaml.safe_load(path.read_text(encoding="utf-8"))
    env = compose["services"]["odysseus"]["environment"]
    return {entry.split("=", 1)[0] for entry in env}


def _upload_limit_env_names() -> set[str]:
    source = (ROOT / "src" / "upload_limits.py").read_text(encoding="utf-8")
    return set(re.findall(r'"(ODYSSEUS_[A-Z_]*BYTES)"', source)) | {
        "ODYSSEUS_CHAT_UPLOAD_MAX_BYTES"
    }


def _cors_allow_methods() -> list[str]:
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "CORS_ALLOW_METHODS" in names:
                return ast.literal_eval(node.value)
    raise AssertionError("CORS_ALLOW_METHODS not found")


def test_compose_files_forward_every_upload_limit_env_var():
    expected = _upload_limit_env_names()
    assert expected
    for path in COMPOSE_FILES:
        assert expected <= _compose_env_names(path), path.name


def test_default_compose_files_do_not_mount_host_docker_socket():
    for path in COMPOSE_FILES:
        text = path.read_text(encoding="utf-8")
        assert "/var/run/docker.sock" not in text, path.name


def test_host_docker_overlay_mounts_socket_and_adds_docker_group():
    overlay = yaml.safe_load(HOST_DOCKER_OVERLAY.read_text(encoding="utf-8"))
    service = overlay["services"]["odysseus"]

    assert "/var/run/docker.sock:/var/run/docker.sock" in service["volumes"]
    assert "${DOCKER_GID:-963}" in service["group_add"]
    assert "ODYSSEUS_ENABLE_HOST_DOCKER=true" in service["environment"]


def test_docker_entrypoint_gates_socket_group_plumbing_on_explicit_opt_in():
    script = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    block_start = script.index("DOCKER_SOCK=\"${DOCKER_SOCK:-/var/run/docker.sock}\"")
    block_end = script.index("\nmount_root_for()", block_start)
    socket_group_block = script[block_start:block_end]

    opt_in_check = socket_group_block.index(
        "[ \"${ODYSSEUS_ENABLE_HOST_DOCKER:-}\" = \"true\" ]"
    )
    socket_check = socket_group_block.index("[ -S \"$DOCKER_SOCK\" ]")
    stat_socket = socket_group_block.index("stat -c")
    add_group = socket_group_block.index("groupadd -g")
    add_user_group = socket_group_block.index("usermod -aG")

    assert opt_in_check < socket_check < stat_socket < add_group < add_user_group


def test_docker_entrypoint_does_not_resolve_root_commands_from_app_local_path():
    script = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    path_export = script.index('export PATH="/app/.local/bin:$PATH"')
    gosu_capture = script.index('GOSU_BIN="$(command -v gosu)"')
    python_capture = script.index('PYTHON_BIN="$(command -v python)"')
    setup_call = script.index('"$GOSU_BIN" "$ODY_USER" "$PYTHON_BIN" /app/setup.py')
    final_exec = script.index('exec "$GOSU_BIN" "$ODY_USER" "$@"')

    assert gosu_capture < path_export < setup_call
    assert python_capture < path_export < setup_call
    assert final_exec > path_export


def test_docker_entrypoint_fails_closed_when_setup_fails():
    script = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")

    setup_guard = (
        'if ! "$GOSU_BIN" "$ODY_USER" "$PYTHON_BIN" /app/setup.py; then'
    )

    assert setup_guard in script
    assert "/app/setup.py || true" not in script

    start = script.index(setup_guard)
    end = script.index(
        'exec "$GOSU_BIN" "$ODY_USER" "$@"',
        start,
    )
    setup_block = script[start:end]

    assert 'exit 1' in setup_block
    assert (
        "refusing to start the application"
        in setup_block
    )
    assert "unset ODYSSEUS_ADMIN_PASSWORD" in setup_block

    unset_pos = setup_block.index(
        "unset ODYSSEUS_ADMIN_PASSWORD"
    )
    guard_pos = setup_block.index(setup_guard)

    assert guard_pos < unset_pos


def test_docker_entrypoint_ownership_repair_stays_inside_expected_mounts():
    script = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    assert "find /app -xdev" in script
    for path in ("/app/data", "/app/logs", "/app/.ssh", "/app/.cache", "/app/.local"):
        assert f"-path {path}" in script
    assert "mount_root_for" in script
    assert "is_broad_mount_root" in script
    assert "Skipping recursive ownership repair" in script


def test_dockerignore_excludes_secrets_editor_backups():
    patterns = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
    assert {
        "secrets.env",
        "secrets.env.*",
        "secrets.env~",
        ".secrets.env.swp",
        ".secrets.env.swo",
        "**/#secrets.env#",
    } <= patterns
    assert "!secrets.env.example" in patterns


def test_admin_bootstrap_docs_never_direct_users_to_password_logs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    setup_doc = (ROOT / "docs" / "setup.md").read_text(encoding="utf-8")
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    docs = (readme + "\n" + setup_doc).lower()

    stale_phrases = (
        "first admin password is printed",
        "prints a temporary password",
        "generated admin password",
        "for docker installs, the same line is in `docker compose logs odysseus`",
        "use that for the first login",
    )

    for phrase in stale_phrases:
        assert phrase not in docs

    assert "odysseus_admin_password" in docs
    assert "refuses to start" in readme.lower()
    assert "never printed to logs" in readme.lower()

    # Generic container-log inspection is legitimate operational
    # troubleshooting. Only password-retrieval instructions are forbidden.
    assert (
        "docker compose logs odysseus | grep"
        in setup_doc.lower()
    )

    assert (
        "required if data/auth.json does not exist"
        in env_example.lower()
    )
    assert "remove this variable from .env" in env_example.lower()


def test_nightly_skill_audit_is_opt_in_by_default():
    import ast

    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    calls = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if not isinstance(node.func, ast.Name):
            continue

        if node.func.id != "get_setting":
            continue

        if not node.args:
            continue

        first = node.args[0]

        if (
            isinstance(first, ast.Constant)
            and first.value == "skill_audit_nightly"
        ):
            calls.append(node)

    assert len(calls) == 1

    call = calls[0]

    assert len(call.args) >= 2
    assert isinstance(call.args[1], ast.Constant)
    assert call.args[1].value is False

    # Preserve the execution gate: false/default skips the autonomous
    # nightly call; an explicitly truthy setting allows execution.
    assert (
        'if not get_setting("skill_audit_nightly", False):'
        in source
    )
    assert (
        'get_setting("skill_audit_nightly", True)'
        not in source
    )


def test_manual_skill_audit_is_independent_of_nightly_gate():
    import ast

    source = (
        ROOT / "routes" / "skills_routes.py"
    ).read_text(encoding="utf-8")

    tree = ast.parse(source)

    functions = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "run_scheduled_skill_audit"
        )
    ]

    assert len(functions) == 1

    function = functions[0]

    nightly_gate_calls = []

    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue

        if not isinstance(node.func, ast.Name):
            continue

        if node.func.id != "get_setting":
            continue

        if not node.args:
            continue

        first = node.args[0]

        if (
            isinstance(first, ast.Constant)
            and first.value == "skill_audit_nightly"
        ):
            nightly_gate_calls.append(node)

    assert nightly_gate_calls == []


def test_gallery_background_removal_never_executes_remote_model_code():
    source = (
        ROOT
        / "routes"
        / "gallery"
        / "gallery_routes.py"
    ).read_text(encoding="utf-8")

    # Background removal may use the explicitly installed rembg
    # dependency, but must never silently download/execute model
    # repository code as a fallback.
    assert "from rembg import remove" in source
    assert "cut = remove(crop)" in source

    assert "trust_remote_code" not in source
    assert "briaai/RMBG-1.4" not in source
    assert "from transformers import pipeline" not in source

    assert (
        "Install rembg from Cookbook"
        in source
    )


def test_gallery_realesrgan_uses_only_verified_local_checkpoints():
    gallery = (
        ROOT
        / "routes"
        / "gallery"
        / "gallery_routes.py"
    ).read_text(encoding="utf-8")

    installer = (
        ROOT
        / "routes"
        / "shell_routes.py"
    ).read_text(encoding="utf-8")

    constants = (
        ROOT
        / "src"
        / "constants.py"
    ).read_text(encoding="utf-8")

    models = (
        ROOT
        / "src"
        / "realesrgan_models.py"
    ).read_text(encoding="utf-8")

    # No RealESRGANer call may receive an HTTP(S) model URL.
    # Other image backends are audited independently.
    import ast

    assert "Real-ESRGAN/releases/download" not in gallery

    tree = ast.parse(gallery)

    realesrgan_calls = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        name = None

        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr

        if name == "RealESRGANer":
            realesrgan_calls.append(node)

    assert len(realesrgan_calls) == 2

    for call in realesrgan_calls:
        model_path_args = [
            keyword
            for keyword in call.keywords
            if keyword.arg == "model_path"
        ]

        assert len(model_path_args) == 1

        expression = ast.get_source_segment(
            gallery,
            model_path_args[0].value,
        )

        assert expression
        assert "http://" not in expression
        assert "https://" not in expression

    assert (
        'verified_realesrgan_model('
        in gallery
    )

    assert (
        '"RealESRGAN_x4plus.pth"'
        in gallery
    )

    assert (
        '"realesr-general-x4v3.pth"'
        in gallery
    )

    assert (
        '"realesr-general-wdn-x4v3.pth"'
        in gallery
    )

    # DNI requires both local checkpoints.
    assert "model_path=[" in gallery
    assert "str(general_path)" in gallery
    assert "str(weak_path)" in gallery
    assert (
        "dni_weight=[strength, 1.0 - strength]"
        in gallery
    )

    # Checkpoints belong to the existing persistent DATA_DIR hierarchy.
    assert (
        'REALESRGAN_MODELS_DIR = '
        'os.path.join(DATA_DIR, "models", "realesrgan")'
        in constants
    )

    # Provisioning is explicit administrative installation work.
    assert (
        'if pip_name == "realesrgan":'
        in installer
    )

    assert (
        "provision_realesrgan_models"
        in installer
    )

    # Integrity enforcement exists in the model store.
    assert "sha256" in models
    assert "compare_digest" in models
    assert "os.replace" in models


def test_cors_allow_methods_include_patch():
    methods = _cors_allow_methods()
    assert "PATCH" in methods


def test_patch_preflight_is_allowed_by_configured_cors_methods():
    async def patched(_request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/api/document/1", patched, methods=["PATCH"])])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://client.local"],
        allow_credentials=True,
        allow_methods=_cors_allow_methods(),
        allow_headers=["Content-Type"],
    )

    response = TestClient(app).options(
        "/api/document/1",
        headers={
            "Origin": "http://client.local",
            "Access-Control-Request-Method": "PATCH",
        },
    )

    assert response.status_code == 200


def test_testing_docs_use_project_venv_for_python_validation():
    stale_patterns = [
        "python3 -m pytest",
        "python3 -m py_compile",
        "Focused `pytest`",
        "`pytest` on neighboring",
        ".venv/bin/python",
    ]
    for path in TEST_DOCS:
        text = path.read_text(encoding="utf-8")
        for stale in stale_patterns:
            assert stale not in text, f"{path.name} still contains {stale!r}"
