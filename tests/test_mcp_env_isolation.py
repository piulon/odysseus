from src.builtin_mcp import builtin_python_env
from src.mcp_manager import _stdio_server_env


SENSITIVE_PARENT_VARS = (
    "OPENAI_API_KEY",
    "GOOGLE_OAUTH_CLIENT_SECRET",
    "HF_TOKEN",
    "HOMELAB_OPERATOR_ACTION_TOKEN",
    "ODYSSEUS_ADMIN_PASSWORD",
)


def test_stdio_server_env_does_not_inherit_parent_environment(monkeypatch):
    for key in SENSITIVE_PARENT_VARS:
        monkeypatch.setenv(key, "must-not-leak")

    explicit = {
        "PYTHONPATH": "/app",
        "ODYSSEUS_MCP_MEMORY_OWNER": "alice",
    }

    result = _stdio_server_env(explicit)

    assert result == explicit
    assert result is not explicit

    for key in SENSITIVE_PARENT_VARS:
        assert key not in result


def test_stdio_server_env_none_remains_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")

    assert _stdio_server_env(None) is None
    assert _stdio_server_env({}) is None


def test_rag_and_image_receive_no_application_secrets(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/existing")

    for key in SENSITIVE_PARENT_VARS:
        monkeypatch.setenv(key, "must-not-leak")

    for server_id in ("rag", "image_gen"):
        env = builtin_python_env("/app", server_id)

        assert set(env) == {"PYTHONPATH"}

        for key in SENSITIVE_PARENT_VARS:
            assert key not in env


def test_memory_receives_only_owner_configuration(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setenv("ODYSSEUS_MCP_MEMORY_OWNER", "alice")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("HOMELAB_OPERATOR_ACTION_TOKEN", "must-not-leak")

    env = builtin_python_env("/app", "memory")

    assert env["ODYSSEUS_MCP_MEMORY_OWNER"] == "alice"
    assert "OPENAI_API_KEY" not in env
    assert "HOMELAB_OPERATOR_ACTION_TOKEN" not in env


def test_email_receives_email_config_but_not_unrelated_secrets(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setenv("ODYSSEUS_MCP_EMAIL_OWNER", "alice")
    monkeypatch.setenv("IMAP_HOST", "imap.example.test")
    monkeypatch.setenv("SMTP_PASSWORD", "email-specific-secret")

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("ODYSSEUS_ADMIN_PASSWORD", "must-not-leak")
    monkeypatch.setenv("HOMELAB_OPERATOR_ACTION_TOKEN", "must-not-leak")

    env = builtin_python_env("/app", "email")

    assert env["ODYSSEUS_MCP_EMAIL_OWNER"] == "alice"
    assert env["IMAP_HOST"] == "imap.example.test"
    assert env["SMTP_PASSWORD"] == "email-specific-secret"

    assert "OPENAI_API_KEY" not in env
    assert "ODYSSEUS_ADMIN_PASSWORD" not in env
    assert "HOMELAB_OPERATOR_ACTION_TOKEN" not in env
