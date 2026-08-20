import json

from src.agent_loop import _homelab_doctor_response
from src.agent_tools.homelab_tools import (
    classify_direct_homelab_request,
)


def test_whole_homelab_diagnostic_routes_to_doctor():
    got = classify_direct_homelab_request(
        "Fes un diagnòstic del meu homelab.",
        {"homelab"},
        continuation=False,
    )

    assert got == {"action": "doctor"}


def test_service_diagnostic_stays_on_agent_path():
    got = classify_direct_homelab_request(
        "Diagnostica Ollama",
        {"homelab"},
        continuation=False,
    )

    assert got is None


def test_doctor_warning_renders_authoritative_catalan_reply():
    output = json.dumps({
        "ok": False,
        "exit_code": 1,
        "checked_services": 18,
        "observed_containers": 21,
        "errors": 0,
        "warnings": 1,
        "issues": [{
            "severity": "warning",
            "code": "container_not_running",
            "message": "El contenedor está en estado exited",
            "service": "comfyui",
        }],
    })

    reply = _homelab_doctor_response(
        output,
        "Fes un diagnòstic del meu homelab.",
    )

    assert reply is not None
    assert "Diagnòstic del homelab" in reply
    assert "Errors: **0**" in reply
    assert "Avisos: **1**" in reply
    assert "**comfyui**" in reply
    assert "`container_not_running`" in reply


def test_healthy_doctor_renders_without_incidents():
    output = json.dumps({
        "ok": True,
        "exit_code": 0,
        "checked_services": 18,
        "observed_containers": 18,
        "errors": 0,
        "warnings": 0,
        "issues": [],
    })

    reply = _homelab_doctor_response(
        output,
        "Fes un diagnòstic del meu homelab.",
    )

    assert reply is not None
    assert "Estat: **correcte**" in reply
    assert "No s'han detectat incidències." in reply


def test_invalid_doctor_payload_is_rejected():
    assert (
        _homelab_doctor_response(
            "not-json",
            "Fes un diagnòstic del meu homelab.",
        )
        is None
    )
