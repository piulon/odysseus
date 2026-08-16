from src import model_capabilities as mc
from src.adaptive_routing import (
    RequestProfile,
    build_routing_decision,
)
from src.adaptive_routing_candidates import (
    candidate_from_capability_record,
    ollama_candidate_from_show_payload,
)
from src.model_capability_readers import ollama


TOWER_URL = "http://tower.example/v1/chat/completions"


def show_payload(
    *,
    capabilities,
    family="qwen3",
    context_tokens=40960,
):
    return {
        "capabilities": list(capabilities),
        "details": {
            "family": family,
        },
        "model_info": {
            f"{family}.context_length": context_tokens,
        },
    }


def test_ollama_show_maps_explicit_capabilities_and_context():
    candidate = ollama_candidate_from_show_payload(
        "qwen3-vl:8b",
        show_payload(
            capabilities=[
                "completion",
                "vision",
                "tools",
                "thinking",
            ],
            family="qwen3vl",
            context_tokens=262144,
        ),
        endpoint_id="tower-ollama",
        endpoint_url=TOWER_URL,
        node="tower",
    )

    assert candidate is not None
    assert set(candidate.capabilities) == {
        mc.CAP_VISION,
        mc.CAP_TOOL_CALL,
        mc.CAP_REASONING,
    }
    assert candidate.context_tokens == 262144


def test_completion_only_does_not_infer_capabilities_from_model_name():
    candidate = ollama_candidate_from_show_payload(
        "looks-like-a-tool-model",
        show_payload(
            capabilities=["completion"],
            family="llama",
            context_tokens=131072,
        ),
        endpoint_id="msi-ollama",
        endpoint_url="http://msi.example/api/chat",
        node="msi",
    )

    assert candidate is not None
    assert candidate.capabilities == ()
    assert candidate.context_tokens == 131072


def test_tool_requirement_uses_provider_evidence_not_name():
    profile = RequestProfile(
        workload="agent",
        required_capabilities=(mc.CAP_TOOL_CALL,),
    )

    plain = ollama_candidate_from_show_payload(
        "bigger-looking-model:99b",
        show_payload(
            capabilities=["completion"],
        ),
        endpoint_id="plain",
        endpoint_url="http://plain.example/api/chat",
        node="msi",
        preference=100,
    )

    tools = ollama_candidate_from_show_payload(
        "small-model",
        show_payload(
            capabilities=["completion", "tools"],
        ),
        endpoint_id="tools",
        endpoint_url="http://tools.example/api/chat",
        node="tower",
        preference=0,
    )

    decision = build_routing_decision(
        profile,
        [plain, tools],
    )

    assert decision.primary is not None
    assert decision.primary.model == "small-model"
    assert decision.fallbacks == ()


def test_realistic_tower_vision_request_selects_vl_candidate():
    profile = RequestProfile(
        workload="vision",
        required_capabilities=(mc.CAP_VISION,),
        preferred_capabilities=(
            mc.CAP_REASONING,
            mc.CAP_TOOL_CALL,
        ),
    )

    qwen14 = ollama_candidate_from_show_payload(
        "qwen3:14b",
        show_payload(
            capabilities=[
                "completion",
                "tools",
                "thinking",
            ],
        ),
        endpoint_id="tower",
        endpoint_url=TOWER_URL,
        node="tower",
        preference=40,
    )

    qwen8 = ollama_candidate_from_show_payload(
        "qwen3:8b",
        show_payload(
            capabilities=[
                "completion",
                "tools",
                "thinking",
            ],
        ),
        endpoint_id="tower",
        endpoint_url=TOWER_URL,
        node="tower",
        preference=20,
    )

    qwen_vl = ollama_candidate_from_show_payload(
        "qwen3-vl:8b",
        show_payload(
            capabilities=[
                "completion",
                "vision",
                "tools",
                "thinking",
            ],
            family="qwen3vl",
            context_tokens=262144,
        ),
        endpoint_id="tower",
        endpoint_url=TOWER_URL,
        node="tower",
        preference=10,
    )

    decision = build_routing_decision(
        profile,
        [qwen14, qwen8, qwen_vl],
    )

    assert decision.primary is not None
    assert decision.primary.model == "qwen3-vl:8b"
    assert decision.fallbacks == ()


def test_record_adapter_preserves_reachability_and_preference():
    record = ollama.record_from_show_payload(
        "qwen3:14b",
        show_payload(
            capabilities=[
                "completion",
                "tools",
                "thinking",
            ],
        ),
        endpoint_id="tower",
        base_url=TOWER_URL,
    )

    assert record is not None

    candidate = candidate_from_capability_record(
        record,
        endpoint_id="tower",
        endpoint_url=TOWER_URL,
        node="tower",
        reachable=False,
        preference=77,
    )

    assert candidate.reachable is False
    assert candidate.preference == 77
    assert candidate.context_tokens == 40960


def test_unreachable_tower_candidate_degrades_to_local_candidate():
    profile = RequestProfile(
        workload="general",
    )

    tower = ollama_candidate_from_show_payload(
        "qwen3:14b",
        show_payload(
            capabilities=[
                "completion",
                "tools",
                "thinking",
            ],
        ),
        endpoint_id="tower",
        endpoint_url=TOWER_URL,
        node="tower",
        reachable=False,
        preference=100,
    )

    local = ollama_candidate_from_show_payload(
        "local-model",
        show_payload(
            capabilities=["completion", "tools"],
            family="llama",
            context_tokens=131072,
        ),
        endpoint_id="msi",
        endpoint_url="http://msi.example/api/chat",
        node="msi",
        reachable=True,
        preference=0,
    )

    decision = build_routing_decision(
        profile,
        [tower, local],
    )

    assert decision.primary is not None
    assert decision.primary.node == "msi"
    assert decision.fallbacks == ()
