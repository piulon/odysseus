from src import model_capabilities as mc

from src.adaptive_routing import (
    POLICY_LOCAL_ONLY,
    POLICY_LOCAL_PREFERRED,
    RequestProfile,
    RoutingCandidate,
    build_routing_decision,
)


def candidate(
    model,
    *,
    node="msi",
    scope="local",
    capabilities=(),
    reachable=True,
    preference=0,
):
    return RoutingCandidate(
        endpoint_id=f"ep-{node}",
        endpoint_url=f"http://{node}.example/chat",
        model=model,
        node=node,
        scope=scope,
        capabilities=tuple(capabilities),
        reachable=reachable,
        preference=preference,
    )


def test_hard_vision_requirement_rejects_non_vision_model():
    profile = RequestProfile(
        workload="vision",
        required_capabilities=(mc.CAP_VISION,),
    )

    decision = build_routing_decision(
        profile,
        [
            candidate(
                "qwen3:14b",
                node="tower",
                capabilities=(
                    mc.CAP_TOOL_CALL,
                    mc.CAP_REASONING,
                ),
                preference=50,
            ),
            candidate(
                "qwen3-vl:8b",
                node="tower",
                capabilities=(
                    mc.CAP_VISION,
                    mc.CAP_TOOL_CALL,
                    mc.CAP_REASONING,
                ),
                preference=20,
            ),
        ],
    )

    assert decision.primary is not None
    assert decision.primary.model == "qwen3-vl:8b"
    assert decision.fallbacks == ()


def test_unreachable_candidate_is_removed():
    profile = RequestProfile(workload="reasoning")

    decision = build_routing_decision(
        profile,
        [
            candidate(
                "qwen3:14b",
                node="tower",
                reachable=False,
                preference=100,
            ),
            candidate(
                "qwen3:4b",
                node="msi",
                preference=10,
            ),
        ],
    )

    assert decision.primary is not None
    assert decision.primary.node == "msi"
    assert decision.fallbacks == ()


def test_local_only_excludes_cloud():
    profile = RequestProfile(
        workload="general",
        policy=POLICY_LOCAL_ONLY,
    )

    decision = build_routing_decision(
        profile,
        [
            candidate(
                "cloud-model",
                node="openai",
                scope="cloud",
                preference=1000,
            ),
            candidate(
                "local-model",
                node="msi",
                preference=1,
            ),
        ],
    )

    assert decision.primary is not None
    assert decision.primary.model == "local-model"
    assert decision.fallbacks == ()


def test_local_preferred_adds_local_bias():
    profile = RequestProfile(
        workload="general",
        policy=POLICY_LOCAL_PREFERRED,
    )

    decision = build_routing_decision(
        profile,
        [
            candidate(
                "cloud-model",
                node="openai",
                scope="cloud",
                preference=15,
            ),
            candidate(
                "local-model",
                node="msi",
                preference=0,
            ),
        ],
    )

    assert decision.primary is not None
    assert decision.primary.model == "local-model"


def test_reasoning_is_soft_preference():
    profile = RequestProfile(
        workload="reasoning",
        preferred_capabilities=(mc.CAP_REASONING,),
    )

    decision = build_routing_decision(
        profile,
        [
            candidate(
                "plain",
                capabilities=(mc.CAP_TOOL_CALL,),
            ),
            candidate(
                "thinker",
                node="tower",
                capabilities=(mc.CAP_REASONING,),
            ),
        ],
    )

    assert decision.primary is not None
    assert decision.primary.model == "thinker"


def test_unknown_capability_does_not_satisfy_hard_requirement():
    profile = RequestProfile(
        workload="agent",
        required_capabilities=(mc.CAP_TOOL_CALL,),
    )

    decision = build_routing_decision(
        profile,
        [
            candidate(
                "unknown",
                capabilities=(),
            ),
        ],
    )

    assert decision.primary is None
    assert decision.reason == "agent:no_viable_candidate"


def test_equal_scores_have_deterministic_order():
    profile = RequestProfile(workload="general")

    decision = build_routing_decision(
        profile,
        [
            candidate(
                "b",
                node="tower",
            ),
            candidate(
                "a",
                node="msi",
            ),
        ],
    )

    assert decision.primary is not None
    assert decision.primary.node == "msi"
    assert [item.node for item in decision.fallbacks] == ["tower"]


def test_invalid_policy_is_rejected():
    try:
        RequestProfile(policy="send_everything_anywhere")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid policy was accepted")


def test_explicit_target_preference_breaks_capability_tie():
    profile = RequestProfile(
        workload="reasoning",
        preferred_capabilities=(
            mc.CAP_REASONING,
        ),
        target_preferences=(
            ("ep-tower", "qwen3:14b", 50),
        ),
    )

    decision = build_routing_decision(
        profile,
        [
            candidate(
                "qwen3:4b",
                node="msi",
                capabilities=(
                    mc.CAP_REASONING,
                ),
            ),
            candidate(
                "qwen3:14b",
                node="tower",
                capabilities=(
                    mc.CAP_REASONING,
                ),
            ),
        ],
    )

    assert decision.primary is not None
    assert decision.primary.node == "tower"
    assert decision.primary.model == "qwen3:14b"


def test_target_preference_never_overrides_hard_capability_requirement():
    profile = RequestProfile(
        workload="vision",
        required_capabilities=(
            mc.CAP_VISION,
        ),
        target_preferences=(
            ("ep-tower", "qwen3:14b", 1000),
        ),
    )

    decision = build_routing_decision(
        profile,
        [
            candidate(
                "qwen3:14b",
                node="tower",
                capabilities=(
                    mc.CAP_REASONING,
                ),
            ),
            candidate(
                "gemma3:4b",
                node="msi",
                capabilities=(
                    mc.CAP_VISION,
                ),
            ),
        ],
    )

    assert decision.primary is not None
    assert decision.primary.node == "msi"
    assert decision.primary.model == "gemma3:4b"


def test_unreachable_preferred_target_degrades_to_viable_candidate():
    profile = RequestProfile(
        workload="agent",
        required_capabilities=(
            mc.CAP_TOOL_CALL,
        ),
        target_preferences=(
            ("ep-tower", "qwen3:14b", 50),
        ),
    )

    decision = build_routing_decision(
        profile,
        [
            candidate(
                "qwen3:14b",
                node="tower",
                capabilities=(
                    mc.CAP_TOOL_CALL,
                ),
                reachable=False,
            ),
            candidate(
                "qwen3:4b",
                node="msi",
                capabilities=(
                    mc.CAP_TOOL_CALL,
                ),
            ),
        ],
    )

    assert decision.primary is not None
    assert decision.primary.node == "msi"
    assert decision.primary.model == "qwen3:4b"
