import pytest

from src.routing_contract_adapter import (
    RoutingContractAdapterError,
    build_legacy_invocation,
    capability_snapshot_to_legacy_view,
    legacy_decision_to_contract,
    request_to_legacy_view,
    route_via_legacy_adapter,
)
from src.routing_contracts import (
    AdaptiveRouteDecisionV1,
    AdaptiveRouteRequestV1,
    ModelCapabilitySnapshotV1,
    RouteTargetRefV1,
)


def _request(
    candidates=(
        "model-c",
        "model-a",
        "model-b",
    ),
):
    return AdaptiveRouteRequestV1(
        request_id="request-001",
        lane="agent",
        task_features=(
            "tool_use",
            "long_context",
        ),
        required_capabilities=(
            "tool_call",
            "text",
        ),
        candidate_models=candidates,
        context_size_estimate=12000,
        workload="agent",
        preferred_capabilities=(
            "reasoning",
        ),
        target_preferences=(
            (
                "endpoint-a",
                "model-a",
                10,
            ),
        ),
        policy="local_preferred",
    )


def _snapshot(
    model,
    *,
    endpoint_id=None,
    version="snapshot-v1",
    context_tokens=32768,
    capabilities=(
        "reasoning",
        "tool_call",
    ),
    reachable=True,
    preference=0,
):
    if endpoint_id is None:
        endpoint_id = "endpoint-" + model

    return ModelCapabilitySnapshotV1(
        model_id=model,
        provider="ollama",
        context_tokens=context_tokens,
        supports_tools="tool_call" in capabilities,
        supports_vision="vision" in capabilities,
        availability="available",
        health="healthy",
        latency_class="local",
        capability_snapshot_version=version,
        endpoint_id=endpoint_id,
        node="tower",
        scope="local",
        capabilities=capabilities,
        reachable=reachable,
        preference=preference,
    )


def _snapshots():
    return (
        _snapshot(
            "model-c",
            endpoint_id="endpoint-c",
            preference=30,
        ),
        _snapshot(
            "model-a",
            endpoint_id="endpoint-a",
            preference=20,
        ),
        _snapshot(
            "model-b",
            endpoint_id="endpoint-b",
            preference=10,
        ),
    )


def _raw_decision(
    selected=(
        "endpoint-a",
        "model-a",
    ),
    fallbacks=(
        ("endpoint-c", "model-c"),
        ("endpoint-b", "model-b"),
    ),
    reason="capability fit",
):
    if selected is None:
        selected_endpoint_id = None
        selected_model = None
    else:
        selected_endpoint_id, selected_model = selected

    return {
        "request_id": "request-001",
        "selected_endpoint_id": selected_endpoint_id,
        "selected_model": selected_model,
        "fallback_order": [
            model
            for _endpoint_id, model in fallbacks
        ],
        "fallback_targets": [
            {
                "endpoint_id": endpoint_id,
                "model": model,
            }
            for endpoint_id, model in fallbacks
        ],
        "reason": reason,
        "routing_trace_id": "trace-001",
        "snapshot_version": "snapshot-v1",
    }


def test_request_translation_is_lossless():
    request = _request()

    assert request_to_legacy_view(
        request
    ) == {
        "request_id": "request-001",
        "lane": "agent",
        "task_features": [
            "tool_use",
            "long_context",
        ],
        "required_capabilities": [
            "tool_call",
            "text",
        ],
        "candidate_models": [
            "model-c",
            "model-a",
            "model-b",
        ],
        "context_size_estimate": 12000,
        "workload": "agent",
        "preferred_capabilities": [
            "reasoning",
        ],
        "target_preferences": [
            [
                "endpoint-a",
                "model-a",
                10,
            ],
        ],
        "policy": "local_preferred",
    }


def test_snapshot_translation_is_lossless():
    snapshot = _snapshot(
        "model-a",
        endpoint_id="endpoint-a",
        preference=7,
    )

    assert capability_snapshot_to_legacy_view(
        snapshot
    ) == {
        "model": "model-a",
        "model_id": "model-a",
        "provider": "ollama",
        "context_tokens": 32768,
        "supports_tools": True,
        "supports_vision": False,
        "availability": "available",
        "health": "healthy",
        "latency_class": "local",
        "capability_snapshot_version": (
            "snapshot-v1"
        ),
        "endpoint_id": "endpoint-a",
        "node": "tower",
        "scope": "local",
        "capabilities": [
            "reasoning",
            "tool_call",
        ],
        "reachable": True,
        "preference": 7,
    }


def test_invocation_preserves_candidate_order():
    request = _request()

    # Preimage compatibility: snapshots may arrive in another order.
    snapshots = (
        _snapshot(
            "model-b",
            endpoint_id="endpoint-b",
        ),
        _snapshot(
            "model-c",
            endpoint_id="endpoint-c",
        ),
        _snapshot(
            "model-a",
            endpoint_id="endpoint-a",
        ),
    )

    invocation = build_legacy_invocation(
        request,
        snapshots,
    )

    assert invocation[
        "request"
    ]["candidate_models"] == [
        "model-c",
        "model-a",
        "model-b",
    ]

    assert [
        item["model"]
        for item in invocation["capabilities"]
    ] == [
        "model-c",
        "model-a",
        "model-b",
    ]

    assert [
        item["endpoint_id"]
        for item in invocation["capabilities"]
    ] == [
        "endpoint-c",
        "endpoint-a",
        "endpoint-b",
    ]

    assert invocation[
        "snapshot_version"
    ] == "snapshot-v1"


def test_missing_candidate_snapshot_rejected():
    with pytest.raises(
        RoutingContractAdapterError
    ):
        build_legacy_invocation(
            _request(),
            (
                _snapshot("model-c"),
                _snapshot("model-a"),
            ),
        )


def test_extra_snapshot_rejected():
    with pytest.raises(
        RoutingContractAdapterError
    ):
        build_legacy_invocation(
            _request(),
            (
                *_snapshots(),
                _snapshot("model-extra"),
            ),
        )


def test_duplicate_snapshot_rejected():
    with pytest.raises(
        RoutingContractAdapterError
    ):
        build_legacy_invocation(
            _request(),
            (
                _snapshot("model-c"),
                _snapshot("model-a"),
                _snapshot("model-a"),
            ),
        )


def test_mixed_snapshot_versions_rejected():
    with pytest.raises(
        RoutingContractAdapterError
    ):
        build_legacy_invocation(
            _request(),
            (
                _snapshot(
                    "model-c",
                    version="snapshot-v1",
                ),
                _snapshot(
                    "model-a",
                    version="snapshot-v2",
                ),
                _snapshot(
                    "model-b",
                    version="snapshot-v1",
                ),
            ),
        )


def test_duplicate_models_exact_order_preserves_endpoint_identity():
    request = _request(
        (
            "same",
            "other",
            "same",
        )
    )
    snapshots = (
        _snapshot(
            "same",
            endpoint_id="endpoint-1",
        ),
        _snapshot(
            "other",
            endpoint_id="endpoint-other",
        ),
        _snapshot(
            "same",
            endpoint_id="endpoint-2",
        ),
    )

    invocation = build_legacy_invocation(
        request,
        snapshots,
    )

    assert [
        (
            item["endpoint_id"],
            item["model"],
        )
        for item in invocation["capabilities"]
    ] == [
        ("endpoint-1", "same"),
        ("endpoint-other", "other"),
        ("endpoint-2", "same"),
    ]


def test_duplicate_models_ambiguous_reorder_fails_closed():
    request = _request(
        (
            "same",
            "other",
            "same",
        )
    )
    snapshots = (
        _snapshot(
            "same",
            endpoint_id="endpoint-1",
        ),
        _snapshot(
            "same",
            endpoint_id="endpoint-2",
        ),
        _snapshot(
            "other",
            endpoint_id="endpoint-other",
        ),
    )

    with pytest.raises(
        RoutingContractAdapterError,
        match="ambiguous",
    ):
        build_legacy_invocation(
            request,
            snapshots,
        )


def test_duplicate_endpoint_model_identity_rejected():
    request = _request(
        (
            "same",
            "same",
        )
    )
    snapshots = (
        _snapshot(
            "same",
            endpoint_id="endpoint-1",
        ),
        _snapshot(
            "same",
            endpoint_id="endpoint-1",
        ),
    )

    with pytest.raises(
        RoutingContractAdapterError
    ):
        build_legacy_invocation(
            request,
            snapshots,
        )


def test_legacy_decision_translation_preserves_order_and_reason():
    request = _request()

    decision = legacy_decision_to_contract(
        request,
        _raw_decision(),
        snapshot_version="snapshot-v1",
        snapshots=_snapshots(),
    )

    assert isinstance(
        decision,
        AdaptiveRouteDecisionV1,
    )

    assert decision.request_id == \
        "request-001"

    assert decision.selected_model == \
        "model-a"

    assert decision.selected_endpoint_id == \
        "endpoint-a"

    assert decision.fallback_order == (
        "model-c",
        "model-b",
    )

    assert decision.fallback_targets == (
        RouteTargetRefV1(
            endpoint_id="endpoint-c",
            model="model-c",
        ),
        RouteTargetRefV1(
            endpoint_id="endpoint-b",
            model="model-b",
        ),
    )

    assert decision.reason == \
        "capability fit"

    assert decision.routing_trace_id == \
        "trace-001"

    assert decision.snapshot_version == \
        "snapshot-v1"


def test_no_primary_decision_is_representable():
    raw = _raw_decision(
        selected=None,
        fallbacks=(
            ("endpoint-c", "model-c"),
            ("endpoint-a", "model-a"),
            ("endpoint-b", "model-b"),
        ),
        reason="no viable primary",
    )

    decision = legacy_decision_to_contract(
        _request(),
        raw,
        snapshot_version="snapshot-v1",
        snapshots=_snapshots(),
    )

    assert decision.selected_model is None
    assert decision.selected_endpoint_id is None

    assert decision.fallback_order == (
        "model-c",
        "model-a",
        "model-b",
    )


def test_selected_model_outside_candidate_set_rejected():
    raw = _raw_decision(
        selected=(
            "wrong-endpoint",
            "model-a",
        ),
        fallbacks=(),
    )

    with pytest.raises(
        RoutingContractAdapterError
    ):
        legacy_decision_to_contract(
            _request(),
            raw,
            snapshot_version="snapshot-v1",
            snapshots=_snapshots(),
        )


def test_selected_identity_must_be_nullable_together():
    raw = _raw_decision()
    raw["selected_endpoint_id"] = None

    with pytest.raises(
        RoutingContractAdapterError
    ):
        legacy_decision_to_contract(
            _request(),
            raw,
            snapshot_version="snapshot-v1",
            snapshots=_snapshots(),
        )


def test_fallback_outside_candidate_set_rejected():
    raw = _raw_decision(
        fallbacks=(
            ("endpoint-x", "model-b"),
        )
    )

    with pytest.raises(
        RoutingContractAdapterError
    ):
        legacy_decision_to_contract(
            _request(),
            raw,
            snapshot_version="snapshot-v1",
            snapshots=_snapshots(),
        )


def test_fallback_projection_mismatch_rejected():
    raw = _raw_decision()
    raw["fallback_order"] = [
        "model-b",
        "model-c",
    ]

    with pytest.raises(
        RoutingContractAdapterError
    ):
        legacy_decision_to_contract(
            _request(),
            raw,
            snapshot_version="snapshot-v1",
            snapshots=_snapshots(),
        )


def test_duplicate_fallbacks_rejected():
    raw = _raw_decision(
        selected=None,
        fallbacks=(
            ("endpoint-b", "model-b"),
            ("endpoint-b", "model-b"),
        ),
    )

    with pytest.raises(
        RoutingContractAdapterError
    ):
        legacy_decision_to_contract(
            _request(),
            raw,
            snapshot_version="snapshot-v1",
            snapshots=_snapshots(),
        )


def test_selected_model_in_fallbacks_rejected():
    raw = _raw_decision(
        selected=(
            "endpoint-a",
            "model-a",
        ),
        fallbacks=(
            ("endpoint-a", "model-a"),
            ("endpoint-b", "model-b"),
        ),
    )

    with pytest.raises(
        RoutingContractAdapterError
    ):
        legacy_decision_to_contract(
            _request(),
            raw,
            snapshot_version="snapshot-v1",
            snapshots=_snapshots(),
        )


def test_request_id_mismatch_rejected():
    raw = _raw_decision()
    raw["request_id"] = "wrong-request"

    with pytest.raises(
        RoutingContractAdapterError
    ):
        legacy_decision_to_contract(
            _request(),
            raw,
            snapshot_version="snapshot-v1",
            snapshots=_snapshots(),
        )


def test_snapshot_version_mismatch_rejected():
    raw = _raw_decision()
    raw["snapshot_version"] = "wrong-version"

    with pytest.raises(
        RoutingContractAdapterError
    ):
        legacy_decision_to_contract(
            _request(),
            raw,
            snapshot_version="snapshot-v1",
            snapshots=_snapshots(),
        )


def test_legacy_router_invoked_exactly_once():
    calls = []

    def legacy_router(invocation):
        calls.append(invocation)

        return _raw_decision(
            reason="legacy-result"
        )

    result = route_via_legacy_adapter(
        _request(),
        _snapshots(),
        legacy_router,
    )

    assert len(calls) == 1
    assert result.reason == "legacy-result"
    assert result.selected_endpoint_id == \
        "endpoint-a"


def test_invoker_receives_explicit_candidate_set_only():
    captured = {}

    def legacy_router(invocation):
        captured.update(invocation)

        return _raw_decision(
            selected=(
                "endpoint-c",
                "model-c",
            ),
            fallbacks=(
                ("endpoint-a", "model-a"),
                ("endpoint-b", "model-b"),
            ),
        )

    route_via_legacy_adapter(
        _request(),
        _snapshots(),
        legacy_router,
    )

    assert captured[
        "request"
    ]["candidate_models"] == [
        "model-c",
        "model-a",
        "model-b",
    ]

    assert [
        item["model"]
        for item in captured["capabilities"]
    ] == [
        "model-c",
        "model-a",
        "model-b",
    ]

    assert all(
        "endpoint_url" not in item
        for item in captured["capabilities"]
    )


def test_generic_capabilities_remain_authoritative_in_legacy_view():
    snapshot = _snapshot(
        "model-a",
        endpoint_id="endpoint-a",
        capabilities=(
            "reasoning",
            "tool_call",
            "vision",
        ),
    )

    view = capability_snapshot_to_legacy_view(
        snapshot
    )

    assert view["capabilities"] == [
        "reasoning",
        "tool_call",
        "vision",
    ]

    assert view["supports_tools"] is True
    assert view["supports_vision"] is True


def test_context_tokens_none_is_preserved_in_legacy_view():
    snapshot = _snapshot(
        "model-a",
        endpoint_id="endpoint-a",
        context_tokens=None,
    )

    view = capability_snapshot_to_legacy_view(
        snapshot
    )

    assert view["context_tokens"] is None


def test_invoker_may_not_return_non_mapping():
    def legacy_router(_invocation):
        return "model-a"

    with pytest.raises(
        RoutingContractAdapterError
    ):
        route_via_legacy_adapter(
            _request(),
            _snapshots(),
            legacy_router,
        )


def test_adapter_requires_callable():
    with pytest.raises(TypeError):
        route_via_legacy_adapter(
            _request(),
            _snapshots(),
            None,
        )
