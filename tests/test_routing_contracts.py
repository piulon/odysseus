from dataclasses import fields

import pytest

from src.routing_contracts import (
    AdaptiveRouteDecisionV1,
    AdaptiveRouteRequestV1,
    ModelCapabilitySnapshotV1,
    RouteTargetRefV1,
)


def _snapshot(
    model="qwen3.5:9b-32k",
    *,
    endpoint_id="endpoint-a",
    context_tokens=32768,
    capabilities=(
        "reasoning",
        "tool_call",
    ),
):
    return ModelCapabilitySnapshotV1(
        model_id=model,
        provider="ollama",
        context_tokens=context_tokens,
        supports_tools="tool_call" in capabilities,
        supports_vision="vision" in capabilities,
        availability="available",
        health="healthy",
        latency_class="local",
        capability_snapshot_version="snapshot-v1",
        endpoint_id=endpoint_id,
        node="tower",
        scope="local",
        capabilities=capabilities,
        reachable=True,
        preference=10,
    )


def _request(
    candidates=(
        "qwen3.5:9b-32k",
        "qwen3:14b",
        "qwen3:8b",
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
                "qwen3.5:9b-32k",
                10,
            ),
        ),
        policy="local_preferred",
    )


def _decision():
    return AdaptiveRouteDecisionV1(
        request_id="request-001",
        selected_model="qwen3.5:9b-32k",
        selected_endpoint_id="endpoint-a",
        fallback_order=(
            "qwen3:14b",
            "qwen3:8b",
        ),
        fallback_targets=(
            RouteTargetRefV1(
                endpoint_id="endpoint-b",
                model="qwen3:14b",
            ),
            RouteTargetRefV1(
                endpoint_id="endpoint-c",
                model="qwen3:8b",
            ),
        ),
        reason="best capability fit",
        routing_trace_id="trace-001",
        snapshot_version="snapshot-v1",
    )


def test_construct_valid_contracts():
    assert _snapshot().model_id == \
        "qwen3.5:9b-32k"

    assert _request().lane == "agent"
    assert _request().workload == "agent"

    assert _decision().selected_model == \
        "qwen3.5:9b-32k"

    assert _decision().selected_endpoint_id == \
        "endpoint-a"


def test_route_target_ref_round_trip():
    value = RouteTargetRefV1(
        endpoint_id="endpoint-a",
        model="model-a",
    )

    assert RouteTargetRefV1.from_json(
        value.to_json()
    ) == value


@pytest.mark.parametrize(
    "cls,data",
    [
        (
            RouteTargetRefV1,
            {
                "endpoint_id": "endpoint-a",
            },
        ),
        (
            ModelCapabilitySnapshotV1,
            {
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
                "capabilities": [],
                "reachable": True,
                "preference": 0,
            },
        ),
        (
            AdaptiveRouteRequestV1,
            {
                "request_id": "request-001",
                "lane": "agent",
                "task_features": [],
                "required_capabilities": [],
                "context_size_estimate": 0,
                "workload": "agent",
                "preferred_capabilities": [],
                "target_preferences": [],
                "policy": "local_preferred",
            },
        ),
        (
            AdaptiveRouteDecisionV1,
            {
                "request_id": "request-001",
                "selected_model": "m1",
                "selected_endpoint_id": "e1",
                "reason": "fit",
                "routing_trace_id": "trace",
                "snapshot_version": "v1",
                "fallback_targets": [],
            },
        ),
    ],
)
def test_reject_missing_required_fields(
    cls,
    data,
):
    with pytest.raises(ValueError):
        cls.from_dict(data)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ModelCapabilitySnapshotV1(
            **{
                **_snapshot().to_dict(),
                "context_tokens": True,
            }
        ),
        lambda: ModelCapabilitySnapshotV1(
            **{
                **_snapshot().to_dict(),
                "context_tokens": 0,
            }
        ),
        lambda: ModelCapabilitySnapshotV1(
            **{
                **_snapshot().to_dict(),
                "context_tokens": -1,
            }
        ),
        lambda: ModelCapabilitySnapshotV1(
            **{
                **_snapshot().to_dict(),
                "supports_tools": 1,
            }
        ),
        lambda: ModelCapabilitySnapshotV1(
            **{
                **_snapshot().to_dict(),
                "preference": True,
            }
        ),
        lambda: AdaptiveRouteRequestV1(
            **{
                **_request().to_dict(),
                "candidate_models": "m1",
            }
        ),
        lambda: AdaptiveRouteRequestV1(
            **{
                **_request().to_dict(),
                "context_size_estimate": "100",
            }
        ),
        lambda: AdaptiveRouteRequestV1(
            **{
                **_request().to_dict(),
                "target_preferences": [
                    ["endpoint-a", "model-a"],
                ],
            }
        ),
        lambda: AdaptiveRouteRequestV1(
            **{
                **_request().to_dict(),
                "target_preferences": [
                    [
                        "endpoint-a",
                        "model-a",
                        True,
                    ],
                ],
            }
        ),
        lambda: AdaptiveRouteDecisionV1(
            **{
                **_decision().to_dict(),
                "fallback_order": "m2",
            }
        ),
    ],
)
def test_reject_invalid_basic_types(factory):
    with pytest.raises(
        (TypeError, ValueError)
    ):
        factory()


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(
            RouteTargetRefV1(
                endpoint_id="endpoint-a",
                model="model-a",
            ),
            id="target",
        ),
        pytest.param(
            _snapshot(),
            id="snapshot",
        ),
        pytest.param(
            _request(),
            id="request",
        ),
        pytest.param(
            _decision(),
            id="decision",
        ),
    ],
)
def test_json_round_trip(value):
    restored = type(value).from_json(
        value.to_json()
    )

    assert restored == value

    assert restored.to_json() == \
        value.to_json()


def test_candidate_order_preservation():
    value = AdaptiveRouteRequestV1.from_dict(
        {
            **_request().to_dict(),
            "candidate_models": [
                "model-c",
                "model-a",
                "model-b",
            ],
        }
    )

    assert value.candidate_models == (
        "model-c",
        "model-a",
        "model-b",
    )

    assert value.to_dict()[
        "candidate_models"
    ] == [
        "model-c",
        "model-a",
        "model-b",
    ]


def test_fallback_order_preservation():
    value = AdaptiveRouteDecisionV1.from_dict(
        {
            "request_id": "r1",
            "selected_model": "model-a",
            "selected_endpoint_id": "endpoint-a",
            "fallback_order": [
                "model-c",
                "model-b",
            ],
            "fallback_targets": [
                {
                    "endpoint_id": "endpoint-c",
                    "model": "model-c",
                },
                {
                    "endpoint_id": "endpoint-b",
                    "model": "model-b",
                },
            ],
            "reason": "fit",
            "routing_trace_id": "trace",
            "snapshot_version": "v1",
        }
    )

    assert value.fallback_order == (
        "model-c",
        "model-b",
    )

    assert value.to_dict()[
        "fallback_order"
    ] == [
        "model-c",
        "model-b",
    ]

    assert value.fallback_targets == (
        RouteTargetRefV1(
            endpoint_id="endpoint-c",
            model="model-c",
        ),
        RouteTargetRefV1(
            endpoint_id="endpoint-b",
            model="model-b",
        ),
    )


@pytest.mark.parametrize(
    "cls,value",
    [
        (
            RouteTargetRefV1,
            RouteTargetRefV1(
                endpoint_id="endpoint-a",
                model="model-a",
            ),
        ),
        (
            ModelCapabilitySnapshotV1,
            _snapshot(),
        ),
        (
            AdaptiveRouteRequestV1,
            _request(),
        ),
        (
            AdaptiveRouteDecisionV1,
            _decision(),
        ),
    ],
)
def test_unknown_fields_rejected(cls, value):
    data = value.to_dict()
    data["unexpected"] = "value"

    with pytest.raises(ValueError):
        cls.from_dict(data)


def test_no_secret_fields():
    forbidden_exact = {
        "secret",
        "password",
        "token",
        "credential",
        "credentials",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "auth_token",
    }

    forbidden_prefixes = (
        "secret_",
        "password_",
        "credential_",
        "credentials_",
        "token_",
        "api_key_",
    )

    forbidden_suffixes = (
        "_secret",
        "_password",
        "_credential",
        "_credentials",
        "_token",
        "_api_key",
    )

    def looks_secret(name):
        name = name.lower()

        return (
            name in forbidden_exact
            or name.startswith(forbidden_prefixes)
            or name.endswith(forbidden_suffixes)
        )

    # Model-capacity metadata is not a credential.
    assert looks_secret("context_tokens") is False

    # Representative credential-bearing names remain forbidden.
    assert looks_secret("access_token") is True
    assert looks_secret("refresh_token") is True
    assert looks_secret("client_secret") is True
    assert looks_secret("database_password") is True
    assert looks_secret("provider_credential") is True
    assert looks_secret("api_key") is True

    for cls in (
        RouteTargetRefV1,
        ModelCapabilitySnapshotV1,
        AdaptiveRouteRequestV1,
        AdaptiveRouteDecisionV1,
    ):
        names = [
            field.name.lower()
            for field in fields(cls)
        ]

        for name in names:
            assert not looks_secret(name), name


def test_canonical_json_is_deterministic():
    value = _request()

    assert value.to_json() == \
        value.to_json()

    assert value.to_json().startswith(
        '{"candidate_models":'
    )


def test_context_tokens_none_round_trip():
    value = ModelCapabilitySnapshotV1(
        **{
            **_snapshot().to_dict(),
            "context_tokens": None,
        }
    )

    assert value.context_tokens is None

    restored = ModelCapabilitySnapshotV1.from_json(
        value.to_json()
    )

    assert restored == value
    assert restored.context_tokens is None


def test_generic_capability_vocabulary_is_preserved():
    value = ModelCapabilitySnapshotV1(
        **{
            **_snapshot().to_dict(),
            "capabilities": [
                "reasoning",
                "tool_call",
                "vision",
            ],
        }
    )

    assert value.capabilities == (
        "reasoning",
        "tool_call",
        "vision",
    )


def test_no_primary_is_representable():
    value = AdaptiveRouteDecisionV1(
        request_id="r",
        selected_model=None,
        selected_endpoint_id=None,
        fallback_order=("model-a",),
        fallback_targets=(
            RouteTargetRefV1(
                endpoint_id="endpoint-a",
                model="model-a",
            ),
        ),
        reason="none viable",
        routing_trace_id="trace",
        snapshot_version="v1",
    )

    assert value.selected_model is None
    assert value.selected_endpoint_id is None

    assert AdaptiveRouteDecisionV1.from_json(
        value.to_json()
    ) == value


def test_selected_identity_must_be_nullable_together():
    with pytest.raises(ValueError):
        AdaptiveRouteDecisionV1(
            request_id="r",
            selected_model="model-a",
            selected_endpoint_id=None,
            fallback_order=(),
            fallback_targets=(),
            reason="bad",
            routing_trace_id="trace",
            snapshot_version="v1",
        )


def test_fallback_projection_must_match_targets():
    with pytest.raises(ValueError):
        AdaptiveRouteDecisionV1(
            request_id="r",
            selected_model=None,
            selected_endpoint_id=None,
            fallback_order=("wrong",),
            fallback_targets=(
                RouteTargetRefV1(
                    endpoint_id="endpoint-a",
                    model="model-a",
                ),
            ),
            reason="bad",
            routing_trace_id="trace",
            snapshot_version="v1",
        )


def test_duplicate_fallback_identity_rejected():
    target = RouteTargetRefV1(
        endpoint_id="endpoint-a",
        model="model-a",
    )

    with pytest.raises(ValueError):
        AdaptiveRouteDecisionV1(
            request_id="r",
            selected_model=None,
            selected_endpoint_id=None,
            fallback_order=(
                "model-a",
                "model-a",
            ),
            fallback_targets=(
                target,
                target,
            ),
            reason="bad",
            routing_trace_id="trace",
            snapshot_version="v1",
        )


def test_selected_target_not_repeated_as_fallback():
    with pytest.raises(ValueError):
        AdaptiveRouteDecisionV1(
            request_id="r",
            selected_model="model-a",
            selected_endpoint_id="endpoint-a",
            fallback_order=("model-a",),
            fallback_targets=(
                RouteTargetRefV1(
                    endpoint_id="endpoint-a",
                    model="model-a",
                ),
            ),
            reason="bad",
            routing_trace_id="trace",
            snapshot_version="v1",
        )


def test_endpoint_url_is_absent_from_contract_surface():
    values = (
        RouteTargetRefV1(
            endpoint_id="endpoint-a",
            model="model-a",
        ),
        _snapshot(),
        _request(),
        _decision(),
    )

    for value in values:
        names = {
            field.name
            for field in fields(type(value))
        }

        assert "endpoint_url" not in names
        assert "endpoint_url" not in value.to_dict()
        assert "endpoint_url" not in value.to_json()


def test_lane_and_workload_are_independent_fields():
    request = AdaptiveRouteRequestV1(
        **{
            **_request().to_dict(),
            "lane": "control-plane-lane",
            "workload": "reasoning",
        }
    )

    assert request.lane == "control-plane-lane"
    assert request.workload == "reasoning"


def test_task_features_and_preferred_capabilities_are_independent():
    request = AdaptiveRouteRequestV1(
        **{
            **_request().to_dict(),
            "task_features": ["long_context"],
            "preferred_capabilities": ["vision"],
        }
    )

    assert request.task_features == (
        "long_context",
    )
    assert request.preferred_capabilities == (
        "vision",
    )
