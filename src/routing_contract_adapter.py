"""Pure translation adapter for the versioned adaptive-routing contract.

This module deliberately does not import or own a concrete routing
implementation. The existing Odysseus router is supplied as an injected
callable at the seam.

The adapter owns only:

    contract DTO -> neutral legacy invocation envelope
    legacy decision -> contract DTO

It performs no network, filesystem, environment, credential, session,
agent-lifecycle, discovery, persistence, or deployment work.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from src.routing_contracts import (
    AdaptiveRouteDecisionV1,
    AdaptiveRouteRequestV1,
    ModelCapabilitySnapshotV1,
    RouteTargetRefV1,
)


LegacyRouterInvoker = Callable[
    [Mapping[str, Any]],
    Mapping[str, Any],
]


class RoutingContractAdapterError(ValueError):
    """Raised when data cannot cross the routing boundary safely."""


def _require_mapping(
    value: Any,
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutingContractAdapterError(
            f"{name} must be a mapping"
        )

    return value


def _require_nonempty_string(
    value: Any,
    *,
    name: str,
) -> str:
    if type(value) is not str:
        raise RoutingContractAdapterError(
            f"{name} must be a string"
        )

    if not value.strip():
        raise RoutingContractAdapterError(
            f"{name} must not be empty"
        )

    return value


def _require_optional_string(
    value: Any,
    *,
    name: str,
) -> str | None:
    if value is None:
        return None

    return _require_nonempty_string(
        value,
        name=name,
    )


def _require_string_sequence(
    value: Any,
    *,
    name: str,
) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise RoutingContractAdapterError(
            f"{name} must be a sequence of strings"
        )

    result = []

    for index, item in enumerate(value):
        result.append(
            _require_nonempty_string(
                item,
                name=f"{name}[{index}]",
            )
        )

    return tuple(result)


def _require_target_sequence(
    value: Any,
    *,
    name: str,
) -> tuple[RouteTargetRefV1, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise RoutingContractAdapterError(
            f"{name} must be a sequence of route targets"
        )

    result = []

    for index, item in enumerate(value):
        try:
            if isinstance(item, RouteTargetRefV1):
                target = item
            else:
                target = RouteTargetRefV1.from_dict(
                    _require_mapping(
                        item,
                        name=f"{name}[{index}]",
                    )
                )
        except (TypeError, ValueError) as exc:
            if isinstance(
                exc,
                RoutingContractAdapterError,
            ):
                raise

            raise RoutingContractAdapterError(
                f"invalid {name}[{index}]: {exc}"
            ) from exc

        result.append(target)

    return tuple(result)


def capability_snapshot_to_legacy_view(
    snapshot: ModelCapabilitySnapshotV1,
) -> dict[str, Any]:
    """Translate one capability snapshot without adding routing policy."""

    if not isinstance(
        snapshot,
        ModelCapabilitySnapshotV1,
    ):
        raise TypeError(
            "snapshot must be ModelCapabilitySnapshotV1"
        )

    return {
        # ``model`` is the frozen legacy RoutingCandidate field.
        # ``model_id`` remains as the original contract convenience view.
        "model": snapshot.model_id,
        "model_id": snapshot.model_id,
        "provider": snapshot.provider,
        "context_tokens": snapshot.context_tokens,
        "supports_tools": snapshot.supports_tools,
        "supports_vision": snapshot.supports_vision,
        "availability": snapshot.availability,
        "health": snapshot.health,
        "latency_class": snapshot.latency_class,
        "capability_snapshot_version": (
            snapshot.capability_snapshot_version
        ),
        "endpoint_id": snapshot.endpoint_id,
        "node": snapshot.node,
        "scope": snapshot.scope,
        "capabilities": list(snapshot.capabilities),
        "reachable": snapshot.reachable,
        "preference": snapshot.preference,
    }


def request_to_legacy_view(
    request: AdaptiveRouteRequestV1,
) -> dict[str, Any]:
    """Translate a request DTO without selecting candidates or policy."""

    if not isinstance(
        request,
        AdaptiveRouteRequestV1,
    ):
        raise TypeError(
            "request must be AdaptiveRouteRequestV1"
        )

    return {
        "request_id": request.request_id,
        "lane": request.lane,
        "task_features": list(request.task_features),
        "required_capabilities": list(
            request.required_capabilities
        ),
        "candidate_models": list(
            request.candidate_models
        ),
        "context_size_estimate": (
            request.context_size_estimate
        ),
        "workload": request.workload,
        "preferred_capabilities": list(
            request.preferred_capabilities
        ),
        "target_preferences": [
            list(item)
            for item in request.target_preferences
        ],
        "policy": request.policy,
    }


def _validate_and_order_snapshots(
    request: AdaptiveRouteRequestV1,
    snapshots: Sequence[ModelCapabilitySnapshotV1],
) -> tuple[ModelCapabilitySnapshotV1, ...]:
    """Validate candidate coverage and return deterministic request order.

    Compatibility rule:
    - when request model ids are unique, snapshots may arrive in any order
      and are reordered to ``request.candidate_models``;
    - when request model ids contain duplicates, model-only reordering is
      ambiguous, so the received projection must already exactly match the
      request projection. Endpoint/model identity is then preserved as-is.
    """

    if (
        isinstance(snapshots, (str, bytes))
        or not isinstance(snapshots, Sequence)
    ):
        raise TypeError(
            "snapshots must be a sequence"
        )

    supplied = tuple(snapshots)

    for snapshot in supplied:
        if not isinstance(
            snapshot,
            ModelCapabilitySnapshotV1,
        ):
            raise TypeError(
                "every snapshot must be "
                "ModelCapabilitySnapshotV1"
            )

    expected_models = tuple(
        request.candidate_models
    )
    supplied_models = tuple(
        snapshot.model_id
        for snapshot in supplied
    )

    expected_counts = Counter(expected_models)
    supplied_counts = Counter(supplied_models)

    missing = list(
        (expected_counts - supplied_counts).elements()
    )
    extras = list(
        (supplied_counts - expected_counts).elements()
    )

    if missing:
        raise RoutingContractAdapterError(
            "missing capability snapshots: "
            + ", ".join(sorted(missing))
        )

    if extras:
        raise RoutingContractAdapterError(
            "capability snapshots outside candidate set: "
            + ", ".join(sorted(extras))
        )

    request_models_are_unique = (
        len(expected_models)
        == len(set(expected_models))
    )

    if request_models_are_unique:
        by_model = {
            snapshot.model_id: snapshot
            for snapshot in supplied
        }

        ordered = tuple(
            by_model[model]
            for model in expected_models
        )
    else:
        if supplied_models != expected_models:
            raise RoutingContractAdapterError(
                "duplicate model ids make snapshot reordering "
                "ambiguous; snapshots must already match "
                "candidate_models order"
            )

        ordered = supplied

    identities = [
        (
            snapshot.endpoint_id,
            snapshot.model_id,
        )
        for snapshot in ordered
    ]

    if len(identities) != len(set(identities)):
        raise RoutingContractAdapterError(
            "duplicate endpoint/model candidate identity"
        )

    versions = {
        snapshot.capability_snapshot_version
        for snapshot in ordered
    }

    if len(versions) != 1:
        raise RoutingContractAdapterError(
            "candidate capability snapshots must share one version"
        )

    return ordered


def build_legacy_invocation(
    request: AdaptiveRouteRequestV1,
    snapshots: Sequence[ModelCapabilitySnapshotV1],
) -> dict[str, Any]:
    """Build the complete explicit input envelope for the legacy seam."""

    if not isinstance(
        request,
        AdaptiveRouteRequestV1,
    ):
        raise TypeError(
            "request must be AdaptiveRouteRequestV1"
        )

    ordered = _validate_and_order_snapshots(
        request,
        snapshots,
    )

    snapshot_version = (
        ordered[0].capability_snapshot_version
    )

    return {
        "request": request_to_legacy_view(request),
        "capabilities": [
            capability_snapshot_to_legacy_view(snapshot)
            for snapshot in ordered
        ],
        "snapshot_version": snapshot_version,
    }


def legacy_decision_to_contract(
    request: AdaptiveRouteRequestV1,
    legacy_decision: Mapping[str, Any],
    *,
    snapshot_version: str,
    snapshots: Sequence[ModelCapabilitySnapshotV1],
) -> AdaptiveRouteDecisionV1:
    """Validate and translate one legacy result into the V1 contract.

    ``snapshots`` is required so executable endpoint/model identities can
    be checked against the explicit candidate set. This is intentionally
    stronger than validating a model-name projection alone.
    """

    if not isinstance(
        request,
        AdaptiveRouteRequestV1,
    ):
        raise TypeError(
            "request must be AdaptiveRouteRequestV1"
        )

    ordered = _validate_and_order_snapshots(
        request,
        snapshots,
    )

    raw = _require_mapping(
        legacy_decision,
        name="legacy_decision",
    )

    selected_model = _require_optional_string(
        raw.get("selected_model"),
        name="selected_model",
    )
    selected_endpoint_id = _require_optional_string(
        raw.get("selected_endpoint_id"),
        name="selected_endpoint_id",
    )

    if (
        (selected_model is None)
        != (selected_endpoint_id is None)
    ):
        raise RoutingContractAdapterError(
            "selected_model and selected_endpoint_id "
            "must both be set or both be None"
        )

    fallback_order = _require_string_sequence(
        raw.get("fallback_order"),
        name="fallback_order",
    )
    fallback_targets = _require_target_sequence(
        raw.get("fallback_targets"),
        name="fallback_targets",
    )

    fallback_projection = tuple(
        target.model
        for target in fallback_targets
    )

    if fallback_projection != fallback_order:
        raise RoutingContractAdapterError(
            "fallback_order must equal fallback_targets "
            "model projection"
        )

    reason = _require_nonempty_string(
        raw.get("reason"),
        name="reason",
    )
    routing_trace_id = _require_nonempty_string(
        raw.get("routing_trace_id"),
        name="routing_trace_id",
    )
    snapshot_version = _require_nonempty_string(
        snapshot_version,
        name="snapshot_version",
    )

    candidate_targets = {
        (
            snapshot.endpoint_id,
            snapshot.model_id,
        )
        for snapshot in ordered
    }

    if selected_model is None:
        selected_pair = None
    else:
        selected_pair = (
            selected_endpoint_id,
            selected_model,
        )

    if (
        selected_pair is not None
        and selected_pair not in candidate_targets
    ):
        raise RoutingContractAdapterError(
            "selected endpoint/model is outside "
            "the explicit candidate set"
        )

    fallback_pairs = [
        (
            target.endpoint_id,
            target.model,
        )
        for target in fallback_targets
    ]

    invalid_fallbacks = [
        pair
        for pair in fallback_pairs
        if pair not in candidate_targets
    ]

    if invalid_fallbacks:
        raise RoutingContractAdapterError(
            "fallback endpoint/model identities "
            "outside candidate set"
        )

    if len(fallback_pairs) != len(set(fallback_pairs)):
        raise RoutingContractAdapterError(
            "fallback targets contain duplicate identities"
        )

    if (
        selected_pair is not None
        and selected_pair in fallback_pairs
    ):
        raise RoutingContractAdapterError(
            "selected target must not also appear "
            "in fallback targets"
        )

    raw_request_id = raw.get("request_id")

    if (
        raw_request_id is not None
        and raw_request_id != request.request_id
    ):
        raise RoutingContractAdapterError(
            "legacy decision request_id does not match request"
        )

    raw_snapshot_version = raw.get(
        "snapshot_version"
    )

    if (
        raw_snapshot_version is not None
        and raw_snapshot_version != snapshot_version
    ):
        raise RoutingContractAdapterError(
            "legacy decision snapshot_version does not "
            "match input snapshot"
        )

    return AdaptiveRouteDecisionV1(
        request_id=request.request_id,
        selected_model=selected_model,
        selected_endpoint_id=selected_endpoint_id,
        fallback_order=fallback_order,
        fallback_targets=fallback_targets,
        reason=reason,
        routing_trace_id=routing_trace_id,
        snapshot_version=snapshot_version,
    )


def route_via_legacy_adapter(
    request: AdaptiveRouteRequestV1,
    snapshots: Sequence[ModelCapabilitySnapshotV1],
    legacy_router: LegacyRouterInvoker,
) -> AdaptiveRouteDecisionV1:
    """Run one legacy routing decision through the translation seam."""

    if not callable(legacy_router):
        raise TypeError(
            "legacy_router must be callable"
        )

    invocation = build_legacy_invocation(
        request,
        snapshots,
    )

    raw_decision = legacy_router(
        invocation
    )

    return legacy_decision_to_contract(
        request,
        raw_decision,
        snapshot_version=(
            invocation["snapshot_version"]
        ),
        snapshots=snapshots,
    )


__all__ = [
    "LegacyRouterInvoker",
    "RoutingContractAdapterError",
    "capability_snapshot_to_legacy_view",
    "request_to_legacy_view",
    "build_legacy_invocation",
    "legacy_decision_to_contract",
    "route_via_legacy_adapter",
]
