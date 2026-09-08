"""Versioned, side-effect-free DTOs for adaptive routing.

These contracts form the boundary between the general agentic control
plane and Odysseus' specialized adaptive-routing core.

Design constraints:
- no network I/O
- no filesystem I/O
- no environment reads
- no credentials
- no session or agent lifecycle ownership
- deterministic serialization
- strict basic-type validation
- lossless representation of the frozen legacy routing decision surface
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping


def _require_mapping(
    value: Any,
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{name} must be a mapping"
        )

    return value


def _require_exact_fields(
    data: Mapping[str, Any],
    expected: tuple[str, ...],
) -> None:
    actual = set(data)
    wanted = set(expected)

    missing = sorted(wanted - actual)
    unknown = sorted(actual - wanted)

    if missing:
        raise ValueError(
            "missing required fields: "
            + ", ".join(missing)
        )

    if unknown:
        raise ValueError(
            "unknown fields: "
            + ", ".join(unknown)
        )


def _require_str(
    value: Any,
    *,
    name: str,
) -> str:
    if type(value) is not str:
        raise TypeError(
            f"{name} must be a string"
        )

    if not value.strip():
        raise ValueError(
            f"{name} must not be empty"
        )

    return value


def _require_optional_str(
    value: Any,
    *,
    name: str,
) -> str | None:
    if value is None:
        return None

    return _require_str(
        value,
        name=name,
    )


def _require_bool(
    value: Any,
    *,
    name: str,
) -> bool:
    if type(value) is not bool:
        raise TypeError(
            f"{name} must be a boolean"
        )

    return value


def _require_int(
    value: Any,
    *,
    name: str,
) -> int:
    if type(value) is not int:
        raise TypeError(
            f"{name} must be an integer"
        )

    return value


def _require_positive_int(
    value: Any,
    *,
    name: str,
) -> int:
    value = _require_int(
        value,
        name=name,
    )

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero"
        )

    return value


def _require_nonnegative_int(
    value: Any,
    *,
    name: str,
) -> int:
    value = _require_int(
        value,
        name=name,
    )

    if value < 0:
        raise ValueError(
            f"{name} must be zero or greater"
        )

    return value


def _require_str_sequence(
    value: Any,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(
        value,
        (list, tuple),
    ):
        raise TypeError(
            f"{name} must be a list or tuple"
        )

    result = []

    for index, item in enumerate(value):
        result.append(
            _require_str(
                item,
                name=f"{name}[{index}]",
            )
        )

    if not allow_empty and not result:
        raise ValueError(
            f"{name} must not be empty"
        )

    return tuple(result)


def _require_target_preferences(
    value: Any,
    *,
    name: str,
) -> tuple[tuple[str, str, int], ...]:
    if not isinstance(
        value,
        (list, tuple),
    ):
        raise TypeError(
            f"{name} must be a list or tuple"
        )

    result = []

    for index, item in enumerate(value):
        if not isinstance(
            item,
            (list, tuple),
        ):
            raise TypeError(
                f"{name}[{index}] must be a list or tuple"
            )

        if len(item) != 3:
            raise ValueError(
                f"{name}[{index}] must contain "
                "endpoint_id, model, preference"
            )

        endpoint_id, model, preference = item

        result.append(
            (
                _require_str(
                    endpoint_id,
                    name=f"{name}[{index}][0]",
                ),
                _require_str(
                    model,
                    name=f"{name}[{index}][1]",
                ),
                _require_int(
                    preference,
                    name=f"{name}[{index}][2]",
                ),
            )
        )

    return tuple(result)


def _loads_object(
    payload: str,
) -> Mapping[str, Any]:
    if type(payload) is not str:
        raise TypeError(
            "JSON payload must be a string"
        )

    value = json.loads(payload)

    return _require_mapping(
        value,
        name="JSON payload",
    )


def _canonical_json(
    value: Mapping[str, Any],
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(
    frozen=True,
    slots=True,
)
class RouteTargetRefV1:
    """Executable route identity without endpoint URL or credentials."""

    endpoint_id: str
    model: str

    _FIELDS = (
        "endpoint_id",
        "model",
    )

    def __post_init__(self) -> None:
        _require_str(
            self.endpoint_id,
            name="endpoint_id",
        )
        _require_str(
            self.model,
            name="model",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "model": self.model,
        }

    def to_json(self) -> str:
        return _canonical_json(
            self.to_dict()
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "RouteTargetRefV1":
        data = _require_mapping(
            data,
            name="RouteTargetRefV1",
        )

        _require_exact_fields(
            data,
            cls._FIELDS,
        )

        return cls(
            endpoint_id=data["endpoint_id"],
            model=data["model"],
        )

    @classmethod
    def from_json(
        cls,
        payload: str,
    ) -> "RouteTargetRefV1":
        return cls.from_dict(
            _loads_object(payload)
        )


def _require_route_target_sequence(
    value: Any,
    *,
    name: str,
) -> tuple[RouteTargetRefV1, ...]:
    if not isinstance(
        value,
        (list, tuple),
    ):
        raise TypeError(
            f"{name} must be a list or tuple"
        )

    result = []

    for index, item in enumerate(value):
        if isinstance(item, RouteTargetRefV1):
            result.append(item)
            continue

        result.append(
            RouteTargetRefV1.from_dict(
                _require_mapping(
                    item,
                    name=f"{name}[{index}]",
                )
            )
        )

    return tuple(result)


@dataclass(
    frozen=True,
    slots=True,
)
class ModelCapabilitySnapshotV1:
    """Lossless, credential-free routing snapshot for one candidate."""

    model_id: str
    provider: str
    context_tokens: int | None
    supports_tools: bool
    supports_vision: bool
    availability: str
    health: str
    latency_class: str
    capability_snapshot_version: str
    endpoint_id: str
    node: str
    scope: str
    capabilities: tuple[str, ...]
    reachable: bool
    preference: int

    _FIELDS = (
        "model_id",
        "provider",
        "context_tokens",
        "supports_tools",
        "supports_vision",
        "availability",
        "health",
        "latency_class",
        "capability_snapshot_version",
        "endpoint_id",
        "node",
        "scope",
        "capabilities",
        "reachable",
        "preference",
    )

    def __post_init__(self) -> None:
        _require_str(
            self.model_id,
            name="model_id",
        )
        _require_str(
            self.provider,
            name="provider",
        )

        if self.context_tokens is not None:
            _require_positive_int(
                self.context_tokens,
                name="context_tokens",
            )

        _require_bool(
            self.supports_tools,
            name="supports_tools",
        )
        _require_bool(
            self.supports_vision,
            name="supports_vision",
        )
        _require_str(
            self.availability,
            name="availability",
        )
        _require_str(
            self.health,
            name="health",
        )
        _require_str(
            self.latency_class,
            name="latency_class",
        )
        _require_str(
            self.capability_snapshot_version,
            name="capability_snapshot_version",
        )
        _require_str(
            self.endpoint_id,
            name="endpoint_id",
        )
        _require_str(
            self.node,
            name="node",
        )
        _require_str(
            self.scope,
            name="scope",
        )

        object.__setattr__(
            self,
            "capabilities",
            _require_str_sequence(
                self.capabilities,
                name="capabilities",
                allow_empty=True,
            ),
        )

        _require_bool(
            self.reachable,
            name="reachable",
        )
        _require_int(
            self.preference,
            name="preference",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "context_tokens": self.context_tokens,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "availability": self.availability,
            "health": self.health,
            "latency_class": self.latency_class,
            "capability_snapshot_version": (
                self.capability_snapshot_version
            ),
            "endpoint_id": self.endpoint_id,
            "node": self.node,
            "scope": self.scope,
            "capabilities": list(self.capabilities),
            "reachable": self.reachable,
            "preference": self.preference,
        }

    def to_json(self) -> str:
        return _canonical_json(
            self.to_dict()
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ModelCapabilitySnapshotV1":
        data = _require_mapping(
            data,
            name="ModelCapabilitySnapshotV1",
        )

        _require_exact_fields(
            data,
            cls._FIELDS,
        )

        return cls(
            model_id=data["model_id"],
            provider=data["provider"],
            context_tokens=data["context_tokens"],
            supports_tools=data["supports_tools"],
            supports_vision=data["supports_vision"],
            availability=data["availability"],
            health=data["health"],
            latency_class=data["latency_class"],
            capability_snapshot_version=(
                data["capability_snapshot_version"]
            ),
            endpoint_id=data["endpoint_id"],
            node=data["node"],
            scope=data["scope"],
            capabilities=data["capabilities"],
            reachable=data["reachable"],
            preference=data["preference"],
        )

    @classmethod
    def from_json(
        cls,
        payload: str,
    ) -> "ModelCapabilitySnapshotV1":
        return cls.from_dict(
            _loads_object(payload)
        )


@dataclass(
    frozen=True,
    slots=True,
)
class AdaptiveRouteRequestV1:
    """Explicit, lossless, credential-free input to adaptive routing."""

    request_id: str
    lane: str
    task_features: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    candidate_models: tuple[str, ...]
    context_size_estimate: int
    workload: str
    preferred_capabilities: tuple[str, ...]
    target_preferences: tuple[tuple[str, str, int], ...]
    policy: str

    _FIELDS = (
        "request_id",
        "lane",
        "task_features",
        "required_capabilities",
        "candidate_models",
        "context_size_estimate",
        "workload",
        "preferred_capabilities",
        "target_preferences",
        "policy",
    )

    def __post_init__(self) -> None:
        _require_str(
            self.request_id,
            name="request_id",
        )
        _require_str(
            self.lane,
            name="lane",
        )

        object.__setattr__(
            self,
            "task_features",
            _require_str_sequence(
                self.task_features,
                name="task_features",
                allow_empty=True,
            ),
        )

        object.__setattr__(
            self,
            "required_capabilities",
            _require_str_sequence(
                self.required_capabilities,
                name="required_capabilities",
                allow_empty=True,
            ),
        )

        object.__setattr__(
            self,
            "candidate_models",
            _require_str_sequence(
                self.candidate_models,
                name="candidate_models",
                allow_empty=False,
            ),
        )

        _require_nonnegative_int(
            self.context_size_estimate,
            name="context_size_estimate",
        )
        _require_str(
            self.workload,
            name="workload",
        )

        object.__setattr__(
            self,
            "preferred_capabilities",
            _require_str_sequence(
                self.preferred_capabilities,
                name="preferred_capabilities",
                allow_empty=True,
            ),
        )

        object.__setattr__(
            self,
            "target_preferences",
            _require_target_preferences(
                self.target_preferences,
                name="target_preferences",
            ),
        )

        _require_str(
            self.policy,
            name="policy",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "lane": self.lane,
            "task_features": list(self.task_features),
            "required_capabilities": list(
                self.required_capabilities
            ),
            "candidate_models": list(
                self.candidate_models
            ),
            "context_size_estimate": (
                self.context_size_estimate
            ),
            "workload": self.workload,
            "preferred_capabilities": list(
                self.preferred_capabilities
            ),
            "target_preferences": [
                list(item)
                for item in self.target_preferences
            ],
            "policy": self.policy,
        }

    def to_json(self) -> str:
        return _canonical_json(
            self.to_dict()
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "AdaptiveRouteRequestV1":
        data = _require_mapping(
            data,
            name="AdaptiveRouteRequestV1",
        )

        _require_exact_fields(
            data,
            cls._FIELDS,
        )

        return cls(
            request_id=data["request_id"],
            lane=data["lane"],
            task_features=data["task_features"],
            required_capabilities=(
                data["required_capabilities"]
            ),
            candidate_models=data["candidate_models"],
            context_size_estimate=(
                data["context_size_estimate"]
            ),
            workload=data["workload"],
            preferred_capabilities=(
                data["preferred_capabilities"]
            ),
            target_preferences=(
                data["target_preferences"]
            ),
            policy=data["policy"],
        )

    @classmethod
    def from_json(
        cls,
        payload: str,
    ) -> "AdaptiveRouteRequestV1":
        return cls.from_dict(
            _loads_object(payload)
        )


@dataclass(
    frozen=True,
    slots=True,
)
class AdaptiveRouteDecisionV1:
    """Deterministic result returned by the specialized router."""

    request_id: str
    selected_model: str | None
    selected_endpoint_id: str | None
    fallback_order: tuple[str, ...]
    fallback_targets: tuple[RouteTargetRefV1, ...]
    reason: str
    routing_trace_id: str
    snapshot_version: str

    _FIELDS = (
        "request_id",
        "selected_model",
        "selected_endpoint_id",
        "fallback_order",
        "fallback_targets",
        "reason",
        "routing_trace_id",
        "snapshot_version",
    )

    def __post_init__(self) -> None:
        _require_str(
            self.request_id,
            name="request_id",
        )

        selected_model = _require_optional_str(
            self.selected_model,
            name="selected_model",
        )
        selected_endpoint_id = _require_optional_str(
            self.selected_endpoint_id,
            name="selected_endpoint_id",
        )

        if (
            (selected_model is None)
            != (selected_endpoint_id is None)
        ):
            raise ValueError(
                "selected_model and selected_endpoint_id "
                "must both be set or both be None"
            )

        object.__setattr__(
            self,
            "fallback_order",
            _require_str_sequence(
                self.fallback_order,
                name="fallback_order",
                allow_empty=True,
            ),
        )

        object.__setattr__(
            self,
            "fallback_targets",
            _require_route_target_sequence(
                self.fallback_targets,
                name="fallback_targets",
            ),
        )

        fallback_projection = tuple(
            target.model
            for target in self.fallback_targets
        )

        if fallback_projection != self.fallback_order:
            raise ValueError(
                "fallback_order must equal the model "
                "projection of fallback_targets"
            )

        fallback_identities = [
            (
                target.endpoint_id,
                target.model,
            )
            for target in self.fallback_targets
        ]

        if len(fallback_identities) != len(
            set(fallback_identities)
        ):
            raise ValueError(
                "fallback_targets contain duplicate "
                "endpoint/model identities"
            )

        if selected_model is not None:
            selected_identity = (
                selected_endpoint_id,
                selected_model,
            )

            if selected_identity in fallback_identities:
                raise ValueError(
                    "selected target must not also appear "
                    "in fallback_targets"
                )

        _require_str(
            self.reason,
            name="reason",
        )
        _require_str(
            self.routing_trace_id,
            name="routing_trace_id",
        )
        _require_str(
            self.snapshot_version,
            name="snapshot_version",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "selected_model": self.selected_model,
            "selected_endpoint_id": (
                self.selected_endpoint_id
            ),
            "fallback_order": list(
                self.fallback_order
            ),
            "fallback_targets": [
                target.to_dict()
                for target in self.fallback_targets
            ],
            "reason": self.reason,
            "routing_trace_id": self.routing_trace_id,
            "snapshot_version": self.snapshot_version,
        }

    def to_json(self) -> str:
        return _canonical_json(
            self.to_dict()
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "AdaptiveRouteDecisionV1":
        data = _require_mapping(
            data,
            name="AdaptiveRouteDecisionV1",
        )

        _require_exact_fields(
            data,
            cls._FIELDS,
        )

        return cls(
            request_id=data["request_id"],
            selected_model=data["selected_model"],
            selected_endpoint_id=(
                data["selected_endpoint_id"]
            ),
            fallback_order=data["fallback_order"],
            fallback_targets=data["fallback_targets"],
            reason=data["reason"],
            routing_trace_id=data["routing_trace_id"],
            snapshot_version=data["snapshot_version"],
        )

    @classmethod
    def from_json(
        cls,
        payload: str,
    ) -> "AdaptiveRouteDecisionV1":
        return cls.from_dict(
            _loads_object(payload)
        )


__all__ = [
    "RouteTargetRefV1",
    "ModelCapabilitySnapshotV1",
    "AdaptiveRouteRequestV1",
    "AdaptiveRouteDecisionV1",
]
