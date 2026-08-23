"""Adapters from canonical capability records to adaptive routing candidates.

This module is deliberately network-free. Provider payloads are normalized by the
existing model capability readers; adaptive routing consumes only that canonical
evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.adaptive_routing import RoutingCandidate
from src.model_capability_readers import ollama
from src.model_capability_readers.base import ModelCapabilityRecord


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def candidate_from_capability_record(
    record: ModelCapabilityRecord,
    *,
    endpoint_id: Any,
    endpoint_url: Any,
    node: Any,
    scope: str = "local",
    reachable: bool = True,
    preference: int = 0,
) -> RoutingCandidate:
    """Build a routing candidate from canonical provider-reported evidence."""

    resolved_endpoint_id = str(endpoint_id or "").strip()
    resolved_endpoint_url = str(endpoint_url or "").strip()
    resolved_node = str(node or "").strip()
    resolved_scope = str(scope or "").strip().lower() or "local"

    if not resolved_endpoint_id:
        raise ValueError("endpoint_id is required")
    if not resolved_endpoint_url:
        raise ValueError("endpoint_url is required")
    if not resolved_node:
        raise ValueError("node is required")

    limits = dict(record.capability.limits or {})

    return RoutingCandidate(
        endpoint_id=resolved_endpoint_id,
        endpoint_url=resolved_endpoint_url,
        model=record.model_id,
        node=resolved_node,
        scope=resolved_scope,
        capabilities=tuple(record.capability.capabilities),
        context_tokens=_positive_int(limits.get("context_tokens")),
        reachable=bool(reachable),
        preference=int(preference),
    )


def ollama_candidate_from_show_payload(
    model_id: str,
    payload: Mapping[str, Any],
    *,
    endpoint_id: Any,
    endpoint_url: Any,
    node: Any,
    scope: str = "local",
    reachable: bool = True,
    preference: int = 0,
) -> RoutingCandidate | None:
    """Normalize one Ollama ``/api/show`` payload and build a candidate."""

    record = ollama.record_from_show_payload(
        model_id,
        payload,
        endpoint_id=endpoint_id,
        base_url=endpoint_url,
    )

    if record is None:
        return None

    return candidate_from_capability_record(
        record,
        endpoint_id=endpoint_id,
        endpoint_url=endpoint_url,
        node=node,
        scope=scope,
        reachable=reachable,
        preference=preference,
    )
