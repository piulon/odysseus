"""Dispatch-time hydration for adaptive routing candidates.

Adaptive snapshots intentionally do not retain credentials. This module
re-hydrates only the headers needed for dispatch from the local endpoint row.
It performs no endpoint probing, DNS/Tailscale resolution, or model discovery.
"""

from __future__ import annotations

import json
from typing import Any

from core.database import ModelEndpoint, SessionLocal
from src.adaptive_routing import RoutingCandidate
from src.endpoint_resolver import build_headers


DispatchCandidate = tuple[str, str, dict]


def _hidden_models(raw: Any) -> frozenset[str]:
    if not raw:
        return frozenset()

    value = raw

    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except Exception:
            return frozenset()

    if not isinstance(value, list):
        return frozenset()

    return frozenset(
        str(item).strip()
        for item in value
        if str(item).strip()
    )


def resolve_adaptive_dispatch_candidate(
    candidate: RoutingCandidate,
    *,
    owner: str | None = None,
) -> DispatchCandidate | None:
    """Return a dispatch tuple without performing network I/O.

    Router v2 currently publishes local Ollama candidates only. Re-check the
    endpoint row at dispatch time for enablement/ownership and hidden-model
    changes, but use the already-resolved snapshot URL so this hot path never
    invokes URL/Tailscale resolution.
    """

    if not isinstance(candidate, RoutingCandidate):
        return None

    if (candidate.scope or "").strip().lower() != "local":
        return None

    endpoint_id = str(candidate.endpoint_id or "").strip()
    endpoint_url = str(candidate.endpoint_url or "").strip()
    model = str(candidate.model or "").strip()

    if not endpoint_id or not endpoint_url or not model:
        return None

    db = SessionLocal()

    try:
        q = db.query(ModelEndpoint).filter(
            ModelEndpoint.id == endpoint_id,
            ModelEndpoint.is_enabled == True,
        )

        if owner:
            from src.auth_helpers import owner_filter

            q = owner_filter(
                q,
                ModelEndpoint,
                owner,
            )

        ep = q.first()

        if ep is None:
            return None

        if str(getattr(ep, "model_type", None) or "llm").lower() != "llm":
            return None

        endpoint_kind = str(
            getattr(ep, "endpoint_kind", None)
            or "auto"
        ).strip().lower()

        if endpoint_kind not in {"auto", "local"}:
            return None

        # Session-backed/OAuth providers may refresh credentials at call time.
        # They are outside the local-only v2 dispatch contract for now.
        if getattr(ep, "provider_auth_id", None):
            return None

        if model in _hidden_models(
            getattr(ep, "hidden_models", None)
        ):
            return None

        base = str(
            getattr(ep, "base_url", "")
            or ""
        ).strip()

        headers = build_headers(
            getattr(ep, "api_key", None),
            base,
        )

        return (
            endpoint_url,
            model,
            dict(headers or {}),
        )
    finally:
        db.close()
