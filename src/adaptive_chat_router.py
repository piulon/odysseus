"""Request-local adapter from Adaptive candidates to the legacy ChatRoute.

This module deliberately has no refresh, credential, or dispatch responsibilities.
The legacy resolver remains the fail-safe and the route authorizer remains the
only authority for endpoint ownership, credentials, and model privileges.
"""

from __future__ import annotations

from typing import Optional

from src.adaptive_routing import (
    POLICY_LOCAL_PREFERRED,
    RequestProfile,
    build_routing_decision,
)
from src.adaptive_routing_snapshot import get_adaptive_routing_snapshot
from src.chat_model_router import ChatRoute, RouteTarget, resolve_chat_route


# Must dominate the current local-preferred +/-20 score adjustment while still
# remaining a soft preference that cannot satisfy a hard capability requirement.
LEGACY_TARGET_PREFERENCE_BONUS = 100
DEFAULT_SNAPSHOT_TTL_SECONDS = 60.0


def _owner_key(owner: Optional[str]) -> str:
    return str(owner or "").strip()


def _target_preference(route: ChatRoute) -> tuple[tuple[str, str, int], ...]:
    if route.reason not in {"auto_chat", "auto_agent"}:
        return ()

    endpoint_id = str(route.target.endpoint_id or "").strip()
    model = str(route.target.model or "").strip()
    if not endpoint_id or not model:
        return ()

    return ((endpoint_id, model, LEGACY_TARGET_PREFERENCE_BONUS),)


def resolve_adaptive_chat_route(
    session,
    *,
    owner: Optional[str] = None,
    agent_mode: bool = False,
    enabled: bool = False,
    snapshot_ttl_seconds: float = DEFAULT_SNAPSHOT_TTL_SECONDS,
) -> ChatRoute:
    """Select an Adaptive target while preserving the legacy route contract.

    The legacy route is resolved first and is returned verbatim for manual
    sessions, disabled Adaptive, missing/stale/empty snapshots, no viable
    candidates, or any Adaptive-internal failure. Snapshot endpoint URLs are
    never executable route data; only endpoint identity and model are copied.
    """

    legacy = resolve_chat_route(session, owner=owner, agent_mode=agent_mode)
    if not legacy.auto or not enabled:
        return legacy

    try:
        owner_key = _owner_key(owner)
        snapshot = get_adaptive_routing_snapshot(
            owner_key,
            max_age_seconds=float(snapshot_ttl_seconds),
        )
        if snapshot is None or snapshot.owner != owner_key or not snapshot.candidates:
            return legacy

        profile = RequestProfile(
            workload="agent" if agent_mode else "chat",
            required_capabilities=(),
            preferred_capabilities=(),
            target_preferences=_target_preference(legacy),
            policy=POLICY_LOCAL_PREFERRED,
        )
        decision = build_routing_decision(profile, snapshot.candidates)
        primary = decision.primary
        if primary is None:
            return legacy

        # Keep the legacy manual fallback verbatim. Snapshot endpoint_url is
        # discovery metadata, not authoritative endpoint identity, so it must
        # never suppress an otherwise valid fallback.
        manual_fallback = legacy.manual_fallback

        return ChatRoute(
            auto=True,
            lane=legacy.lane,
            target=RouteTarget(
                endpoint_id=primary.endpoint_id,
                model=primary.model,
            ),
            reason=f"adaptive_{legacy.lane}",
            manual_fallback=manual_fallback,
        )
    except Exception:
        # Adaptive is optional and request-scoped. Any malformed snapshot,
        # scoring error, or TTL failure must preserve the exact legacy route.
        return legacy
