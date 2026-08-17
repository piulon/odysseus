"""Pure, request-scoped selection for legacy automatic chat routing.

This module selects endpoint identifiers and models only. Credential hydration,
privilege enforcement, capability checks, and LLM dispatch belong to later
layers. Resolution never mutates the persistent session.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from core.database import ModelEndpoint, SessionLocal
from src.settings import get_user_setting


RouteLane = Literal["manual", "chat", "agent"]


@dataclass(frozen=True)
class RouteTarget:
    """A selected target without hydrated credentials."""

    model: str
    endpoint_id: Optional[str] = None
    endpoint_url: Optional[str] = None


@dataclass(frozen=True)
class ChatRoute:
    """Immutable routing decision for one request."""

    auto: bool
    lane: RouteLane
    target: RouteTarget
    reason: str
    manual_fallback: Optional[RouteTarget] = None


def _clean(value) -> str:
    return str(value or "").strip()


def _manual_target(session) -> RouteTarget:
    return RouteTarget(
        model=_clean(getattr(session, "model", "")),
        endpoint_url=_clean(getattr(session, "endpoint_url", "")),
    )


def _configured_target(lane: Literal["chat", "agent"], owner: Optional[str]):
    prefix = f"auto_{lane}"
    db = None
    try:
        endpoint_id = _clean(
            get_user_setting(
                f"{prefix}_endpoint_id",
                owner or "",
                "",
                inherit_global=False,
            )
        )
        model = _clean(
            get_user_setting(
                f"{prefix}_model",
                owner or "",
                "",
                inherit_global=False,
            )
        )
        if not endpoint_id or not model:
            return None, None, "unconfigured"

        db = SessionLocal()
        query = db.query(ModelEndpoint).filter(
            ModelEndpoint.id == endpoint_id,
            ModelEndpoint.is_enabled == True,  # noqa: E712
        )
        if owner:
            query = query.filter(
                (ModelEndpoint.owner == owner) | (ModelEndpoint.owner == None)  # noqa: E711
            )
        else:
            # Anonymous/single-user resolution may use shared endpoints only;
            # it must never select a row explicitly owned by another account.
            query = query.filter(ModelEndpoint.owner == None)  # noqa: E711
        endpoint = query.first()
        if endpoint is None:
            return None, None, "unavailable"

        return RouteTarget(model=model, endpoint_id=endpoint_id), endpoint.base_url, None
    except Exception:
        return None, None, "unavailable"
    finally:
        if db is not None:
            db.close()


def _same_target(
    auto_target: RouteTarget,
    endpoint_base_url: str,
    manual_target: RouteTarget,
) -> bool:
    if auto_target.model != manual_target.model:
        return False
    try:
        from src.endpoint_resolver import build_chat_url, normalize_base

        auto_url = build_chat_url(normalize_base(endpoint_base_url or ""))
    except Exception:
        return False
    return auto_url.rstrip("/") == (manual_target.endpoint_url or "").rstrip("/")


def resolve_chat_route(
    session,
    *,
    owner: Optional[str] = None,
    agent_mode: bool = False,
) -> ChatRoute:
    """Select a manual, chat, or agent target without mutating ``session``."""
    manual = _manual_target(session)
    if not bool(getattr(session, "auto_route", False)):
        return ChatRoute(
            auto=False,
            lane="manual",
            target=manual,
            reason="manual",
        )

    lane: Literal["chat", "agent"] = "agent" if agent_mode else "chat"
    auto_target, endpoint_base_url, error = _configured_target(lane, owner)
    if auto_target is None:
        return ChatRoute(
            auto=True,
            lane=lane,
            target=manual,
            reason=f"auto_{lane}_{error}",
        )

    manual_fallback = (
        None
        if _same_target(auto_target, endpoint_base_url, manual)
        else manual
    )
    return ChatRoute(
        auto=True,
        lane=lane,
        target=auto_target,
        reason=f"auto_{lane}",
        manual_fallback=manual_fallback,
    )
