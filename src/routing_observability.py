"""Safe, request-scoped routing telemetry.

Routing traces contain only target identity and categorical decisions.  They
never include request content, users, sessions, URLs, headers, or credentials.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

logger = logging.getLogger(__name__)


def new_routing_trace() -> str:
    """Return an opaque per-request identifier unrelated to user input."""
    return secrets.token_hex(8)


def log_routing_decision(trace: str, route: Any) -> None:
    logger.info(
        "event=routing_decision routing_trace=%s lane=%s auto=%s reason=%s "
        "endpoint_id=%s selected_model=%s",
        trace,
        getattr(route, "lane", "manual"),
        bool(getattr(route, "auto", False)),
        getattr(route, "reason", "unknown"),
        getattr(getattr(route, "target", None), "endpoint_id", None) or "none",
        getattr(getattr(route, "target", None), "model", None) or "none",
    )


def log_routing_authorized(trace: str, candidate: Any) -> None:
    logger.info(
        "event=routing_authorized routing_trace=%s endpoint_id=%s authorized_model=%s",
        trace,
        getattr(candidate, "endpoint_id", None) or "none",
        getattr(candidate, "model", None) or "none",
    )


def log_manual_authorized(trace: str, model: str) -> None:
    """Record legacy manual hydration when no registered endpoint ID exists."""
    logger.info(
        "event=routing_authorized routing_trace=%s endpoint_id=none authorized_model=%s",
        trace,
        model or "none",
    )


def log_llm_dispatch(
    trace: str | None,
    *,
    lane: str,
    endpoint_id: str | None,
    model: str,
    endpoint_url: str | None = None,
) -> None:
    if not trace:
        return
    provider = "unknown"
    try:
        from src.llm_core import _detect_provider

        provider = _detect_provider(endpoint_url or "") or "unknown"
    except Exception:
        pass
    logger.info(
        "event=llm_dispatch routing_trace=%s lane=%s endpoint_id=%s "
        "dispatch_model=%s dispatch_provider=%s",
        trace,
        lane,
        endpoint_id or "none",
        model or "none",
        provider,
    )


def log_routing_fallback(
    trace: str | None,
    *,
    from_model: str,
    to_model: str,
    reason: str,
) -> None:
    if not trace:
        return
    logger.info(
        "event=routing_fallback routing_trace=%s from_model=%s to_model=%s reason=%s",
        trace,
        from_model or "none",
        to_model or "none",
        reason,
    )
