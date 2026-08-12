"""Request-scoped model routing for interactive chat.

Manual sessions preserve their configured endpoint/model exactly.

Sessions with ``auto_route=True`` may choose a configured chat or agent
target for the current request without mutating the persistent session.
"""

from dataclasses import dataclass
from typing import Any, Iterable


Candidate = tuple[str, str, dict]


@dataclass(frozen=True)
class ChatRoute:
    endpoint_url: str
    model: str
    headers: dict
    fallbacks: tuple[Candidate, ...]
    reason: str
    lane: str
    auto: bool


def _candidate(
    endpoint_url: Any,
    model: Any,
    headers: Any,
) -> Candidate:
    return (
        str(endpoint_url or "").strip(),
        str(model or "").strip(),
        dict(headers or {}),
    )


def _candidate_key(candidate: Candidate) -> tuple[str, str]:
    url, model, _headers = candidate
    return url.rstrip("/"), model


def _dedupe_candidates(
    primary: Candidate,
    candidates: Iterable[Candidate],
) -> tuple[Candidate, ...]:
    seen = {_candidate_key(primary)}
    result: list[Candidate] = []

    for raw in candidates:
        candidate = _candidate(*raw)

        if not candidate[0] or not candidate[1]:
            continue

        key = _candidate_key(candidate)

        if key in seen:
            continue

        seen.add(key)
        result.append(candidate)

    return tuple(result)


def _read_setting(key: str, owner: str | None = None):
    from src.settings import get_user_setting, load_settings

    settings = load_settings()

    return get_user_setting(
        key,
        owner or "",
        settings.get(key),
    )


def _configured_target(
    prefix: str,
    owner: str | None = None,
) -> Candidate | None:
    endpoint_id = str(
        _read_setting(
            f"{prefix}_endpoint_id",
            owner,
        )
        or ""
    ).strip()

    model = str(
        _read_setting(
            f"{prefix}_model",
            owner,
        )
        or ""
    ).strip()

    if not endpoint_id:
        return None

    from src.endpoint_resolver import resolve_endpoint_by_id

    resolved = resolve_endpoint_by_id(
        endpoint_id,
        model,
        owner=owner,
    )

    if not resolved:
        return None

    return _candidate(*resolved)


def _default_fallbacks(
    owner: str | None = None,
) -> list[Candidate]:
    from src.endpoint_resolver import (
        resolve_chat_fallback_candidates,
    )

    return [
        _candidate(*candidate)
        for candidate in resolve_chat_fallback_candidates(
            owner=owner,
        )
    ]


def _direct_homelab_fastpath(message: Any) -> bool:
    if not isinstance(message, str) or not message.strip():
        return False

    try:
        from src.agent_tools.homelab_tools import (
            classify_direct_homelab_request,
        )

        command = classify_direct_homelab_request(
            message,
            {"homelab"},
            continuation=False,
        )

        return command is not None
    except Exception:
        return False


def resolve_chat_route(
    sess,
    *,
    owner: str | None = None,
    agent_mode: bool = False,
    message: Any = "",
    allow_auto: bool = True,
) -> ChatRoute:
    """Resolve the effective model route for one request.

    ``sess.endpoint_url/model/headers`` remain the persistent/manual
    selection and are never modified here.
    """

    session_primary = _candidate(
        getattr(sess, "endpoint_url", ""),
        getattr(sess, "model", ""),
        getattr(sess, "headers", {}),
    )

    auto_requested = bool(getattr(sess, "auto_route", False))

    if not auto_requested:
        configured_fallbacks = _default_fallbacks(owner)

        return ChatRoute(
            endpoint_url=session_primary[0],
            model=session_primary[1],
            headers=session_primary[2],
            fallbacks=_dedupe_candidates(
                session_primary,
                configured_fallbacks,
            ),
            reason="manual",
            lane="manual",
            auto=False,
        )

    if not allow_auto:
        return ChatRoute(
            endpoint_url=session_primary[0],
            model=session_primary[1],
            headers=session_primary[2],
            fallbacks=(),
            reason="auto_capability_bypass",
            lane="manual",
            auto=False,
        )

    homelab_fastpath = (
        bool(agent_mode)
        and _direct_homelab_fastpath(message)
    )

    lane = (
        "agent"
        if agent_mode and not homelab_fastpath
        else "chat"
    )

    target = _configured_target(
        f"auto_{lane}",
        owner,
    )

    if not target:
        return ChatRoute(
            endpoint_url=session_primary[0],
            model=session_primary[1],
            headers=session_primary[2],
            fallbacks=(),
            reason=f"auto_{lane}_unconfigured",
            lane=lane,
            auto=True,
        )

    fallback_seed: list[Candidate] = []

    if lane == "agent":
        chat_fallback = _configured_target(
            "auto_chat",
            owner,
        )
        if chat_fallback:
            fallback_seed.append(chat_fallback)

    reason = (
        "auto_homelab_fastpath"
        if homelab_fastpath
        else f"auto_{lane}"
    )

    return ChatRoute(
        endpoint_url=target[0],
        model=target[1],
        headers=target[2],
        fallbacks=_dedupe_candidates(
            target,
            fallback_seed,
        ),
        reason=reason,
        lane=lane,
        auto=True,
    )
