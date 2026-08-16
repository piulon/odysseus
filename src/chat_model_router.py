"""Request-scoped model routing for interactive chat.

Manual sessions preserve their configured endpoint/model exactly.

Sessions with ``auto_route=True`` may choose a configured chat or agent
target for the current request without mutating the persistent session.
"""

from dataclasses import dataclass
from typing import Any, Iterable


Candidate = tuple[str, str, dict]

_ADAPTIVE_SNAPSHOT_MAX_AGE_SECONDS = 150.0
_ADAPTIVE_MAX_FALLBACKS = 2


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


def _auto_routing_mode(
    owner: str | None = None,
) -> str:
    mode = str(
        _read_setting(
            "auto_routing_mode",
            owner,
        )
        or "legacy"
    ).strip().lower()

    return "adaptive" if mode == "adaptive" else "legacy"


def _configured_identity(
    prefix: str,
    owner: str | None = None,
) -> tuple[str, str] | None:
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

    if not endpoint_id or not model:
        return None

    return endpoint_id, model


def _adaptive_target_preferences(
    lane: str,
    owner: str | None = None,
) -> tuple[tuple[str, str, int], ...]:
    weighted: list[tuple[str, str, int]] = []

    if lane == "agent":
        agent_target = _configured_identity(
            "auto_agent",
            owner,
        )
        if agent_target:
            weighted.append(
                (
                    agent_target[0],
                    agent_target[1],
                    50,
                )
            )

        chat_target = _configured_identity(
            "auto_chat",
            owner,
        )
        if chat_target:
            weighted.append(
                (
                    chat_target[0],
                    chat_target[1],
                    20,
                )
            )
    else:
        chat_target = _configured_identity(
            "auto_chat",
            owner,
        )
        if chat_target:
            weighted.append(
                (
                    chat_target[0],
                    chat_target[1],
                    50,
                )
            )

    result: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()

    for endpoint_id, model, bonus in weighted:
        key = (endpoint_id, model)

        if key in seen:
            continue

        seen.add(key)
        result.append(
            (
                endpoint_id,
                model,
                bonus,
            )
        )

    return tuple(result)


def _adaptive_profile(
    lane: str,
    owner: str | None = None,
):
    from src import model_capabilities as mc
    from src.adaptive_routing import RequestProfile

    target_preferences = _adaptive_target_preferences(
        lane,
        owner,
    )

    if lane == "agent":
        return RequestProfile(
            workload="agent",
            required_capabilities=(
                mc.CAP_TOOL_CALL,
            ),
            preferred_capabilities=(
                mc.CAP_REASONING,
            ),
            target_preferences=target_preferences,
        )

    return RequestProfile(
        workload="general",
        target_preferences=target_preferences,
    )


def _resolve_adaptive_candidate(
    routing_candidate,
    owner: str | None = None,
) -> Candidate | None:
    from src.adaptive_routing_dispatch import (
        resolve_adaptive_dispatch_candidate,
    )

    resolved = resolve_adaptive_dispatch_candidate(
        routing_candidate,
        owner=owner,
    )

    if not resolved:
        return None

    candidate = _candidate(*resolved)

    if (
        not candidate[0]
        or not candidate[1]
        or candidate[1] != routing_candidate.model
    ):
        return None

    return candidate


def _resolve_adaptive_route(
    *,
    owner: str | None,
    lane: str,
) -> ChatRoute | None:
    from src.adaptive_routing import build_routing_decision
    from src.adaptive_routing_snapshot import (
        get_adaptive_routing_snapshot,
    )

    snapshot = get_adaptive_routing_snapshot(
        owner or "",
        max_age_seconds=_ADAPTIVE_SNAPSHOT_MAX_AGE_SECONDS,
    )

    if snapshot is None:
        return None

    profile = _adaptive_profile(
        lane,
        owner,
    )

    decision = build_routing_decision(
        profile,
        snapshot.candidates,
    )

    if decision.primary is None:
        return None

    ranked = (
        decision.primary,
        *decision.fallbacks,
    )

    hydrated: list[tuple[Any, Candidate]] = []

    for routing_candidate in ranked:
        dispatch = _resolve_adaptive_candidate(
            routing_candidate,
            owner,
        )

        if dispatch is None:
            continue

        # Dispatch-time hydration may add credentials/headers, but it must
        # never change the endpoint/model identity selected by the scorer.
        # Re-check this invariant at the consumer boundary as defence in
        # depth even though the normal hydration helper enforces it too.
        dispatch = _candidate(*dispatch)
        expected_key = (
            str(routing_candidate.endpoint_url or "").strip().rstrip("/"),
            str(routing_candidate.model or "").strip(),
        )

        if _candidate_key(dispatch) != expected_key:
            continue

        hydrated.append(
            (
                routing_candidate,
                dispatch,
            )
        )

        if len(hydrated) >= 1 + _ADAPTIVE_MAX_FALLBACKS:
            break

    if not hydrated:
        return None

    primary_candidate, primary = hydrated[0]

    fallbacks = _dedupe_candidates(
        primary,
        (
            dispatch
            for _candidate_meta, dispatch in hydrated[1:]
        ),
    )[:_ADAPTIVE_MAX_FALLBACKS]

    return ChatRoute(
        endpoint_url=primary[0],
        model=primary[1],
        headers=primary[2],
        fallbacks=fallbacks,
        reason=(
            f"adaptive_{lane}:"
            f"{profile.workload}:"
            f"{primary_candidate.node}/"
            f"{primary_candidate.model}"
        ),
        lane=lane,
        auto=True,
    )


def _resolve_legacy_auto_route(
    session_primary: Candidate,
    *,
    owner: str | None,
    agent_mode: bool,
    homelab_fastpath: bool,
) -> ChatRoute:
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

    if homelab_fastpath:
        return _resolve_legacy_auto_route(
            session_primary,
            owner=owner,
            agent_mode=agent_mode,
            homelab_fastpath=True,
        )

    lane = "agent" if agent_mode else "chat"

    if _auto_routing_mode(owner) == "adaptive":
        adaptive_route = _resolve_adaptive_route(
            owner=owner,
            lane=lane,
        )

        if adaptive_route is not None:
            return adaptive_route

    return _resolve_legacy_auto_route(
        session_primary,
        owner=owner,
        agent_mode=agent_mode,
        homelab_fastpath=False,
    )
