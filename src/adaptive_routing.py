"""Pure scoring primitives for adaptive chat routing.

This module intentionally performs no network I/O and does not mutate settings.
Runtime discovery/capability evidence is supplied by callers so routing decisions
remain deterministic and easy to test.
"""

from dataclasses import dataclass
from typing import Iterable

from src import model_capabilities as mc


POLICY_LOCAL_ONLY = "local_only"
POLICY_LOCAL_PREFERRED = "local_preferred"
POLICY_QUALITY_PREFERRED = "quality_preferred"

POLICIES = frozenset(
    {
        POLICY_LOCAL_ONLY,
        POLICY_LOCAL_PREFERRED,
        POLICY_QUALITY_PREFERRED,
    }
)


@dataclass(frozen=True)
class RequestProfile:
    workload: str = "general"
    required_capabilities: tuple[str, ...] = ()
    preferred_capabilities: tuple[str, ...] = ()
    target_preferences: tuple[tuple[str, str, int], ...] = ()
    policy: str = POLICY_LOCAL_PREFERRED

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(
                f"unknown routing policy: {self.policy}"
            )


@dataclass(frozen=True)
class RoutingCandidate:
    endpoint_id: str
    endpoint_url: str
    model: str
    node: str
    scope: str
    capabilities: tuple[str, ...] = ()
    context_tokens: int | None = None
    reachable: bool = True
    preference: int = 0


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: RoutingCandidate
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RoutingDecision:
    primary: RoutingCandidate | None
    fallbacks: tuple[RoutingCandidate, ...]
    scored: tuple[ScoredCandidate, ...]
    reason: str


def _normalized_caps(
    values: Iterable[str],
) -> frozenset[str]:
    result = set()

    for value in values:
        cap = mc.normalize_capability(value)

        if cap:
            result.add(cap)

    return frozenset(result)


def score_candidate(
    profile: RequestProfile,
    candidate: RoutingCandidate,
) -> ScoredCandidate | None:
    """Score one candidate or reject it when it cannot satisfy the request."""

    if not candidate.reachable:
        return None

    scope = (candidate.scope or "").strip().lower()

    if (
        profile.policy == POLICY_LOCAL_ONLY
        and scope != "local"
    ):
        return None

    caps = _normalized_caps(candidate.capabilities)
    required = _normalized_caps(
        profile.required_capabilities
    )

    missing = required - caps

    if missing:
        return None

    score = int(candidate.preference)
    reasons = [
        f"preference={candidate.preference}"
    ]

    preferred = _normalized_caps(
        profile.preferred_capabilities
    )

    matched_preferred = preferred & caps

    if matched_preferred:
        bonus = 10 * len(matched_preferred)
        score += bonus

        reasons.append(
            "preferred_capabilities="
            + ",".join(sorted(matched_preferred))
        )

    for endpoint_id, model, bonus in profile.target_preferences:
        if (
            candidate.endpoint_id == endpoint_id
            and candidate.model == model
        ):
            bonus = int(bonus)
            score += bonus
            reasons.append(
                f"target_preference={bonus}"
            )
            break

    if profile.policy == POLICY_LOCAL_PREFERRED:
        if scope == "local":
            score += 20
            reasons.append("local_preferred")
        else:
            score -= 20
            reasons.append("cloud_penalty")

    return ScoredCandidate(
        candidate=candidate,
        score=score,
        reasons=tuple(reasons),
    )


def build_routing_decision(
    profile: RequestProfile,
    candidates: Iterable[RoutingCandidate],
) -> RoutingDecision:
    """Rank viable candidates deterministically and build a fallback chain."""

    scored = [
        result
        for candidate in candidates
        if (
            result := score_candidate(
                profile,
                candidate,
            )
        )
        is not None
    ]

    scored.sort(
        key=lambda item: (
            -item.score,
            item.candidate.node,
            item.candidate.endpoint_id,
            item.candidate.model,
        )
    )

    if not scored:
        return RoutingDecision(
            primary=None,
            fallbacks=(),
            scored=(),
            reason=(
                f"{profile.workload}:"
                "no_viable_candidate"
            ),
        )

    primary = scored[0].candidate
    fallbacks = tuple(
        item.candidate
        for item in scored[1:]
    )

    return RoutingDecision(
        primary=primary,
        fallbacks=fallbacks,
        scored=tuple(scored),
        reason=(
            f"{profile.workload}:"
            f"{primary.node}/{primary.model}:"
            f"score={scored[0].score}"
        ),
    )
