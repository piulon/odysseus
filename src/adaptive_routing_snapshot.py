"""Process-local snapshot store for adaptive model routing.

The store performs no network or database I/O. Refresh code publishes immutable,
owner-scoped candidate snapshots; request-time routing only reads a snapshot for
the same owner and only while it is fresh enough for the caller's policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
import time
from typing import Iterable

from src.adaptive_routing import RoutingCandidate


@dataclass(frozen=True)
class AdaptiveRoutingSnapshot:
    owner: str
    candidates: tuple[RoutingCandidate, ...]
    generated_at: float


_lock = RLock()
_snapshots: dict[str, AdaptiveRoutingSnapshot] = {}


def _owner_key(owner: str | None) -> str:
    return str(owner or "").strip()


def publish_adaptive_routing_snapshot(
    owner: str | None,
    candidates: Iterable[RoutingCandidate],
    *,
    generated_at: float | None = None,
) -> AdaptiveRoutingSnapshot:
    """Atomically replace the current snapshot for one owner."""

    key = _owner_key(owner)
    snapshot = AdaptiveRoutingSnapshot(
        owner=key,
        candidates=tuple(candidates),
        generated_at=(
            float(time.time())
            if generated_at is None
            else float(generated_at)
        ),
    )

    with _lock:
        _snapshots[key] = snapshot

    return snapshot


def get_adaptive_routing_snapshot(
    owner: str | None,
    *,
    max_age_seconds: float,
    now: float | None = None,
) -> AdaptiveRoutingSnapshot | None:
    """Return one owner's snapshot only when it is within the caller TTL."""

    ttl = float(max_age_seconds)
    if ttl <= 0:
        raise ValueError("max_age_seconds must be positive")

    current_time = (
        float(time.time())
        if now is None
        else float(now)
    )

    key = _owner_key(owner)

    with _lock:
        snapshot = _snapshots.get(key)

    if snapshot is None:
        return None

    age = max(0.0, current_time - snapshot.generated_at)
    if age > ttl:
        return None

    return snapshot


def clear_adaptive_routing_snapshot(
    owner: str | None = None,
) -> None:
    """Clear one owner snapshot, or all snapshots when owner is omitted."""

    with _lock:
        if owner is None:
            _snapshots.clear()
            return

        _snapshots.pop(_owner_key(owner), None)
