"""Periodic producer loop for owner-scoped adaptive routing snapshots.

This module owns scheduling only. It has no database, authentication, or HTTP
knowledge beyond calling ``refresh_owner_adaptive_snapshot`` outside the event
loop. Application wiring supplies the current owner set.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from typing import Any

from src.adaptive_routing_refresh import refresh_owner_adaptive_snapshot


logger = logging.getLogger(__name__)


def _normalized_owners(owners: Iterable[Any] | None) -> tuple[str, ...]:
    """Return stable, de-duplicated owner keys.

    The empty owner is retained because auth-disabled/single-user deployments
    use it as their legitimate owner scope.
    """

    result: list[str] = []
    seen: set[str] = set()

    if isinstance(owners, (str, bytes)):
        source: Iterable[Any] = (owners,)
    else:
        source = owners or ()

    for raw in source:
        owner = str(raw or "").strip()
        if owner in seen:
            continue
        seen.add(owner)
        result.append(owner)

    return tuple(result)


def run_adaptive_routing_refresh_cycle(
    owners: Iterable[Any] | None,
    *,
    refresh_timeout: float = 6.0,
    refresh_func: Callable[..., Any] | None = None,
) -> dict[str, bool]:
    """Refresh each owner independently and report which publishes succeeded.

    A failure for one owner is logged and does not prevent later owners from
    refreshing. Network/database work belongs to ``refresh_func``; callers must
    run this cycle outside an asyncio event loop thread.
    """

    timeout = float(refresh_timeout)
    if timeout <= 0:
        raise ValueError("refresh_timeout must be positive")

    refresh = refresh_func or refresh_owner_adaptive_snapshot
    results: dict[str, bool] = {}

    for owner in _normalized_owners(owners):
        try:
            results[owner] = refresh(owner, timeout=timeout) is not None
        except Exception as exc:
            logger.warning(
                "Adaptive routing snapshot refresh failed for owner %r: %s",
                owner,
                exc,
            )
            results[owner] = False

    return results


async def adaptive_routing_refresh_loop(
    owners_provider: Callable[[], Iterable[Any] | None],
    *,
    interval_seconds: float = 60.0,
    refresh_timeout: float = 6.0,
    refresh_func: Callable[..., Any] | None = None,
    sleep_func: Callable[[float], Any] | None = None,
    to_thread_func: Callable[..., Any] | None = None,
) -> None:
    """Refresh immediately, then repeat at a fixed cadence until cancelled.

    The synchronous refresh cycle is dispatched through ``asyncio.to_thread``
    by default, so endpoint probes never block the application's event loop.
    Cancellation is propagated unchanged for clean FastAPI shutdown.
    """

    interval = float(interval_seconds)
    timeout = float(refresh_timeout)

    if interval <= 0:
        raise ValueError("interval_seconds must be positive")
    if timeout <= 0:
        raise ValueError("refresh_timeout must be positive")

    sleep = sleep_func or asyncio.sleep
    to_thread = to_thread_func or asyncio.to_thread
    refresh = refresh_func or refresh_owner_adaptive_snapshot

    while True:
        try:
            owners = owners_provider()
            await to_thread(
                run_adaptive_routing_refresh_cycle,
                owners,
                refresh_timeout=timeout,
                refresh_func=refresh,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Adaptive routing refresh cycle failed: %s", exc)

        await sleep(interval)
