"""Lifecycle-managed background refresh for Adaptive routing snapshots.

The chat request path never calls this module. Discovery runs only from this
background worker and only while the global Adaptive feature gate is enabled.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Iterable
from typing import Any

from src.adaptive_routing_refresh import refresh_owner_adaptive_snapshot
from src.adaptive_routing_snapshot import clear_adaptive_routing_snapshot
from src.settings import get_setting

logger = logging.getLogger(__name__)

DEFAULT_REFRESH_INTERVAL_SECONDS = 30.0
DEFAULT_REFRESH_TIMEOUT_SECONDS = 3.0
DEFAULT_MAX_CONCURRENCY = 2

Owner = str | None
OwnerProvider = Callable[[], Iterable[Owner]]

_worker_task: asyncio.Task[Any] | None = None


def _operationally_enabled() -> bool:
    """Operator kill-switch for the in-process Adaptive refresh worker."""
    raw = os.getenv("ODYSSEUS_ADAPTIVE_ROUTING_REFRESH", "1")
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _feature_enabled() -> bool:
    """Fail closed: only literal boolean True enables network discovery."""
    try:
        return get_setting("adaptive_routing_enabled", False) is True
    except Exception:
        return False


def _normalize_owners(values: Iterable[Owner]) -> tuple[Owner, ...]:
    result: list[Owner] = []
    seen: set[str] = set()

    for value in values:
        key = str(value or "").strip()
        if key in seen:
            continue
        seen.add(key)
        result.append(key or None)

    return tuple(result)


async def refresh_adaptive_routing_once(
    owner_provider: OwnerProvider,
    *,
    timeout: float = DEFAULT_REFRESH_TIMEOUT_SECONDS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> int:
    """Refresh all known owners once.

    Owners without configured Auto endpoints are safe: the underlying refresh
    publishes an empty snapshot without performing network discovery.
    """
    if not _feature_enabled():
        return 0

    try:
        owners = _normalize_owners(owner_provider())
    except Exception as exc:
        logger.warning(
            "Adaptive routing owner enumeration failed: %s",
            type(exc).__name__,
        )
        return 0

    if not owners:
        return 0

    concurrency = max(1, int(max_concurrency))
    semaphore = asyncio.Semaphore(concurrency)

    async def _refresh(owner: Owner) -> None:
        async with semaphore:
            try:
                await asyncio.to_thread(
                    refresh_owner_adaptive_snapshot,
                    owner,
                    timeout=timeout,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Adaptive routing refresh failed for owner=%r: %s",
                    owner,
                    type(exc).__name__,
                )

    await asyncio.gather(*(_refresh(owner) for owner in owners))
    return len(owners)


async def _worker_loop(owner_provider: OwnerProvider) -> None:
    while True:
        try:
            await refresh_adaptive_routing_once(owner_provider)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Adaptive routing worker tick failed: %s",
                type(exc).__name__,
            )

        await asyncio.sleep(DEFAULT_REFRESH_INTERVAL_SECONDS)


def start_adaptive_routing_worker(
    owner_provider: OwnerProvider,
) -> asyncio.Task[Any] | None:
    """Start the process-local worker once; return its task."""
    global _worker_task

    if not _operationally_enabled():
        logger.info(
            "Adaptive routing refresh worker disabled "
            "(ODYSSEUS_ADAPTIVE_ROUTING_REFRESH=0)"
        )
        return None

    if _worker_task is not None and not _worker_task.done():
        return _worker_task

    _worker_task = asyncio.create_task(_worker_loop(owner_provider))
    logger.info(
        "Adaptive routing refresh worker started "
        "(interval=%ss timeout=%ss concurrency=%s)",
        DEFAULT_REFRESH_INTERVAL_SECONDS,
        DEFAULT_REFRESH_TIMEOUT_SECONDS,
        DEFAULT_MAX_CONCURRENCY,
    )
    return _worker_task


async def stop_adaptive_routing_worker() -> None:
    """Cancel and await the worker during application shutdown."""
    global _worker_task

    # Invalidate before cancelling the asyncio task.  Cancellation cannot stop
    # a refresh already running in asyncio.to_thread(); its captured snapshot
    # generation will therefore reject any publication after shutdown begins.
    clear_adaptive_routing_snapshot()

    task = _worker_task
    _worker_task = None

    if task is None:
        return

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    logger.info("Adaptive routing refresh worker stopped")


def adaptive_routing_worker_running() -> bool:
    task = _worker_task
    return task is not None and not task.done()
