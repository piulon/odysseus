import asyncio
import threading
import time

from src import adaptive_routing_worker as worker


def test_feature_gate_off_performs_no_owner_enumeration_or_refresh(monkeypatch):
    monkeypatch.setattr(worker, "get_setting", lambda key, default=None: False)

    def owner_provider():
        raise AssertionError("owners must not be enumerated while Adaptive is off")

    def refresh(*args, **kwargs):
        raise AssertionError("network refresh must not run while Adaptive is off")

    monkeypatch.setattr(worker, "refresh_owner_adaptive_snapshot", refresh)

    result = asyncio.run(worker.refresh_adaptive_routing_once(owner_provider))

    assert result == 0


def test_feature_gate_requires_literal_true(monkeypatch):
    for malformed in ("true", "1", 1, {}, []):
        monkeypatch.setattr(
            worker,
            "get_setting",
            lambda key, default=None, value=malformed: value,
        )

        result = asyncio.run(
            worker.refresh_adaptive_routing_once(
                lambda: (_ for _ in ()).throw(
                    AssertionError("owner provider must not run")
                )
            )
        )

        assert result == 0


def test_owner_normalization_deduplicates_and_preserves_ownerless():
    assert worker._normalize_owners(
        [" alice ", "alice", None, "", " bob ", "bob"]
    ) == (
        "alice",
        None,
        "bob",
    )


def test_refresh_once_is_owner_scoped_and_failure_isolated(monkeypatch):
    monkeypatch.setattr(worker, "get_setting", lambda key, default=None: True)

    calls = []

    def refresh(owner, *, timeout):
        calls.append((owner, timeout))
        if owner == "bob":
            raise RuntimeError("offline")

    monkeypatch.setattr(worker, "refresh_owner_adaptive_snapshot", refresh)

    result = asyncio.run(
        worker.refresh_adaptive_routing_once(
            lambda: ["alice", "bob", "alice", None],
            timeout=2.5,
            max_concurrency=2,
        )
    )

    assert result == 3
    assert sorted(
        calls,
        key=lambda item: "" if item[0] is None else item[0],
    ) == [
        (None, 2.5),
        ("alice", 2.5),
        ("bob", 2.5),
    ]


def test_refresh_once_respects_concurrency_bound(monkeypatch):
    monkeypatch.setattr(worker, "get_setting", lambda key, default=None: True)

    lock = threading.Lock()
    active = 0
    peak = 0
    seen = []

    def refresh(owner, *, timeout):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            seen.append(owner)
        try:
            time.sleep(0.03)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(worker, "refresh_owner_adaptive_snapshot", refresh)

    result = asyncio.run(
        worker.refresh_adaptive_routing_once(
            lambda: ["a", "b", "c", "d"],
            max_concurrency=2,
        )
    )

    assert result == 4
    assert set(seen) == {"a", "b", "c", "d"}
    assert 1 <= peak <= 2


def test_operational_kill_switch_prevents_start(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_ADAPTIVE_ROUTING_REFRESH", "0")
    worker._worker_task = None

    async def scenario():
        task = worker.start_adaptive_routing_worker(lambda: ["alice"])
        assert task is None
        assert worker.adaptive_routing_worker_running() is False

    asyncio.run(scenario())


def test_start_is_idempotent_and_stop_cancels_cleanly(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_ADAPTIVE_ROUTING_REFRESH", "1")
    worker._worker_task = None

    async def fake_loop(owner_provider):
        await asyncio.Event().wait()

    monkeypatch.setattr(worker, "_worker_loop", fake_loop)

    async def scenario():
        first = worker.start_adaptive_routing_worker(lambda: ["alice"])
        second = worker.start_adaptive_routing_worker(lambda: ["bob"])

        try:
            assert first is not None
            assert first is second
            assert worker.adaptive_routing_worker_running() is True
        finally:
            await worker.stop_adaptive_routing_worker()

        assert worker.adaptive_routing_worker_running() is False
        assert first.cancelled()

    asyncio.run(scenario())
