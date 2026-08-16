import asyncio

import pytest

import src.adaptive_routing_refresher as refresher


def test_cycle_refreshes_distinct_owners_in_order():
    calls = []

    def fake_refresh(owner, *, timeout):
        calls.append((owner, timeout))
        return object()

    result = refresher.run_adaptive_routing_refresh_cycle(
        ["alice", "alice", " bob ", None, ""],
        refresh_timeout=4.5,
        refresh_func=fake_refresh,
    )

    assert calls == [
        ("alice", 4.5),
        ("bob", 4.5),
        ("", 4.5),
    ]
    assert result == {
        "alice": True,
        "bob": True,
        "": True,
    }


def test_cycle_treats_single_owner_string_as_one_owner():
    calls = []

    refresher.run_adaptive_routing_refresh_cycle(
        "alice",
        refresh_func=lambda owner, *, timeout: calls.append(owner) or object(),
    )

    assert calls == ["alice"]


def test_cycle_isolates_owner_failure():
    calls = []

    def fake_refresh(owner, *, timeout):
        calls.append(owner)
        if owner == "alice":
            raise RuntimeError("offline")
        return object()

    result = refresher.run_adaptive_routing_refresh_cycle(
        ["alice", "bob"],
        refresh_func=fake_refresh,
    )

    assert calls == ["alice", "bob"]
    assert result == {"alice": False, "bob": True}


def test_cycle_reports_none_as_unsuccessful_publish():
    result = refresher.run_adaptive_routing_refresh_cycle(
        ["alice"],
        refresh_func=lambda owner, *, timeout: None,
    )

    assert result == {"alice": False}


def test_cycle_with_no_owners_performs_no_refresh():
    calls = []

    result = refresher.run_adaptive_routing_refresh_cycle(
        None,
        refresh_func=lambda owner, *, timeout: calls.append(owner),
    )

    assert result == {}
    assert calls == []


@pytest.mark.parametrize("timeout", [0, -1])
def test_cycle_rejects_non_positive_timeout(timeout):
    with pytest.raises(ValueError, match="refresh_timeout"):
        refresher.run_adaptive_routing_refresh_cycle(
            ["alice"],
            refresh_timeout=timeout,
        )


def test_loop_refreshes_immediately_then_sleeps_and_propagates_cancel():
    events = []

    def owners_provider():
        events.append("owners")
        return ["alice"]

    def fake_refresh(owner, *, timeout):
        events.append(("refresh", owner, timeout))
        return object()

    async def immediate_to_thread(func, *args, **kwargs):
        events.append("to_thread")
        return func(*args, **kwargs)

    async def cancel_on_sleep(seconds):
        events.append(("sleep", seconds))
        raise asyncio.CancelledError

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await refresher.adaptive_routing_refresh_loop(
                owners_provider,
                interval_seconds=12.0,
                refresh_timeout=3.0,
                refresh_func=fake_refresh,
                sleep_func=cancel_on_sleep,
                to_thread_func=immediate_to_thread,
            )

    asyncio.run(run())

    assert events == [
        "owners",
        "to_thread",
        ("refresh", "alice", 3.0),
        ("sleep", 12.0),
    ]


def test_loop_survives_owner_provider_error_until_cancel():
    events = []

    def owners_provider():
        events.append("owners_error")
        raise RuntimeError("auth source unavailable")

    async def should_not_thread(*args, **kwargs):
        raise AssertionError("cycle must not run when owner provider fails")

    async def cancel_on_sleep(seconds):
        events.append(("sleep", seconds))
        raise asyncio.CancelledError

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await refresher.adaptive_routing_refresh_loop(
                owners_provider,
                interval_seconds=9.0,
                sleep_func=cancel_on_sleep,
                to_thread_func=should_not_thread,
            )

    asyncio.run(run())

    assert events == ["owners_error", ("sleep", 9.0)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("interval_seconds", 0),
        ("interval_seconds", -1),
        ("refresh_timeout", 0),
        ("refresh_timeout", -1),
    ],
)
def test_loop_rejects_non_positive_timing(field, value):
    kwargs = {field: value}

    async def run():
        with pytest.raises(ValueError):
            await refresher.adaptive_routing_refresh_loop(
                lambda: [],
                **kwargs,
            )

    asyncio.run(run())
