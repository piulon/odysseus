from src.adaptive_routing import RoutingCandidate
from src.adaptive_routing_snapshot import (
    clear_adaptive_routing_snapshot,
    get_adaptive_routing_snapshot,
    publish_adaptive_routing_snapshot,
)


def candidate(model="model-a"):
    return RoutingCandidate(
        endpoint_id="ep-a",
        endpoint_url="http://example.invalid/chat",
        model=model,
        node="msi",
        scope="local",
    )


def setup_function():
    clear_adaptive_routing_snapshot()


def teardown_function():
    clear_adaptive_routing_snapshot()


def test_missing_snapshot_returns_none():
    assert (
        get_adaptive_routing_snapshot(
            "alice",
            max_age_seconds=30,
            now=100,
        )
        is None
    )


def test_fresh_snapshot_is_returned():
    published = publish_adaptive_routing_snapshot(
        "alice",
        [candidate()],
        generated_at=100,
    )

    current = get_adaptive_routing_snapshot(
        "alice",
        max_age_seconds=30,
        now=129,
    )

    assert current == published
    assert current is not None
    assert current.owner == "alice"
    assert current.candidates == (candidate(),)
    assert current.generated_at == 100


def test_stale_snapshot_is_rejected_without_being_destroyed():
    published = publish_adaptive_routing_snapshot(
        "alice",
        [candidate()],
        generated_at=100,
    )

    assert (
        get_adaptive_routing_snapshot(
            "alice",
            max_age_seconds=5,
            now=106,
        )
        is None
    )

    assert (
        get_adaptive_routing_snapshot(
            "alice",
            max_age_seconds=10,
            now=106,
        )
        == published
    )


def test_publish_replaces_only_same_owner_snapshot():
    first = publish_adaptive_routing_snapshot(
        "alice",
        [candidate("model-a")],
        generated_at=100,
    )
    second = publish_adaptive_routing_snapshot(
        "alice",
        [candidate("model-b")],
        generated_at=101,
    )

    assert first != second
    assert (
        get_adaptive_routing_snapshot(
            "alice",
            max_age_seconds=30,
            now=102,
        )
        == second
    )


def test_owner_snapshots_are_isolated():
    alice = publish_adaptive_routing_snapshot(
        "alice",
        [candidate("alice-model")],
        generated_at=100,
    )
    bob = publish_adaptive_routing_snapshot(
        "bob",
        [candidate("bob-model")],
        generated_at=100,
    )

    assert (
        get_adaptive_routing_snapshot(
            "alice",
            max_age_seconds=30,
            now=101,
        )
        == alice
    )
    assert (
        get_adaptive_routing_snapshot(
            "bob",
            max_age_seconds=30,
            now=101,
        )
        == bob
    )
    assert alice.candidates != bob.candidates


def test_invalid_max_age_is_rejected():
    publish_adaptive_routing_snapshot(
        "alice",
        [candidate()],
        generated_at=100,
    )

    try:
        get_adaptive_routing_snapshot(
            "alice",
            max_age_seconds=0,
            now=100,
        )
    except ValueError as exc:
        assert "max_age_seconds" in str(exc)
    else:
        raise AssertionError("expected ValueError")
