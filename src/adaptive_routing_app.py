"""Application wiring helpers for adaptive routing snapshot refresh.

These helpers intentionally contain no endpoint probing or routing policy.
They only interpret the opt-in environment gate and derive owner scopes from
the existing authentication file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_REFRESH_ENV = "ODYSSEUS_ADAPTIVE_ROUTING_REFRESH"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def adaptive_routing_refresh_enabled(
    environ: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether periodic adaptive snapshot refresh is explicitly enabled."""

    env = os.environ if environ is None else environ
    return str(env.get(_REFRESH_ENV, "") or "").strip().lower() in _TRUTHY


def adaptive_routing_owners_from_auth_file(
    auth_file: str | os.PathLike[str],
) -> tuple[str, ...]:
    """Return normalized owner scopes from Odysseus' auth file.

    A missing file or an explicitly empty users mapping represents the
    auth-disabled/single-user deployment and therefore returns the legitimate
    empty owner scope. Corrupt or structurally invalid auth data fails closed
    with no owners.
    """

    path = Path(auth_file)

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return ("",)
    except Exception:
        return ()

    if not isinstance(payload, dict):
        return ()

    users = payload.get("users", {})
    if not isinstance(users, dict):
        return ()

    owners = sorted(
        {
            str(raw or "").strip()
            for raw in users.keys()
            if str(raw or "").strip()
        }
    )

    return tuple(owners) if owners else ("",)
