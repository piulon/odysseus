"""Request-local authorization and hydration for one selected chat route."""

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from core.database import ModelEndpoint, SessionLocal
from src.chat_model_router import ChatRoute
from src.endpoint_resolver import (
    StrictEndpointResolutionError,
    build_chat_url,
    normalize_base,
    resolve_endpoint_by_id_strict,
)
from src.model_authorization import authorize_model


RouteLane = Literal["manual", "chat", "agent"]


class _ImmutablePrivileges(Mapping[str, Any]):
    """Defensive privilege snapshot whose mutable values are never shared."""

    __slots__ = ("_data",)

    def __init__(self, privileges: Mapping[str, Any]):
        self._data = deepcopy(dict(privileges))

    def __getitem__(self, key: str) -> Any:
        return deepcopy(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


@dataclass(frozen=True)
class ChatRouteAuthContext:
    """Resolved request authority required before route authorization."""

    owner: Optional[str]
    privileges: Optional[Mapping[str, Any]]
    is_admin: bool = False
    single_user: bool = False

    def __post_init__(self):
        owner = str(self.owner or "").strip() or None
        object.__setattr__(self, "owner", owner)
        if self.privileges is not None:
            object.__setattr__(
                self,
                "privileges",
                _ImmutablePrivileges(self.privileges),
            )


class _ImmutableHeaders(Mapping[str, str]):
    """Defensive, immutable runtime headers with a redacted representation."""

    __slots__ = ("_data",)

    def __init__(self, headers: Optional[Mapping[str, str]] = None):
        self._data = dict(headers or {})

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return "<runtime headers>"

    def __deepcopy__(self, memo):
        return dict(self._data)


@dataclass(frozen=True)
class AuthorizedChatRoute:
    """One executable target authorized for the current request only."""

    auto: bool
    lane: RouteLane
    reason: str
    model: str
    endpoint_id: Optional[str]
    endpoint_url: str = field(repr=False)
    headers: Mapping[str, str] = field(repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "headers", _ImmutableHeaders(self.headers))

    def to_safe_metadata(self) -> dict[str, Any]:
        """Return the explicit allowlist of fields safe for internal telemetry."""
        return {
            "auto": self.auto,
            "lane": self.lane,
            "reason": self.reason,
            "model": self.model,
            "endpoint_id": self.endpoint_id,
        }


class ChatRouteAuthorizationError(Exception):
    """Sanitized route-authorization failure identified only by stable code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _validate_auth_context(auth: ChatRouteAuthContext) -> None:
    if not isinstance(auth, ChatRouteAuthContext):
        raise ChatRouteAuthorizationError("invalid_auth_context")
    if auth.single_user:
        if auth.owner is not None or auth.is_admin:
            raise ChatRouteAuthorizationError("invalid_auth_context")
        return
    if auth.owner is None:
        raise ChatRouteAuthorizationError("invalid_auth_context")
    if not auth.privileges:
        raise ChatRouteAuthorizationError("privileges_unavailable")


def _manual_url_matches_endpoint(session_url: str, endpoint_base: str) -> bool:
    try:
        current = str(session_url or "").strip().rstrip("/")
        base = normalize_base(endpoint_base or "").rstrip("/")
        if not current or not base:
            return False
        return current in {
            base,
            base + "/chat/completions",
            build_chat_url(base, resolve_host=False).rstrip("/"),
        }
    except Exception:
        return False


def _registered_manual_endpoint(session_url: str, owner: Optional[str]):
    db = None
    try:
        db = SessionLocal()
        matches = [
            endpoint
            for endpoint in db.query(ModelEndpoint).all()
            if _manual_url_matches_endpoint(session_url, endpoint.base_url or "")
        ]
        if not matches:
            return None, False

        enabled = [endpoint for endpoint in matches if endpoint.is_enabled]
        if owner:
            eligible = [
                endpoint
                for endpoint in enabled
                if endpoint.owner in (None, owner)
            ]
            eligible.sort(key=lambda endpoint: endpoint.owner != owner)
        else:
            eligible = [endpoint for endpoint in enabled if endpoint.owner is None]

        if eligible:
            return eligible[0].id, True
        if owner is None and enabled:
            # Single-user compatibility: an old owned row remains usable as a
            # raw manual target, but is not reclassified as strict Auto/shared.
            return None, False
        raise ChatRouteAuthorizationError("manual_endpoint_not_allowed")
    except ChatRouteAuthorizationError:
        raise
    except Exception:
        raise ChatRouteAuthorizationError("manual_endpoint_unavailable") from None
    finally:
        if db is not None:
            db.close()


def _hydrate_manual_target(
    session,
    *,
    auth: ChatRouteAuthContext,
):
    model = str(getattr(session, "model", "") or "").strip()
    endpoint_url = str(getattr(session, "endpoint_url", "") or "").strip()
    endpoint_id, registered = _registered_manual_endpoint(endpoint_url, auth.owner)

    if registered and endpoint_id:
        try:
            return resolve_endpoint_by_id_strict(endpoint_id, model, owner=auth.owner)
        except StrictEndpointResolutionError as exc:
            raise ChatRouteAuthorizationError(exc.code) from None

    if not (auth.single_user or auth.is_admin):
        raise ChatRouteAuthorizationError("manual_endpoint_not_allowed")
    if not endpoint_url:
        raise ChatRouteAuthorizationError("manual_endpoint_unavailable")

    return None, model, endpoint_url, dict(getattr(session, "headers", {}) or {})


def authorize_chat_route(
    route: ChatRoute,
    session,
    *,
    auth: ChatRouteAuthContext,
) -> AuthorizedChatRoute:
    """Hydrate and authorize exactly one candidate without dispatch or fallback."""
    _validate_auth_context(auth)
    endpoint_id = getattr(route.target, "endpoint_id", None)
    if route.auto and endpoint_id:
        try:
            hydrated = resolve_endpoint_by_id_strict(
                endpoint_id,
                route.target.model,
                owner=auth.owner,
            )
        except StrictEndpointResolutionError as exc:
            raise ChatRouteAuthorizationError(exc.code) from None
        effective_endpoint_id = hydrated.endpoint_id
        model = hydrated.model
        endpoint_url = hydrated.endpoint_url
        headers = hydrated.headers
    else:
        manual = _hydrate_manual_target(
            session,
            auth=auth,
        )
        if isinstance(manual, tuple):
            effective_endpoint_id, model, endpoint_url, headers = manual
        else:
            effective_endpoint_id = manual.endpoint_id
            model = manual.model
            endpoint_url = manual.endpoint_url
            headers = manual.headers

    authorization = authorize_model(model, auth.privileges)
    if not authorization.allowed:
        raise ChatRouteAuthorizationError(authorization.reason)

    return AuthorizedChatRoute(
        auto=route.auto,
        lane=route.lane,
        reason=route.reason,
        model=model,
        endpoint_id=effective_endpoint_id,
        endpoint_url=endpoint_url,
        headers=headers,
    )
