"""Bounded, owner-scoped capability discovery for adaptive routing.

This module is a synchronous producer.  It discovers metadata only; the
snapshot is never an authorization decision and must be revalidated by the
chat route authorizer before dispatch.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from src.adaptive_routing_candidates import ollama_candidate_from_show_payload
from src.adaptive_routing_snapshot import (
    AdaptiveRoutingSnapshot,
    publish_adaptive_routing_snapshot,
)
from src.endpoint_resolver import build_chat_url, resolve_endpoint_runtime
from src.llm_core import _detect_provider, _ollama_api_root
from src.model_capability_readers import ollama

logger = logging.getLogger(__name__)

MAX_PROBE_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_MODELS_PER_ENDPOINT = 128
MAX_LOG_TOKEN_LENGTH = 96

RequestJson = Callable[..., Mapping[str, Any]]


def _safe_log_token(value: Any) -> str:
    text = str(value or "")
    text = "".join(char if ord(char) >= 32 else "_" for char in text)
    return text[:MAX_LOG_TOKEN_LENGTH]


def _owner_key(owner: Any) -> str:
    return str(owner or "").strip()


def _configured_endpoint_ids(owner: str | None) -> tuple[str, ...]:
    from src.settings import get_user_setting

    owner_key = _owner_key(owner)
    result: list[str] = []
    for prefix in ("auto_chat", "auto_agent"):
        key = f"{prefix}_endpoint_id"
        endpoint_id = str(
            get_user_setting(
                key,
                owner_key,
                "",
                inherit_global=False,
            )
            or ""
        ).strip()
        if endpoint_id and endpoint_id not in result:
            result.append(endpoint_id)
    return tuple(result)


def _safe_probe_url(value: Any) -> str:
    raw = str(value or "")
    if not raw or any(ord(char) < 32 or ord(char) == 127 for char in raw):
        raise ValueError("invalid probe URL")
    text = raw.strip()
    if "\\" in text or any(char.isspace() for char in text):
        raise ValueError("invalid probe URL")
    try:
        parsed = urlparse(text)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        username = parsed.username
        password = parsed.password
    except (TypeError, ValueError):
        raise ValueError("invalid probe URL") from None
    if scheme not in {"http", "https"} or not hostname:
        raise ValueError("invalid probe URL")
    if username is not None or password is not None:
        raise ValueError("probe URL userinfo is not allowed")
    if parsed.query or parsed.fragment:
        raise ValueError("probe URL query/fragment is not allowed")
    return text


def _scope_for_endpoint(endpoint_kind: Any, base_url: str) -> str:
    kind = str(endpoint_kind or "auto").strip().lower()
    if kind in {"api", "proxy"}:
        return "cloud"
    if kind == "local":
        return "local"

    host = (urlparse(base_url).hostname or "").lower().rstrip(".")
    if host in {"localhost", "host.docker.internal"} or "." not in host:
        return "local"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return "local" if host.endswith(".local") else "cloud"
    if (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or ipaddress.ip_address(host) in ipaddress.ip_network("100.64.0.0/10")
    ):
        return "local"
    return "cloud"


@dataclass(frozen=True, repr=True)
class _ProbeEndpoint:
    endpoint_id: str
    scope: str
    hidden_models: frozenset[str] = field(repr=True)
    base_url: str = field(repr=False)
    models_url: str = field(repr=False)
    show_url: str = field(repr=False)
    chat_url: str = field(repr=False)
    headers: Mapping[str, str] = field(repr=False, compare=False)


def _load_ollama_endpoint(endpoint_id: str, owner: str | None) -> _ProbeEndpoint | None:
    from core.database import ModelEndpoint, SessionLocal
    from src.auth_helpers import owner_filter
    from src.endpoint_resolver import _endpoint_hidden_models

    db = SessionLocal()
    try:
        query = db.query(ModelEndpoint).filter(
            ModelEndpoint.id == endpoint_id,
            ModelEndpoint.is_enabled == True,  # noqa: E712
        )
        owner_key = _owner_key(owner)
        if owner_key:
            query = owner_filter(query, ModelEndpoint, owner_key)
        else:
            # Match legacy/strict routing: ownerless resolution may only
            # inspect shared endpoints, never rows owned by another account.
            query = query.filter(ModelEndpoint.owner == None)  # noqa: E711
        endpoint = query.first()
        if endpoint is None:
            return None
        model_type = str(getattr(endpoint, "model_type", None) or "llm").strip().lower()
        if model_type != "llm":
            return None

        configured_base = _safe_probe_url(getattr(endpoint, "base_url", ""))
        runtime_base, api_key = resolve_endpoint_runtime(
            endpoint,
            owner=owner_key or None,
        )
        runtime_base = _safe_probe_url(runtime_base)
        if _detect_provider(runtime_base) != "ollama":
            return None

        chat_url = _safe_probe_url(build_chat_url(runtime_base, resolve_host=False))
        if not urlparse(chat_url).path.rstrip("/").endswith("/api/chat"):
            return None
        api_root = _safe_probe_url(_ollama_api_root(runtime_base).rstrip("/"))
        models_url = _safe_probe_url(api_root + "/tags")
        show_url = _safe_probe_url(api_root + "/show")
        # Validate configured provenance too; runtime credentials may change the base.
        _safe_probe_url(configured_base)

        from src.endpoint_resolver import build_headers
        headers = build_headers(api_key, runtime_base)
        return _ProbeEndpoint(
            endpoint_id=str(endpoint.id),
            scope=_scope_for_endpoint(getattr(endpoint, "endpoint_kind", None), configured_base),
            hidden_models=frozenset(_endpoint_hidden_models(endpoint)),
            base_url=runtime_base,
            models_url=models_url,
            show_url=show_url,
            chat_url=chat_url,
            headers=dict(headers or {}),
        )
    finally:
        db.close()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


_PROBE_OPENER = urllib.request.build_opener(_NoRedirect)


def _request_json(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 3.0,
) -> Mapping[str, Any]:
    safe_url = _safe_probe_url(url)
    timeout_value = float(timeout)
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        raise ValueError("timeout must be finite and positive")
    request_headers = dict(headers or {})
    data = None
    if payload is not None:
        data = json.dumps(dict(payload)).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(safe_url, data=data, headers=request_headers)
    response = _PROBE_OPENER.open(request, timeout=timeout_value)
    try:
        raw = response.read(MAX_PROBE_RESPONSE_BYTES + 1)
    finally:
        response.close()
    if len(raw) > MAX_PROBE_RESPONSE_BYTES:
        raise ValueError("probe response too large")
    value = json.loads(raw)
    return value if isinstance(value, Mapping) else {}


def _validate_refresh_timeout(timeout: Any) -> float:
    value = float(timeout)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be finite and positive")
    return value


def refresh_owner_adaptive_snapshot(
    owner: str | None,
    *,
    request_json: RequestJson | None = None,
    generated_at: float | None = None,
    timeout: float = 3.0,
) -> AdaptiveRoutingSnapshot | None:
    owner_key = _owner_key(owner)
    budget = _validate_refresh_timeout(timeout)
    deadline = time.monotonic() + budget
    fetch = request_json or _request_json

    try:
        endpoint_ids = _configured_endpoint_ids(owner_key)
        if not endpoint_ids:
            return publish_adaptive_routing_snapshot(owner_key, (), generated_at=generated_at)
        endpoints: list[_ProbeEndpoint] = []
        for endpoint_id in endpoint_ids:
            try:
                endpoint = _load_ollama_endpoint(endpoint_id, owner_key)
            except Exception:
                return None
            if endpoint is not None:
                endpoints.append(endpoint)
        if not endpoints:
            return None
    except Exception:
        return None

    candidates = []
    tags_succeeded = False
    discovery_incomplete = False

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("adaptive refresh deadline exceeded")
        return left

    for endpoint in endpoints:
        try:
            tags_timeout = remaining()
        except TimeoutError:
            discovery_incomplete = True
            break
        try:
            tags_payload = fetch(
                endpoint.models_url,
                headers=endpoint.headers,
                timeout=min(budget, tags_timeout),
            )
            tags_succeeded = True
            records = ollama.records_from_tags_payload(
                tags_payload,
                endpoint_id=endpoint.endpoint_id,
                base_url=endpoint.base_url,
            )
            records = tuple(
                record
                for record in records
                if record.model_id not in endpoint.hidden_models
            )[:MAX_MODELS_PER_ENDPOINT]
        except Exception:
            continue

        for record in records:
            model_id = record.model_id
            try:
                show_timeout = remaining()
            except TimeoutError:
                discovery_incomplete = True
                break
            try:
                show_payload = fetch(
                    endpoint.show_url,
                    payload={"model": model_id},
                    headers=endpoint.headers,
                    timeout=min(budget, show_timeout),
                )
                candidate = ollama_candidate_from_show_payload(
                    model_id,
                    show_payload,
                    endpoint_id=endpoint.endpoint_id,
                    endpoint_url=endpoint.chat_url,
                    node=endpoint.endpoint_id,
                    scope=endpoint.scope,
                    reachable=True,
                )
            except Exception:
                continue
            if candidate is not None:
                candidates.append(candidate)

        if discovery_incomplete:
            break

    if discovery_incomplete or not tags_succeeded:
        return None
    return publish_adaptive_routing_snapshot(
        owner_key,
        candidates,
        generated_at=generated_at,
    )
