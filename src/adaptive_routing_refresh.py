"""Owner-scoped adaptive routing snapshot refresh.

This module performs bounded network I/O outside the chat routing hot path.
The first implementation intentionally supports configured Ollama endpoints
only. Provider payloads are normalized through the existing canonical
capability readers before candidates are published.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from src.adaptive_routing_candidates import ollama_candidate_from_show_payload
from src.adaptive_routing_snapshot import (
    AdaptiveRoutingSnapshot,
    publish_adaptive_routing_snapshot,
)
from src.model_capability_readers import ollama

logger = logging.getLogger(__name__)

RequestJson = Callable[..., Mapping[str, Any]]


def _configured_endpoint_ids(owner: str | None) -> tuple[str, ...]:
    """Return de-duplicated auto-chat/auto-agent endpoint ids for one owner."""

    from src.settings import get_user_setting, load_settings

    settings = load_settings()
    owner_key = str(owner or "").strip()
    endpoint_ids: list[str] = []

    for prefix in ("auto_chat", "auto_agent"):
        key = f"{prefix}_endpoint_id"
        endpoint_id = str(
            get_user_setting(
                key,
                owner_key,
                settings.get(key, ""),
            )
            or ""
        ).strip()

        if endpoint_id and endpoint_id not in endpoint_ids:
            endpoint_ids.append(endpoint_id)

    return tuple(endpoint_ids)


def _load_ollama_endpoint(
    endpoint_id: str,
    owner: str | None,
) -> dict[str, Any] | None:
    """Load one owner-visible Ollama endpoint as transient refresh metadata."""

    from core.database import ModelEndpoint, SessionLocal
    from src.auth_helpers import owner_filter
    from src.endpoint_resolver import (
        _endpoint_hidden_models,
        build_chat_url,
        build_headers,
        build_models_url,
        normalize_base,
        resolve_endpoint_runtime,
    )

    db = SessionLocal()
    try:
        query = db.query(ModelEndpoint).filter(
            ModelEndpoint.id == endpoint_id,
            ModelEndpoint.is_enabled == True,
        )

        owner_key = str(owner or "").strip()
        if owner_key:
            query = owner_filter(
                query,
                ModelEndpoint,
                owner_key,
            )

        endpoint = query.first()
        if endpoint is None:
            return None

        model_type = str(
            getattr(endpoint, "model_type", None)
            or "llm"
        ).strip().lower()
        if model_type != "llm":
            return None

        configured_base = normalize_base(
            getattr(endpoint, "base_url", "")
            or ""
        )
        configured_models_url = build_models_url(
            configured_base,
        )

        if (
            not configured_models_url
            or not configured_models_url.rstrip("/").endswith("/api/tags")
        ):
            return None

        base, api_key = resolve_endpoint_runtime(
            endpoint,
            owner=owner_key or None,
        )
        base = normalize_base(base)

        models_url = build_models_url(base)
        if (
            not models_url
            or not models_url.rstrip("/").endswith("/api/tags")
        ):
            return None

        api_root = models_url.rstrip("/").rsplit("/", 1)[0]

        return {
            "endpoint_id": str(endpoint.id),
            "base_url": base,
            "models_url": models_url,
            "show_url": api_root + "/show",
            "chat_url": build_chat_url(base),
            "headers": build_headers(api_key, base),
            "hidden_models": frozenset(
                _endpoint_hidden_models(endpoint)
            ),
        }
    finally:
        db.close()


def _request_json(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 3.0,
) -> Mapping[str, Any]:
    request_headers = dict(headers or {})
    data = None

    if payload is not None:
        data = json.dumps(dict(payload)).encode("utf-8")
        request_headers.setdefault(
            "Content-Type",
            "application/json",
        )

    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
    )

    with urllib.request.urlopen(
        request,
        timeout=float(timeout),
    ) as response:
        value = json.load(response)

    return value if isinstance(value, Mapping) else {}


def refresh_owner_adaptive_snapshot(
    owner: str | None,
    *,
    request_json: RequestJson | None = None,
    generated_at: float | None = None,
    timeout: float = 3.0,
) -> AdaptiveRoutingSnapshot | None:
    """Probe configured Ollama auto targets and atomically publish one snapshot.

    Per-endpoint network failures are availability observations: the failed
    endpoint contributes no reachable candidates, while successful endpoints
    are still published. Setup/configuration failures preserve the previous
    snapshot by returning without publishing.
    """

    owner_key = str(owner or "").strip()
    fetch = request_json or _request_json

    try:
        endpoint_ids = _configured_endpoint_ids(
            owner_key,
        )
        endpoints = [
            endpoint
            for endpoint_id in endpoint_ids
            if (
                endpoint := _load_ollama_endpoint(
                    endpoint_id,
                    owner_key,
                )
            )
            is not None
        ]
    except Exception as exc:
        logger.warning(
            "Adaptive routing refresh setup failed for owner %r: %s",
            owner_key,
            exc,
        )
        return None

    candidates = []

    for endpoint in endpoints:
        try:
            tags_payload = fetch(
                endpoint["models_url"],
                headers=endpoint["headers"],
                timeout=timeout,
            )

            tag_records = ollama.records_from_tags_payload(
                tags_payload,
                endpoint_id=endpoint["endpoint_id"],
                base_url=endpoint["base_url"],
            )
        except Exception as exc:
            logger.debug(
                "Adaptive routing tags probe failed for endpoint %s: %s",
                endpoint["endpoint_id"],
                exc,
            )
            continue

        for tag_record in tag_records:
            model_id = tag_record.model_id

            if model_id in endpoint["hidden_models"]:
                continue

            try:
                show_payload = fetch(
                    endpoint["show_url"],
                    payload={"model": model_id},
                    headers=endpoint["headers"],
                    timeout=timeout,
                )

                candidate = ollama_candidate_from_show_payload(
                    model_id,
                    show_payload,
                    endpoint_id=endpoint["endpoint_id"],
                    endpoint_url=endpoint["chat_url"],
                    node=endpoint["endpoint_id"],
                    scope="local",
                    reachable=True,
                )
            except Exception as exc:
                logger.debug(
                    "Adaptive routing show probe failed for endpoint %s model %s: %s",
                    endpoint["endpoint_id"],
                    model_id,
                    exc,
                )
                continue

            if candidate is not None:
                candidates.append(candidate)

    return publish_adaptive_routing_snapshot(
        owner_key,
        candidates,
        generated_at=generated_at,
    )
