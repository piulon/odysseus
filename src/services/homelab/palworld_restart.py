"""Confirmed and verified Palworld restart workflow."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from core.atomic_io import atomic_write_json
from src.constants import DATA_DIR
from src.services.homelab.action_client import (
    HomelabActionClient,
)
from src.services.homelab.client import (
    HomelabClient,
    HomelabClientError,
)
from src.services.homelab.palworld_backup import (
    create_verified_palworld_backup,
)


PENDING_ACTIONS_FILE = (
    Path(DATA_DIR)
    / "pending_homelab_actions.json"
)

PENDING_ACTIONS_LOCK_FILE = (
    Path(DATA_DIR)
    / ".pending_homelab_actions.lock"
)

RESTART_CONFIRMATION_TTL = 300

_CONFIRMATION_ALPHABET = (
    "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
)


class PalworldRestartError(
    HomelabClientError
):
    """Base error for the controlled restart workflow."""


class PalworldRestartBlockedError(
    PalworldRestartError
):
    """Raised when the server cannot safely restart."""


class PalworldRestartConfirmationError(
    PalworldRestartError
):
    """Raised when a confirmation is invalid or expired."""


class PalworldRestartVerificationError(
    PalworldRestartError
):
    """Raised when the restarted server cannot be verified."""


def _normalize_command(text: str) -> str:
    normalized = unicodedata.normalize(
        "NFKD",
        str(text or ""),
    )

    folded = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    ).casefold()

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        folded,
    ).strip()


_RESTART_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        (
            r"(?:por favor )?"
            r"(?:reinicia|reiniciar) "
            r"(?:el )?"
            r"(?:servidor (?:de )?)?"
            r"palworld"
            r"(?: ahora| ya)?"
        ),
        (
            r"(?:si us plau )?"
            r"reinicia "
            r"(?:el )?"
            r"(?:servidor (?:de )?)?"
            r"palworld"
            r"(?: ara| ja)?"
        ),
        (
            r"(?:please )?"
            r"restart "
            r"(?:the )?"
            r"(?:palworld server|"
            r"server (?:for|of) palworld|"
            r"palworld)"
            r"(?: now)?"
        ),
    )
)

_CONFIRMATION_PATTERN = re.compile(
    r"^\s*"
    r"confirmar\s+reinicio\s+palworld\s+"
    r"(PRW-[A-Z2-9]{12})"
    r"\s*[.!]?\s*$",
    re.IGNORECASE,
)


def classify_palworld_restart_turn(
    text: str,
    *,
    continuation: bool = False,
) -> Optional[Dict[str, str]]:
    """Classify an explicit request or exact confirmation."""

    raw = str(text or "")

    confirmation = (
        _CONFIRMATION_PATTERN.fullmatch(raw)
    )

    if confirmation:
        return {
            "kind": "confirmation",
            "code": (
                confirmation
                .group(1)
                .upper()
            ),
        }

    if continuation:
        return None

    normalized = _normalize_command(raw)

    if not normalized:
        return None

    if any(
        pattern.fullmatch(normalized)
        for pattern in _RESTART_PATTERNS
    ):
        return {
            "kind": "request",
        }

    return None


def _empty_store() -> Dict[str, Any]:
    return {
        "version": 1,
        "actions": [],
    }


def _load_store() -> Dict[str, Any]:
    if not PENDING_ACTIONS_FILE.exists():
        return _empty_store()

    try:
        payload = json.loads(
            PENDING_ACTIONS_FILE.read_text(
                encoding="utf-8",
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return _empty_store()

    if not isinstance(payload, dict):
        return _empty_store()

    actions = payload.get("actions")

    if not isinstance(actions, list):
        return _empty_store()

    return {
        "version": 1,
        "actions": [
            action
            for action in actions
            if isinstance(action, dict)
        ],
    }


def _save_store(
    store: Dict[str, Any],
) -> None:
    PENDING_ACTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_write_json(
        str(PENDING_ACTIONS_FILE),
        store,
        indent=2,
    )

    os.chmod(
        PENDING_ACTIONS_FILE,
        0o600,
    )


@contextmanager
def _store_lock() -> Iterator[None]:
    PENDING_ACTIONS_LOCK_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PENDING_ACTIONS_LOCK_FILE.open(
        "a+",
        encoding="utf-8",
    ) as handle:
        os.chmod(
            PENDING_ACTIONS_LOCK_FILE,
            0o600,
        )

        fcntl.flock(
            handle.fileno(),
            fcntl.LOCK_EX,
        )

        try:
            yield
        finally:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_UN,
            )


def _code_digest(code: str) -> str:
    return hashlib.sha256(
        str(code).upper().encode("utf-8")
    ).hexdigest()


def _active_actions(
    actions: list[Dict[str, Any]],
    now: float,
) -> list[Dict[str, Any]]:
    active = []

    for action in actions:
        try:
            expires_at = float(
                action.get("expires_at") or 0
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if expires_at > now:
            active.append(action)

    return active


def issue_pending_restart(
    *,
    owner: str,
    session_id: str,
    ttl: int = RESTART_CONFIRMATION_TTL,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    owner_value = str(owner or "").strip()
    session_value = str(
        session_id or ""
    ).strip()

    if not owner_value:
        raise PalworldRestartConfirmationError(
            "Falta el propietario de la sesión"
        )

    if not session_value:
        raise PalworldRestartConfirmationError(
            "Falta una sesión persistente"
        )

    ttl_value = int(ttl)

    if not 30 <= ttl_value <= 900:
        raise PalworldRestartConfirmationError(
            "TTL de confirmación no permitido"
        )

    issued_at = (
        float(now)
        if now is not None
        else time.time()
    )

    code = (
        "PRW-"
        + "".join(
            secrets.choice(
                _CONFIRMATION_ALPHABET
            )
            for _ in range(12)
        )
    )

    record = {
        "action": "palworld.restart",
        "owner": owner_value,
        "session_id": session_value,
        "code_digest": _code_digest(code),
        "created_at": issued_at,
        "expires_at": (
            issued_at + ttl_value
        ),
    }

    with _store_lock():
        store = _load_store()

        actions = _active_actions(
            store.get("actions", []),
            issued_at,
        )

        actions = [
            action
            for action in actions
            if not (
                action.get("action")
                == "palworld.restart"
                and action.get("owner")
                == owner_value
                and action.get("session_id")
                == session_value
            )
        ]

        actions.append(record)

        _save_store({
            "version": 1,
            "actions": actions,
        })

    return {
        "code": code,
        "created_at": issued_at,
        "expires_at": (
            issued_at + ttl_value
        ),
        "expires_in_seconds": ttl_value,
    }


def consume_pending_restart(
    *,
    owner: str,
    session_id: str,
    code: str,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    owner_value = str(owner or "").strip()
    session_value = str(
        session_id or ""
    ).strip()

    code_value = str(
        code or ""
    ).strip().upper()

    current_time = (
        float(now)
        if now is not None
        else time.time()
    )

    digest = _code_digest(code_value)

    with _store_lock():
        store = _load_store()

        actions = _active_actions(
            store.get("actions", []),
            current_time,
        )

        matched_index = None

        for index, action in enumerate(actions):
            stored_digest = str(
                action.get("code_digest") or ""
            )

            if (
                action.get("action")
                == "palworld.restart"
                and action.get("owner")
                == owner_value
                and action.get("session_id")
                == session_value
                and hmac.compare_digest(
                    stored_digest,
                    digest,
                )
            ):
                matched_index = index
                break

        if matched_index is None:
            _save_store({
                "version": 1,
                "actions": actions,
            })

            raise (
                PalworldRestartConfirmationError(
                    "Confirmación inválida, "
                    "caducada o perteneciente "
                    "a otra sesión"
                )
            )

        consumed = actions.pop(
            matched_index
        )

        _save_store({
            "version": 1,
            "actions": actions,
        })

    return consumed


def _extract_players(
    payload: Dict[str, Any],
) -> int:
    candidates = [payload]

    server = payload.get("server")

    if isinstance(server, dict):
        candidates.append(server)

    for candidate in candidates:
        for key in (
            "players",
            "player_count",
            "current_players",
            "online_players",
        ):
            if key not in candidate:
                continue

            try:
                return int(
                    candidate.get(key) or 0
                )
            except (
                TypeError,
                ValueError,
            ):
                raise PalworldRestartVerificationError(
                    "El contador de jugadores "
                    "no es válido"
                )

    return 0


def _extract_status(
    payload: Dict[str, Any],
) -> str:
    server = payload.get("server")

    if isinstance(server, dict):
        nested = str(
            server.get("status") or ""
        ).strip()

        if nested:
            return nested

    return str(
        payload.get("status") or ""
    ).strip()


def _assert_restartable(
    payload: Dict[str, Any],
) -> None:
    if not isinstance(payload, dict):
        raise PalworldRestartVerificationError(
            "El estado de Palworld no es válido"
        )

    if payload.get("ok") is False:
        raise PalworldRestartBlockedError(
            "Palworld no está disponible"
        )

    players = _extract_players(payload)

    if players > 0:
        raise PalworldRestartBlockedError(
            "No se puede reiniciar Palworld "
            "mientras haya jugadores "
            f"conectados: {players}"
        )

    state = _extract_status(
        payload
    ).casefold()

    if state in {
        "failed",
        "inactive",
        "stopped",
        "stopping",
        "starting",
        "unavailable",
    }:
        raise PalworldRestartBlockedError(
            "Palworld no está en un estado "
            f"reiniciable: {state}"
        )


def prepare_palworld_restart_confirmation(
    *,
    owner: str,
    session_id: str,
    read_client: Optional[
        HomelabClient
    ] = None,
    ttl: int = RESTART_CONFIRMATION_TTL,
) -> Dict[str, Any]:
    reader = (
        read_client
        if read_client is not None
        else HomelabClient(
            timeout=20.0
        )
    )

    status = reader.palworld_status()

    _assert_restartable(status)

    authorization = issue_pending_restart(
        owner=owner,
        session_id=session_id,
        ttl=ttl,
    )

    return {
        "status": status,
        "authorization": authorization,
    }


def _assert_action_completed(
    payload: Dict[str, Any],
) -> None:
    if (
        not isinstance(payload, dict)
        or payload.get("ok") is not True
    ):
        raise PalworldRestartError(
            "El operador no confirmó "
            "el reinicio"
        )

    result = payload.get("result")

    if not isinstance(result, dict):
        raise PalworldRestartError(
            "El resultado del reinicio "
            "no es válido"
        )

    if result.get("ok") is False:
        raise PalworldRestartError(
            "El agente del host rechazó "
            "el reinicio"
        )

    status_value = str(
        result.get("status") or ""
    ).casefold()

    result_value = str(
        result.get("result") or ""
    ).casefold()

    if (
        status_value
        and status_value
        not in {
            "completed",
            "success",
        }
    ):
        raise PalworldRestartError(
            "El reinicio no terminó: "
            f"{status_value}"
        )

    if (
        result_value
        and result_value
        not in {
            "success",
            "ok",
        }
    ):
        raise PalworldRestartError(
            "El reinicio devolvió: "
            f"{result_value}"
        )


def _assert_server_ready(
    payload: Dict[str, Any],
) -> None:
    if not isinstance(payload, dict):
        raise PalworldRestartVerificationError(
            "La verificación posterior "
            "no es válida"
        )

    if payload.get("ok") is False:
        raise PalworldRestartVerificationError(
            "Palworld no responde después "
            "del reinicio"
        )

    state = _extract_status(
        payload
    ).casefold()

    if state in {
        "failed",
        "inactive",
        "stopped",
        "stopping",
        "starting",
        "unavailable",
    }:
        raise PalworldRestartVerificationError(
            "Palworld no recuperó un estado "
            f"operativo: {state}"
        )


def execute_confirmed_palworld_restart(
    *,
    owner: str,
    session_id: str,
    code: str,
    read_client: Optional[
        HomelabClient
    ] = None,
    action_client: Optional[
        HomelabActionClient
    ] = None,
) -> Dict[str, Any]:
    """Consume one confirmation, back up, restart and verify."""

    consume_pending_restart(
        owner=owner,
        session_id=session_id,
        code=code,
    )

    reader = (
        read_client
        if read_client is not None
        else HomelabClient(
            timeout=20.0
        )
    )

    actor = (
        action_client
        if action_client is not None
        else HomelabActionClient(
            timeout=240.0
        )
    )

    before = reader.palworld_status()

    _assert_restartable(before)

    backup = create_verified_palworld_backup(
        read_client=reader,
        action_client=actor,
    )

    restart = actor.restart_palworld()

    _assert_action_completed(restart)

    after = reader.palworld_status()

    _assert_server_ready(after)

    return {
        "before": before,
        "backup": backup,
        "restart": restart,
        "after": after,
    }


def format_restart_confirmation(
    result: Dict[str, Any],
    user_text: str,
) -> str:
    authorization = result.get(
        "authorization"
    )

    if not isinstance(
        authorization,
        dict,
    ):
        raise PalworldRestartConfirmationError(
            "Falta la autorización pendiente"
        )

    code = str(
        authorization.get("code") or ""
    ).strip()

    ttl = int(
        authorization.get(
            "expires_in_seconds"
        )
        or RESTART_CONFIRMATION_TTL
    )

    if not code:
        raise PalworldRestartConfirmationError(
            "Falta el código de confirmación"
        )

    minutes = max(
        ttl // 60,
        1,
    )

    normalized = _normalize_command(
        user_text
    )

    command = (
        "CONFIRMAR REINICIO PALWORLD "
        + code
    )

    if normalized.startswith(
        "please restart"
    ) or normalized.startswith(
        "restart"
    ):
        return (
            "Palworld has no connected players. "
            "A verified backup will be created "
            "before restarting. Reply exactly: "
            f"`{command}`. The authorization "
            f"expires in {minutes} minutes and "
            "only works in this session."
        )

    if (
        "si us plau" in normalized
        or normalized.endswith(" ara")
        or normalized.endswith(" ja")
    ):
        return (
            "Palworld no té jugadors connectats. "
            "Abans del reinici es crearà un backup "
            "verificat. Respon exactament: "
            f"`{command}`. L'autorització caduca "
            f"en {minutes} minuts i només funciona "
            "en aquesta sessió."
        )

    return (
        "Palworld no tiene jugadores conectados. "
        "Antes del reinicio se creará un backup "
        "verificado. Responde exactamente: "
        f"`{command}`. La autorización caduca "
        f"en {minutes} minutos y solo funciona "
        "en esta sesión."
    )


def format_verified_palworld_restart(
    result: Dict[str, Any],
    user_text: str,
) -> str:
    backup = result.get("backup")

    if not isinstance(backup, dict):
        raise PalworldRestartVerificationError(
            "Faltan los datos del backup previo"
        )

    backup_after = backup.get("after")

    if not isinstance(
        backup_after,
        dict,
    ):
        raise PalworldRestartVerificationError(
            "Falta el inventario posterior "
            "del backup"
        )

    after = result.get("after")

    if not isinstance(after, dict):
        raise PalworldRestartVerificationError(
            "Falta el estado posterior "
            "del servidor"
        )

    backup_name = str(
        backup_after.get(
            "latest_backup"
        )
        or ""
    ).strip()

    backup_size = str(
        backup_after.get("size")
        or ""
    ).strip()

    server_state = (
        _extract_status(after)
        or "operativo"
    )

    players = _extract_players(after)

    if not backup_name:
        raise PalworldRestartVerificationError(
            "Falta el nombre del backup previo"
        )

    normalized = _normalize_command(
        user_text
    )

    details = backup_name

    if backup_size:
        details += f", {backup_size}"

    if normalized.startswith(
        "confirm restart"
    ):
        return (
            "Palworld restarted successfully. "
            f"Previous backup: `{details}`. "
            f"Server status: {server_state}; "
            f"connected players: {players}."
        )

    return (
        "Palworld reiniciado correctamente. "
        f"Backup previo: `{details}`. "
        f"Estado del servidor: {server_state}; "
        f"jugadores conectados: {players}."
    )
