"""Deterministic start and confirmed stop workflows for Palworld."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Optional

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
from src.services.homelab.palworld_restart import (
    RESTART_CONFIRMATION_TTL,
    consume_pending_stop,
    issue_pending_stop,
)


_RUNNING_STATES = {
    "active",
    "online",
    "ready",
    "running",
}

_STOPPED_STATES = {
    "dead",
    "inactive",
    "offline",
    "stopped",
    "unavailable",
}

_TRANSITION_STATES = {
    "activating",
    "deactivating",
    "starting",
    "stopping",
}

_CONFIRM_STOP_PATTERN = re.compile(
    r"^\s*"
    r"confirmar\s+parada\s+palworld\s+"
    r"(PST-[A-Z2-9]{12})"
    r"\s*[.!]?\s*$",
    re.IGNORECASE,
)


class PalworldLifecycleError(
    HomelabClientError
):
    """Base error for Palworld start/stop workflows."""


class PalworldLifecycleBlockedError(
    PalworldLifecycleError
):
    """Raised when an action is unsafe for the current state."""


class PalworldLifecycleVerificationError(
    PalworldLifecycleError
):
    """Raised when post-action verification fails."""


def _normalize(text: str) -> str:
    value = unicodedata.normalize(
        "NFKD",
        str(text or ""),
    )

    folded = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    ).casefold()

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        folded,
    ).strip()


_START_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        (
            r"(?:por favor )?"
            r"(?:inicia|iniciar|arranca|arrancar|enciende) "
            r"(?:el )?"
            r"(?:servidor (?:de )?)?"
            r"palworld"
            r"(?: ahora| ya)?"
        ),
        (
            r"(?:si us plau )?"
            r"(?:inicia|arrenca|engega) "
            r"(?:el )?"
            r"(?:servidor (?:de )?)?"
            r"palworld"
            r"(?: ara| ja)?"
        ),
        (
            r"(?:please )?"
            r"(?:start|launch) "
            r"(?:the )?"
            r"(?:palworld server|"
            r"server (?:for|of) palworld|"
            r"palworld)"
            r"(?: now)?"
        ),
    )
)

_STOP_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        (
            r"(?:por favor )?"
            r"(?:deten|detener|detiene|para|parar|apaga) "
            r"(?:el )?"
            r"(?:servidor (?:de )?)?"
            r"palworld"
            r"(?: ahora| ya)?"
        ),
        (
            r"(?:si us plau )?"
            r"(?:atura|para|apaga) "
            r"(?:el )?"
            r"(?:servidor (?:de )?)?"
            r"palworld"
            r"(?: ara| ja)?"
        ),
        (
            r"(?:please )?"
            r"(?:stop|shut down) "
            r"(?:the )?"
            r"(?:palworld server|"
            r"server (?:for|of) palworld|"
            r"palworld)"
            r"(?: now)?"
        ),
    )
)


def classify_palworld_lifecycle_turn(
    text: str,
    *,
    continuation: bool = False,
) -> Optional[Dict[str, str]]:
    raw = str(text or "")

    confirmation = (
        _CONFIRM_STOP_PATTERN.fullmatch(raw)
    )

    if confirmation:
        return {
            "action": "stop",
            "kind": "confirmation",
            "code": (
                confirmation
                .group(1)
                .upper()
            ),
        }

    if continuation:
        return None

    normalized = _normalize(raw)

    if any(
        pattern.fullmatch(normalized)
        for pattern in _START_PATTERNS
    ):
        return {
            "action": "start",
            "kind": "request",
        }

    if any(
        pattern.fullmatch(normalized)
        for pattern in _STOP_PATTERNS
    ):
        return {
            "action": "stop",
            "kind": "request",
        }

    return None


def _status_candidates(
    payload: Dict[str, Any],
) -> list[Dict[str, Any]]:
    candidates = [payload]

    server = payload.get("server")

    if isinstance(server, dict):
        candidates.insert(0, server)

    return candidates


def _state(
    payload: Dict[str, Any],
) -> str:
    if not isinstance(payload, dict):
        raise PalworldLifecycleVerificationError(
            "El estado de Palworld no es válido"
        )

    for candidate in _status_candidates(
        payload
    ):
        value = str(
            candidate.get("status") or ""
        ).strip()

        if value:
            return value

    return "unknown"


def _players(
    payload: Dict[str, Any],
) -> int:
    if not isinstance(payload, dict):
        raise PalworldLifecycleVerificationError(
            "El estado de Palworld no es válido"
        )

    for candidate in _status_candidates(
        payload
    ):
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
                raise (
                    PalworldLifecycleVerificationError(
                        "El contador de jugadores "
                        "no es válido"
                    )
                )

    return 0


def _require_no_players(
    payload: Dict[str, Any],
) -> None:
    count = _players(payload)

    if count > 0:
        raise PalworldLifecycleBlockedError(
            "No se puede detener Palworld "
            "mientras haya jugadores "
            f"conectados: {count}"
        )


def _require_running(
    payload: Dict[str, Any],
) -> None:
    state = _state(
        payload
    ).casefold()

    if state in _TRANSITION_STATES:
        raise PalworldLifecycleBlockedError(
            "Palworld está realizando otra "
            f"transición: {state}"
        )

    if state not in _RUNNING_STATES:
        raise PalworldLifecycleBlockedError(
            "Palworld no está activo: "
            f"{state}"
        )


def _require_startable(
    payload: Dict[str, Any],
) -> None:
    state = _state(
        payload
    ).casefold()

    if state in _RUNNING_STATES:
        raise PalworldLifecycleBlockedError(
            "Palworld ya está activo"
        )

    if state in _TRANSITION_STATES:
        raise PalworldLifecycleBlockedError(
            "Palworld está realizando otra "
            f"transición: {state}"
        )

    if state not in _STOPPED_STATES:
        raise PalworldLifecycleBlockedError(
            "Palworld no está en un estado "
            f"iniciable: {state}"
        )


def _require_ready(
    payload: Dict[str, Any],
) -> None:
    state = _state(
        payload
    ).casefold()

    if state not in _RUNNING_STATES:
        raise PalworldLifecycleVerificationError(
            "Palworld no recuperó un estado "
            f"operativo: {state}"
        )


def _require_stopped(
    payload: Dict[str, Any],
) -> None:
    state = _state(
        payload
    ).casefold()

    if state not in _STOPPED_STATES:
        raise PalworldLifecycleVerificationError(
            "Palworld no quedó detenido: "
            f"{state}"
        )


def _require_action_success(
    payload: Dict[str, Any],
    *,
    expected_action: str,
) -> None:
    if (
        not isinstance(payload, dict)
        or payload.get("ok") is not True
    ):
        raise PalworldLifecycleError(
            "El operador no confirmó la acción"
        )

    result = payload.get("result")

    if not isinstance(result, dict):
        raise PalworldLifecycleError(
            "La respuesta del host-agent "
            "no es válida"
        )

    if result.get("ok") is False:
        raise PalworldLifecycleError(
            "El host-agent rechazó la acción"
        )

    observed_action = str(
        result.get("action") or ""
    ).strip()

    if (
        observed_action
        and observed_action != expected_action
    ):
        raise PalworldLifecycleError(
            "El host-agent ejecutó una "
            "acción distinta"
        )

    observed_status = str(
        result.get("status") or ""
    ).casefold()

    if observed_status in {
        "failed",
        "failure",
        "error",
    }:
        raise PalworldLifecycleError(
            "La acción terminó con estado "
            f"{observed_status}"
        )

    observed_result = str(
        result.get("result") or ""
    ).casefold()

    if observed_result in {
        "failed",
        "failure",
        "error",
    }:
        raise PalworldLifecycleError(
            "La acción terminó con resultado "
            f"{observed_result}"
        )


def start_palworld_verified(
    *,
    read_client: Optional[
        HomelabClient
    ] = None,
    action_client: Optional[
        HomelabActionClient
    ] = None,
) -> Dict[str, Any]:
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

    _require_startable(before)

    action_result = actor.start_palworld()

    _require_action_success(
        action_result,
        expected_action="palworld.start",
    )

    after = reader.palworld_status()

    _require_ready(after)

    return {
        "before": before,
        "action": action_result,
        "after": after,
    }


def prepare_palworld_stop_confirmation(
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

    _require_running(status)
    _require_no_players(status)

    authorization = issue_pending_stop(
        owner=owner,
        session_id=session_id,
        ttl=ttl,
    )

    return {
        "status": status,
        "authorization": authorization,
    }


def execute_confirmed_palworld_stop(
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
    consume_pending_stop(
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

    _require_running(before)
    _require_no_players(before)

    backup = create_verified_palworld_backup(
        read_client=reader,
        action_client=actor,
    )

    action_result = actor.stop_palworld()

    _require_action_success(
        action_result,
        expected_action="palworld.stop",
    )

    after = reader.palworld_status()

    _require_stopped(after)

    return {
        "before": before,
        "backup": backup,
        "action": action_result,
        "after": after,
    }


def format_start_result(
    result: Dict[str, Any],
    user_text: str,
) -> str:
    after = result.get("after")

    if not isinstance(after, dict):
        raise PalworldLifecycleVerificationError(
            "Falta el estado posterior"
        )

    state = _state(after)
    players = _players(after)
    normalized = _normalize(user_text)

    if normalized.startswith(
        "start"
    ) or normalized.startswith(
        "please start"
    ):
        return (
            "Palworld started successfully. "
            f"Server status: {state}; "
            f"connected players: {players}."
        )

    return (
        "Palworld iniciado correctamente. "
        f"Estado del servidor: {state}; "
        f"jugadores conectados: {players}."
    )


def format_stop_confirmation(
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
        raise PalworldLifecycleVerificationError(
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
        raise PalworldLifecycleVerificationError(
            "Falta el código de confirmación"
        )

    minutes = max(
        ttl // 60,
        1,
    )

    command = (
        "CONFIRMAR PARADA PALWORLD "
        + code
    )

    normalized = _normalize(user_text)

    if normalized.startswith(
        "stop"
    ) or normalized.startswith(
        "please stop"
    ):
        return (
            "Palworld has no connected players. "
            "A verified backup will be created "
            "before stopping. Reply exactly: "
            f"`{command}`. The authorization "
            f"expires in {minutes} minutes and "
            "only works in this session."
        )

    return (
        "Palworld no tiene jugadores conectados. "
        "Antes de detenerlo se creará un backup "
        "verificado. Responde exactamente: "
        f"`{command}`. La autorización caduca "
        f"en {minutes} minutos y solo funciona "
        "en esta sesión."
    )


def format_stop_result(
    result: Dict[str, Any],
    user_text: str,
) -> str:
    backup = result.get("backup")

    if not isinstance(backup, dict):
        raise PalworldLifecycleVerificationError(
            "Falta el backup previo"
        )

    backup_after = backup.get("after")

    if not isinstance(
        backup_after,
        dict,
    ):
        raise PalworldLifecycleVerificationError(
            "Falta el inventario posterior "
            "del backup"
        )

    after = result.get("after")

    if not isinstance(after, dict):
        raise PalworldLifecycleVerificationError(
            "Falta el estado posterior"
        )

    name = str(
        backup_after.get(
            "latest_backup"
        )
        or ""
    ).strip()

    size = str(
        backup_after.get("size")
        or ""
    ).strip()

    if not name:
        raise PalworldLifecycleVerificationError(
            "Falta el nombre del backup previo"
        )

    details = name

    if size:
        details += f", {size}"

    state = _state(after)
    normalized = _normalize(user_text)

    if normalized.startswith(
        "confirm stop"
    ):
        return (
            "Palworld stopped successfully. "
            f"Previous backup: `{details}`. "
            f"Server status: {state}."
        )

    return (
        "Palworld detenido correctamente. "
        f"Backup previo: `{details}`. "
        f"Estado del servidor: {state}."
    )
