"""Deterministic and verified Palworld backup action."""

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


class PalworldBackupVerificationError(
    HomelabClientError
):
    """Raised when a completed backup cannot be verified."""


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


_EXPLICIT_BACKUP_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        (
            r"(?:por favor )?"
            r"(?:crea(?:me)?|crear|haz(?:me)?|hacer|"
            r"genera(?:me)?|generar|realiza|realizar|"
            r"lanza|lanzar) "
            r"(?:(?:un|una) )?"
            r"(?:copia(?: de seguridad)?|backup|respaldo) "
            r"(?:de|del) palworld"
            r"(?: ahora| ya)?"
        ),
        (
            r"(?:si us plau )?"
            r"(?:crea|fes(?: me)?|genera|realitza|llanca) "
            r"(?:una )?"
            r"(?:copia de seguretat|copia|backup) "
            r"(?:de|del) palworld"
            r"(?: ara| ja)?"
        ),
        (
            r"(?:please )?"
            r"(?:create|make|run|take) "
            r"(?:a )?palworld backup"
            r"(?: now)?"
        ),
        (
            r"(?:please )?"
            r"(?:create|make|run|take) "
            r"(?:a )?backup "
            r"(?:of|for) palworld"
            r"(?: now)?"
        ),
    )
)


def classify_explicit_palworld_backup_request(
    text: str,
    *,
    continuation: bool = False,
) -> bool:
    """Accept only complete, explicit backup commands."""

    if continuation:
        return False

    normalized = _normalize_command(text)

    if not normalized:
        return False

    return any(
        pattern.fullmatch(normalized)
        for pattern in _EXPLICIT_BACKUP_PATTERNS
    )


def _count(payload: Dict[str, Any]) -> int:
    try:
        return int(payload.get("count") or 0)
    except (
        TypeError,
        ValueError,
    ):
        return 0


def _action_succeeded(
    payload: Dict[str, Any],
) -> bool:
    result = payload.get("result")

    return bool(
        payload.get("ok") is True
        and isinstance(result, dict)
        and result.get("ok") is True
        and result.get("status") == "completed"
        and result.get("result") == "success"
    )


def create_verified_palworld_backup(
    *,
    read_client: Optional[HomelabClient] = None,
    action_client: Optional[
        HomelabActionClient
    ] = None,
) -> Dict[str, Any]:
    """Create one backup and verify that the inventory changed."""

    reader = (
        read_client
        if read_client is not None
        else HomelabClient(timeout=20.0)
    )

    actor = (
        action_client
        if action_client is not None
        else HomelabActionClient(timeout=180.0)
    )

    before = reader.palworld_backups()

    if not isinstance(before, dict):
        raise PalworldBackupVerificationError(
            "El estado previo de los backups "
            "no es válido"
        )

    action = actor.create_palworld_backup()

    if (
        not isinstance(action, dict)
        or not _action_succeeded(action)
    ):
        raise HomelabClientError(
            "El operador no confirmó que el backup "
            "terminara correctamente"
        )

    try:
        after = reader.palworld_backups()
    except Exception as exc:
        raise PalworldBackupVerificationError(
            "El operador ejecutó la acción, pero "
            "no se pudo verificar el backup creado"
        ) from exc

    if not isinstance(after, dict):
        raise PalworldBackupVerificationError(
            "El estado posterior de los backups "
            "no es válido"
        )

    before_name = str(
        before.get("latest_backup") or ""
    )

    after_name = str(
        after.get("latest_backup") or ""
    )

    inventory_changed = bool(
        after_name
        and (
            after_name != before_name
            or _count(after) > _count(before)
        )
    )

    if not inventory_changed:
        raise PalworldBackupVerificationError(
            "La acción terminó, pero el inventario "
            "no muestra un backup nuevo"
        )

    if after.get("ok") is not True:
        raise PalworldBackupVerificationError(
            "El backup aparece en el inventario, "
            "pero su estado no es válido"
        )

    integrity = str(
        after.get("integrity") or ""
    ).casefold()

    if integrity not in {
        "correcta",
        "correct",
        "ok",
    }:
        raise PalworldBackupVerificationError(
            "El backup nuevo no tiene una "
            "integridad confirmada"
        )

    return {
        "before": before,
        "action": action,
        "after": after,
    }


def format_verified_palworld_backup(
    result: Dict[str, Any],
    user_text: str,
) -> str:
    after = result.get("after")

    if not isinstance(after, dict):
        raise PalworldBackupVerificationError(
            "Faltan los metadatos del backup"
        )

    name = str(
        after.get("latest_backup") or ""
    ).strip()

    size = str(
        after.get("size") or ""
    ).strip()

    integrity = str(
        after.get("integrity") or ""
    ).strip()

    count = _count(after)

    if not name:
        raise PalworldBackupVerificationError(
            "Falta el nombre del backup creado"
        )

    normalized = _normalize_command(user_text)

    if any(
        marker in normalized
        for marker in (
            "si us plau",
            "copia de seguretat",
            "fes ",
            " ara",
        )
    ):
        details = []

        if size:
            details.append(size)

        if integrity:
            details.append(
                f"integritat {integrity}"
            )

        suffix = (
            f" ({', '.join(details)})"
            if details
            else ""
        )

        return (
            "Backup de Palworld creat "
            f"correctament: `{name}`{suffix}. "
            f"Total: {count} còpies."
        )

    if any(
        marker in normalized.split()
        for marker in (
            "please",
            "create",
            "make",
            "run",
            "take",
        )
    ):
        details = []

        if size:
            details.append(size)

        if integrity:
            details.append(
                f"integrity {integrity}"
            )

        suffix = (
            f" ({', '.join(details)})"
            if details
            else ""
        )

        return (
            "Palworld backup created "
            f"successfully: `{name}`{suffix}. "
            f"Total: {count} backups."
        )

    details = []

    if size:
        details.append(size)

    if integrity:
        details.append(
            f"integridad {integrity}"
        )

    suffix = (
        f" ({', '.join(details)})"
        if details
        else ""
    )

    return (
        "Backup de Palworld creado "
        f"correctamente: `{name}`{suffix}. "
        f"Total: {count} copias."
    )
