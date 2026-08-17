"""Pure model authorization policy shared by request-level gates."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ModelAuthorization:
    allowed: bool
    reason: str


def authorize_model(
    model: str,
    privileges: Optional[Mapping[str, Any]],
) -> ModelAuthorization:
    """Apply exact-match model privileges without quota or external I/O."""
    privileges = privileges or {}
    if privileges.get("block_all_models"):
        return ModelAuthorization(False, "block_all_models")

    allowed_raw = privileges.get("allowed_models")
    allowed = allowed_raw if isinstance(allowed_raw, list) else []
    restricted = bool(privileges.get("allowed_models_restricted")) or bool(allowed)
    if restricted and model not in allowed:
        return ModelAuthorization(False, "model_not_allowed")
    return ModelAuthorization(True, "allowed")
