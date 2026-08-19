import pytest

from core.auth import ADMIN_PRIVILEGES
from src.model_authorization import authorize_model


@pytest.mark.parametrize(
    ("privileges", "model", "allowed", "reason"),
    [
        ({"block_all_models": True}, "model-a", False, "block_all_models"),
        (
            {"allowed_models_restricted": True, "allowed_models": []},
            "model-a",
            False,
            "model_not_allowed",
        ),
        (
            {"allowed_models_restricted": True, "allowed_models": ["model-a"]},
            "model-a",
            True,
            "allowed",
        ),
        (
            {"allowed_models_restricted": True, "allowed_models": ["model-a"]},
            "model-b",
            False,
            "model_not_allowed",
        ),
        ({"allowed_models": []}, "model-a", True, "allowed"),
        ({"allowed_models": ["model-a"]}, "model-b", False, "model_not_allowed"),
    ],
)
def test_authorize_model_matches_existing_policy(privileges, model, allowed, reason):
    result = authorize_model(model, privileges)
    assert result.allowed is allowed
    assert result.reason == reason


def test_authorize_model_uses_exact_match_only():
    privileges = {
        "allowed_models_restricted": True,
        "allowed_models": ["provider/model-a"],
    }

    assert authorize_model("provider/model-a", privileges).allowed is True
    assert authorize_model("model-a", privileges).allowed is False
    assert authorize_model("provider/model-a:latest", privileges).allowed is False


def test_restricted_empty_allowlist_denies_an_empty_model_too():
    result = authorize_model(
        "",
        {"allowed_models_restricted": True, "allowed_models": []},
    )
    assert result.allowed is False
    assert result.reason == "model_not_allowed"


def test_admin_privileges_remain_unrestricted():
    assert authorize_model("any-model", ADMIN_PRIVILEGES).allowed is True


def test_authorize_model_ignores_quota_fields():
    result = authorize_model(
        "model-a",
        {
            "allowed_models": [],
            "allowed_models_restricted": False,
            "block_all_models": False,
            "max_messages_per_day": 1,
        },
    )
    assert result.allowed is True
