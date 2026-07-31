import json

from src.services.homelab.client import (
    HomelabClient,
)


READ_ONLY_ACTIONS = {
    "status",
    "doctor",
    "service",
}

BLOCKED_ACTIONS = {
    "start",
    "restart",
    "stop",
}


class HomelabTool:
    async def execute(
        self,
        content: str,
        ctx: dict,
    ) -> dict:
        del ctx

        raw = (content or "").strip()

        try:
            args = (
                json.loads(raw)
                if raw
                else {}
            )
        except (
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            args = {}

        if not isinstance(args, dict):
            args = {}

        action = str(
            args.get("action", "status")
        ).lower()

        service = str(
            args.get("service", "")
        ).strip()

        if action in BLOCKED_ACTIONS:
            return {
                "error": (
                    "homelab: mutating actions are "
                    "disabled for the Odysseus "
                    "read-only tool"
                ),
                "exit_code": 1,
            }

        if action not in READ_ONLY_ACTIONS:
            return {
                "error": (
                    "homelab: unknown action "
                    f"'{action}'"
                ),
                "exit_code": 1,
            }

        try:
            client = HomelabClient()

            if action == "status":
                result = client.status()

            elif action == "doctor":
                result = client.doctor()

            else:
                if not service:
                    return {
                        "error": (
                            "homelab: service "
                            "is required"
                        ),
                        "exit_code": 1,
                    }

                result = client.service(service)

            return {
                "output": json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                ),
                "exit_code": 0,
            }

        except Exception as exc:
            return {
                "error": f"homelab: {exc}",
                "exit_code": 1,
            }
