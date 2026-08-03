import json
from typing import Any

from src.services.homelab.client import HomelabClient


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

CATEGORY_ORDER = (
    "ai",
    "infrastructure",
    "monitoring",
    "games",
)

CATEGORY_LABELS = {
    "ai": "IA",
    "infrastructure": "Infraestructura",
    "monitoring": "Monitorización",
    "games": "Juegos",
}


def _as_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_number(
    value: Any,
    decimals: int = 2,
) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/d"

    if number.is_integer():
        return str(int(number))

    return (
        f"{number:.{decimals}f}"
        .rstrip("0")
        .rstrip(".")
    )


def _service_state(
    service: dict[str, Any],
) -> str:
    health = str(
        service.get("health")
        or ""
    ).lower()

    if health == "unhealthy":
        return "UNHEALTHY"

    if service.get("running"):
        return "OK"

    if service.get("present"):
        return "PARADO"

    if service.get("expected_running"):
        return "AUSENTE"

    return "AUSENTE OPCIONAL"


def format_homelab_status(
    payload: dict[str, Any],
) -> str:
    """Build the terminal status response without an LLM pass."""
    if not isinstance(payload, dict):
        return (
            "## Estado del homelab\n\n"
            "No se recibió una respuesta válida del operador."
        )

    services = [
        service
        for service in payload.get("services", [])
        if isinstance(service, dict)
    ]

    summary = (
        payload.get("summary")
        if isinstance(payload.get("summary"), dict)
        else {}
    )

    inventory = (
        summary.get("inventory")
        if isinstance(summary.get("inventory"), dict)
        else {}
    )

    docker = (
        summary.get("docker")
        if isinstance(summary.get("docker"), dict)
        else {}
    )

    category_summary = (
        summary.get("categories")
        if isinstance(summary.get("categories"), dict)
        else {}
    )

    inventory_total = _as_int(
        inventory.get("total"),
        len(services),
    )

    inventory_running = _as_int(
        inventory.get("running"),
        sum(
            bool(service.get("running"))
            for service in services
        ),
    )

    inventory_stopped = _as_int(
        inventory.get("stopped"),
    )

    inventory_missing = _as_int(
        inventory.get("missing"),
    )

    inventory_unhealthy = _as_int(
        inventory.get("unhealthy"),
    )

    docker_running = _as_int(
        docker.get("running"),
    )

    docker_total = _as_int(
        docker.get("total"),
        docker_running,
    )

    expected_issues = sorted({
        str(name)
        for name in summary.get(
            "expected_issues",
            [],
        )
        if str(name).strip()
    })

    optional_missing = sorted({
        str(name)
        for name in summary.get(
            "optional_missing",
            [],
        )
        if str(name).strip()
    })

    lines = [
        "## Estado del homelab",
        "",
        "**Resumen**",
        (
            f"- Inventario: **{inventory_running}/"
            f"{inventory_total}** servicios activos"
        ),
        (
            f"- Docker: **{docker_running}/"
            f"{docker_total}** contenedores activos"
        ),
        (
            "- Estados del inventario: "
            f"{inventory_stopped} parados · "
            f"{inventory_missing} ausentes · "
            f"{inventory_unhealthy} unhealthy"
        ),
        (
            "- Incidencias esperadas: "
            + (
                ", ".join(expected_issues)
                if expected_issues
                else "ninguna"
            )
        ),
        (
            "- Ausentes opcionales: "
            + (
                ", ".join(optional_missing)
                if optional_missing
                else "ninguno"
            )
        ),
    ]

    grouped: dict[str, list[dict[str, Any]]] = {}

    for service in services:
        category = str(
            service.get("category")
            or "uncategorized"
        )

        grouped.setdefault(
            category,
            [],
        ).append(service)

    ordered_categories = [
        category
        for category in CATEGORY_ORDER
        if category in grouped
    ]

    ordered_categories.extend(
        sorted(
            set(grouped)
            - set(ordered_categories)
        )
    )

    lines.extend([
        "",
        "## Servicios inventariados",
    ])

    for category in ordered_categories:
        category_services = sorted(
            grouped[category],
            key=lambda item: str(
                item.get("service")
                or ""
            ),
        )

        counters = (
            category_summary.get(category)
            if isinstance(
                category_summary.get(category),
                dict,
            )
            else {}
        )

        total = _as_int(
            counters.get("total"),
            len(category_services),
        )

        running = _as_int(
            counters.get("running"),
            sum(
                bool(service.get("running"))
                for service in category_services
            ),
        )

        label = CATEGORY_LABELS.get(
            category,
            category.replace("_", " ").title(),
        )

        lines.extend([
            "",
            f"### {label} — {running}/{total} activos",
        ])

        for service in category_services:
            name = str(
                service.get("service")
                or "servicio-sin-nombre"
            )

            container = str(
                service.get("container")
                or ""
            ).strip()

            marker = _service_state(service)

            suffix = ""

            if container and container != name:
                suffix = f" — contenedor: `{container}`"

            lines.append(
                f"- [{marker}] **{name}**{suffix}"
            )

    outside_inventory = sorted({
        str(name)
        for name in docker.get(
            "outside_inventory_running",
            [],
        )
        if str(name).strip()
    })

    lines.extend([
        "",
        "## Contenedores activos fuera del inventario",
    ])

    if outside_inventory:
        lines.extend(
            f"- **{name}**"
            for name in outside_inventory
        )
    else:
        lines.append("- Ninguno")

    gpu = (
        payload.get("gpu")
        if isinstance(payload.get("gpu"), dict)
        else {}
    )

    lines.extend([
        "",
        "## GPU",
    ])

    devices = [
        device
        for device in gpu.get("devices", [])
        if isinstance(device, dict)
    ]

    if gpu.get("available") and devices:
        devices.sort(
            key=lambda device: _as_int(
                device.get("index"),
            )
        )

        for device in devices:
            model = str(
                device.get("model")
                or device.get("device")
                or "GPU"
            )

            temperature = _format_number(
                device.get("temperature_c"),
            )

            utilization = _format_number(
                device.get("utilization_percent"),
            )

            memory_used = _format_number(
                device.get("memory_used_mib"),
            )

            memory_total = _format_number(
                device.get("memory_total_mib"),
            )

            power = _format_number(
                device.get("power_w"),
            )

            lines.append(
                f"- **{model}** · "
                f"{temperature} °C · "
                f"uso {utilization} % · "
                f"VRAM {memory_used}/{memory_total} MiB · "
                f"{power} W"
            )
    else:
        lines.append("- No disponible")

    return "\n".join(lines)


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
                direct_response = format_homelab_status(
                    result
                )

                return {
                    "output": direct_response,
                    "direct_response": direct_response,
                    "terminal_response": True,
                    "exit_code": 0,
                }

            if action == "doctor":
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
