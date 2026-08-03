import json
from typing import Any

from src.services.homelab.client import HomelabClient


READ_ONLY_ACTIONS = {
    "doctor",
    "gpu",
    "palworld_backups",
    "palworld_status",
    "service",
    "services",
    "status",
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



def format_homelab_service(
    payload: dict[str, Any],
) -> str:
    service = (
        payload.get("service")
        if isinstance(payload, dict)
        else None
    )

    if not isinstance(service, dict):
        return (
            "## Estado del servicio\n\n"
            "No se recibió una respuesta válida del operador."
        )

    name = str(
        service.get("service")
        or "servicio"
    )

    title = (
        name.replace("-", " ")
        .replace("_", " ")
        .title()
    )

    marker = _service_state(service)

    status = str(
        service.get("status")
        or "desconocido"
    )

    health = str(
        service.get("health")
        or "n/d"
    )

    container = str(
        service.get("container")
        or "n/d"
    )

    category = str(
        service.get("category")
        or "n/d"
    )

    image = str(
        service.get("image")
        or "n/d"
    )

    networks = [
        str(network)
        for network in service.get("networks", [])
        if str(network).strip()
    ]

    return "\n".join([
        f"## Estado de {title}",
        "",
        f"- Estado: **{marker}**",
        f"- Runtime: `{status}`",
        f"- Salud: `{health}`",
        f"- Contenedor: `{container}`",
        f"- Categoría: `{category}`",
        f"- Imagen: `{image}`",
        (
            "- Redes: "
            + (
                ", ".join(
                    f"`{network}`"
                    for network in networks
                )
                if networks
                else "ninguna"
            )
        ),
    ])


def format_homelab_gpu(
    payload: dict[str, Any],
) -> str:
    if not isinstance(payload, dict):
        gpu = {}
    elif isinstance(payload.get("gpu"), dict):
        gpu = payload["gpu"]
    else:
        gpu = payload

    devices = [
        device
        for device in gpu.get("devices", [])
        if isinstance(device, dict)
    ]

    lines = [
        "## Estado de la GPU",
        "",
        (
            "- Fuente: "
            f"`{gpu.get('source') or 'n/d'}`"
        ),
    ]

    if not gpu.get("available") or not devices:
        lines.append("- GPU: **no disponible**")
        return "\n".join(lines)

    devices.sort(
        key=lambda device: _as_int(
            device.get("index")
        )
    )

    for device in devices:
        model = str(
            device.get("model")
            or device.get("device")
            or "GPU"
        )

        used = _as_int(
            device.get("memory_used_mib")
        )

        free = _as_int(
            device.get("memory_free_mib")
        )

        reserved = _as_int(
            device.get("memory_reserved_mib")
        )

        total = _as_int(
            device.get("memory_total_mib")
        )

        percentage = (
            (used / total) * 100
            if total > 0
            else 0.0
        )

        lines.extend([
            "",
            f"### {model}",
            (
                "- Temperatura: "
                f"**{_format_number(device.get('temperature_c'))} °C**"
            ),
            (
                "- Uso de GPU: "
                f"**{_format_number(device.get('utilization_percent'))} %**"
            ),
            (
                "- VRAM usada: "
                f"**{used}/{total} MiB "
                f"({_format_number(percentage, 1)} %)**"
            ),
            f"- VRAM libre: **{free} MiB**",
            f"- VRAM reservada: **{reserved} MiB**",
            (
                "- Potencia: "
                f"**{_format_number(device.get('power_w'))} W**"
            ),
            (
                "- Driver: "
                f"`{device.get('driver_version') or 'n/d'}`"
            ),
        ])

    return "\n".join(lines)


def format_palworld_status(
    payload: dict[str, Any],
) -> str:
    if not isinstance(payload, dict):
        return (
            "## Estado de Palworld\n\n"
            "No se recibió una respuesta válida del operador."
        )

    raw_status = str(
        payload.get("status")
        or "desconocido"
    ).lower()

    status_label = {
        "online": "EN LÍNEA",
        "offline": "FUERA DE LÍNEA",
        "starting": "INICIANDO",
        "stopping": "DETENIÉNDOSE",
    }.get(
        raw_status,
        raw_status.upper(),
    )

    players = (
        payload.get("players_display")
        or (
            f"{payload.get('players', 'n/d')} / "
            f"{payload.get('max_players', 'n/d')}"
        )
    )

    lines = [
        "## Estado de Palworld",
        "",
        f"- Estado: **{status_label}**",
        (
            "- Servidor: "
            f"**{payload.get('server') or 'n/d'}**"
        ),
        (
            "- Versión: "
            f"`{payload.get('version') or 'n/d'}`"
        ),
        f"- Jugadores: **{players}**",
        f"- FPS: **{payload.get('fps', 'n/d')}**",
        (
            "- Tiempo activo: "
            f"**{payload.get('uptime') or 'n/d'}**"
        ),
        (
            "- Días del mundo: "
            f"**{payload.get('world_days', 'n/d')}**"
        ),
    ]

    if payload.get("observed_at"):
        lines.append(
            "- Observado: "
            f"`{payload['observed_at']}`"
        )

    return "\n".join(lines)


def format_palworld_backups(
    payload: dict[str, Any],
) -> str:
    if not isinstance(payload, dict):
        return (
            "## Copias de Palworld\n\n"
            "No se recibió una respuesta válida del operador."
        )

    status = str(
        payload.get("status")
        or "desconocido"
    ).upper()

    lines = [
        "## Copias de Palworld",
        "",
        f"- Estado: **{status}**",
        (
            "- Copias disponibles: "
            f"**{payload.get('count', 'n/d')}**"
        ),
        (
            "- Última copia: "
            f"`{payload.get('latest_backup') or 'n/d'}`"
        ),
        (
            "- Antigüedad: "
            f"**{payload.get('age') or 'n/d'}**"
        ),
        (
            "- Tamaño: "
            f"**{payload.get('size') or 'n/d'}**"
        ),
        (
            "- Integridad: "
            f"**{payload.get('integrity') or 'n/d'}**"
        ),
        (
            "- Retención: "
            f"**{payload.get('retention') or 'n/d'}**"
        ),
        (
            "- Programación: "
            f"`{payload.get('schedule') or 'n/d'}`"
        ),
    ]

    if payload.get("observed_at"):
        lines.append(
            "- Observado: "
            f"`{payload['observed_at']}`"
        )

    return "\n".join(lines)


DIRECT_SERVICE_ALIASES = {
    "grafana": "grafana",
    "prometheus": "prometheus",
    "ollama": "ollama",
    "caddy": "caddy",
    "chromadb": "chromadb",
    "chroma db": "chromadb",
    "searxng": "searxng",
    "searx ng": "searxng",
    "portainer": "portainer",
    "homepage": "homepage",
    "home page": "homepage",
    "open webui": "open-webui",
    "open web ui": "open-webui",
    "openwebui": "open-webui",
    "comfyui": "comfyui",
    "comfy ui": "comfyui",
}


def _normalize_direct_homelab_text(
    text: str,
) -> str:
    import re
    import unicodedata

    normalized = unicodedata.normalize(
        "NFKD",
        str(text or ""),
    )

    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    ).lower()

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        normalized,
    ).strip()


def _direct_service_mentions(
    normalized: str,
) -> list[str]:
    """Return canonical service names in textual order."""
    padded = f" {normalized} "
    matches: list[tuple[int, int, str]] = []

    for alias, service in DIRECT_SERVICE_ALIASES.items():
        position = padded.find(
            f" {alias} "
        )

        if position >= 0:
            matches.append(
                (
                    position,
                    -len(alias),
                    service,
                )
            )

    services: list[str] = []

    for _, _, service in sorted(matches):
        if service not in services:
            services.append(service)

    return services



def classify_direct_homelab_request(
    text: str,
    domains,
    continuation: bool = False,
) -> dict[str, Any] | None:
    """Return a canonical deterministic command or None.

    Only unequivocal read-only status requests are accepted. Causal,
    diagnostic, multi-target and mutating requests stay on the agent path.
    """
    if continuation:
        return None

    domain_set = set(domains or ())

    homelab_domain_allowed = not (
        domain_set - {"homelab"}
    )

    default_service_domains = {
        "homelab",
    }

    service_domain_overrides = {
        "ollama": {
            "homelab",
            "cookbook",
        },
        "open-webui": {
            "homelab",
            "ui",
        },
    }

    normalized = _normalize_direct_homelab_text(
        text
    )

    if not normalized:
        return None

    if len(normalized.split()) > 16:
        return None

    padded = f" {normalized} "

    causal_phrases = (
        " por que ",
        " porque ",
        " why ",
    )

    multi_target_connectors = (
        " y ",
        " and ",
        " i ",
    )

    blocked_stems = (
        "diagnost",
        "analiz",
        "investig",
        "problema",
        "error",
        "falla",
        "solucion",
        "arregl",
        "reinici",
        "restart",
        "deten",
        " stop ",
        "arranc",
        " start ",
        "apaga",
        "enciende",
        "crear",
        "crea ",
        "borr",
        "elimin",
    )

    if any(
        phrase in padded
        for phrase in causal_phrases
    ):
        return None

    if any(
        stem in padded
        for stem in blocked_stems
    ):
        return None

    status_terms = (
        " estado ",
        " estat ",
        " status ",
        " salud ",
        " health ",
        " como esta ",
        " como va ",
        " com esta ",
        " how is ",
        " funciona ",
        " funcionando ",
        " activo ",
        " activa ",
        " online ",
        " caido ",
        " caida ",
    )

    service_mentions = (
        _direct_service_mentions(normalized)
    )

    if (
        homelab_domain_allowed
        and len(service_mentions) >= 2
        and any(
            term in padded
            for term in status_terms
        )
    ):
        return {
            "action": "services",
            "services": service_mentions,
        }

    if any(
        connector in padded
        for connector in multi_target_connectors
    ):
        return None

    backup_terms = (
        " backup ",
        " backups ",
        " copia ",
        " copias ",
        " respaldo ",
        " respaldos ",
    )

    if (
        homelab_domain_allowed
        and " palworld " in padded
        and any(
            term in padded
            for term in backup_terms
        )
    ):
        return {
            "action": "palworld_backups",
        }

    palworld_terms = status_terms + (
        " jugadores ",
        " fps ",
        " uptime ",
        " version ",
    )

    if (
        homelab_domain_allowed
        and " palworld " in padded
        and any(
            term in padded
            for term in palworld_terms
        )
    ):
        return {
            "action": "palworld_status",
        }

    gpu_subject = (
        " gpu " in padded
        or " vram " in padded
    )

    gpu_terms = status_terms + (
        " uso ",
        " us ",
        " utilizacion ",
        " temperatura ",
        " consumo ",
        " memoria ",
        " libre ",
        " disponible ",
        " lliure ",
        " lliures ",
        " cuanta ",
        " cuanto ",
        " quant ",
        " quanta ",
        " quants ",
        " quantes ",
    )

    if (
        homelab_domain_allowed
        and gpu_subject
        and any(
            term in padded
            for term in gpu_terms
        )
    ):
        return {
            "action": "gpu",
        }

    for alias, service in (
        DIRECT_SERVICE_ALIASES.items()
    ):
        if (
            not (
                domain_set
                - service_domain_overrides.get(
                    service,
                    default_service_domains,
                )
            )
            and f" {alias} " in padded
            and any(
                term in padded
                for term in status_terms
            )
        ):
            return {
                "action": "service",
                "service": service,
            }

    homelab_subject = (
        " homelab " in padded
        or " home lab " in padded
    )

    general_detail_terms = (
        " servicio ",
        " servicios ",
        " service ",
        " services ",
        " contenedor ",
        " contenedores ",
        " container ",
        " containers ",
    )

    if (
        homelab_domain_allowed
        and homelab_subject
        and any(
            term in padded
            for term in status_terms
        )
        and not any(
            term in padded
            for term in general_detail_terms
        )
    ):
        return {
            "action": "status",
        }

    return None


def should_include_homelab_tool(
    text: str,
    domains=None,
    continuation: bool = False,
) -> bool:
    """Identify read-only homelab requests that need the agent."""
    del domains

    if continuation:
        return False

    normalized = _normalize_direct_homelab_text(
        text
    )

    if not normalized:
        return False

    if len(normalized.split()) > 40:
        return False

    padded = f" {normalized} "

    mutating_stems = (
        "reinici",
        "restart",
        "deten",
        " stop ",
        "arranc",
        " start ",
        "apaga",
        "enciende",
        "crear",
        "crea ",
        "borr",
        "elimin",
        "actualiz",
        " update ",
        " upgrade ",
        "instal",
        "configur",
        "cambia",
        " change ",
    )

    if any(
        stem in padded
        for stem in mutating_stems
    ):
        return False

    service_mentions = (
        _direct_service_mentions(normalized)
    )

    subject_present = (
        bool(service_mentions)
        or " homelab " in padded
        or " home lab " in padded
        or " gpu " in padded
        or " vram " in padded
        or " palworld " in padded
    )

    if not subject_present:
        return False

    read_only_terms = (
        " estado ",
        " estat ",
        " status ",
        " salud ",
        " health ",
        " funciona ",
        " funcionando ",
        " activo ",
        " activa ",
        " online ",
        " caido ",
        " caida ",
        " por que ",
        " porque ",
        " why ",
        " diagnost",
        " analiz",
        " investig",
        " problema",
        " error",
        " falla",
        " log ",
        " logs ",
        " registro ",
        " registros ",
        " revisa ",
        " revisar ",
    )

    return any(
        term in padded
        for term in read_only_terms
    )



def canonicalize_homelab_service(
    value: Any,
) -> str | None:
    """Resolve case-insensitive service names and known aliases."""
    normalized = _normalize_direct_homelab_text(
        str(value or "")
    )

    if not normalized:
        return None

    service = DIRECT_SERVICE_ALIASES.get(
        normalized
    )

    if service:
        return service

    for canonical in set(
        DIRECT_SERVICE_ALIASES.values()
    ):
        if (
            _normalize_direct_homelab_text(
                canonical
            )
            == normalized
        ):
            return canonical

    return None



def is_direct_homelab_status_request(
    text: str,
    domains,
    continuation: bool = False,
) -> bool:
    """Backward-compatible wrapper for whole-homelab status."""
    return (
        classify_direct_homelab_request(
            text,
            domains,
            continuation=continuation,
        )
        == {"action": "status"}
    )


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

        raw_service = str(
            args.get("service", "")
        ).strip()

        service = canonicalize_homelab_service(
            raw_service
        )

        raw_services = args.get(
            "services",
            [],
        )

        services: list[str] = []
        invalid_services: list[str] = []

        if isinstance(raw_services, list):
            for item in raw_services:
                raw_candidate = str(
                    item or ""
                ).strip()

                candidate = (
                    canonicalize_homelab_service(
                        raw_candidate
                    )
                )

                if (
                    raw_candidate
                    and candidate is None
                ):
                    invalid_services.append(
                        raw_candidate
                    )

                elif (
                    candidate
                    and candidate not in services
                ):
                    services.append(candidate)

        if (
            action == "service"
            and raw_service
            and service is None
        ):
            return {
                "error": (
                    "homelab: unknown service "
                    f"'{raw_service}'"
                ),
                "exit_code": 1,
            }

        if (
            action == "services"
            and invalid_services
        ):
            return {
                "error": (
                    "homelab: unknown services: "
                    + ", ".join(invalid_services)
                ),
                "exit_code": 1,
            }

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

            elif action == "gpu":
                result = client.status()
                direct_response = format_homelab_gpu(
                    result
                )

            elif action == "palworld_status":
                result = client.palworld_status()
                direct_response = format_palworld_status(
                    result
                )

            elif action == "palworld_backups":
                result = client.palworld_backups()
                direct_response = format_palworld_backups(
                    result
                )

            elif action == "services":
                if not (
                    2 <= len(services) <= 10
                ):
                    return {
                        "error": (
                            "homelab: services must contain "
                            "between 2 and 10 unique names"
                        ),
                        "exit_code": 1,
                    }

                formatted_services = []

                for service_name in services:
                    result = client.service(
                        service_name
                    )

                    formatted_services.append(
                        format_homelab_service(
                            result
                        )
                    )

                direct_response = (
                    "\n\n---\n\n".join(
                        formatted_services
                    )
                )

            elif action == "service":
                if not service:
                    return {
                        "error": (
                            "homelab: service "
                            "is required"
                        ),
                        "exit_code": 1,
                    }

                result = client.service(service)
                direct_response = format_homelab_service(
                    result
                )

            else:
                result = client.doctor()

                return {
                    "output": json.dumps(
                        result,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    "exit_code": 0,
                }

            return {
                "output": direct_response,
                "direct_response": direct_response,
                "terminal_response": True,
                "exit_code": 0,
            }

        except Exception as exc:
            return {
                "error": f"homelab: {exc}",
                "exit_code": 1,
            }
