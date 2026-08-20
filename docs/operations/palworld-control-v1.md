# Palworld Control v1

Hito operativo cerrado el 4 de agosto de 2026.

## Versión funcional

- Commit: `7b983dc`
- Imagen estable: `odysseus-odysseus:palworld-control-v1`
- Imagen por commit: `odysseus-odysseus:palworld-control-7b983dc`
- Tag Git: `palworld-control-v1.0`

## Comandos disponibles en Odysseus

- `Estado de Palworld`
- `Muéstrame los backups de Palworld`
- `Crea un backup de Palworld`
- `Reinicia Palworld`
- `Detén Palworld`
- `Inicia Palworld`

## Controles de seguridad

- Acciones restringidas al administrador.
- Token de lectura separado del token de acciones.
- Endpoints específicos para backup, reinicio, parada e inicio.
- Las acciones no están expuestas al esquema general del LLM.
- Reinicio y parada requieren confirmación de un solo uso.
- La confirmación está ligada al propietario y a la conversación.
- La autorización caduca en cinco minutos.
- Reinicio y parada se bloquean con jugadores conectados.
- Backup verificado antes de reiniciar o detener.
- Verificación obligatoria del estado posterior.
- `offline` se acepta como estado detenido e iniciable.

## Arquitectura

Odysseus → clasificador determinista → autorización → operador del
homelab → host-agent con allowlist → systemd → verificación posterior.

## Comprobación de solo lectura

    cd /opt/stacks/odysseus
    ./scripts/verify-palworld-control.sh

## Rollback disponible

La imagen anterior permanece etiquetada como:

`odysseus-odysseus:pre-palworld-offline-fix-20260804-015840`

El rollback de Odysseus no revierte una acción de Palworld que ya
haya sido ejecutada.
