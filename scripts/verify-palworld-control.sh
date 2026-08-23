#!/usr/bin/env bash
set -euo pipefail

cd /opt/stacks/odysseus

TAG="${1:-odysseus-odysseus:palworld-control-v1}"

EXPECTED="$(
  docker image inspect "$TAG" \
    --format '{{.Id}}'
)"

ACTIVE="$(
  docker inspect odysseus-odysseus-1 \
    --format '{{.Image}}'
)"

[ "$ACTIVE" = "$EXPECTED" ] || {
  echo "ERROR: imagen activa distinta de $TAG"
  exit 1
}

HTTP="$(
  curl -sS -o /dev/null \
    -w '%{http_code}' \
    --max-time 5 \
    http://127.0.0.1:7000/login ||
  true
)"

[ "$HTTP" = "200" ] || {
  echo "ERROR: /login devuelve HTTP $HTTP"
  exit 1
}

UNIT="$(
  systemctl is-active palworld.service ||
  true
)"

[ "$UNIT" = "active" ] || {
  echo "ERROR: palworld.service está $UNIT"
  exit 1
}

docker exec \
  --workdir /app \
  --env PYTHONPATH=/app \
  odysseus-odysseus-1 \
  python -c '
from src.services.homelab.client import HomelabClient

client = HomelabClient(timeout=20)
status = client.palworld_status()
backups = client.palworld_backups()

server = status.get("server")
state = ""

if isinstance(server, dict):
    state = str(server.get("status") or "")

if not state:
    state = str(status.get("status") or "")

state = state.casefold()

if state not in {"active", "online", "ready", "running"}:
    raise SystemExit(f"ERROR: estado inesperado: {state}")

count = int(backups.get("count") or 0)
latest = str(backups.get("latest_backup") or "")
integrity = str(backups.get("integrity") or "")

if count < 1 or not latest:
    raise SystemExit("ERROR: inventario de backups no válido")

print("Estado del operador:", state)
print("Backups:", count)
print("Último backup:", latest)
print("Integridad:", integrity)
'

echo "Imagen activa: $ACTIVE"
echo "HTTP /login:  $HTTP"
echo "Palworld:     $UNIT"
echo "PALWORLD CONTROL V1: OK"
