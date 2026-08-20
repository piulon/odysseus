#!/usr/bin/env sh
#
# Configure the exact Debian archive used by Odysseus image builds.
#
# A historical snapshot makes apt dependency resolution stable between
# otherwise identical builds.

set -eu

SNAPSHOT='20260815T180000Z'
KEYRING='/usr/share/keyrings/debian-archive-keyring.pgp'
SOURCES='/etc/apt/sources.list.d/debian.sources'

if [ ! -r /etc/os-release ]; then
    echo "ERROR: /etc/os-release missing" >&2
    exit 1
fi

. /etc/os-release

if [ "${ID:-}" != "debian" ]; then
    echo "ERROR: expected Debian base, got ${ID:-unknown}" >&2
    exit 1
fi

if [ "${VERSION_CODENAME:-}" != "trixie" ]; then
    echo "ERROR: expected Debian trixie, got ${VERSION_CODENAME:-unknown}" >&2
    exit 1
fi

if [ ! -r "$KEYRING" ]; then
    echo "ERROR: Debian archive keyring missing: $KEYRING" >&2
    exit 1
fi

{
    printf '%s\n' \
        'Types: deb' \
        "URIs: https://snapshot.debian.org/archive/debian/${SNAPSHOT}/" \
        'Suites: trixie trixie-updates' \
        'Components: main' \
        "Signed-By: ${KEYRING}" \
        'Check-Valid-Until: no' \
        '' \
        'Types: deb' \
        "URIs: https://snapshot.debian.org/archive/debian-security/${SNAPSHOT}/" \
        'Suites: trixie-security' \
        'Components: main' \
        "Signed-By: ${KEYRING}" \
        'Check-Valid-Until: no'
} > "$SOURCES"

rm -rf /var/lib/apt/lists/*

echo "Debian snapshot configured: ${SNAPSHOT}"
