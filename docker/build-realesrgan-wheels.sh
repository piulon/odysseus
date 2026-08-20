#!/usr/bin/env bash
# Build deterministic patched wheels for Real-ESRGAN's unmaintained
# dependencies on Python 3.14.
#
# Security/reproducibility properties:
# - exact upstream versions
# - exact upstream URLs
# - SHA256 verification before extraction
# - fixed setuptools wheel + SHA256
# - setup_requires removed so legacy setup.py cannot resolve build deps
# - no dependency resolution during wheel construction
# - SOURCE_DATE_EPOCH fixes generated version.py and ZIP timestamps
# - expected final wheel SHA256 values are verified
#
# Usage:
#   build-realesrgan-wheels.sh [OUTPUT_DIR]
#
# Default output:
#   /wheels

set -euo pipefail

OUT="${1:-/wheels}"

SOURCE_DATE_EPOCH_VALUE=1704067200

BASICSR_URL='https://files.pythonhosted.org/packages/86/41/00a6b000f222f0fa4c6d9e1d6dcc9811a374cabb8abb9d408b77de39648c/basicsr-1.4.2.tar.gz'
BASICSR_SHA256='b89b595a87ef964cda9913b4d99380ddb6554c965577c0c10cb7b78e31301e87'

FACEXLIB_URL='https://files.pythonhosted.org/packages/e1/93/c820cd2c6315b635934770808e0b01ed4db257ec33bcf803909dcf4bce15/facexlib-0.3.0.tar.gz'
FACEXLIB_SHA256='7ae784a520eb52e05583e8bf9f68f77f45083239ac754d646d635017b49e7763'

GFPGAN_URL='https://files.pythonhosted.org/packages/6b/e9/b2db24ed840f188792581d217229022ff85e0ae3055a708e9f28430b8083/gfpgan-1.3.8.tar.gz'
GFPGAN_SHA256='21618b06ce8ea6230448cb526b012004f23a9ab956b55c833f69b9fc8a60c4f9'

REALESRGAN_WHEEL_URL='https://files.pythonhosted.org/packages/b2/3e/e2f79917a04991b9237df264f7abab2b58cf94748e7acfb6677b55232ca1/realesrgan-0.3.0-py3-none-any.whl'
REALESRGAN_WHEEL_SHA256='59336c16c30dd5130eff350dd27424acb9b7281d18a6810130e265606c9a6088'
REALESRGAN_WHEEL_SIZE=26012

SETUPTOOLS_URL='https://files.pythonhosted.org/packages/5d/40/e1e72872c6354b306daef1703549e8e83b4d43cfea356311bf722a043752/setuptools-83.0.0-py3-none-any.whl'
SETUPTOOLS_SHA256='29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3'

BASICSR_WHEEL_SHA256='148ebcf7ebd20c09dc94a639bdcb3b5a057509e8ac5d8a4d56b376d07c61d254'
FACEXLIB_WHEEL_SHA256='755cb04ec22ab197fa88d84685e76433e899545d7dd01c81a398de06ef5f38eb'
GFPGAN_WHEEL_SHA256='142952efd4586806dab2f75219976033824d8b2bea908ff62c6c0dc62f411b13'

# The output must contain only artifacts produced by this invocation.
rm -rf -- "$OUT"
mkdir -p "$OUT"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

input="$work/input"
src="$work/src"
venv="$work/venv"

mkdir -p "$input" "$src"

echo ">> fetching pinned Real-ESRGAN 0.3.0 wheel"

curl   --fail   --silent   --show-error   --location   --proto '=https'   --tlsv1.2   "$REALESRGAN_WHEEL_URL"   -o "$input/realesrgan-0.3.0-py3-none-any.whl"

test "$(
  stat     --printf='%s'     "$input/realesrgan-0.3.0-py3-none-any.whl"
)" -eq "$REALESRGAN_WHEEL_SIZE"

printf '%s  %s\n'   "$REALESRGAN_WHEEL_SHA256"   "$input/realesrgan-0.3.0-py3-none-any.whl"   | sha256sum -c -


echo ">> downloading fixed source artifacts"

curl -fL --retry 3 \
  "$BASICSR_URL" \
  -o "$input/basicsr-1.4.2.tar.gz"

curl -fL --retry 3 \
  "$FACEXLIB_URL" \
  -o "$input/facexlib-0.3.0.tar.gz"

curl -fL --retry 3 \
  "$GFPGAN_URL" \
  -o "$input/gfpgan-1.3.8.tar.gz"

curl -fL --retry 3 \
  "$SETUPTOOLS_URL" \
  -o "$input/setuptools-83.0.0-py3-none-any.whl"

echo ">> verifying downloaded SHA256 values"

(
cd "$input"

cat > checksums.sha256 <<EOF_SHA
${BASICSR_SHA256}  basicsr-1.4.2.tar.gz
${FACEXLIB_SHA256}  facexlib-0.3.0.tar.gz
${GFPGAN_SHA256}  gfpgan-1.3.8.tar.gz
${SETUPTOOLS_SHA256}  setuptools-83.0.0-py3-none-any.whl
EOF_SHA

sha256sum -c checksums.sha256
)

echo ">> extracting verified sources"

cd "$src"

tar --no-same-owner -xzf "$input/basicsr-1.4.2.tar.gz"
tar --no-same-owner -xzf "$input/facexlib-0.3.0.tar.gz"
tar --no-same-owner -xzf "$input/gfpgan-1.3.8.tar.gz"

echo ">> applying deterministic Python 3.14 patches"

python - <<'PY'
from pathlib import Path
import re

old_exec = "exec(compile(f.read(), version_file, 'exec'))"

new_exec = (
    "_ver_ns = {}\n"
    "        exec(compile(f.read(), version_file, 'exec'), _ver_ns)"
)

old_ret = "return locals()['__version__']"
new_ret = "return _ver_ns['__version__']"

old_time = (
    "version_file_str = "
    "content.format(time.asctime(), SHORT_VERSION, sha, VERSION_INFO)"
)

new_time = (
    "build_epoch = int(os.environ['SOURCE_DATE_EPOCH'])\n"
    "    build_time = time.asctime(time.gmtime(build_epoch))\n"
    "    version_file_str = "
    "content.format(build_time, SHORT_VERSION, sha, VERSION_INFO)"
)

version_patches = 0
setup_requires_patches = 0
time_patches = 0

for setup in sorted(Path(".").glob("*/setup.py")):
    s = setup.read_text()

    if old_exec in s and old_ret in s:
        s = s.replace(old_exec, new_exec)
        s = s.replace(old_ret, new_ret)
        version_patches += 1

    s, n = re.subn(
        r"setup_requires\s*=\s*\[[^\]]*\]",
        "setup_requires=[]",
        s,
    )
    setup_requires_patches += n

    if old_time in s:
        s = s.replace(old_time, new_time)
        time_patches += 1

    setup.write_text(s)

print("version_patches =", version_patches)
print("setup_requires_patches =", setup_requires_patches)
print("time_patches =", time_patches)

if version_patches != 3:
    raise SystemExit(
        f"expected 3 version patches, got {version_patches}"
    )

if setup_requires_patches != 3:
    raise SystemExit(
        "expected 3 setup_requires patches, "
        f"got {setup_requires_patches}"
    )

if time_patches != 3:
    raise SystemExit(
        f"expected 3 deterministic time patches, got {time_patches}"
    )
PY

echo ">> preparing fixed offline build environment"

python -m venv "$venv"

PIP_NO_INDEX=1 \
PIP_DISABLE_PIP_VERSION_CHECK=1 \
"$venv/bin/python" -m pip install \
  --no-index \
  --no-deps \
  "$input/setuptools-83.0.0-py3-none-any.whl"

echo ">> building deterministic wheels"

export SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH_VALUE"
export TZ=UTC
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONDONTWRITEBYTECODE=1

"$venv/bin/python" -m pip wheel \
  --no-index \
  --no-deps \
  --no-build-isolation \
  -w "$OUT" \
  ./basicsr-* \
  ./facexlib-* \
  ./gfpgan-*

echo ">> publishing verified Real-ESRGAN main wheel"

install   -m 0644   "$input/realesrgan-0.3.0-py3-none-any.whl"   "$OUT/realesrgan-0.3.0-py3-none-any.whl"

echo ">> verifying final wheel identities"

(
cd "$OUT"

cat > "$work/wheels.sha256" <<EOF_SHA
${BASICSR_WHEEL_SHA256}  basicsr-1.4.2-py3-none-any.whl
${FACEXLIB_WHEEL_SHA256}  facexlib-0.3.0-py3-none-any.whl
${GFPGAN_WHEEL_SHA256}  gfpgan-1.3.8-py3-none-any.whl
${REALESRGAN_WHEEL_SHA256}  realesrgan-0.3.0-py3-none-any.whl
EOF_SHA

sha256sum -c "$work/wheels.sha256"
)

echo ">> deterministic Real-ESRGAN wheels ready"
ls -lh "$OUT"
