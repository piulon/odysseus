# ---- builder: patch + build wheels for Real-ESRGAN's broken-on-3.14 deps ----
# basicsr/gfpgan/facexlib read their version via exec()+locals()['__version__'],
# which raises KeyError on Python 3.13+ (PEP 667). Build patched wheels here so
# the final image / Cookbook never has to compile the broken sdists. See
# docker/build-realesrgan-wheels.sh for the full rationale.
FROM python:3.14.6-slim@sha256:7bec7ddcddeff7975d6ba9b4be7dd6f6b2f55e7491539145e2978f7f97ce9144 AS realesrgan-wheels

COPY docker/configure-debian-snapshot.sh /usr/local/bin/configure-debian-snapshot

RUN chmod 0755 /usr/local/bin/configure-debian-snapshot \
    && /usr/local/bin/configure-debian-snapshot \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
    && rm -rf /var/lib/apt/lists/*

COPY docker/build-realesrgan-wheels.sh /usr/local/bin/build-realesrgan-wheels.sh
RUN bash /usr/local/bin/build-realesrgan-wheels.sh /wheels

# ---- builder: locked Browser MCP + verified browser artifacts ----
FROM python:3.14.6-slim@sha256:7bec7ddcddeff7975d6ba9b4be7dd6f6b2f55e7491539145e2978f7f97ce9144 AS playwright-bundle

# The audited Chrome-for-Testing and Playwright FFmpeg artifacts below are
# linux64/x86_64 artifacts. Fail closed rather than silently producing an
# invalid or architecture-mismatched runtime image.
RUN ARCH="$(dpkg --print-architecture)" \
    && if [ "$ARCH" != "amd64" ]; then \
         echo "unsupported Playwright runtime architecture: $ARCH (expected amd64)" >&2; \
         exit 1; \
       fi \
    && echo "Playwright runtime architecture: $ARCH"

COPY docker/configure-debian-snapshot.sh /usr/local/bin/configure-debian-snapshot

RUN chmod 0755 /usr/local/bin/configure-debian-snapshot \
    && /usr/local/bin/configure-debian-snapshot \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
       nodejs \
       npm \
       unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/playwright-mcp

COPY docker/playwright/package.json ./
COPY docker/playwright/package-lock.json ./

RUN printf '%s  %s\n' \
       '6a6d86accb31ba6512ff886abb7d6ecb5368651a2d1aab29b95579168f2c9e73' \
       package-lock.json \
       | sha256sum -c - \
    && npm ci \
       --ignore-scripts \
       --omit=optional \
       --no-audit \
       --no-fund \
    && test ! -e node_modules/fsevents \
    && test -x node_modules/.bin/playwright \
    && test -x node_modules/.bin/playwright-mcp \
    && test "$(node -p "require('./node_modules/@playwright/mcp/package.json').version")" = "0.0.78" \
    && test "$(node -p "require('./node_modules/playwright/package.json').version")" = "1.62.0-alpha-1783623505000" \
    && test "$(node -p "require('./node_modules/playwright-core/package.json').version")" = "1.62.0-alpha-1783623505000" \
    && rm -rf /root/.npm

RUN mkdir -p \
       /opt/ms-playwright/chromium-1232 \
       /opt/ms-playwright/ffmpeg-1011 \
    && curl -fsSL --retry 3 \
       'https://cdn.playwright.dev/builds/cft/151.0.7922.10/linux64/chrome-linux64.zip' \
       -o /tmp/chrome-linux64.zip \
    && printf '%s  %s\n' \
       '273b48734c09bb171dfe67f0766c0c2333f2394da13dbe44050377bf1867d9a2' \
       /tmp/chrome-linux64.zip \
       | sha256sum -c - \
    && unzip -q \
       /tmp/chrome-linux64.zip \
       -d /opt/ms-playwright/chromium-1232 \
    && curl -fsSL --retry 3 \
       'https://cdn.playwright.dev/dbazure/download/playwright/builds/ffmpeg/1011/ffmpeg-linux.zip' \
       -o /tmp/ffmpeg-linux.zip \
    && printf '%s  %s\n' \
       'ebc74fc5b94830176a3c2914ae96bd8bc7f6a91f4f33890230f84a172ee61ccc' \
       /tmp/ffmpeg-linux.zip \
       | sha256sum -c - \
    && unzip -q \
       /tmp/ffmpeg-linux.zip \
       -d /opt/ms-playwright/ffmpeg-1011 \
    && test -x \
       /opt/ms-playwright/chromium-1232/chrome-linux64/chrome \
    && test -x \
       /opt/ms-playwright/ffmpeg-1011/ffmpeg-linux \
    && touch \
       /opt/ms-playwright/chromium-1232/INSTALLATION_COMPLETE \
       /opt/ms-playwright/ffmpeg-1011/INSTALLATION_COMPLETE \
    && chmod -R a+rX \
       /opt/playwright-mcp \
       /opt/ms-playwright \
    && rm -f \
       /tmp/chrome-linux64.zip \
       /tmp/ffmpeg-linux.zip


# ---- final runtime ----
FROM python:3.14.6-slim@sha256:7bec7ddcddeff7975d6ba9b4be7dd6f6b2f55e7491539145e2978f7f97ce9144

COPY docker/configure-debian-snapshot.sh /usr/local/bin/configure-debian-snapshot
COPY docker/playwright/debian-deps.txt /tmp/playwright-debian-deps.txt

# System deps. tmux is required by Cookbook for background downloads/serves.
# openssh-client is required for Cookbook remote server tests, setup, probes,
# downloads, and serves from Docker installs.
# git/cmake are required when Cookbook builds llama.cpp on first llama.cpp
# launch inside Docker.
# nodejs runs the locked Browser MCP; npm is confined to the builder stage.
# gosu lets the entrypoint drop privileges cleanly so signals still reach
# uvicorn directly (no extra shell layer like `su`/`sudo` would add).
RUN chmod 0755 /usr/local/bin/configure-debian-snapshot \
    && /usr/local/bin/configure-debian-snapshot \
    && apt-get update \
    && PW_DEPS="$(tr '\n' ' ' < /tmp/playwright-debian-deps.txt)" \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       cmake \
       curl \
       git \
       nodejs \
       tmux \
       openssh-client \
       gosu \
       libgl1 \
       libglib2.0-0t64 \
       libxcb1 \
       libmagic1t64 \
       $PW_DEPS \
    && rm -f /tmp/playwright-debian-deps.txt \
    && rm -f /usr/bin/corepack \
    && rm -rf /usr/share/nodejs/corepack \
    && test ! -e /usr/bin/corepack \
    && test ! -e /usr/share/nodejs/corepack \
    && ! command -v npm >/dev/null 2>&1 \
    && ! command -v npx >/dev/null 2>&1 \
    && ! command -v corepack >/dev/null 2>&1 \
    && ! command -v yarn >/dev/null 2>&1 \
    && ! command -v yarnpkg >/dev/null 2>&1 \
    && ! command -v pnpm >/dev/null 2>&1 \
    && ! command -v pnpx >/dev/null 2>&1 \
    && node --version \
    && rm -rf /var/lib/apt/lists/*

# libgl1/libglib2.0-0t64/libxcb1 are runtime shared libs (libGL.so.1,
# libglib-2.0/libgthread, libxcb.so.1) that opencv-python (cv2) loads. The
# slim base omits them, so the Cookbook "install realesrgan" path imports cv2
# and dies with `libxcb.so.1: cannot open shared object file` despite a clean
# pip install. Using full opencv-python (not -headless) because basicsr/gfpgan/
# facexlib/realesrgan all depend on the `opencv-python` distribution by name.
#
# libmagic1t64 is the shared lib (libmagic.so.1) that python-magic dlopens for
# content-based MIME sniffing in src/upload_handler.py. We install both here
# (libmagic1t64 + the python-magic wrapper, below) rather than in requirements.txt
# because python-magic resolves libmagic at import time: where the lib is
# absent the import can block or raise, so keeping it image-only avoids
# regressing pip/venv installs on hosts without libmagic. Debian always has the
# lib here, so the import is instant and detection actually works.

# Debian's nodejs package depends on node-corepack. Keep the dpkg dependency
# satisfied, but remove Corepack's executable/package-manager bootstrap
# implementation from the runtime image. This intentionally causes
# dpkg -V node-corepack to report removed package-owned files.
# Built-in Browser MCP.
#
# npm exists only in the isolated playwright-bundle build stage. The runtime
# receives the package-lock-resolved node_modules tree and the browser
# artifacts whose SHA256 values were verified before extraction.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

COPY --from=playwright-bundle /opt/playwright-mcp /opt/playwright-mcp
COPY --from=playwright-bundle /opt/ms-playwright /opt/ms-playwright

RUN ln -sf \
       /opt/playwright-mcp/node_modules/.bin/playwright-mcp \
       /usr/local/bin/playwright-mcp \
    && ln -sf \
       /opt/playwright-mcp/node_modules/.bin/playwright \
       /usr/local/bin/playwright \
    && ln -sf \
       /opt/ms-playwright/chromium-1232/chrome-linux64/chrome \
       /usr/local/bin/odysseus-chromium \
    && test -x /usr/local/bin/playwright-mcp \
    && test -x /usr/local/bin/odysseus-chromium \
    && test -x /opt/ms-playwright/ffmpeg-1011/ffmpeg-linux \
    && playwright-mcp --version \
    && /usr/local/bin/odysseus-chromium --version \
    && ldd /usr/local/bin/odysseus-chromium \
       > /tmp/odysseus-chromium.ldd \
    && ! grep -F 'not found' /tmp/odysseus-chromium.ldd \
    && rm -f /tmp/odysseus-chromium.ldd

# Docker CLI (client only — daemon stays on the host via the
# /var/run/docker.sock mount). The Debian `docker.io` package ships
# dockerd but not the client binary on slim, so grab the static client
# tarball from download.docker.com instead.
ARG DOCKER_CLI_VERSION=29.6.2
RUN ARCH="$(dpkg --print-architecture)" \
    && case "$ARCH" in \
         amd64) \
           DARCH=x86_64; \
           DOCKER_CLI_SHA256=d6204aea92238e2453d5445c885b9d2e5eb8f82915568ec50edf9dbe12a3ac74 \
           ;; \
         arm64) \
           DARCH=aarch64; \
           DOCKER_CLI_SHA256=8d16d8b3b158c132a9fb9963d4b4345746f925e287e154c9ed880ac257baf292 \
           ;; \
         *) \
           echo "unsupported arch $ARCH"; \
           exit 1 \
           ;; \
       esac \
    && curl -fsSL "https://download.docker.com/linux/static/stable/${DARCH}/docker-${DOCKER_CLI_VERSION}.tgz" \
       -o /tmp/docker.tgz \
    && printf '%s  %s\n' "$DOCKER_CLI_SHA256" /tmp/docker.tgz \
       | sha256sum -c - \
    && tar -xzf /tmp/docker.tgz -C /tmp \
    && install -m 0755 /tmp/docker/docker /usr/local/bin/docker \
    && rm -rf /tmp/docker /tmp/docker.tgz

WORKDIR /app

# Install Python deps first (layer cache). Optional extras (PyMuPDF AGPL, etc.)
# are opt-in so the default image stays MIT-core; see requirements-optional.txt.
ARG INSTALL_OPTIONAL=false
COPY requirements.txt requirements.lock.txt requirements-optional.txt requirements-optional.lock.txt ./
RUN pip install \
      --no-cache-dir \
      --require-hashes \
      -r requirements.lock.txt \
    && if [ "$INSTALL_OPTIONAL" = "true" ]; then \
         pip install \
           --no-cache-dir \
           --only-binary=:all: \
           --require-hashes \
           -r requirements-optional.lock.txt; \
       fi \
    && pip check

# python-magic powers content-based MIME sniffing in src/upload_handler.py.
# Keep it image-only because it requires the system libmagic runtime above.
# Install the exact audited PyPI wheel by immutable SHA256; pip receives only
# the already-verified local artifact and cannot resolve anything from an index.
RUN PYTHON_MAGIC_WHEEL=/tmp/python_magic-0.4.27-py2.py3-none-any.whl \
    && curl -fsSL --retry 3 \
       'https://files.pythonhosted.org/packages/6c/73/9f872cb81fc5c3bb48f7227872c28975f998f3e7c2b1c16e95e6432bbb90/python_magic-0.4.27-py2.py3-none-any.whl' \
       -o "$PYTHON_MAGIC_WHEEL" \
    && printf '%s  %s\n' \
       'c212960ad306f700aa0d01e5d7a325d20548ff97eb9920dcd29513174f0294d3' \
       "$PYTHON_MAGIC_WHEEL" \
       | sha256sum -c - \
    && python -m pip install \
       --no-cache-dir \
       --no-index \
       --no-deps \
       "$PYTHON_MAGIC_WHEEL" \
    && python -c "from importlib.metadata import version; import magic; assert version('python-magic') == '0.4.27'; assert magic.from_buffer(b'%PDF-1.4\\n%%EOF\\n', mime=True) == 'application/pdf'" \
    && python -m pip check \
    && rm -f "$PYTHON_MAGIC_WHEEL"

# Keep the deterministic Python-3.14-compatible helper wheels plus the exact
# upstream Real-ESRGAN wheel in an immutable image wheelhouse, but do NOT
# install them into the base Python
# environment. Installing them with --no-deps leaves their declared
# torch/torchvision/OpenCV/SciPy/etc. requirements unsatisfied and makes
# `pip check` fail before the user has even enabled Real-ESRGAN.
#
# The Cookbook installer explicitly supplies these exact local wheels when the
# user requests realesrgan/gfpgan. Their heavy dependencies are resolved only
# at that point.
COPY --from=realesrgan-wheels /wheels/ /opt/odysseus-wheelhouse/
RUN test -f /opt/odysseus-wheelhouse/basicsr-1.4.2-py3-none-any.whl \
    && test -f /opt/odysseus-wheelhouse/facexlib-0.3.0-py3-none-any.whl \
    && test -f /opt/odysseus-wheelhouse/gfpgan-1.3.8-py3-none-any.whl \
    && test -f /opt/odysseus-wheelhouse/realesrgan-0.3.0-py3-none-any.whl \
    && test "$(find /opt/odysseus-wheelhouse -maxdepth 1 -type f -name '*.whl' | wc -l)" -eq 4 \
    && chmod -R a+rX /opt/odysseus-wheelhouse \
    && python -m pip check

# Copy app code
# Copy only the application runtime surface.
#
# Do not use `COPY . .` here. Production source must be an explicit allowlist
# so host/development/build metadata cannot silently become executable runtime
# content merely by appearing in the Docker build context.
COPY app.py setup.py LICENSE ./
COPY core/ ./core/
COPY companion/ ./companion/
COPY integrations/ ./integrations/
COPY mcp_servers/ ./mcp_servers/
COPY routes/ ./routes/
COPY services/ ./services/
COPY src/ ./src/
COPY static/ ./static/
COPY scripts/ ./scripts/
COPY licenses/ ./licenses/

# Create data directory (mount a volume here for persistence)
RUN mkdir -p data logs services/cache/search

# Entrypoint that drops to PUID/PGID (default 1000:1000) and repairs
# ownership on the bind-mounted /app/data and /app/logs. Without this,
# the container runs as root and writes root-owned files into host
# bind mounts — any later non-root run (or a host user trying to
# update them) silently fails on EPERM, breaking skill extraction,
# prefs persistence, mail attachments, etc.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 7000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000"]
