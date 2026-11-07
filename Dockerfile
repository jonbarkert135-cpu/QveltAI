FROM ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 AS uv
FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS base
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=hardlink PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/qvelt-matplotlib
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./

FROM base AS training
# Immutable signed Debian snapshot pins OS packages;the optional CA is not retained.
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    rm -f /etc/apt/sources.list.d/debian.sources; \
    printf '%s\n' \
      'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20261106T000000Z/ bookworm main' \
      'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/20261106T000000Z/ bookworm-security main' > /etc/apt/sources.list; \
    apt-get -o "Acquire::https::CaInfo=${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}" update && \
    apt-get -o "Acquire::https::CaInfo=${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}" install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
COPY src ./src
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    uv sync --frozen --extra cpu --no-dev --no-editable && uv cache clean
USER 10001:10001
ENTRYPOINT ["/app/.venv/bin/qvelt"]
CMD ["--help"]

FROM base AS serving
COPY src ./src
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    uv sync --frozen --extra cpu --extra serving --no-dev --no-editable && uv cache clean
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["/app/.venv/bin/qvelt"]
CMD ["serve", "/config/serving.yaml"]

FROM serving AS serving-quantized
USER root
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    uv sync --frozen --extra cpu --extra serving --extra hf --extra quantization --no-dev --no-editable && uv cache clean
USER 10001:10001
