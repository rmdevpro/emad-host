# emad-host-langgraph — Application container
# All LangGraph flows, queue workers, Imperator, and ASGI server.
#
# Build context: project root (.)

FROM python:3.12.10-slim

ARG USER_NAME=emad-host
ARG USER_UID=1000
ARG USER_GID=1000

# Root phase: system packages, user creation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        libpq-dev \
        git && \
    # Install GitHub CLI
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends gh && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid ${USER_GID} ${USER_NAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} --shell /bin/bash --create-home ${USER_NAME}

USER ${USER_NAME}
WORKDIR /app

# REQ-001 §10: Enable --user pip installs for runtime StateGraph packages
ENV PYTHONUSERBASE=/home/${USER_NAME}/.local
ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

# Copy requirements and install dependencies
COPY --chown=${USER_NAME}:${USER_NAME} requirements.txt ./

# Package source is configurable: local wheels, pypi, or devpi
# Default: pypi (wheels can be placed in /app/packages for local mode)
ARG PACKAGE_SOURCE=pypi
ARG DEVPI_URL=""

ARG PIP_INDEX_URL="http://192.168.1.110:3141/root/pypi/+simple/"
ARG PIP_TRUSTED_HOST="192.168.1.110"

RUN if [ "$PACKAGE_SOURCE" = "local" ]; then \
        pip install --no-cache-dir --no-index --find-links=/app/packages -r requirements.txt; \
    elif [ "$PACKAGE_SOURCE" = "devpi" ] && [ -n "$DEVPI_URL" ]; then \
        pip install --no-cache-dir --index-url "$DEVPI_URL" --trusted-host "$PIP_TRUSTED_HOST" -r requirements.txt; \
    else \
        pip install --no-cache-dir --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" -r requirements.txt; \
    fi

USER root
RUN apt-get purge -y --auto-remove build-essential && rm -rf /var/lib/apt/lists/*
USER ${USER_NAME}

# Copy application code and entrypoint
COPY --chown=${USER_NAME}:${USER_NAME} app/ ./app/
COPY --chown=${USER_NAME}:${USER_NAME} entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# AE and TE packages are installed at startup by entrypoint.sh from the
# configured source (PyPI, devpi, or local). No pre-built wheels in the image.

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
