# Linux with python 3.12
FROM python:3.12.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ARG MAKEFLAGS=-j$(nproc)

# Install system dependencies
RUN apt update && \
    apt purge -y hwloc-nox libhwloc-dev libhwloc-plugins && \
    apt install -y zip nano htop screen libgl1 libglib2.0-0 libmpich-dev jpeginfo curl && \
    rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

RUN python -m pip install --no-cache-dir uv

# Copy only dependency files first (enables layer caching)
COPY pyproject.toml uv.lock ./

# Install deps into a project-local venv
ENV UV_PROJECT_ENVIRONMENT=/uv_venv
RUN uv sync --frozen --no-dev

# Copy the app
COPY . .

# Ensure the venv is first on PATH for CMD/RUN that follow
# This means that "python" will be resolved to our venv
ENV PATH="/uv_venv/bin:${PATH}"

# Run from /uv_venv
ENV TZ=Europe/Amsterdam
CMD ["python", "/app/main.py"]
