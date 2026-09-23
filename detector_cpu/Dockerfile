FROM python:3.12-bookworm 
# DEBIAN 12

# NOTE: Build context is the repo root, not this folder!

# 1. Install system dependencies -----------------------------------------------

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    openssh-client \
    curl \
    git \
    cmake \
    build-essential \
    unzip \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 2. Install Python dependencies -----------------------------------------------

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /usr/local/bin/uv
# The uv cache is a BuildKit cache mount on another filesystem
ENV UV_LINK_MODE=copy

# Install Python dependencies
WORKDIR /uv_env

# Copy only dependency files first (enables layer caching)
COPY ./detector_cpu/pyproject.toml ./detector_cpu/uv.lock ./

# Install deps into a project-local venv
RUN uv venv --system-site-packages
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked

# Ensure the venv is first on PATH for CMD/RUN that follow
ENV PATH="/uv_env/.venv/bin:${PATH}"

# Install YOLOv5 dependencies before its code
COPY ./trainer/app_code/yolov5/requirements.txt /app/app_code/yolov5/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -r /app/app_code/yolov5/requirements.txt
ADD ./trainer/app_code/yolov5 /app/app_code/yolov5

# 3. Copy the app --------------------------------------------------------------

ADD ./detector_cpu /app

# Remove dev-only files
RUN rm -f /app/.env


WORKDIR /app

EXPOSE 80
ENV TZ=Europe/Amsterdam

ARG VERSION=unknown
ENV NODE_VERSION=$VERSION

CMD ["python", "/app/main.py"]
