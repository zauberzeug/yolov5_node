# syntax=docker/dockerfile:1
# check=skip=InvalidDefaultArgInFrom

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG INSTALL_OPENCV=true

# 1. Install system dependencies -----------------------------------------------

# Install build tools and runtime dependencies (needed for runtime engine compilation)
# OpenCV is conditionally installed based on INSTALL_OPENCV (Jetson base images already include the build with NEON support)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    make \
    libjpeg8 \
    zlib1g \
    ca-certificates

RUN if [ "$INSTALL_OPENCV" = "true" ]; then apt-get install -y --no-install-recommends libopencv-dev; fi

RUN rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/share/doc/* \
    && rm -rf /usr/share/man/* \
    && rm -rf /usr/share/locale/* \
    && apt-get clean

# 2. Install Python dependencies -----------------------------------------------

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies
WORKDIR /uv_env

# Copy only dependency files first (enables layer caching)
COPY pyproject.toml uv.lock ./

# Install deps into a project-local venv
RUN uv venv .venv --system-site-packages --no-managed-python --python /usr/bin/python3 && \
    uv sync --frozen

# Ensure the venv is first on PATH for CMD/RUN that follow
ENV PATH="/uv_env/.venv/bin:${PATH}"

# 3. Copy TensorRT sources -----------------------------------------------------

WORKDIR /
COPY ./tensorrtx /tensorrtx
RUN mkdir -p /tensorrtx/yolov5/build

# 4. Copy the app --------------------------------------------------------------
COPY . /app

# Optional: remove dev-only files
RUN rm -f /app/.env

WORKDIR /app/

EXPOSE 80
ENV TZ=Europe/Amsterdam
CMD ["python", "/app/main.py"]
