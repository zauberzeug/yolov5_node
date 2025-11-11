ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install build tools and runtime dependencies (needed for runtime engine compilation)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    make \
    libopencv-dev \
    libjpeg8 \
    zlib1g \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/share/doc/* \
    && rm -rf /usr/share/man/* \
    && rm -rf /usr/share/locale/* \
    && apt-get clean

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy and build TensorRT (initial build, will be rebuilt at runtime with model-specific config)
WORKDIR /
COPY ./tensorrtx /tensorrtx

WORKDIR /tensorrtx/yolov5/build
RUN cmake \
    -DCMAKE_CUDA_FLAGS="--diag-suppress=997 -Xcompiler=-Wno-deprecated-declarations" \
    -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations" \
    .. && make -j$(nproc)

ENV PYTHONPATH="${PYTHONPATH:-}:/tensorrtx/yolov5/"

# Install Python dependencies
WORKDIR /app

# Copy only dependency files first (enables layer caching)
COPY pyproject.toml uv.lock ./

# Install deps into a project-local venv
RUN pip install --no-cache-dir uv && \
    uv venv --system-site-packages && \
    uv sync --locked

# Now copy the rest of the app
COPY . /app

# Optional: remove dev-only files
RUN rm -f /app/.env

# Ensure the venv is first on PATH for CMD/RUN that follow
ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 80
ENV TZ=Europe/Amsterdam
CMD ["python", "/app/main.py"]
