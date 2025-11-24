# Linux with python 3.12
FROM nvcr.io/nvidia/pytorch:25.01-py3 
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:${LD_LIBRARY_PATH}

# 1. Install system dependencies -----------------------------------------------

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ARG MAKEFLAGS=-j$(nproc)

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3-pip \
#     python3-setuptools \
#     python3-matplotlib \
#     curl \
#     && rm -rf /var/lib/apt/lists/* 

RUN apt update && \
    apt purge -y hwloc-nox libhwloc-dev libhwloc-plugins && \
    apt install -y zip nano htop screen libgl1 libmpich-dev jpeginfo && \
    rm -rf /var/lib/apt/lists/* \
    && apt-get clean


# 2. Install Python dependencies -----------------------------------------------

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies
WORKDIR /uv_env

# Copy only dependency files first (enables layer caching)
COPY pyproject.toml uv.lock ./

# Install deps into a project-local venv
RUN pip install --no-cache-dir uv 
RUN uv venv --system-site-packages 
RUN uv sync --locked

# Ensure the venv is first on PATH for CMD/RUN that follow
ENV PATH="/uv_env/.venv/bin:${PATH}"

# Install YOLOv5 dependencies
ADD /app_code/yolov5 /app/app_code/yolov5
RUN uv pip install -r /app/app_code/yolov5/requirements.txt

# 3. Copy the app --------------------------------------------------------------
COPY . /app

# Remove dev-only files
RUN rm -f /app/.env

WORKDIR /app

EXPOSE 80
ENV TZ=Europe/Amsterdam
CMD ["python", "/app/main.py"]


