ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS release

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt-get install -y --no-install-recommends \
    libjpeg8-dev zlib1g-dev \
    ca-certificates \
    openssh-client \
    curl \
    git \
    cmake \
    build-essential \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Download the coco model. 
# RUN pip3 install --no-cache-dir gdown==4.6.3
# WORKDIR /data/models
# RUN gdown --fuzzy https://drive.google.com/file/d/1KGZe7OUX9QWZm-dnkssSV9lSXxCn7nD_/view?usp=sharing  -O coco.zip && unzip coco.zip && rm coco.zip
# WORKDIR /data/
# RUN ln -s models/coco model

# Install TensorRT and use it as working directory
WORKDIR /
COPY ./tensorrtx /tensorrtx

WORKDIR /tensorrtx/yolov5/build
RUN cmake \
    -DCMAKE_CUDA_FLAGS="--diag-suppress=997 -Xcompiler=-Wno-deprecated-declarations" \
    -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations" \
    .. && make -j$(nproc)
ENV PYTHONPATH="${PYTHONPATH:-}:/tensorrtx/yolov5/"

# LL_NODE-Library can be overwritten by local version if environment variable LINKLL is set to TRUE
ARG NODE_LIB_VERSION
RUN uv pip install --system "learning_loop_node==${NODE_LIB_VERSION}"

ADD ./ /yolov5_node/detector/
RUN rm -f /yolov5_node/detector/.env
RUN ln -sf /yolov5_node/detector /app

WORKDIR /app
RUN uv sync install

EXPOSE 80

ENV TZ=Europe/Amsterdam
CMD ["/app/start.sh"]
