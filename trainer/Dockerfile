ARG BASE_IMAGE
ARG CUDA=cuda

FROM $BASE_IMAGE AS release

# Re-declare the ARG so it's available in this stage
ARG CUDA

ENV DEBIAN_FRONTEND=noninteractive
ARG MAKEFLAGS=-j$(nproc)

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools \
    python3-matplotlib \
    curl \
 #   git \
    && rm -rf /var/lib/apt/lists/* 

RUN apt update && \
    apt purge -y hwloc-nox libhwloc-dev libhwloc-plugins && \
    apt install -y zip nano htop screen libgl1-mesa-glx libmpich-dev jpeginfo && \
    rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /

# Install requirements
RUN pip install --upgrade pip
# IF CUDA=cpu install pytorch 2.1.0 cpu + torchvision cpu
RUN if [ "$CUDA" = "cpu" ]; then echo "Installing torch 2.1.0 cpu"; else echo "Skipping torch install"; fi
RUN if [ "$CUDA" = "cpu" ]; then pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cpu ; fi
RUN pip install opencv-python==4.5.5.64
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof

# fetch trainer code and install requirements
COPY ./ /yolov5_node/trainer/
RUN pip install --no-cache -r yolov5_node/trainer/requirements.txt

# Install yolov5 requirements
RUN pip install --no-cache -r /yolov5_node/trainer/app_code/yolov5/requirements.txt

# LL_NODE-Library can be overwritten by local version if environment variable LINKLL is set to TRUE
ARG NODE_LIB_VERSION
RUN pip3 install --no-cache-dir "learning_loop_node==${NODE_LIB_VERSION}"

# link trainer code to /app
RUN ln -sf /yolov5_node/trainer /app 
#RUN git config --global --add safe.directory /yolov5_node && git config --global --add safe.directory /learning_loop_node

WORKDIR /app

EXPOSE 80

ENV PYTHONPATH="${PYTHONPATH:-}:/"

ENV TZ=Europe/Amsterdam
CMD ["/app/start.sh"]
