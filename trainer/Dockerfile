ARG BASE_IMAGE
FROM $BASE_IMAGE as release

ENV DEBIAN_FRONTEND=noninteractive
ARG MAKEFLAGS=-j$(nproc)

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools \
    python3-matplotlib \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt update && \
    apt purge -y hwloc-nox libhwloc-dev libhwloc-plugins && \
    apt install -y zip htop screen libgl1-mesa-glx libmpich-dev jpeginfo && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install seaborn thop coremltools onnx gsutil notebook wandb>=0.12.2 opencv-python==4.5.5.64 torchinfo python-dotenv httptools dacite
# https://githubmemory.com/repo/ultralytics/yolov5/issues/5374

# TODO: Check if installing torch is needed because the image is already based on pytorch
RUN pip install --no-cache -U torch torchvision numpy Pillow

RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN python3 -m pip install autopep8 debugpy gunicorn pyyaml uvloop pytest-asyncio ruamel.yaml
# LL_NODE-Library can be overwritten by local version if environment variable LINKLL is set to TRUE
RUN python3 -m pip install "learning_loop_node==v0.8.7"

WORKDIR /
# fetch trainer code
COPY ./ /yolov5_node/trainer/
# Install yolov5 requirements
RUN python3 -m pip install --no-cache -r /yolov5_node/trainer/app_code/yolov5/requirements.txt


# link trainer code to /app and yolov5 into /app/app_code/yolov5
RUN ln -sf /yolov5_node/trainer /app 
#RUN ln -sf /yolov5 /app/app_code/yolov5
RUN git config --global --add safe.directory /yolov5_node && git config --global --add safe.directory /learning_loop_node

WORKDIR /app

EXPOSE 80

#ENV PYTHONPATH="$PYTHONPATH:/yolov5"
ENV PYTHONPATH="$PYTHONPATH:/"
# TODO CHECK
#ENV PYTHONPATH="$PYTHONPATH:/opt/conda/lib/python3.10/site-packages" 
ENV TZ=Europe/Amsterdam
CMD /app/start.sh
