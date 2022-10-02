ARG BASE_IMAGE
FROM $BASE_IMAGE as release

ENV DEBIAN_FRONTEND=noninteractive
ARG MAKEFLAGS=-j$(nproc)

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools \
    python3-matplotlib \
    curl \
    python3-opencv \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt update && \
    apt purge -y hwloc-nox libhwloc-dev libhwloc-plugins && \
    apt install -y zip htop screen libgl1-mesa-glx libmpich-dev jpeginfo && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install seaborn thop coremltools onnx gsutil notebook wandb>=0.12.2

WORKDIR /

# https://githubmemory.com/repo/ultralytics/yolov5/issues/5374
RUN pip install --no-cache -U torch torchvision numpy Pillow

RUN pip install --no-cache torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# fetch yolov5 code
RUN git clone https://github.com/ultralytics/yolov5.git
WORKDIR /yolov5
RUN git checkout db6ec66a602a0b64a7db1711acd064eda5daf2b3
COPY *.patch ./
RUN git apply confusion_matrices.patch && git apply clear_epochs.patch && git apply ignore_previous_fitness.patch && rm *.patch
RUN python3 -m pip install --no-cache -r requirements.txt
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
# wts file generator for detector format
RUN wget https://raw.githubusercontent.com/wang-xinyu/tensorrtx/master/yolov5/gen_wts.py

RUN python3 -m pip install autopep8 debugpy gunicorn pyyaml uvloop pytest-asyncio ruamel.yaml
RUN python3 -m pip install "learning_loop_node==v0.7.31"

ADD ./ /yolov5_node/trainer/
RUN ln -sf /yolov5_node/trainer /app
git config --global --add safe.directory /yolov5_node
git config --global --add safe.directory /learning_loop_node

WORKDIR /app

EXPOSE 80

ENV PYTHONPATH="$PYTHONPATH:/yolov5"
ENV PYTHONPATH="$PYTHONPATH:/"
ENV PYTHONPATH="$PYTHONPATH:/opt/conda/lib/python3.8/site-packages"
CMD /app/start.sh

FROM release as dev


RUN curl -sSL https://gist.githubusercontent.com/b01/0a16b6645ab7921b0910603dfb85e4fb/raw/5186ea07a06eac28937fd914a9c8f9ce077a978e/download-vs-code-server.sh | sed "s/server-linux-x64/server-linux-$(dpkg --print-architecture)/" | sed "s/amd64/x64/" | bash
ENV VSCODE_SERVER=/root/.vscode-server/bin/*/server.sh

RUN $VSCODE_SERVER --install-extension ms-python.vscode-pylance \
    $VSCODE_SERVER --install-extension ms-python.python \
    $VSCODE_SERVER --install-extension himanoa.python-autopep8 \
    $VSCODE_SERVER --install-extension esbenp.prettier-vscode \
    $VSCODE_SERVER --install-extension littlefoxteam.vscode-python-test-adapter
