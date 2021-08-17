ARG BASE_IMAGE
FROM $BASE_IMAGE

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

RUN curl -sSL https://gist.githubusercontent.com/b01/0a16b6645ab7921b0910603dfb85e4fb/raw/5186ea07a06eac28937fd914a9c8f9ce077a978e/download-vs-code-server.sh | sed "s/server-linux-x64/server-linux-$(dpkg --print-architecture)/" | sed "s/amd64/x64/" | sh

ENV VSCODE_SERVER=/root/.vscode-server/bin/*/server.sh

RUN $VSCODE_SERVER --install-extension ms-python.vscode-pylance \
    $VSCODE_SERVER --install-extension ms-python.python \
    $VSCODE_SERVER --install-extension himanoa.python-autopep8 \
    $VSCODE_SERVER --install-extension esbenp.prettier-vscode \
    $VSCODE_SERVER --install-extension littlefoxteam.vscode-python-test-adapter

RUN apt update && apt install -y zip htop screen libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install seaborn thop

WORKDIR /

# install mish-cuda to use mish activation
# https://github.com/thomasbrandon/mish-cuda
# https://github.com/JunnYu/mish-cuda
RUN git clone https://github.com/JunnYu/mish-cuda && cd mish-cuda && python3 setup.py build install

# install pytorch_wavelets to use dwt down-sampling module
# https://github.com/fbcotter/pytorch_wavelets
RUN git clone https://github.com/fbcotter/pytorch_wavelets && cd pytorch_wavelets && pip3 install .

# fetch yolor code
RUN git clone https://github.com/WongKinYiu/yolor.git && cd /yolor && git checkout paper

RUN python3 -m pip install autopep8 debugpy gunicorn pyyaml uvloop
RUN python3 -m pip install "learning_loop_node==0.3.3"
RUN python3 -m pip install utils tqdm pycocotools

WORKDIR /

# --plugins do not build (see https://github.com/NVIDIA-AI-IOT/torch2trt/issues/558)
# cloning fork from https://github.com/NVIDIA-AI-IOT/torch2trt which supports TensorRT 8.0.1 (which comes with JetPack 4.6)
RUN git clone https://github.com/gcunhase/torch2trt.git && cd torch2trt && python3 -m pip install .

ADD ./trainer/ /app/
RUN cd /yolor && git apply /app/yolor.patch

WORKDIR /app

EXPOSE 80

ENV PYTHONPATH="$PYTHONPATH:/yolor"
ENV PYTHONPATH="$PYTHONPATH:/"
CMD /app/start.sh
