ARG BASE_IMAGE
FROM ${BASE_IMAGE} as release

RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
        apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        curl \
        git \
        cmake \
        build-essential \
        python3-markupsafe \
        python3-dev \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV PYTHONPATH "${PYTHONPATH}:/usr/local/lib/python3.6/dist-packages/"

RUN pip3 install --no-cache-dir wheel
RUN pip3 install --no-cache-dir pycuda
RUN pip3 install --no-cache-dir "uvicorn" 

RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
        apt-get install -y --no-install-recommends \
        libjpeg8-dev zlib1g-dev \
        ca-certificates \
        openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install --no-cache-dir async_generator aiofiles psutil pillow multidict attrs yarl async_timeout idna_ssl cchardet aiosignal
RUN pip3 install --no-cache-dir "learning-loop-node==0.7.12"
RUN pip3 install --no-cache-dir "gdown"

WORKDIR /data/models
RUN gdown --fuzzy https://drive.google.com/file/d/1KGZe7OUX9QWZm-dnkssSV9lSXxCn7nD_/view?usp=sharing  -O coco.zip && unzip coco.zip && rm coco.zip
WORKDIR /data/
RUN ln -s models/coco model

WORKDIR /

# RUN pip3 install --no-cache torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# fetch yolov5 code
RUN git clone https://github.com/ultralytics/yolov5.git
WORKDIR /yolov5
# Tag 6.1        3752807c0b8af03d42de478fbcbf338ec4546a6c
RUN git fetch --all --tags && git checkout tags/v6.1
ENV PYTHONPATH=$PYTHONPATH:/yolov5/

#Torch
WORKDIR /
RUN apt-get update && apt-get install -y libopenblas-base libopenmpi-dev libpython3-dev
RUN wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
RUN pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

# Torchvision
RUN git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision
RUN export BUILD_VERSION=0.9.9   
RUN cd torchvision && python3 setup.py install --user


ADD ./ /yolov5_node/detector/
RUN ln -sf /yolov5_node/detector /app

WORKDIR /app

EXPOSE 80

ENV HOST=learning-loop.ai
CMD /app/start.sh

FROM release

RUN python3 -m pip install --no-cache-dir retry debugpy pytest-asyncio icecream pytest autopep8
RUN python3 -m pip install --no-cache-dir starlette==0.16.0

RUN curl -sSL https://gist.githubusercontent.com/b01/0a16b6645ab7921b0910603dfb85e4fb/raw/5186ea07a06eac28937fd914a9c8f9ce077a978e/download-vs-code-server.sh | sed "s/server-linux-x64/server-linux-$(dpkg --print-architecture)/" | sed "s/amd64/x64/" | bash

ENV VSCODE_SERVER=/root/.vscode-server/bin/*/server.sh

RUN $VSCODE_SERVER --install-extension ms-python.vscode-pylance \
    $VSCODE_SERVER --install-extension ms-python.python \
    $VSCODE_SERVER --install-extension himanoa.python-autopep8 \
    $VSCODE_SERVER --install-extension esbenp.prettier-vscode \
    $VSCODE_SERVER --install-extension littlefoxteam.vscode-python-test-adapter

ENV PYTHONFAULTHANDLER=1

RUN apt-get update && apt-get install gnupg2 -y
