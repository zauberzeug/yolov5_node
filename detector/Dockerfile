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


RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y python3.8 python3.8-dev python3.8-venv curl python3-tk
ENV PYTHONPATH "${PYTHONPATH}:/usr/local/lib/python3.8/dist-packages/"

RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3
RUN python3 --version
RUN python3 -m pip install --upgrade pip setuptools

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
RUN pip3 install --no-cache-dir "learning_loop_node==v0.7.13rc1"
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
COPY ./requirements.txt /yolov5/requirements.txt
RUN apt-get update && apt-get install -y libpcap-dev libpq-dev 
RUN pip3 install scikit-learn
# RUN pip3 install git+https://github.com/scikit-learn/scikit-learn.git
RUN cat requirements.txt
RUN pip3 --version
RUN python3 -m pip install -r requirements.txt

#Torch
RUN pip3 install setuptools==59.5.0 opencv-python
WORKDIR /
RUN git clone --recursive --branch v1.10.0 https://github.com/pytorch/pytorch
WORKDIR /pytorch
RUN python3 setup.py install

# WORKDIR /
# RUN apt-get update && apt-get install -y libopenblas-base libopenmpi-dev libpython3-dev
# RUN wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
# RUN pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

# Torchvision
RUN git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
RUN export BUILD_VERSION=0.11.1   
RUN cd torchvision && python3 setup.py install --user

# TensorRT
RUN git clone -b master https://github.com/nvidia/TensorRT TensorRT
RUN cd TensorRT && git submodule update --init --recursive
RUN export TRT_OSSPATH=/TensorRT

RUN export EXT_PATH=~/external
RUN mkdir -p $EXT_PATH && cd $EXT_PATH
RUN git clone https://github.com/pybind/pybind11.git

RUN wget https://www.python.org/ftp/python/3.8.0/Python-3.8.0.tgz
RUN tar -xvf Python-3.8.0.tgz
RUN mkdir -p $EXT_PATH/python3.8.0/include
RUN cp -r Python-3.8.0/Include/* $EXT_PATH/python3.8.0/include/

RUN cp -r /usr/include/python3.8/pyconfig.h $EXT_PATH/python3.8.0/include/
RUN ln -sv /usr/include/python3.8/* /usr/include/
RUN cd $TRT_OSSPATH/python
RUN PYTHON_MAJOR_VERSION=3 PYTHON_MINOR_VERSION=8 TARGET_ARCHITECTURE=aarch64 ./build.sh
RUN python3 -m pip install build/dist/tensorrt-*.whl


# RUN git clone --recursive --branch v1.8.0 http://github.com/pytorch/pytorch
# RUN cd pytorch
# RUN python3 -m pip install -r requirements.txt
# RUN python3 setup.py install

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
