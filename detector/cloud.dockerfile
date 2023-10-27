ARG BASE_IMAGE
FROM ${BASE_IMAGE} as release

# RUN pkg-config --cflags --libs opencv


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

# ENV PYTHONPATH "${PYTHONPATH}:/usr/local/lib/python3.10/dist-packages/"

# RUN apt-get update && \
#     DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
#     apt-get install -y --no-install-recommends \
#     libjpeg8-dev zlib1g-dev \
#     ca-certificates \
#     openssh-client \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir wheel
RUN pip3 install --no-cache-dir pycuda==2022.2.2
RUN pip3 install --no-cache-dir "uvicorn" 
RUN pip3 install --no-cache-dir async_generator aiofiles psutil pillow multidict attrs yarl async_timeout idna_ssl cchardet aiosignal
# LL_NODE-Library can be overwritten by local version if environment variable LINKLL is set to TRUE
RUN pip3 install --no-cache-dir "learning_loop_node==v0.8.3"
RUN pip3 install --no-cache-dir "gdown"
RUN pip3 install --no-cache-dir starlette==0.16.0

# Download the coco model. 
WORKDIR /data/models
RUN gdown --fuzzy https://drive.google.com/file/d/1KGZe7OUX9QWZm-dnkssSV9lSXxCn7nD_/view?usp=sharing  -O coco.zip && unzip coco.zip && rm coco.zip
WORKDIR /data/
RUN ln -s models/coco model

# Install TensorRT and use it as working directory
WORKDIR /
RUN git clone https://github.com/wang-xinyu/tensorrtx.git
WORKDIR /tensorrtx/yolov5/src
RUN git checkout c997e35710ff0230ae6361d9ba3b9ae82ed3a7d8

# Edit calibrator.cpp to make it compile (comment out some lines)
RUN sed -i 's|^#include <opencv2/dnn/dnn.hpp>|\/\/&|' calibrator.cpp
RUN sed -i '72s/^/\/\//' calibrator.cpp
RUN sed -i '74s/^/\/\//' calibrator.cpp

WORKDIR /tensorrtx/yolov5/build
RUN cmake .. && make -j6
ENV PYTHONPATH=$PYTHONPATH:/tensorrtx/yolov5/

ADD ./ /yolov5_node/detector/
RUN ln -sf /yolov5_node/detector /app

WORKDIR /app

EXPOSE 80

ENV HOST=learning-loop.ai
ENV TZ=Europe/Amsterdam
CMD /app/start.sh

FROM release

RUN python3 -m pip install --no-cache-dir retry debugpy pytest-asyncio icecream pytest autopep8

ENV PYTHONFAULTHANDLER=1

RUN apt-get update && apt-get install gnupg2 -y
