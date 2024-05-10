ARG OPENCV_IMAGE
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

RUN pip3 install --no-cache-dir gdown==4.6.3

WORKDIR /data/models
RUN gdown --fuzzy https://drive.google.com/file/d/1KGZe7OUX9QWZm-dnkssSV9lSXxCn7nD_/view?usp=sharing  -O coco.zip && unzip coco.zip && rm coco.zip
WORKDIR /data/
RUN ln -s models/coco model

WORKDIR /

RUN git clone https://github.com/wang-xinyu/tensorrtx.git
WORKDIR /tensorrtx
RUN git checkout 9243edf59e527bb25e5b966c2d1ae4d1b0c78d5f
WORKDIR /tensorrtx/yolov5/build

RUN cmake ..
RUN make -j6
ENV PYTHONPATH=$PYTHONPATH:/tensorrtx/yolov5/

ADD ./ /yolov5_node/detector/
RUN ln -sf /yolov5_node/detector /app

ARG NODE_LIB_VERSION

RUN pip3 install --no-cache-dir --ignore-installed pyyaml numpy==1.22.4
RUN pip3 install --no-cache-dir "learning_loop_node==${NODE_LIB_VERSION}" 

WORKDIR /app

EXPOSE 80

ENV HOST=learning-loop.ai
ENV TZ=Europe/Amsterdam
CMD /app/start.sh
