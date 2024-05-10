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
RUN pip3 install --no-cache-dir "learning_loop_node==v0.7.54"
RUN pip3 install --no-cache-dir "gdown"
RUN pip3 install --no-cache-dir starlette==0.16.0

WORKDIR /data/models
RUN gdown --fuzzy https://drive.google.com/file/d/1KGZe7OUX9QWZm-dnkssSV9lSXxCn7nD_/view?usp=sharing  -O coco.zip && unzip coco.zip && rm coco.zip
WORKDIR /data/
RUN ln -s models/coco model

WORKDIR /

RUN git clone https://github.com/wang-xinyu/tensorrtx.git
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
