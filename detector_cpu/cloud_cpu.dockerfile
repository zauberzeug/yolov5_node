FROM python:3.12-bookworm 
# DEBIAN 12

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    openssh-client \
    curl \
    git \
    cmake \
    build-essential \
    unzip \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install --upgrade pip
RUN pip3 install wheel
RUN pip3 install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install ultralytics
RUN pip3 install uvicorn
RUN pip3 install async_generator aiofiles psutil pillow multidict attrs yarl async_timeout idna_ssl aiosignal
RUN pip3 install IPython
# RUN pip3 install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

# LL_NODE-Library can be overwritten by local version if environment variable LINKLL is set to TRUE
ARG NODE_LIB_VERSION
RUN pip3 install "learning_loop_node==${NODE_LIB_VERSION}"

ADD ./trainer/app_code/yolov5 /yolov5_node/detector_cpu/app_code/yolov5
RUN pip install --no-cache -r /yolov5_node/detector_cpu/app_code/yolov5/requirements.txt

ADD ./detector_cpu /yolov5_node/detector_cpu/
RUN rm -f /yolov5_node/detector_cpu/.env
RUN ln -sf /yolov5_node/detector_cpu /app



WORKDIR /app

EXPOSE 80

ENV TZ=Europe/Amsterdam
CMD ["/app/start.sh"]
