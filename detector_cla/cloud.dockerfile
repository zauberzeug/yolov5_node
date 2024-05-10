ARG BASE_IMAGE
FROM ${BASE_IMAGE} as release

# RUN pkg-config --cflags --libs opencv


RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt-get install -y --no-install-recommends \
    libjpeg8-dev libgl1 zlib1g-dev \
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
#     libjpeg8-dev libgl1 zlib1g-dev \
#     ca-certificates \
#     openssh-client \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir wheel
RUN pip3 install --no-cache-dir pycuda
RUN pip3 install --no-cache-dir "uvicorn" 
RUN pip3 install --no-cache-dir async_generator aiofiles psutil pillow multidict attrs yarl async_timeout idna_ssl cchardet aiosignal
RUN pip3 install --no-cache-dir "learning_loop_node==v0.8.3"
RUN pip3 install --no-cache-dir "gdown"
RUN pip3 install --no-cache-dir starlette==0.16.0


# fetch yolov5 code (no longer required?)
# RUN git clone https://github.com/ultralytics/yolov5.git
# WORKDIR /yolov5
# RUN git checkout 342fe05e6c88221750ce7e90b7d2e8baabd397dc

WORKDIR /
ADD ./ /yolov5_node/detector/
RUN ln -sf /yolov5_node/detector /app

WORKDIR /app
RUN python3 -m pip install --no-cache -r yolo_requirements.txt
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip3 install opencv-python==4.5.5.64

EXPOSE 80

ENV HOST=learning-loop.ai
ENV TZ=Europe/Amsterdam
CMD /app/start.sh
