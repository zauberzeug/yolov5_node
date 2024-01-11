ARG OPENCV_IMAGE
ARG BASE_IMAGE
FROM ${BASE_IMAGE} as builder

# Source: https://github.com/dusty-nv/jetson-containers/blob/master/Dockerfile.ml 
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_CONFIG="/usr/bin/llvm-config-9"

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#
# apt packages
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-distutils \
    python3-dev \
    python3-setuptools \
    python3-matplotlib \
    build-essential \
    gfortran \
    git \
    cmake \
    curl \
    unzip \
    vim \
    gnupg \
    libopencv-dev \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    zlib1g-dev \
    zip \
    pkg-config \
    libavcodec-dev \ 
    libavformat-dev \ 
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libv4l-dev \
    v4l-utils \
    libdc1394-22-dev \

    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \

    libgtk2.0-dev \
    libjpeg8-dev \
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common \
    protobuf-compiler \
    libprotoc-dev \
    llvm-9 \
    llvm-9-dev \
    && apt-get -y purge *libopencv* \
    && apt -y autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/bin/python3 && ln -sf /usr/bin/python3.8 /usr/bin/python

ARG OPENCV_VERSION
ARG MAKEFLAGS

WORKDIR /root

RUN curl -L https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -o opencv-${OPENCV_VERSION}.zip && \
    curl -L https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -o opencv_contrib-${OPENCV_VERSION}.zip && \
    unzip opencv-${OPENCV_VERSION}.zip && \
    unzip opencv_contrib-${OPENCV_VERSION}.zip

WORKDIR /root/opencv-${OPENCV_VERSION}/build

RUN cmake -D WITH_VTK=OFF -D BUILD_opencv_viz=OFF -DWITH_QT=OFF -DWITH_GTK=OFF -D WITH_CUDA=ON -D WITH_CUDNN=ON -D CUDA_ARCH_BIN="5.3,6.2,7.2" -D CUDA_ARCH_PTX="" -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/opencv ..

RUN make $MAKEFLAGS 

RUN make install

CMD bash

FROM ${BASE_IMAGE} as release

COPY --from=builder /usr/local/opencv /usr/local

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
# NOTE: currently we can not use newer version of learning_loop_node because it requires a higher python version
RUN pip3 install --no-cache-dir "learning_loop_node" 
RUN pip3 install --no-cache-dir "gdown"
RUN pip3 install --no-cache-dir starlette==0.16.0

WORKDIR /data/models
RUN gdown --fuzzy https://drive.google.com/file/d/1KGZe7OUX9QWZm-dnkssSV9lSXxCn7nD_/view?usp=sharing  -O coco.zip && unzip coco.zip && rm coco.zip
WORKDIR /data/
RUN ln -s models/coco model

WORKDIR /

RUN git clone https://github.com/wang-xinyu/tensorrtx.git
WORKDIR /tensorrtx/yolov5/build
RUN git checkout 9243edf59e527bb25e5b966c2d1ae4d1b0c78d5f

RUN cmake ..
RUN make -j6
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


# RUN curl -sSL https://gist.githubusercontent.com/b01/0a16b6645ab7921b0910603dfb85e4fb/raw/5186ea07a06eac28937fd914a9c8f9ce077a978e/download-vs-code-server.sh | sed "s/server-linux-x64/server-linux-$(dpkg --print-architecture)/" | sed "s/amd64/x64/" | bash

# ENV VSCODE_SERVER=/root/.vscode-server/bin/*/server.sh

# RUN $VSCODE_SERVER --install-extension ms-python.vscode-pylance \
#     $VSCODE_SERVER --install-extension ms-python.python \
#     $VSCODE_SERVER --install-extension himanoa.python-autopep8 \
#     $VSCODE_SERVER --install-extension esbenp.prettier-vscode \
#     $VSCODE_SERVER --install-extension littlefoxteam.vscode-python-test-adapter

ENV PYTHONFAULTHANDLER=1

RUN apt-get update && apt-get install gnupg2 -y
