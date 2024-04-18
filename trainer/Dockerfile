ARG BASE_IMAGE
FROM $BASE_IMAGE as release

ENV DEBIAN_FRONTEND=noninteractive
ARG MAKEFLAGS=-j$(nproc)

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools \
    python3-matplotlib \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt update && \
    apt purge -y hwloc-nox libhwloc-dev libhwloc-plugins && \
    apt install -y zip htop screen libgl1-mesa-glx libmpich-dev jpeginfo && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /
# fetch trainer code
COPY ./ /yolov5_node/trainer/

# Install requirements
RUN pip install --upgrade pip
RUN pip install opencv-python==4.5.5.64
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache -r yolov5_node/trainer/requirements.txt
RUN pip install "learning_loop_node==v0.10.1"

# Install yolov5 requirements
RUN pip install --no-cache -r /yolov5_node/trainer/app_code/yolov5/requirements.txt


# link trainer code to /app
RUN ln -sf /yolov5_node/trainer /app 
RUN git config --global --add safe.directory /yolov5_node && git config --global --add safe.directory /learning_loop_node

WORKDIR /app

EXPOSE 80

#ENV PYTHONPATH="$PYTHONPATH:/yolov5"
ENV PYTHONPATH="$PYTHONPATH:/"
# TODO CHECK
#ENV PYTHONPATH="$PYTHONPATH:/opt/conda/lib/python3.10/site-packages" 
ENV TZ=Europe/Amsterdam
CMD /app/start.sh
