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
RUN python3 -m pip install --upgrade pip

# TODO httptools is required but the test run even if it is not installed ... we need to extend the tests 
RUN python3 -m pip install seaborn thop coremltools onnx gsutil notebook wandb>=0.12.2 opencv-python==4.5.5.64 torchinfo python-dotenv httptools

WORKDIR /

# https://githubmemory.com/repo/ultralytics/yolov5/issues/5374
RUN pip install --no-cache -U torch torchvision numpy Pillow

# fetch yolov5 code
RUN git clone https://github.com/ultralytics/yolov5.git
WORKDIR /yolov5
RUN git checkout 342fe05e6c88221750ce7e90b7d2e8baabd397dc
COPY yolov5_patches/*.patch ./
RUN git apply confusion_matrices.patch && git apply clear_epochs.patch && git apply ignore_previous_fitness.patch && rm *.patch
# TODO there is no requirements.txt file ?!
RUN python3 -m pip install --no-cache -r requirements.txt
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof

# wts file generator for detector format
RUN wget https://raw.githubusercontent.com/wang-xinyu/tensorrtx/master/yolov5/gen_wts.py

RUN python3 -m pip install autopep8 debugpy gunicorn pyyaml uvloop pytest-asyncio ruamel.yaml
RUN python3 -m pip install "learning_loop_node==v0.7.55"

COPY ./ /yolov5_node/trainer/

RUN ln -sf /yolov5_node/trainer /app
RUN ln -sf /yolov5 /app/yolov5
RUN git config --global --add safe.directory /yolov5_node && git config --global --add safe.directory /learning_loop_node

WORKDIR /app

EXPOSE 80

#ENV PYTHONPATH="$PYTHONPATH:/yolov5"
ENV PYTHONPATH="$PYTHONPATH:/"
# TODO CHECK
#ENV PYTHONPATH="$PYTHONPATH:/opt/conda/lib/python3.8/site-packages" 
ENV TZ=Europe/Amsterdam
CMD /app/start.sh
