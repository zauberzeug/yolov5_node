ARG BASE_IMAGE
FROM ${BASE_IMAGE} as release


ENV PYTHONPATH "${PYTHONPATH}:/usr/local/lib/python3.8/dist-packages/"

RUN python3 --version
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
