FROM zauberzeug/l4t-opencv:4.5.2-on-nano-32.6.1 as release

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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV PYTHONPATH "${PYTHONPATH}:/usr/local/lib/python3.6/dist-packages/"

RUN pip3 install --no-cache-dir wheel
RUN pip3 install --no-cache-dir pycuda
RUN pip3 install --no-cache-dir "uvicorn[standard]" 

RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
        apt-get install -y --no-install-recommends \
        libjpeg8-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install --no-cache-dir async_generator aiofiles psutil pillow
RUN pip3 install --no-cache-dir multidict attrs yarl async_timeout
RUN pip3 install --no-cache-dir idna_ssl cchardet aiosignal
RUN pip3 install --no-cache-dir "learning-loop-node==0.6.1"

WORKDIR /

RUN git clone https://github.com/wang-xinyu/tensorrtx.git

ADD ./ /yolov5_node/detector/
RUN ln -sf /yolov5_node/detector /app

WORKDIR /app

EXPOSE 80

ENV HOST=learning-loop.ai

CMD /app/start.sh

FROM release

RUN python3 -m pip install --no-cache-dir retry debugpy pytest-asyncio icecream pytest autopep8

RUN curl -sSL https://gist.githubusercontent.com/b01/0a16b6645ab7921b0910603dfb85e4fb/raw/5186ea07a06eac28937fd914a9c8f9ce077a978e/download-vs-code-server.sh | sed "s/server-linux-x64/server-linux-$(dpkg --print-architecture)/" | sed "s/amd64/x64/" | bash

ENV VSCODE_SERVER=/root/.vscode-server/bin/*/server.sh

RUN $VSCODE_SERVER --install-extension ms-python.vscode-pylance \
    $VSCODE_SERVER --install-extension ms-python.python \
    $VSCODE_SERVER --install-extension himanoa.python-autopep8 \
    $VSCODE_SERVER --install-extension esbenp.prettier-vscode \
    $VSCODE_SERVER --install-extension littlefoxteam.vscode-python-test-adapter

ENV PYTHONFAULTHANDLER=1