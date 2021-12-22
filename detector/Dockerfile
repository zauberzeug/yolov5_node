FROM zauberzeug/l4t-opencv:4.5.2-on-nano-32.6.1

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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /

RUN git clone https://github.com/wang-xinyu/tensorrtx.git
WORKDIR /tensorrtx/yolov5/build
RUN cmake ..
RUN make -j6
COPY ./yolov5s6.wts ./
RUN ./yolov5 -s yolov5s6.wts yolov5s6.engine s6

RUN pip3 install --no-cache-dir wheel
RUN pip3 install --no-cache-dir pycuda

RUN mkdir ./images
RUN curl -o ./images/crowd.jpg "https://www.securitymagazine.com/ext/resources/Issues/2019/March/SEC0319-Events-Feat-slide1_900px.jpg?t=1552326688&width=900"
RUN for i in $(seq 1 10); do cp images/crowd.jpg images/$i.jpg; done

CMD ./yolov5 -d yolov5s6.engine ./images