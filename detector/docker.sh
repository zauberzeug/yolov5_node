#!/usr/bin/env bash

# This script is used to build, run, stop, kill, attach, etc. the docker container for the detector node.

if [ $# -eq 0 ]
then
    echo "Usage:"
    echo
    echo "  `basename $0` (b | build)            Build or rebuild"
    echo "  `basename $0` (bnc | build-no-cache) Build or rebuild without cache"
    echo "  `basename $0` (p | push)             Push image"
    echo "  ------------------------------"
    echo "  `basename $0` (U | update)           Download latest image"
    echo "  `basename $0` (r | run)              Run"
    echo "  `basename $0` (u | up)               Run in background"
    echo "  `basename $0` (s | stop)             Stop"
    echo "  `basename $0` (k | kill)             Kill"
    echo "  `basename $0` (rm)                   Kill and remove"
    echo
    echo "  `basename $0` (l | log)              Show log tail (last 100 lines)"
    echo "  `basename $0` (e | exec) <command>   Execute command"
    echo "  `basename $0` (a | attach)           Attach to container with shell"
    echo
    echo "Arguments:"
    echo "  command       Command to be executed inside a container"
    exit
fi

# ========================== BUILD CONFIGURATION / IMAGE SELECTION =======================

SEMANTIC_VERSION=0.1.10
NODE_LIB_VERSION=0.13.3
build_args=" --build-arg NODE_LIB_VERSION=$NODE_LIB_VERSION"

if [ -f /etc/nv_tegra_release ] # Check if we are on a Jetson device
then
    dockerfile="jetson.dockerfile"

    # version discovery borrowed from https://github.com/dusty-nv/jetson-containers/blob/master/scripts/l4t_version.sh
    L4T_VERSION_STRING=$(head -n 1 /etc/nv_tegra_release)
    L4T_RELEASE=$(echo $L4T_VERSION_STRING | cut -f 2 -d ' ' | grep -Po '(?<=R)[^;]+')
    L4T_REVISION=$(echo $L4T_VERSION_STRING | cut -f 2 -d ',' | grep -Po '(?<=REVISION: )[^;]+')
    L4T_VERSION="$L4T_RELEASE.$L4T_REVISION"
    
    if [ "$L4T_VERSION" == "32.6.1" ]; then
        # do nothing
        echo "Using exact L4T version 32.6.1"
        build_args+=" --build-arg BASE_IMAGE=zauberzeug/l4t-nn-inference-base:OCV4.6.0-L4T32.6.1-PY3.6"
    elif [ "$L4T_RELEASE" == "35" ]; then 
        # available versions of the dusty images: 32.7.1, 35.2.1, 35.3.1, 35.4.1
        # L4T R35.x containers can run on other versions of L4T R35.x (JetPack 5.1+)
        L4T_VERSION="35.4.1"
        echo "Using L4T version 35.4.1 (dusty image for exact version $L4T_VERSION)"
        build_args+=" --build-arg BASE_IMAGE=dustynv/opencv:r$L4T_VERSION"
    else
        echo "Unsupported L4T version: $L4T_VERSION"
        exit 1
    fi

    image="zauberzeug/yolov5-detector:$SEMANTIC_VERSION-nlv$NODE_LIB_VERSION-$L4T_VERSION"
else # ----------------------------------------------------------------------- This is cloud (linux) (python 3.10)
    dockerfile="cloud.dockerfile"

    build_args+=" --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:23.07-py3"
    image="zauberzeug/yolov5-detector:$SEMANTIC_VERSION-nlv$NODE_LIB_VERSION-cloud"
fi

# ========================== RUN CONFIGURATION =========================================

# sourcing .env file to get configuration (see README.md)
. .env || echo "you should provide an .env file to configure the detector"

run_args="-it" 
# run_args+=" -v $(pwd)/../:/yolov5_node"
run_args+=" -v $HOME/node_data/$DETECTOR_NAME:/data"
run_args+=" -h ${HOSTNAME}_DEV"
run_args+=" -e HOST=$LOOP_HOST -e ORGANIZATION=$LOOP_ORGANIZATION -e PROJECT=$LOOP_PROJECT"
run_args+=" -e USE_BACKDOOR_CONTROLS=$USE_BACKDOOR_CONTROLS"
run_args+=" --name $DETECTOR_NAME"
run_args+=" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all"
run_args+=" -p 8004:80"

# Link Learning Loop Node library if requested
if [ "$LINKLL" == "TRUE" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    if [ -f /etc/nv_tegra_release ]
    then
        run_args+=" -v $SCRIPT_DIR/../../learning_loop_node/learning_loop_node:/usr/local/lib/python3.8/dist-packages/learning_loop_node"
    else
        run_args+=" -v $SCRIPT_DIR/../../learning_loop_node/learning_loop_node:/usr/local/lib/python3.10/dist-packages/learning_loop_node"
    fi
    echo "Linking Learning Loop from $SCRIPT_DIR/../../learning_loop_node"
fi

# ========================== COMMAND EXECUTION =========================================

cmd=$1
cmd_args=${@:2}
set -x
case $cmd in
    b | build)
        DOCKER_BUILDKIT=0 docker build . -f $dockerfile --target release -t $image $build_args $cmd_args
        ;;
    bnc | build-no-cache)
        docker build --no-cache . -f $dockerfile --target release -t $image $build_args $cmd_args
        ;;
    U | update)
	    docker pull ${image}
	    ;;
    p | push)
        docker push $image
        ;;
    r | run)
        docker run $run_args $image $cmd_args
	    ;;
    u | up)
        docker run -d --restart always $run_args $image $cmd_args
	    ;;
    s | stop)
        docker stop $DETECTOR_NAME $cmd_args
        ;;
    k | kill)
        docker kill $DETECTOR_NAME $cmd_args
        ;;
    rm)
        docker kill $DETECTOR_NAME
        docker rm $DETECTOR_NAME $cmd_args
        ;;
    l | log | logs)
        docker logs -f -n 100 $cmd_args $DETECTOR_NAME
        ;;
    e | exec)
        docker exec $DETECTOR_NAME $cmd_args 
        ;;
    a | attach)
        docker exec -it $cmd_args $DETECTOR_NAME /bin/bash
        ;;
    *)
        echo "Unsupported command \"$cmd\""
        exit 1
esac
