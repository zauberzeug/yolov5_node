#!/usr/bin/env bash

# This script is used to build, run, stop, kill, attach, etc. the docker container for the detector node.

if [ $# -eq 0 ]
then
    echo "Usage:"
    echo
    echo "  `basename $0` (b | build)            Build or rebuild"
    echo "  `basename $0` (bx | buildx)          Build multi-arch (arm64 + amd64) and push"
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

SEMANTIC_VERSION=$(grep -oP '^version\s*=\s*"\K[0-9.]+' pyproject.toml)
NODE_LIB_VERSION=$(grep -oP 'learning_loop_node==\K[0-9.]+' pyproject.toml)
build_args=" --build-arg NODE_LIB_VERSION=$NODE_LIB_VERSION"

BASE_JETSON="dustynv/l4t-ml:r36.4.0"
BASE_CLOUD="nvcr.io/nvidia/tensorrt:25.01-py3"

TARGET_JETSON="zauberzeug/yolov5-detector:$SEMANTIC_VERSION-nlv$NODE_LIB_VERSION-jetson"
TARGET_CLOUD="zauberzeug/yolov5-detector:$SEMANTIC_VERSION-nlv$NODE_LIB_VERSION-cloud"


if [ -f /etc/nv_tegra_release ] # Check if we are on a Jetson device
then
    # version discovery borrowed from https://github.com/dusty-nv/jetson-containers/blob/master/scripts/l4t_version.sh
    L4T_VERSION_STRING=$(head -n 1 /etc/nv_tegra_release)
    L4T_RELEASE=$(echo $L4T_VERSION_STRING | cut -f 2 -d ' ' | grep -Po '(?<=R)[^;]+')
    L4T_REVISION=$(echo $L4T_VERSION_STRING | cut -f 2 -d ',' | grep -Po '(?<=REVISION: )[^;]+')
    L4T_VERSION="$L4T_RELEASE.$L4T_REVISION"

    if [ "$L4T_RELEASE" == "36" ]; then
        # L4T R36.x containers can run on other versions of L4T R36.x (JetPack 6.0+)
        echo "Using L4T version 36.4.0 (dusty image for exact version $L4T_VERSION)"
        build_args+=" --build-arg BASE_IMAGE=$BASE_JETSON"
        build_args+=" --build-arg INSTALL_OPENCV=false"
    else
        echo "Unsupported L4T version: $L4T_VERSION"
        exit 1
    fi

    image=$TARGET_JETSON
else # ----------------------------------------------------------------------- This is cloud (linux) (python 3.12)
    build_args+=" --build-arg BASE_IMAGE=$BASE_CLOUD"
    build_args+=" --build-arg INSTALL_OPENCV=true"
    image=$TARGET_CLOUD
fi

# ========================== RUN CONFIGURATION =========================================

# sourcing .env file to get configuration (see README.md)
. .env || echo "you should provide an .env file to configure the detector"

run_args="-it" 
# run_args+=" -v $(pwd)/:/app"
# run_args+=" -v $HOME/node_data/$DETECTOR_NAME:/data"
run_args+=" -h ${HOSTNAME}_DEV"
run_args+=" -e HOST=$LOOP_HOST -e ORGANIZATION=$LOOP_ORGANIZATION -e PROJECT=$LOOP_PROJECT"
run_args+=" -e USE_BACKDOOR_CONTROLS=$USE_BACKDOOR_CONTROLS"
run_args+=" --name $DETECTOR_NAME"
run_args+=" --gpus all"
run_args+=" -p 8004:80"

# Link Learning Loop Node library if requested
if [ "$LINKLL" == "TRUE" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    if [ -f /etc/nv_tegra_release ]
    then
        # TODO: check that this is correct for new jetson versions
        run_args+=" -v $SCRIPT_DIR/../../learning_loop_node/learning_loop_node:/usr/local/lib/python3.8/dist-packages/learning_loop_node"
    else
        run_args+=" -v $SCRIPT_DIR/../../learning_loop_node/learning_loop_node:/uv_env/.venv/lib/python3.12/dist-packages/learning_loop_node"
    fi
    echo "Linking Learning Loop from $SCRIPT_DIR/../../learning_loop_node"
fi

# ========================== COMMAND EXECUTION =========================================

cmd=$1
cmd_args=${@:2}
set -x
case $cmd in
    b | build)
        docker build . -t $image $build_args $cmd_args
        ;;
    bx | buildx)
        # Ensure buildx builder exists and supports multi-platform
        docker buildx create --name multiarch-builder --use 2>/dev/null || docker buildx use multiarch-builder
        
        # Build ARM64 image (Jetson)
        echo "Building ARM64 image (Jetson)..."
        docker buildx build \
            --platform linux/arm64 \
            --push \
            -t $TARGET_JETSON \
            --build-arg BASE_IMAGE=$BASE_JETSON \
            --build-arg INSTALL_OPENCV=false \
            --build-arg NODE_LIB_VERSION=$NODE_LIB_VERSION \
            $cmd_args \
            .
        
        # Build AMD64 image (Cloud)
        echo "Building AMD64 image (Cloud)..."
        docker buildx build \
            --platform linux/amd64 \
            --push \
            -t $TARGET_CLOUD \
            --build-arg BASE_IMAGE=$BASE_CLOUD \
            --build-arg INSTALL_OPENCV=true \
            --build-arg NODE_LIB_VERSION=$NODE_LIB_VERSION \
            $cmd_args \
            .
        
        echo "Created images: $target-jetson and $target-cloud"
        ;;
    bnc | build-no-cache)
        docker build --no-cache . -t $image $build_args $cmd_args
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
