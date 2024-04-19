#!/usr/bin/env bash

if [ $# -eq 0 ]
then
    echo "Usage:"
    echo
    echo "  `basename $0` (b | build)     [<containers>]      Build or rebuild"
    echo "  `basename $0` (bnc)           [<containers>]      Build or rebuild without using cache"
    echo "  `basename $0` (d | debug)     [<containers>]      Start interactive with debug mode"
    echo "  `basename $0` (p | push)      [<containers>]      Push images"
    echo "  `basename $0` (r | run)       [<containers>]      Run"
    echo "  `basename $0` (ri | run-image)[<containers>]      Run image (no dev)"
    echo "  `basename $0` (s | stop)      [<containers>]      Stop"
    echo "  `basename $0` (U | update)    [<containers>]      Download latest images"
    echo "  `basename $0` (k | kill)      [<containers>]      Kill"
    echo "  `basename $0` ps              [<containers>]      List"
    echo "  `basename $0` rm              [<containers>]      Kill and remove"
    echo "  `basename $0` stats                               Show statistics"
    echo
    echo "  `basename $0` (l | log)    <container>            Show log tail (last 100 lines)"
    echo "  `basename $0` (e | exec)   <container> <command>  Execute command"
    echo "  `basename $0` (a | attach) <container>            Attach to container with shell"
    echo
    echo "Arguments:"
    echo
    echo "  containers    One or more containers (omit to affect all containers)"
    echo "  container     Excactly one container to be affected"
    echo "  command       Command to be executed inside a container"
    exit
fi

# sourcing .env file to get configuration (see README.md)
. .env || echo "you should provide an .env file to configure the detector"


run_args="-it --restart always" 
run_args+=" -v $HOME/node_data/$DETECTOR_NAME:/data"
run_args+=" -h $HOSTNAME"
run_args+=" -e HOST=$LOOP_HOST"
run_args+=" -e ORGANIZATION=$LOOP_ORGANIZATION"
run_args+=" -e PROJECT=$LOOP_PROJECT"
run_args+=" --name $DETECTOR_NAME"
run_args+=" --runtime=nvidia"
run_args+=" -e NVIDIA_VISIBLE_DEVICES=all"
run_args+=" -p 8004:80"
# run_args+=" -v $(pwd)/../:/yolov5_node"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if [ "$LINKLL" == "TRUE" ]; then
    echo "Linking Learning Loop from $SCRIPT_DIR/../../learning_loop_node"
    if [ -f /etc/nv_tegra_release ]
    then
        run_args+=" -v $SCRIPT_DIR/../../learning_loop_node/learning_loop_node:/usr/local/lib/python3.9/dist-packages/learning_loop_node"
    else
        run_args+=" -v $SCRIPT_DIR/../../learning_loop_node/learning_loop_node:/usr/local/lib/python3.10/dist-packages/learning_loop_node"
    fi
fi

NODE_LIB_VERSION=0.10.2

# Check if we are on a Jetson device
build_args=""
build_args+=" --build-arg NODE_LIB_VERSION=$NODE_LIB_VERSION"
if [ -f /etc/nv_tegra_release ]
then
    # version discovery borrowed from https://github.com/dusty-nv/jetson-containers/blob/master/scripts/l4t_version.sh
    L4T_VERSION_STRING=$(head -n 1 /etc/nv_tegra_release)
    L4T_RELEASE=$(echo $L4T_VERSION_STRING | cut -f 2 -d ' ' | grep -Po '(?<=R)[^;]+')
    L4T_REVISION=$(echo $L4T_VERSION_STRING | cut -f 2 -d ',' | grep -Po '(?<=REVISION: )[^;]+')
    L4T_VERSION="$L4T_RELEASE.$L4T_REVISION"
    
    if [ "$L4T_VERSION" == "32.6.1" ]; then # -------------------------------- This is jetson nano (python 3.9)
        build_args+=" --build-arg BASE_IMAGE=zauberzeug/l4t-nn-inference-base:OCV4.6.0-L4T$L4T_VERSION-PY3.9"
    else # ------------------------------------------------------------------- This is jetson orin (python 3.8??)
        build_args+=" --build-arg BASE_IMAGE=dustynv/opencv:r$L4T_VERSION"
    fi

    image="zauberzeug/yolov5-detector:nlv$NODE_LIB_VERSION-$L4T_VERSION"
    dockerfile="jetson.dockerfile"
else # ----------------------------------------------------------------------- This is cloud (linux) (python 3.10)
    build_args+=" --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:23.07-py3"
    image="zauberzeug/yolov5-detector:nlv$NODE_LIB_VERSION-cloud"
    dockerfile="cloud.dockerfile"
fi


cmd=$1
cmd_args=${@:2}
set -x
case $cmd in
    b | build)
        DOCKER_BUILDKIT=0 docker build . -f $dockerfile --target release -t $image $build_args $cmd_args
        DOCKER_BUILDKIT=0 docker build . -f $dockerfile -t ${image}-dev $build_args $cmd_args
        ;;
    bnc | build-no-cache)
        docker build --no-cache . -f $dockerfile --target release -t $image $build_args $cmd_args
        docker build . -f $dockerfile -t ${image}-dev $build_args $cmd_args
        ;;
    d | debug)
        docker run $run_args $image-dev /app/start.sh debug
        ;;
    U | update)
	    docker pull ${image}
        docker pull ${image}-dev
	;;
    p | push)
        docker push ${image}-dev 
        docker push $image
        ;;
    r | run)
        docker run $run_args $image-dev $cmd_args
	;;
    ri | run-image)
    	docker run $run_args $image $cmd_args
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
        docker logs -f --tail 100 $cmd_args $DETECTOR_NAME
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
