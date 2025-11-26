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

TRAINER_VERSION=$(grep -oP '^version\s*=\s*"\K[0-9.]+' pyproject.toml)
NODE_LIB_VERSION=$(grep -oP 'learning_loop_node==\K[0-9.]+' pyproject.toml)


dockerfile="Dockerfile"
if [ "$2" = "test_latest" ]; then
    image="zauberzeug/yolov5-detector:latest-cpu" #TODO why is this required?
else
    image="zauberzeug/yolov5-detector:$TRAINER_VERSION-nlv$NODE_LIB_VERSION-cloud-cpu"
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
run_args+=" -p 8004:80"

# Link Learning Loop Node library if requested
if [ "$LINKLL" == "TRUE" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    run_args+=" -v $SCRIPT_DIR/../../learning_loop_node/learning_loop_node:/usr/local/lib/python3.10/dist-packages/learning_loop_node"
    echo "Linking Learning Loop from $SCRIPT_DIR/../../learning_loop_node"
fi

# ========================== COMMAND EXECUTION =========================================

cmd=$1
if [ "$2" = "test_latest" ]; then
    cmd_args=${@:3}
else
    cmd_args=${@:2}
fi
set -x
case $cmd in
    b | build)
        docker build .. -f $dockerfile -t $image $cmd_args
        ;;
    bnc | build-no-cache)
        docker build --no-cache . -f $dockerfile -t $image $cmd_args
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
