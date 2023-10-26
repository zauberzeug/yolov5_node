#!/usr/bin/env bash

if [ $# -eq 0 ]
then
    echo "Usage:"
    echo
    echo "  `basename $0` (a | attach)  [<containers>]      Exec bash"
    echo "  `basename $0` (b | build)   [<containers>]      Build or rebuild"
    echo "  `basename $0` (bnc | build-no-cache) [<c>]      Build or rebuild without cache"
    echo "  `basename $0` (d | debug)   [<containers>]      Start in debug mode"
    echo "  `basename $0` (e | exec)    [<containers>]      Exec cmd"
    echo "  `basename $0` (k | kill)    [<containers>]      Kill"
    echo "  `basename $0` (l | log)     [<containers>]      Attach to log"
    echo "  `basename $0` (p | push)    [<containers>]      Push image"
    echo "  `basename $0` (r | run)     [<containers>]      Run"
    echo "  `basename $0` (rm)          [<containers>]      Kill and remove"
    echo "  `basename $0` (s | stop)    [<containers>]      Stop"
    echo "  `basename $0` (u | up)      [<containers>]      Start detached"
    echo "  `basename $0` ps            [<containers>]      List"
    echo "  `basename $0` rm            [<containers>]      Remove"
    echo "  `basename $0` stats                             Show statistics"
    echo
    echo "  `basename $0` (l | log)    <container>            Show log tail (last 100 lines)"
    echo "  `basename $0` (e | exec)   <container> <command>  Execute command"
    echo "  `basename $0` (a | attach) <container>            Attach to container with shell"
    echo
    echo "  `basename $0` prune      Remove all unused containers, networks and images"
    echo "  `basename $0` stopall    Stop all running containers (system-wide!)"
    echo "  `basename $0` killall    Kill all running containers (system-wide!)"
    echo
    echo "Arguments:"
    echo
    echo "  containers    One or more containers (omit to affect all containers)"
    echo "  container     Excactly one container to be affected"
    echo "  command       Command to be executed inside a container"
    exit
fi

# sourcing .env file to get configuration (see README.md)
. .env || echo "you should provide an .env file for the Learning Loop"

# if [ "$YOLOV5_MODE" == "CLASSIFICATION" ]; then
#     echo "Mode is set to CLASSIFICATION"
#     name="yolov5_cla_trainer_node"
# else
#     echo "Mode is not set to CLASSIFICATION"
#     name="yolov5_trainer_node"
# fi


run_args="-it --rm" 
run_args+=" -v $(pwd)/../:/yolov5_node/"
run_args+=" -v $HOME/trainer_nodes_data:/data"
run_args+=" -e HOST=$HOST"
run_args+=" -h ${HOSTNAME}_DEV"
run_args+=" -e USERNAME=$USERNAME -e PASSWORD=$PASSWORD"
run_args+=" -e BATCH_SIZE=$BATCH_SIZE"
run_args+=" -e NODE_TYPE=trainer"
run_args+=" --name $TRAINER_NAME"
run_args+=" --gpus all"
run_args+=" --ipc host"
run_args+=" -p 7442:80"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if [ "$LINKLL" == "TRUE" ]; then
    echo "Linking Learning Loop from"
    echo "$SCRIPT_DIR/../../learning_loop_node"
    run_args+=" -v $SCRIPT_DIR/../../learning_loop_node/learning_loop_node:/usr/local/lib/python3.10/dist-packages/learning_loop_node"
    # run_args+=" -v $SCRIPT_DIR/../../learning_loop_node:/learning_loop_node"
fi

image="zauberzeug/yolov5-trainer:latest"

build_args=" --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:23.07-py3" # this is python 3.10

cmd=$1
cmd_args=${@:2}
case $cmd in
    b | build)
        docker build . -t $image $build_args $cmd_args
        ;;
    bnc | build-no-cache)
        docker build --no-cache . -t $image $build_args $cmd_args
        ;;
    d | debug)
        docker run $run_args $image /app/start.sh debug
        ;;
    p | push)
        docker push $image
        ;;
    r | run)
        docker run -it $run_args $image $cmd_args # WARNING: in this mode the GPU may not be available
        ;;
    s | stop)
        docker stop $TRAINER_NAME $cmd_args
        ;;
    u | up)
        docker run -d $run_args $image $cmd_args
        ;;
    k | kill)
        docker kill $TRAINER_NAME $cmd_args
        ;;
    rm)
        docker kill $TRAINER_NAME
        docker rm $TRAINER_NAME $cmd_args
        ;;
    l | log | logs)
        docker logs -f -n 100 $cmd_args $TRAINER_NAME
        ;;
    e | exec)
        docker exec $TRAINER_NAME $cmd_args 
        ;;
    a | attach)
        docker exec -it $cmd_args $TRAINER_NAME /bin/bash
        ;;
    *)
        echo "Unsupported command \"$cmd\""
        exit 1
esac
