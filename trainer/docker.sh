#!/usr/bin/env bash

# This script is used to build, run, stop, kill, attach, etc. the docker container for the trainer node.

# .env may contain (for details see readme.md):
# USERNAME=<loop username>
# PASSWORD=<loop password>
# HOST=<host ip or url>
# TRAINER_NAME=dennis_test_trainer
# LINKLL=<FALSE/TRUE> (default: FALSE)
# UVICORN_RELOAD=<FALSE/TRUE/0/1> (default: FALSE)
# RESTART_AFTER_TRAINING=<FALSE/TRUE/0/1> (default: FALSE)
# KEEP_OLD_TRAININGS=<FALSE/TRUE/0/1> (default: FALSE)


if [ $# -eq 0 ]
then
    echo "Usage:"
    echo
    echo "  `basename $0` (b | build)            Build or rebuild"
    echo "  `basename $0` (bnc | build-no-cache) Build or rebuild without cache"
    echo "  `basename $0` (p | push)             Push image"
    echo "  ------------------------------"
    echo "  `basename $0` (r | run)              Run"
    echo "  `basename $0` (u | up)               Start detached"
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

if [ "$2" = "test_latest" ]; then
    image="zauberzeug/yolov5-trainer:latest"
else
    image="zauberzeug/yolov5-trainer:$SEMANTIC_VERSION-nlv$NODE_LIB_VERSION"
fi


# ========================== RUN CONFIGURATION =========================================

# sourcing .env file to get configuration (see README.md)
. .env || echo "you should provide an .env file for the trainer"

run_args="-it" 
#run_args+=" -v $(pwd)/:/app/"
run_args+=" -v $HOME/trainer_nodes_data:/data"
run_args+=" -h ${HOSTNAME}_DEV"
run_args+=" -e HOST=$HOST -e USERNAME=$USERNAME -e PASSWORD=$PASSWORD -e LOOP_SSL_CERT_PATH=$LOOP_SSL_CERT_PATH"
run_args+=" -e BATCH_SIZE=$BATCH_SIZE -e UVICORN_RELOAD=$UVICORN_RELOAD -e KEEP_OLD_TRAININGS=$KEEP_OLD_TRAININGS"
run_args+=" -e NODE_TYPE=trainer -e YOLOV5_MODE=$YOLOV5_MODE -e RESTART_AFTER_TRAINING=$RESTART_AFTER_TRAINING -e TRAINER_IDLE_TIMEOUT_SEC=$TRAINER_IDLE_TIMEOUT_SEC"
run_args+=" -e USE_BACKDOOR_CONTROLS=$USE_BACKDOOR_CONTROLS"
run_args+=" --name $TRAINER_NAME"
run_args+=" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --gpus all"
run_args+=" --ipc host"
run_args+=" -p 7443:80"

# Link Learning Loop Node library if requested
if [ "$LINKLL" == "TRUE" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    run_args+=" -v $SCRIPT_DIR/../../learning_loop_node/learning_loop_node:/uv_env/.venv/lib/python3.12/dist-packages/learning_loop_node"
    echo "Linked Learning Loop from $SCRIPT_DIR/../../learning_loop_node"
fi

# ========================== COMMAND EXECUTION =========================================

cmd=$1
if [ "$2" = "test_latest" ]; then
    cmd_args=${@:3}
else
    cmd_args=${@:2}
fi
case $cmd in
    b | build)
        docker build . -t $image $cmd_args
        ;;
    bnc | build-no-cache)
        docker build --no-cache . -t $image $cmd_args
        ;;
    p | push)
        docker push $image
        ;;
    r | run)
        docker run -it $run_args $image $cmd_args
        ;;
    u | up)
        docker run -d  --restart always $run_args $image $cmd_args
        ;;
    s | stop)
        docker stop $TRAINER_NAME $cmd_args
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
