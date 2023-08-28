#!/usr/bin/env bash

if [ $# -eq 0 ]
then
    echo "Usage:"
    echo
    echo "  `basename $0` (b | build)   [<containers>]      Build or rebuild"
    echo "  `basename $0` (u | up)      [<containers>]      Create and start"
    echo "  `basename $0` (U | upbuild) [<containers>]      Create and start (force build)"
    echo "  `basename $0` (d | down)    [<containers>]      Stop and remove"
    echo "  `basename $0` (s | start)   [<containers>]      Start"
    echo "  `basename $0` (r | restart) [<containers>]      Restart"
    echo "  `basename $0` (h | stop)    [<containers>]      Stop (halt)"
    echo "  `basename $0` (k | kill)    [<containers>]      Kill"
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
. .env || echo "you should provide an .env file with USERNAME and PASSWORD for the Learning Loop"

name="yolov5_trainer_node"

run_args="-it --rm" 
run_args+=" -v $(pwd)/../:/yolov5_node/"
run_args+=" -v $HOME/data:/data"
run_args+=" -v $HOME/learning_loop_node/learning_loop_node:/opt/conda/lib/python3.8/site-packages/learning_loop_node"
run_args+=" -v $HOME/learning_loop_node:/learning_loop_node"
run_args+=" -v $HOME/.vscode-server:/root/.vscode-server"
run_args+=" -e HOST=$HOST"
run_args+=" -h ${HOSTNAME}_DEV"
run_args+=" -e USERNAME=$USERNAME -e PASSWORD=$PASSWORD"
run_args+=" -e BATCH_SIZE=$BATCH_SIZE"
run_args+=" -e NODE_TYPE=trainer"
run_args+=" --name $name"
run_args+=" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all"
run_args+=" --gpus all"
run_args+=" --ipc host"
run_args+=" -p 7442:80"


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
        docker run -it $run_args $image $cmd_args
        ;;
    ri | run-image)
        docker run -it $run_args $image $cmd_args
        ;;
    s | stop)
        docker stop $name $cmd_args
        ;;
    k | kill)
        docker kill $name $cmd_args
        ;;
    rm)
        docker kill $name
        docker rm $name $cmd_args
        ;;
    l | log | logs)
        docker logs -f --tail 100 $cmd_args $name
        ;;
    e | exec)
        docker exec $name $cmd_args 
        ;;
    a | attach)
        docker exec -it $cmd_args $name /bin/bash
        ;;
    *)
        echo "Unsupported command \"$cmd\""
        exit 1
esac
