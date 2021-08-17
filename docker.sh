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

name="yolor_trainer"

compose_args="-it --rm" 
compose_args+=" -v $(pwd)/trainer:/app"
compose_args+=" -v $HOME/data:/data"
compose_args+=" -v $(pwd)/../learning_loop_node/learning_loop_node:/usr/local/lib/python3.8/dist-packages/learning_loop_node"
compose_args+=" -e HOST=$HOST"
compose_args+=" -e USERNAME=$USERNAME -e PASSWORD=$PASSWORD"
compose_args+=" --name $name"
compose_args+=" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all"
compose_args+=" --gpus all"
compose_args+=" --ipc host"

image="zauberzeug/yolor-trainer-node:latest"

build_args="-t $image"
[ -f /etc/nv_tegra_release ] && build_args+=" --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3"
( nvidia-smi > /dev/null 2>&1 ) && build_args+=" --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:21.07-py3"

cmd=$1
cmd_args=${@:2}
case $cmd in
    b | build)
        docker kill $name
        docker rm $name # remove existing container
        docker build . $build_args
        ;;
    d | debug)
        nvidia-docker run $compose_args $image /app/start.sh debug
        ;;
    r | run)
        nvidia-docker run -it $compose_args $image $cmd_args
        ;;
    s | stop)
        docker stop $name $cmd_args
        ;;
    k | kill)
        docker kill $name $cmd_args
        ;;
    d | rm)
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
        docker exec -it $cmd_args darknet_trainer /bin/bash
        ;;
    *)
        echo "Unsupported command \"$cmd\""
        exit 1
esac
