#!/usr/bin/env bash

if [ $# -eq 0 ]
then
    echo "Usage:"
    echo
    echo "  `basename $0` (d | detect)    [<containers>]      start detecting"
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

name="yolov5_background_detector"

run_args="-it --rm" 
run_args+=" -v $HOME/data:/data"
run_args+=" -e HOST=$HOST"
run_args+=" -h $HOSTNAME"
run_args+=" -e USERNAME=$USERNAME -e PASSWORD=$PASSWORD"
run_args+=" --name $name"
run_args+=" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all"
run_args+=" --gpus all"
run_args+=" --ipc host"

image="zauberzeug/yolov5-trainer:latest"

build_args=""
[ -f /etc/nv_tegra_release ] && build_args+=" --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3"
( nvidia-smi > /dev/null 2>&1 ) && build_args+=" --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:21.10-py3"

cmd=$1
cmd_args=${@:2}
case $cmd in
    d | detect)
        docker run $run_args $image /app/start.sh detect
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
        docker exec -it $cmd_args $name /bin/bash
        ;;
    *)
        echo "Unsupported command \"$cmd\""
        exit 1
esac
