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
name="yolov5_detector_node"

run_args="-it --rm" 
run_args+=" -v $(pwd)/../:/yolov5_node"
run_args+=" -v $HOME/data:/data"
#run_args+=" -v $HOME/learning_loop_node/learning_loop_node:/usr/local/lib/python3.6/dist-packages/learning_loop_node"
run_args+=" -h $HOSTNAME"
run_args+=" -e HOST=n7.zauberzeug.com"
run_args+=" -e ORGANIZATION=zuckerruebe"
run_args+=" -e PROJECT=roboter"
run_args+=" --name $name"
run_args+=" --runtime=nvidia"
run_args+=" -e NVIDIA_VISIBLE_DEVICES=all"
run_args+=" -p 8004:80"


# version discovery borrowed from https://github.com/dusty-nv/jetson-containers/blob/master/scripts/l4t_version.sh
L4T_VERSION_STRING=$(head -n 1 /etc/nv_tegra_release)
L4T_RELEASE=$(echo $L4T_VERSION_STRING | cut -f 2 -d ' ' | grep -Po '(?<=R)[^;]+')
L4T_REVISION=$(echo $L4T_VERSION_STRING | cut -f 2 -d ',' | grep -Po '(?<=REVISION: )[^;]+')
[ "$L4T_REVISION" = "5.1" ] && L4T_REVISION=5.0
L4T_VERSION="$L4T_RELEASE.$L4T_REVISION"

build_args="--build-arg BASE_IMAGE=zauberzeug/l4t-python38-pytorch-trt:$L4T_VERSION"
image="zauberzeug/yolov5-detector:$L4T_VERSION"

cmd=$1
cmd_args=${@:2}
set -x
case $cmd in
    b | build)
	docker pull . ${image}-dev
        docker build . --target release -t $image $build_args $cmd_args
        docker build . -t ${image}-dev $build_args $cmd_args
        ;;
    d | debug)
        docker run $run_args $image-dev /app/start.sh debug
        ;;
    r | run)
        docker run $run_args $image-dev $cmd_args
	;;
    ri | run-image)
    	docker run $run_args $image $cmd_args
        ;;
    p | push)
        docker push ${image}-dev 
        docker push $image
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
