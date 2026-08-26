#!/usr/bin/env bash
# NOTE this file is duplicated verbatim across the node repositories. Nothing enforces that
# today, so a change here has to be copied to the other two by hand.
# Shared driver for every node sub-project's docker.sh.
#
# Each sub-project has a 5-line docker.sh that sources .env, then its docker.conf, then this
# file. The conf declares what differs about that sub-project; everything here is what does not.
#
# Required in docker.conf:  IMAGE_REPO, CONTAINER_NAME
# Optional:                 IMAGE_TAG_SUFFIX, TEST_LATEST_TAG, DATA_VOLUME, HOST_PORT,
#                           CONTAINER_PORT, BUILD_CONTEXT, DOCKERFILE, GPU, IPC_HOST, ENV_FILE,
#                           EXTRA_RUN_ARGS, EXTRA_BUILD_ARGS, LINKLL_SITE_PACKAGES
#                           and a select_image function for host-dependent image choices
#
# NOTE the conf is sourced before this file, so it cannot reference SEMANTIC_VERSION or
# NODE_LIB_VERSION. The tag convention A.B.C-nlvX.Y.Z is shared, so it is applied here; a conf
# contributes only IMAGE_REPO and IMAGE_TAG_SUFFIX.

set -u

: "${IMAGE_REPO:?docker.conf must set IMAGE_REPO}"
: "${CONTAINER_NAME:?docker.conf must set CONTAINER_NAME}"
: "${IMAGE_TAG_SUFFIX:=}"
: "${CONTAINER_PORT:=80}"
: "${BUILD_CONTEXT:=.}"
: "${DOCKERFILE:=}"
: "${GPU:=FALSE}"
: "${IPC_HOST:=FALSE}"
: "${ENV_FILE:=FALSE}"
: "${TEST_LATEST_TAG:=}"

if [ $# -eq 0 ]; then
    self=$(basename "$0")
    cat <<USAGE
Usage:

  $self (b | build)            Build or rebuild
  $self (bnc | build-no-cache) Build or rebuild without cache
  $self (p | push)             Push image
  ------------------------------
  $self (r | run)              Run
  $self (u | up)               Start detached
  $self (s | stop)             Stop
  $self (k | kill)             Kill
  $self (rm)                   Kill and remove

  $self (l | log)              Show log tail (last 100 lines)
  $self (e | exec) <command>   Execute command
  $self (a | attach)           Attach to container with shell

Arguments:
  command       Command to be executed inside a container
USAGE
    exit
fi

# ========================== BUILD CONFIGURATION / IMAGE SELECTION =======================
# The version pair every image tag is built from: A.B.C-nlvX.Y.Z
# NOTE sed, not `grep -oP`: -P is a GNU extension, so on a host whose PATH finds BSD grep
# first both lookups failed silently and every image was tagged ":-nlv".
SEMANTIC_VERSION=$(sed -n 's/^version[[:space:]]*=[[:space:]]*"\([0-9.]*\)".*/\1/p' pyproject.toml | head -1)
NODE_LIB_VERSION=$(sed -n 's/.*learning_loop_node==\([0-9.]*\).*/\1/p' pyproject.toml | head -1)
: "${SEMANTIC_VERSION:?no version found in pyproject.toml}"
: "${NODE_LIB_VERSION:?no learning_loop_node== pin found in pyproject.toml}"

# A conf whose image depends on the host (base image, Jetson vs cloud) defines select_image,
# which runs here -- after the versions exist -- and may set IMAGE_TAG_SUFFIX,
# LINKLL_SITE_PACKAGES or EXTRA_BUILD_ARGS.
if declare -F select_image >/dev/null; then
    select_image
fi

if [ -n "$TEST_LATEST_TAG" ] && [ "${2:-}" = "test_latest" ]; then
    image="$IMAGE_REPO:$TEST_LATEST_TAG"
else
    image="$IMAGE_REPO:$SEMANTIC_VERSION-nlv$NODE_LIB_VERSION$IMAGE_TAG_SUFFIX"
fi

# ========================== RUN CONFIGURATION =========================================
# NOTE .env is sourced by docker.sh, before docker.conf, so the conf can use its values.

run_args="-it"
[ -n "${DATA_VOLUME:-}" ] && run_args+=" -v $DATA_VOLUME:/data"
run_args+=" -h ${HOSTNAME}_DEV"
[ "$ENV_FILE" = "TRUE" ] && [ -f .env ] && run_args+=" --env-file .env"

run_args+=" --name $CONTAINER_NAME"
[ "$GPU" = "TRUE" ] && run_args+=" --device=nvidia.com/gpu=all"
[ "$IPC_HOST" = "TRUE" ] && run_args+=" --ipc host"
[ -n "${HOST_PORT:-}" ] && run_args+=" -p $HOST_PORT:$CONTAINER_PORT"
run_args+="${EXTRA_RUN_ARGS:-}"

# Link Learning Loop Node library if requested
if [ "${LINKLL:-FALSE}" == "TRUE" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[1]}" )" &> /dev/null && pwd )"
    run_args+=" -v $SCRIPT_DIR/../../learning_loop_node/learning_loop_node:${LINKLL_SITE_PACKAGES:?docker.conf must set LINKLL_SITE_PACKAGES}/learning_loop_node"
    echo "Linked Learning Loop from $SCRIPT_DIR/../../learning_loop_node"
fi

# ========================== COMMAND EXECUTION =========================================

cmd=$1
if [ -n "$TEST_LATEST_TAG" ] && [ "${2:-}" = "test_latest" ]; then
    cmd_args=${@:3}
else
    cmd_args=${@:2}
fi

build_args="$BUILD_CONTEXT"
[ -n "$DOCKERFILE" ] && build_args+=" -f $DOCKERFILE"

case $cmd in
    b | build)              docker build $build_args -t $image ${EXTRA_BUILD_ARGS:-} $cmd_args ;;
    bnc | build-no-cache)   docker build --no-cache $build_args -t $image ${EXTRA_BUILD_ARGS:-} $cmd_args ;;
    p | push)               docker push $image ;;
    r | run)                docker run $run_args $image $cmd_args ;;
    u | up)                 docker run -d  --restart always $run_args $image $cmd_args ;;
    s | stop)               docker stop $CONTAINER_NAME $cmd_args ;;
    k | kill)               docker kill $CONTAINER_NAME $cmd_args ;;
    rm)                     docker kill $CONTAINER_NAME
                            docker rm $CONTAINER_NAME $cmd_args ;;
    l | log | logs)         docker logs -f -n 100 $cmd_args $CONTAINER_NAME ;;
    e | exec)               docker exec $CONTAINER_NAME $cmd_args ;;
    a | attach)             docker exec -it $cmd_args $CONTAINER_NAME /bin/bash ;;
    *)                      echo "Unsupported command \"$cmd\""; exit 1 ;;
esac
