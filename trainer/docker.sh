#!/usr/bin/env bash
# Build, run, stop, ... the docker container for the YOLOv5 trainer node.
# What differs about this sub-project is in docker.conf; the rest is scripts/node-docker.sh.
# .env supplies credentials and local overrides -- see README.md.
# Every variable in .env reaches the container, so a setting declared in main.py is set by
# naming it here -- no list to keep in step. LOOP_HOST and HOST are both accepted.
# CONTAINER_NAME=<name for `docker ps`, `docker stop`, ...> -- host-side only. The name the
#   Learning Loop shows is built in main.py from the container hostname, not from this.
# HOST_PORT=<port on this machine, mapped to the container's port 80> (default: 7443)
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
. ./.env || echo "you should provide an .env file for the trainer"
. ./docker.conf
. ../scripts/node-docker.sh "$@"
