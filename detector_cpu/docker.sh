#!/usr/bin/env bash
# Build, run, stop, ... the docker container for the YOLOv5 CPU detector node.
# What differs about this sub-project is in docker.conf; the rest is scripts/node-docker.sh.
# .env supplies credentials and local overrides -- see README.md.
# HOST_PORT=<port on this machine, mapped to the container's port 80> (default: 8004)
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
. ./.env || echo "you should provide an .env file for the detector"
. ./docker.conf
. ../scripts/node-docker.sh "$@"
