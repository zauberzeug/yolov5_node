#!/usr/bin/env bash
# Build, run, stop, ... the docker container for the YOLOv5 trainer node.
# What differs about this sub-project is in docker.conf; the rest is scripts/node-docker.sh.
# .env supplies credentials and local overrides -- see README.md.
# LOOP_HOST / LOOP_USERNAME / LOOP_PASSWORD / LOOP_ORGANIZATION / LOOP_PROJECT may also be
# written without the LOOP_ prefix; either spelling is accepted.
# HOST_PORT=<port on this machine, mapped to the container's port 80> (default: 7443)
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
. ./.env || echo "you should provide an .env file for the trainer"
. ./docker.conf
. ../scripts/node-docker.sh "$@"
