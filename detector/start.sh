#!/usr/bin/env bash

if [[ $1 = "debug" ]]; then
   python3 -m debugpy --listen 5678 /app/main.py
else
   catchsegv python3 /app/main.py
fi