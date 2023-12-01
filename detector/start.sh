#!/usr/bin/env bash

if [[ $1 = "debug" ]]; then
   python3 -m debugpy --listen 5678 /app/main.py
elif [[ $1 = "catchsegv" ]]; then
   catchsegv python3 /app/main.py
else
   python3 /app/main.py
fi