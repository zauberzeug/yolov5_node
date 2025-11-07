#!/usr/bin/env bash

if [[ $1 = "debug" ]]; then
   /app/.venv/bin/python -m debugpy --listen 5678 /app/main.py
elif [[ $1 = "catchsegv" ]]; then
   catchsegv /app/.venv/bin/python /app/main.py
else
   /app/.venv/bin/python /app/main.py
fi