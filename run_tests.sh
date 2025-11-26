#!/bin/bash
# shell script to run all tests

# source local .env
set -o allexport; source .env; set +o allexport
# Check if argument is provided
if [ $# -eq 1 ]; then
    # Run tests with filter
    python -m pytest trainer -v -s -k "$1"
    python -m pytest detector -v -s -k "$1" 
    python -m pytest detector_cpu -v -s -k "$1"
    exit 0
fi


# Run the tests
cd trainer && uv run python -m pytest -v
# python -m pytest detector -v
# python -m pytest detector_cpu -v