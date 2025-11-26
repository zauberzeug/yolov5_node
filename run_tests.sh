#!/bin/bash
# shell script to run all tests

# source local .env
set -o allexport; source .env; set +o allexport
# Check if argument is provided
if [ $# -eq 1 ]; then
    # Run tests with filter
    ( cd python && uv run pytest trainer -v -s -k "$1" )
    ( cd detector && uv run python -m pytest -v -s -k "$1" )
    ( cd detector_cpu &&  uv run python -m pytest -v -s -k "$1" )
    exit 0
fi


# Run the tests
# ( cd trainer && uv run python -m pytest -v )
# ( cd detector && uv run python -m pytest -v )
( cd detector_cpu && uv run python -m pytest -v )
