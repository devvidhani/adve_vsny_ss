#!/bin/bash

COMMAND=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -e|--env) CONDA_ENV_NAME="$2"; shift 2 ;;
        -p|--path) CONDA_ACTIVATE_PATH="$2"; shift 2 ;;
        *) COMMAND+=("$1"); shift ;;
    esac
done

# Debug prints
echo "CONDA_ENV_NAME: $CONDA_ENV_NAME"
echo "CONDA_ACTIVATE_PATH: $CONDA_ACTIVATE_PATH"

# Activate the conda environment
source "$CONDA_ACTIVATE_PATH" "$CONDA_ENV_NAME"

# Run the command passed to this script
"${COMMAND[@]}"
# "$@"

