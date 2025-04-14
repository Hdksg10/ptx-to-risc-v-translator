#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")

BUILD_DIR="$PARENT_DIR/build"

if [ -z "$1" ]; then
  echo "Usage: $0 <remote_address>"
  exit 1
fi

CRICKET_CLIENT="$HOME/cricket/cpu/cricket-client.so"
REMOTE_ADDRESS="$1"

echo "Running samples on remote address: $REMOTE_ADDRESS"

SKIP_SAMPLES=("binomialOptions" "quasirandomGenerator" "reduction")
echo "Skipping samples: ${SKIP_SAMPLES[@]}"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory does not exist. Please build the project first."
    exit 1
fi
cd "$BUILD_DIR" || { echo "Failed to enter build directory"; exit 1; }

SAMPLES_DIR="$BUILD_DIR/test/cuda9.0"
if [ ! -d "$SAMPLES_DIR" ]; then
    echo "Samples directory does not exist. Please build the samples first."
    exit 1
fi
echo "Running samples in directory: $SAMPLES_DIR"
for SAMPLE_DIR in "$SAMPLES_DIR"/*; do
    if [ -d "$SAMPLE_DIR" ]; then
        SAMPLE_NAME=$(basename "$SAMPLE_DIR")
        if [[ " ${SKIP_SAMPLES[@]} " =~ " ${SAMPLE_NAME} " ]]; then
            echo "Skipping sample: $SAMPLE_NAME"
            continue
        fi
        echo "Running sample: $SAMPLE_NAME"
        SAMPLE_PROGRAM="samples_$SAMPLE_NAME"
        (cd "$SAMPLE_DIR" && REMOTE_GPU_ADDRESS="$REMOTE_ADDRESS" LD_PRELOAD="$CRICKET_CLIENT" ./$SAMPLE_PROGRAM)
    fi
done