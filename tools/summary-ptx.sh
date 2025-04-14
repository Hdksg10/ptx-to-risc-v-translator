#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")

BUILD_DIR="$PARENT_DIR/build"


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
echo "Samples Directory: $SAMPLES_DIR"
for SAMPLE_DIR in "$SAMPLES_DIR"/*; do
    if [ -d "$SAMPLE_DIR" ]; then
        SAMPLE_NAME=$(basename "$SAMPLE_DIR")
        if [[ " ${SKIP_SAMPLES[@]} " =~ " ${SAMPLE_NAME} " ]]; then
            echo "Skipping sample: $SAMPLE_NAME"
            continue
        fi
        SAMPLE_PROGRAM="samples_$SAMPLE_NAME"
        cd "$SAMPLE_DIR" 
        LINE_COUNT=$(cuobjdump --dump-ptx ./$SAMPLE_PROGRAM | wc -l)
        echo "PTX line count for $SAMPLE_NAME: $LINE_COUNT"
    fi
done