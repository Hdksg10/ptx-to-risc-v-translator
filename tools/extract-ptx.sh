#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")

BUILD_DIR="$PARENT_DIR/build"
PTX_OUTPUT_DIR="$BUILD_DIR/ptx_output"
mkdir -p "$PTX_OUTPUT_DIR"
SKIP_SAMPLES=("binomialOptions" "quasirandomGenerator" "reduction")
echo "Skipping samples: ${SKIP_SAMPLES[@]}"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory does not exist. Please build the project first."
    exit 1
fi
cd "$BUILD_DIR" || { echo "Failed to enter build directory"; exit 1; }

SAMPLES_DIR="$BUILD_DIR/test/cuda"
if [ ! -d "$SAMPLES_DIR" ]; then
    echo "Samples directory does not exist. Please build the samples first."
    exit 1
fi
echo "Samples Directory: $SAMPLES_DIR"
for SAMPLE_DIR in "$SAMPLES_DIR"/*; do
    if [ -d "$SAMPLE_DIR" ]; then
        SAMPLE_NAME=$(basename "$SAMPLE_DIR")
        if [[ " ${SKIP_SAMPLES[@]} " =~ " ${SAMPLE_NAME} " ]]; then
            echo "Skipping test: $SAMPLE_NAME"
            continue
        fi
        PTX_FILE="$PTX_OUTPUT_DIR/${SAMPLE_NAME}.ptx"
        SAMPLE_PROGRAM="test_$SAMPLE_NAME"
        cd "$SAMPLE_DIR" 
        cuobjdump --dump-ptx ./$SAMPLE_PROGRAM | awk '
            BEGIN { keep=0 }
            /^Fatbin ptx code:/ { keep=0 }
            /^$/ { if (keep==1) print "" }
            /^\/\*/ { if (keep==1) print $0 }
            /^arch =/ { next }
            /^code version =/ { next }
            /^host =/ { next }
            /^compile_size =/ { next }
            /^compressed$/ { next }
            /^[=]+$/ { next }
            /^Fatbin elf code:/ { keep=0 }
            /^Fatbin ptx code:/ { keep=1; next }
            { if (keep==1) print $0 }
        ' > "$PTX_FILE"
        LINE_COUNT=$(wc -l < "$PTX_FILE")
        echo "PTX line count for $SAMPLE_NAME: $LINE_COUNT"
    fi
done