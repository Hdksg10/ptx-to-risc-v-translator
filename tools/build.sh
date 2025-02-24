#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")

BUILD_DIR="$PARENT_DIR/build"
if [ "$1" == "clean" ]; then
    echo "Cleaning build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    echo "Cleaning cricket build directory..."
    (cd $PARENT_DIR/deps/3rd/cricket && make clean)
    echo "Build directory removed."
    exit 0
fi

echo "Building project in directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR" || { echo "Failed to enter build directory"; exit 1; }

cmake .. && make
