#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")

BUILD_DIR="$PARENT_DIR/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory does not exist. Please build the project first."
    exit 1
fi
cd "$BUILD_DIR" || { echo "Failed to enter build directory"; exit 1; }

TESTS_DIR="$BUILD_DIR/test/driver"
if [ ! -d "$TESTS_DIR" ]; then
    echo "Samples directory does not exist. Please build the samples first."
    exit 1
fi

cd "$TESTS_DIR" || { echo "Failed to enter tests directory"; exit 1; }

for TEST_FILE in driver_test_*; do
    if [ -f "$TEST_FILE" ] && [ -x "$TEST_FILE" ]; then
        echo "Running test: $TEST_FILE"
        ./"$TEST_FILE"
        if [ $? -ne 0 ]; then
            echo "Test $TEST_FILE failed."
            exit 1
        fi
    fi
done