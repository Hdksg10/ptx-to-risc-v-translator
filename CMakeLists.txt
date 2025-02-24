cmake_minimum_required(VERSION 3.10)
project(ptx-to-risc-v-translator)

set(CMAKE_CXX_STANDARD 17)

# Build cricket server
set(CRICKET_SUBDIR "deps/3rd/cricket")

add_custom_target(
    build_subdir ALL
    COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_SOURCE_DIR}/${CRICKET_SUBDIR} cpu
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/${CRICKET_SUBDIR}
    COMMENT "Building subdirectory using Makefile"
)