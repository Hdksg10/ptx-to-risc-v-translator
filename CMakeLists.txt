cmake_minimum_required(VERSION 3.10)
project(ptx-to-risc-v-translator)

set(CMAKE_CXX_STANDARD 17)

# Build cuda-risc-v api

file(GLOB CUDA_SRC_FILES 
    "src/cuda.cpp"
    "src/cuda_driver.cpp"
    "src/EmulatedCUDADevice.cpp"
    "src/CUDAContext.cpp"
    )
add_library(cuda_riscv SHARED ${CUDA_SRC_FILES})
target_include_directories(cuda_riscv PUBLIC include include/cuda/include)

# Build cricket server
set(CRICKET_SUBDIR "deps/3rd/cricket")

add_custom_target(
    cricket_server ALL
    COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_SOURCE_DIR}/${CRICKET_SUBDIR} server
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/${CRICKET_SUBDIR}
    COMMENT "Building subdirectory using Makefile"
)
