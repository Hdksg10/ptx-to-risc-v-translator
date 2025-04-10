cmake_minimum_required(VERSION 3.18)

project(asyncAPI LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)
add_custom_target(${DIR_NAME}_compile_cuda ALL
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${SAMPLE}/asyncAPI.cu -o samples_${DIR_NAME} -arch=sm_52 -I${SAMPLES_INCLUDE_DIR} -lcudart
    DEPENDS ${SAMPLE}/asyncAPI.cu
    COMMENT "Compiling CUDA source"
)
