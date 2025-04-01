cmake_minimum_required(VERSION 3.18)

project(transpose LANGUAGES CXX CUDA)
# find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
add_custom_target(${DIR_NAME}_compile_cuda ALL
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${SAMPLE}/transpose.cu -o samples_${DIR_NAME} -arch=sm_35 -I${SAMPLES_INCLUDE_DIR} -lcudart
    DEPENDS ${SAMPLE}/transpose.cu
    COMMENT "Compiling CUDA source"
)
