cmake_minimum_required(VERSION 3.18)

project(reduction LANGUAGES CXX CUDA)
# find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
# TODO: implement SHFL instruction
# Manual set __CUDA__ARCH__ macro to avoid SHFL instruction
add_custom_target(${DIR_NAME}_compile_cuda ALL
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${SAMPLE}/reduction.cpp ${SAMPLE}/reduction_kernel.cu -o samples_${DIR_NAME} -arch=sm_35 -D__CUDA_ARCH__=250 -I${SAMPLES_INCLUDE_DIR} -I${SAMPLE} -lcudart
    DEPENDS ${SAMPLE}/reduction_kernel.cu ${SAMPLE}/reduction.cpp
    COMMENT "Compiling CUDA source"
)
