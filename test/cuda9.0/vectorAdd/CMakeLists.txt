cmake_minimum_required(VERSION 3.18)

project(vectorAdd LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)
add_custom_target(${DIR_NAME}_compile_cuda ALL
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${SAMPLE}/vectorAdd.cu -o samples_${DIR_NAME} -arch=sm_52 -I${SAMPLES_INCLUDE_DIR} -lcudart
    DEPENDS ${SAMPLE}/vectorAdd.cu
    COMMENT "Compiling CUDA source"
)
