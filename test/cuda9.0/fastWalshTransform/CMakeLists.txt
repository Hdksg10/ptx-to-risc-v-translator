cmake_minimum_required(VERSION 3.18)

project(fastWalshTransform LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)
add_custom_target(${DIR_NAME}_compile_cuda ALL
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${SAMPLE}/fastWalshTransform.cu ${SAMPLE}/fastWalshTransform_gold.cpp -o samples_${DIR_NAME} -arch=sm_35 -I${SAMPLES_INCLUDE_DIR} -I${SAMPLE} -lcudart
    DEPENDS ${SAMPLE}/fastWalshTransform.cu ${SAMPLE}/fastWalshTransform_gold.cpp
    COMMENT "Compiling CUDA source"
)
