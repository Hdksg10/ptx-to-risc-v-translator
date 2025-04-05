cmake_minimum_required(VERSION 3.18)

project(scalarProd LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)
add_custom_target(${DIR_NAME}_compile_cuda ALL
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${SAMPLE}/scalarProd.cu ${SAMPLE}/scalarProd_cpu.cpp -o samples_${DIR_NAME} -arch=sm_35 -I${SAMPLES_INCLUDE_DIR} -I${SAMPLE} -lcudart -O0
    DEPENDS ${SAMPLE}/scalarProd.cu ${SAMPLE}/scalarProd_cpu.cpp
    COMMENT "Compiling CUDA source"
)
