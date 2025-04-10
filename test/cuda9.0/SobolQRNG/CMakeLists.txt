cmake_minimum_required(VERSION 3.18)

project(SobolQRNG LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)
add_custom_target(${DIR_NAME}_compile_cuda ALL
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${SAMPLE}/sobol_gpu.cu ${SAMPLE}/sobol_gold.cpp ${SAMPLE}/sobol_primitives.cpp ${SAMPLE}/sobol.cpp -o samples_${DIR_NAME} -arch=sm_52 -I${SAMPLES_INCLUDE_DIR} -I${SAMPLE} -lcudart
    DEPENDS ${SAMPLE}/sobol_gpu.cu ${SAMPLE}/sobol_gold.cpp ${SAMPLE}/sobol_primitives.cpp ${SAMPLE}/sobol.cpp
    COMMENT "Compiling CUDA source"
)
