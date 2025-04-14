cmake_minimum_required(VERSION 3.18)

project(vectorNrm2Test LANGUAGES CXX CUDA)
# find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
add_custom_target(test_${DIR_NAME}_compile_cuda ALL
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${SAMPLE}/vectorNrm2.cu -o test_${DIR_NAME} -arch=sm_52 -I${SAMPLES_INCLUDE_DIR} -lcudart
    DEPENDS ${SAMPLE}/vectorNrm2.cu
    COMMENT "Compiling CUDA source"
)
