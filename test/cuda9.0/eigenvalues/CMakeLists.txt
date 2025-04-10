cmake_minimum_required(VERSION 3.18)

project(eigenValues LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)
add_custom_target(${DIR_NAME}_compile_cuda ALL
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${SAMPLE}/bisect_large.cu ${SAMPLE}/bisect_small.cu ${SAMPLE}/bisect_util.cu ${SAMPLE}/main.cu ${SAMPLE}/gerschgorin.cpp ${SAMPLE}/matlab.cpp -o samples_${DIR_NAME} -arch=sm_35 -I${SAMPLES_INCLUDE_DIR} -I${SAMPLE} -lcudart
    DEPENDS ${SAMPLE}/bisect_large.cu ${SAMPLE}/bisect_small.cu ${SAMPLE}/bisect_util.cu ${SAMPLE}/main.cu ${SAMPLE}/gerschgorin.cpp ${SAMPLE}/matlab.cpp
    COMMENT "Compiling CUDA source"
)
add_custom_command(TARGET ${DIR_NAME}_compile_cuda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${SAMPLE}/eigenvalues.dat ${CMAKE_BINARY_DIR}/eigenvalues.dat
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${SAMPLE}/data ${CMAKE_BINARY_DIR}/data
    COMMENT "Copying data files to build directory"
)
