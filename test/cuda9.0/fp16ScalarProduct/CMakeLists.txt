cmake_minimum_required(VERSION 3.18)

project(fp16ScalarProduct LANGUAGES CXX CUDA)
# find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
# add_executable(samples_${DIR_NAME} ${SAMPLE}/fp16ScalarProduct.cu)
# set_property(TARGET samples_${DIR_NAME} PROPERTY CUDA_ARCHITECTURES 62)
# target_include_directories(samples_${DIR_NAME} PRIVATE ${SAMPLES_INCLUDE_DIR})
# target_include_directories(samples_${DIR_NAME} PRIVATE ${CUDA_INCLUDE_DIR})
# target_link_libraries(samples_${DIR_NAME} PRIVATE CUDA::cudart)
add_custom_target(${DIR_NAME}_compile_cuda ALL
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${SAMPLE}/fp16ScalarProduct.cu -o ${DIR_NAME}.o -c -arch=sm_62 -I${SAMPLES_INCLUDE_DIR}
    DEPENDS ${SAMPLE}/fp16ScalarProduct.cu
    COMMENT "Compiling CUDA source"
)
add_custom_target(${DIR_NAME}_link_cuda ALL
    COMMAND ${CMAKE_CXX_COMPILER} ${DIR_NAME}.o -o samples_${DIR_NAME} -lcudart
    DEPENDS ${DIR_NAME}_compile_cuda
    COMMENT "Linking CUDA object file"
)

add_dependencies(${DIR_NAME}_link_cuda ${DIR_NAME}_compile_cuda)