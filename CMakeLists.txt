cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ptx-to-risc-v-translator)

set(CMAKE_CXX_STANDARD 17)

# Build Ocelot
set(OCELOT_PATH ${CMAKE_SOURCE_DIR}/deps/3rd/gpuocelot/ocelot)
set(OCELOT_BUILD_PATH ${OCELOT_PATH}/build)
add_subdirectory(${OCELOT_PATH}, ${OCELOT_BUILD_PATH}) 

# Build cuda-risc-v api

file(GLOB CUDA_SRC_FILES 
    "src/cuda.cpp"
    "src/cuda_runtime.cpp"
    "src/EmulatedCUDADevice.cpp"
    "src/CUDAContext.cpp"
    "src/CUDAFunction.cpp"
    "src/CUDAModule.cpp"
    "src/Interface.cpp"
    )
add_library(cuda_riscv SHARED ${CUDA_SRC_FILES})
target_include_directories(cuda_riscv PUBLIC include include/cuda/include)
target_include_directories(cuda_riscv PUBLIC ${OCELOT_PATH}/include)
target_link_libraries(cuda_riscv PRIVATE dl pthread)

target_link_libraries(cuda_riscv PRIVATE
    gpuocelot_analysis
	gpuocelot_api
	gpuocelot_cal
	gpuocelot_executive
	gpuocelot_ir
	gpuocelot_parser
	gpuocelot_trace
	gpuocelot_transforms
	gpuocelot_translator
	gpuocelot_util
)
# Build cricket server
set(CRICKET_PATH ${CMAKE_SOURCE_DIR}/deps/3rd/cricket)

add_custom_target(
    cricket_server ALL
    COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CRICKET_PATH} server
    WORKING_DIRECTORY ${CRICKET_PATH}
    COMMENT "Building subdirectory using Makefile"
)
# copy cricket server binary to build directory
add_custom_command(
    TARGET cricket_server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CRICKET_PATH}/cpu/cricket-rpc-server ${CMAKE_BINARY_DIR}/cricket-rpc-server
)
