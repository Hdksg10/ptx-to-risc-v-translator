cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ptx-to-risc-v-translator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
option(DEBUG_BUILD "BUILD DEBUG VERSION" ON)
if (DEBUG_BUILD)
    set(CMAKE_BUILD_TYPE Debug)
endif()
option(BUILD_CRICKET "BUILD CRICKET RPC SERVER" OFF)
option(BUILD_TEST "BUILD TESTS" ON)
# Build Ocelot
set(OCELOT_PATH ${CMAKE_SOURCE_DIR}/deps/3rd/gpuocelot/ocelot)
set(OCELOT_BUILD_PATH ${OCELOT_PATH}/build)
add_subdirectory(${OCELOT_PATH} ${OCELOT_BUILD_PATH}) 

# Build fatbin decompression
set(FATBIN_PATH ${CMAKE_SOURCE_DIR}/deps/3rd/decompression)
set(FATBIN_BUILD_PATH ${FATBIN_PATH}/build)
add_subdirectory(${FATBIN_PATH} ${FATBIN_BUILD_PATH}) 

# set(OCELOT_LIB     
#     gpuocelot_analysis
#     gpuocelot_api
#     gpuocelot_cal
#     gpuocelot_executive
#     gpuocelot_ir
#     gpuocelot_parser
#     gpuocelot_trace
#     gpuocelot_transforms
#     gpuocelot_translator
#     gpuocelot_util
#     )
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
target_include_directories(cuda_riscv PUBLIC ${FATBIN_PATH})
target_link_libraries(cuda_riscv PRIVATE dl pthread)
target_link_libraries(cuda_riscv PRIVATE fatbin)

# target_link_libraries(cuda_riscv PRIVATE ${OCELOT_LIB})
target_link_libraries(cuda_riscv PRIVATE
gpuocelot
)
# Build cricket server
set(CRICKET_PATH ${CMAKE_SOURCE_DIR}/deps/3rd/cricket)
if (BUILD_CRICKET)
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
endif()

# Build test
if (BUILD_TEST)
    file(GLOB_RECURSE TEST_SOURCES "test/*.cpp")
    add_executable(test ${TEST_SOURCES})
    target_include_directories(test PUBLIC include include/cuda/include)
    # target_link_libraries(test ${OCELOT_LIB})
    # target_link_libraries(test PRIVATE
    #         gpuocelot_analysis
    #         gpuocelot_api
    #         gpuocelot_cal
    #         gpuocelot_executive
    #         gpuocelot_ir
    #         gpuocelot_parser
    #         gpuocelot_trace
    #         gpuocelot_transforms
    #         gpuocelot_translator
    #         gpuocelot_util
    #     )
    target_link_libraries(test PRIVATE cuda_riscv)

endif()