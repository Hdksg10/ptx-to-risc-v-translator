cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ptx-to-risc-v-translator)

set(CMAKE_CXX_STANDARD 17)
option(DEBUG_BUILD "BUILD DEBUG VERSION" OFF)
if (DEBUG_BUILD)
    set(CMAKE_BUILD_TYPE Debug)
endif()
option(BUILD_CRICKET "BUILD CRICKET RPC SERVER" OFF)
option(BUILD_TEST "BUILD TESTS" ON)
option(BUILD_SAMPLES "BUILD CUDA SAMPLES" OFF)
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
set(CUDA_INCLUDE_DIR 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cuda/include)
target_include_directories(cuda_riscv PUBLIC ${CUDA_INCLUDE_DIR})
target_include_directories(cuda_riscv PUBLIC ${OCELOT_PATH}/include)
target_include_directories(cuda_riscv PUBLIC ${FATBIN_PATH})
target_link_libraries(cuda_riscv PRIVATE dl pthread)
target_link_libraries(cuda_riscv PRIVATE fatbin)

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
    file(GLOB TEST_SOURCES "test/*.cpp")
    foreach(SOURCE_FILE ${TEST_SOURCES})
        string(REGEX REPLACE ".*/([^/]+)\\.[^.]+$" "\\1" TEST_NAME ${SOURCE_FILE})
        message(STATUS "Adding test source: ${TEST_NAME}")
        add_executable(${TEST_NAME} ${SOURCE_FILE})
        target_include_directories(${TEST_NAME} PUBLIC include include/cuda/include)
        target_link_libraries(${TEST_NAME} PRIVATE cuda_riscv)
    endforeach()
    
endif()

# Build samples
if (BUILD_SAMPLES)
    # find_package(CUDAToolkit)
    set(SAMPLES_DIR ${CMAKE_SOURCE_DIR}/test/cuda9.0)
    set(SAMPLES_INCLUDE_DIR ${SAMPLES_DIR}/common/inc)
    file(GLOB SAMPLES LIST_DIRECTORIES true ${SAMPLES_DIR}/*)
    foreach(SAMPLE ${SAMPLES})
        if (IS_DIRECTORY ${SAMPLE})
            get_filename_component(DIR_NAME ${SAMPLE} NAME)
            if (DIR_NAME STREQUAL common)
                continue()
            endif()
            string(FIND ${DIR_NAME} "Drv" FOUND_POS)
            if (NOT ${FOUND_POS} EQUAL -1)
                message(STATUS "Processing Drv Directory: ${DIR_NAME}")
            else()
                message(STATUS "Processing Normal Directory: ${DIR_NAME}")
                add_subdirectory(${SAMPLE})
            endif()
        endif()
    endforeach()
endif()