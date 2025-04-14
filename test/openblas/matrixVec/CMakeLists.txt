cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(OpenBLAS_matrixVec C)

set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

add_executable(OpenBLAS_matrixVec matrixVec.c)

find_library(OpenBLAS_LIB openblas)

if (OpenBLAS_LIB)
    target_link_libraries(OpenBLAS_matrixVec ${OpenBLAS_LIB})
    message(STATUS "Found OpenBLAS: ${OpenBLAS_LIB}")
else()
    message(FATAL_ERROR "OpenBLAS not found. Please install it and set LIBRARY_PATH if needed.")
endif()
