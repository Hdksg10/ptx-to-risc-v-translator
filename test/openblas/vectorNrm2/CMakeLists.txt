cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(OpenBLAS_vectorNrm2 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(OpenBLAS_vectorNrm2 vectorNrm2.c)

find_library(OpenBLAS_LIB openblas)

if (OpenBLAS_LIB)
    target_link_libraries(OpenBLAS_vectorNrm2 ${OpenBLAS_LIB})
    message(STATUS "Found OpenBLAS: ${OpenBLAS_LIB}")
else()
    message(FATAL_ERROR "OpenBLAS not found. Please install it and set LIBRARY_PATH if needed.")
endif()
