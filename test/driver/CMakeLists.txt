cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(driver-test)
set(CMAKE_CXX_STANDARD 17)

file(GLOB SOURCES "*.cpp")

set(PTX_DIR "${CMAKE_SOURCE_DIR}/test/driver/ptx")
# copy ptx files to build directory
add_custom_target(post_build_cp ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${PTX_DIR} ${CMAKE_BINARY_DIR}/test/driver
    COMMENT "Copying PTX file to build directory"
)


# set(DRIVER_TEST_INCLUDE_DIR )
foreach(SOURCE ${SOURCES})
    message(STATUS "Adding source: ${SOURCE}")
    get_filename_component(TEST_NAME ${SOURCE} NAME_WE)
    add_executable(driver_test_${TEST_NAME} ${SOURCE})
    target_include_directories(driver_test_${TEST_NAME} PUBLIC ${CUDA_INCLUDE_DIR})
    target_link_libraries(driver_test_${TEST_NAME} PRIVATE cuda_riscv)
endforeach()
