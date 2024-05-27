cmake_minimum_required(VERSION 3.8)
project(project)

find_package(CUDA REQUIRED)

message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_executable(project matrix_multi.cu)
target_link_libraries(project ${CUDA_LIBRARIES})