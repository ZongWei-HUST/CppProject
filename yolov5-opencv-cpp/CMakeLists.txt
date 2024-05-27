cmake_minimum_required(VERSION 3.10)
project(facedDetector)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..)

find_package(OpenCV 4 REQUIRED)
message("OpenCV version : ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

add_executable(facedDetector yolo.cpp)

target_link_libraries(facedDetector ${OpenCV_LIBS})
