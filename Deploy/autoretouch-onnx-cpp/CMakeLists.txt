cmake_minimum_required(VERSION 3.1)
project(autoretouch-deploy)

# 设置输出目录
# set(EXECUTABLE_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/bin)
# set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)

find_package( OpenCV REQUIRED )

# 添加头文件路径
include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories( ${PROJECT_SOURCE_DIR}/3rdparty/onnxruntime-linux-x64-1.10.0/include)

# 添加库文件路径
link_directories( ${PROJECT_SOURCE_DIR}/3rdparty/onnxruntime-linux-x64-1.10.0/lib)

# 添加可执行文件或库
add_executable( autoretouch src/main.cpp src/autoretouch.h src/autoretouch.cpp)

# 链接库
target_link_libraries( autoretouch libonnxruntime.so ${OpenCV_LIBS})

# message(STATUS "${OpenCV_INCLUDE_DIRS}")
# message(STATUS "${OpenCV_LIBS}")
message(STATUS "${PROJECT_SOURCE_DIR}") # 源代码路径
message(STATUS "${CMAKE_BINARY_DIR}") # cmake时路径
message(STATUS "${CMAKE_INSTALL_PREFIX}") # 安装路径

# 运行时加载路径
# set(CMAKE_INSTALL_RPATH /home/duzongwei/CppWorkPlace/CppProject/Deploy/srcnn-onnx-cpp/lib)
# set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# 安装规则
# set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
# message(STATUS "${CMAKE_INSTALL_PREFIX}") # 安装路径

# install(TARGETS srcnn DESTINATION bin)
# install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/onnxruntime-linux-x64-1.10.0/lib/libonnxruntime.so.1.10.0 DESTINATION lib)
# install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/opencv4/lib/libopencv_imgcodecs.so.407 DESTINATION lib)
# install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/opencv4/lib/libopencv_core.so.407 DESTINATION lib)
