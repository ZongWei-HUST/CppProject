cmake_minimum_required(VERSION 3.1)
project( test )

# SET(CMAKE_CXX_FLAGS "-g")

find_package( OpenCV REQUIRED )

# 添加头文件路径
include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories( /home/duzongwei/CppWorkPlace/CppProject/Deploy_test/onnxruntime-linux-x64-1.10.0/include)

# 添加库文件路径
link_directories( /home/duzongwei/CppWorkPlace/CppProject/Deploy_test/onnxruntime-linux-x64-1.10.0/lib)

# 添加可执行文件或库
add_executable( test1 cpp_onnx_test2.cpp utils/utils.h)

# 链接库
target_link_libraries( test1 libonnxruntime.so ${OpenCV_LIBS})


message(STATUS "${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_LIBS}")