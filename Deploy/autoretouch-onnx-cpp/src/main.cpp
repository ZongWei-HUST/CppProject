#include "autoretouch.h"

int main(int argc, char* argv[]) {
  const char* onnx_path = "../models/autoretouch.onnx";
  const char* img_path = argv[1];
  // load model
  AutoRetouch AutoRetouch(onnx_path);
  // get img
  cv::Mat result = AutoRetouch.retouch(img_path);
  cv::imwrite("../assets/HDF_result_cpp_onnx.png", result);
}