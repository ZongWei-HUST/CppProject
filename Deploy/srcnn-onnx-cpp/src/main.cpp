#include "srcnn.h"

int main(int argc, char* argv[]) {
  const char* onnx_path = "../models/srcnn.onnx";
  const char* img_path = argv[1];
  // load model
  SRCNN srcnn(onnx_path);
  // get img
  cv::Mat result = srcnn.apply(img_path);
  cv::imwrite("face_result_cpp.png", result);
}