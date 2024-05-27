#pragma once

#include <iostream>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class AutoRetouch {
 public:
  AutoRetouch(const char* onnx_path);
  Mat retouch(const char* img_path);

 private:
  int ori_h, ori_w;
  Ort::Env env;
  Ort::SessionOptions session_options;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::Value input_tensor, output_tensor;
  Ort::Session session;

  vector<const char*> input_node_names{"input"};
  vector<const char*> output_node_names{"output"};
  vector<int64_t> input_shape, output_shape;

  Mat get_img(const char* img_path);
  Mat preprocess(cv::Mat& img);
  Mat postprocess(Ort::Value& output_tensor);
};