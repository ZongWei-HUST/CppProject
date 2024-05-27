#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class SRCNN {
 public:
  SRCNN(const char* onnx_path);
  cv::Mat apply(const char* img_path);

 private:
  Ort::Env env;
  Ort::SessionOptions session_options;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::Value input_tensor, output_tensor;
  Ort::Session session;

  vector<const char*> input_node_names{"input"};
  vector<const char*> output_node_names{"output"};
  vector<int64_t> input_shape, output_shape;

  cv::Mat get_img(const char*);
  vector<float> preprocess(cv::Mat& img);
  cv::Mat postprocess(Ort::Value& output_tensor);
};