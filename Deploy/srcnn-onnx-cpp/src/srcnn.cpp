#include "srcnn.h"

#include <stdexcept>

SRCNN::SRCNN(const char* onnx_path)
    : input_tensor(nullptr), output_tensor(nullptr), session(nullptr) {
  cout << "user net: " << onnx_path << endl;
  env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session = Ort::Session(env, onnx_path, session_options);
  // Get the input and output shapes and types
  Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
  Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
  auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
  auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
  input_shape = input_tensor_info.GetShape();
  output_shape = output_tensor_info.GetShape();
  ONNXTensorElementDataType input_type = input_tensor_info.GetElementType();
  ONNXTensorElementDataType output_type = output_tensor_info.GetElementType();
}

cv::Mat SRCNN::get_img(const char* img_path) {
  cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
  if (img.empty()) {
    printf("No image data \n");
    throw std::runtime_error("Failed to read image: " + string(img_path));
  }
  return img;
}

vector<float> SRCNN::preprocess(cv::Mat& img) {
  // (H, W, C) -> (N, C, H, W)
  img.convertTo(img, CV_32FC3);
  vector<float> img_trans(img.total() * img.channels());
  int channels = img.channels();
  int height = img.rows;
  int width = img.cols;
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        img_trans[c * height * width + h * width + w] = img.at<Vec3f>(h, w)[c];
      }
    }
  }

  return img_trans;
}

cv::Mat SRCNN::postprocess(Ort::Value& output_tensor) {
  float* output_data = output_tensor.GetTensorMutableData<float>();
  int out_h = output_shape[2], out_w = output_shape[3];
  cv::Mat result(out_h, out_w, CV_32FC3);
  for (int i = 0; i < out_h; ++i) {
    for (int j = 0; j < out_w; ++j) {
      int idx = (i * out_w + j);
      float b = output_data[idx];
      float g = output_data[idx + out_h * out_w];
      float r = output_data[idx + out_h * out_w * 2];
      result.at<Vec3f>(i, j) = cv::Vec3f(b, g, r);
    }
  }
  result.convertTo(result, CV_8UC3);

  return result;
}

cv::Mat SRCNN::apply(const char* img_path) {
  cv::Mat img = get_img(img_path);
  vector<float> img_trans = preprocess(img);
  input_tensor = Ort::Value::CreateTensor<float>(
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
      img_trans.data(), img_trans.size(), input_shape.data(),
      input_shape.size());
  output_tensor = Ort::Value::CreateTensor<float>(
      allocator, output_shape.data(), output_shape.size());
  session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor,
              1, output_node_names.data(), &output_tensor, 1);
  cv::Mat result = postprocess(output_tensor);
  cout << "apply done!" << endl;
  return result;
}