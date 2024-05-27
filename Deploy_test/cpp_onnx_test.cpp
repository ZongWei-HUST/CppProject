#include <iostream>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

// 显示图片的shape,size方法只适用于二维矩阵
void show_shape(const Mat& img) {
  for (int i = 0; i < img.dims; ++i) {
    if (i) std::cout << " x ";
    std::cout << img.size[i];
  }
  std::cout << std::endl;
}

template <class T>
void show_vector(std::vector<T>& v) {
  for (auto&& item : v) {
    std::cout << item << " ";
  }
  std::cout << endl;
}

int main(int argc, char* argv[]) {
  // 1. Load the model
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  Ort::Session session(env, "../model/srcnn.onnx", session_options);

  // 2. Get the input and output names
  std::vector<const char*> input_node_names{"input"};
  std::vector<const char*> output_node_names{"output"};

  // 3. Get the input and output shapes and types
  Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
  Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
  auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
  auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> input_shape = input_tensor_info.GetShape();
  std::vector<int64_t> output_shape = output_tensor_info.GetShape();
  ONNXTensorElementDataType input_type = input_tensor_info.GetElementType();
  ONNXTensorElementDataType output_type = output_tensor_info.GetElementType();

  show_vector(input_shape);
  show_vector(output_shape);
  // std::cout << input_type << endl;
  // Enum(1) : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT

  // 4. Create the input and output tensors
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      allocator, input_shape.data(), input_shape.size());
  cv::Mat img = cv::imread("../model/face.png", cv::IMREAD_UNCHANGED);
  show_shape(img);

  // 自定义变换
  cv::Mat image = cv::dnn::blobFromImage(img, 1., Size(256, 256),
                                         Scalar(0, 0, 0), true, false);

  // show_shape(blob);

  // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  // image.convertTo(image, CV_32F, 1.0f / 255.0f);
  show_shape(image);
  std::memcpy(input_tensor.GetTensorMutableData<float>(), image.data,
              image.total() * image.elemSize());
  // std::cout << "----" << image.total() * image.elemSize() << std::endl;

  Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
      allocator, output_shape.data(), output_shape.size());
  std::cout << *output_shape.data() << " " << output_shape.size() << std::endl;

  std::cout << input_tensor_info.GetElementCount() << endl;
  std::cout << output_tensor_info.GetElementCount() << endl;

  // 5. Run the model
  session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor,
              1, output_node_names.data(), &output_tensor, 1);

  // 6. Get the output tensor and process the result
  float* output_data = output_tensor.GetTensorMutableData<float>();

  cv::Mat result(output_shape[2], output_shape[3], CV_32FC3, output_data);
  result.convertTo(result, CV_8UC3);
  // // result.convertTo(result, CV_8UC1, 255.0);
  cv::imwrite("../face_ort_srcnn.png", result);
  // // cv::waitKey(0);

  return 0;
}
