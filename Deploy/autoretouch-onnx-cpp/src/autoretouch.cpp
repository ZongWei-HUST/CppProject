#include "autoretouch.h"

#include <stdexcept>

AutoRetouch::AutoRetouch(const char* onnx_path)
    : input_tensor(nullptr), output_tensor(nullptr), session(nullptr) {
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

Mat AutoRetouch::get_img(const char* img_path) {
  Mat input_img = imread(img_path);
  this->ori_h = input_img.rows;
  this->ori_w = input_img.cols;
  Mat img_rgb, img;
  cvtColor(input_img, img_rgb, COLOR_BGR2RGB);
  resize(img_rgb, img, Size(512, 512), 0, 0, INTER_CUBIC);
  return img;
}

Mat AutoRetouch::preprocess(cv::Mat& img) {
  /*
  transforms.ToTensor(),  # /255, (H, W, C) -> (C, H, W)
  transforms.Normalize(0.5, 0.5),  # -0.5/0.5
  unsqueeze(0)  #  (C, H, W) -> (1, C, H, W)
  */

  // Method 1.
  img.convertTo(img, CV_32FC3);
  img = img / 255.;
  Mat img_trans =
      cv::dnn::blobFromImage(img, 2.0, Size(512, 512), Scalar(0.5, 0.5, 0.5));

  // Method 2:
  // First split: HWC -> NCHW
  //   img.convertTo(img, CV_32FC3);
  //   int height = img.rows;
  //   int width = img.cols;
  //   vector<float> img_trans(img.total() * img.channels());

  //   vector<Mat> channels;
  //   split(img, channels);
  //   Mat R = channels.at(0);
  //   Mat G = channels.at(1);
  //   Mat B = channels.at(2);

  //   R = (R / 255. - 0.5) / 0.5;
  //   G = (G / 255. - 0.5) / 0.5;
  //   B = (B / 255. - 0.5) / 0.5;

  //   int len = R.total() * sizeof(float);
  //   memcpy(img_trans.data(), R.ptr<float>(0), len);
  //   memcpy(img_trans.data() + R.total(), G.ptr<float>(0), len);
  //   memcpy(img_trans.data() + R.total() + G.total(), B.ptr<float>(0), len);

  return img_trans;
}

Mat AutoRetouch::postprocess(Ort::Value& output_tensor) {
  Ort::TypeInfo type_info = output_tensor.GetTypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  auto tensor_shape = tensor_info.GetShape();
  auto tensor_size = tensor_info.GetElementCount();
  auto tensor_data_type = type_info.GetONNXType();
  float* tensor_data = output_tensor.GetTensorMutableData<float>();
  int out_h = tensor_shape[2], out_w = tensor_shape[3];
  //   cout << out_w << out_h;
  Mat result(tensor_shape[2], tensor_shape[3], CV_32FC3);
  for (int i = 0; i < out_h; ++i) {
    for (int j = 0; j < out_w; ++j) {
      int idx = (i * out_w + j);
      float r = tensor_data[idx] + 1.0;
      float g = tensor_data[idx + out_h * out_w] + 1.0;
      float b = tensor_data[idx + out_h * out_w * 2] + 1.0;
      result.at<Vec3f>(i, j) = cv::Vec3f(b, g, r);
    }
  }
  result = result / 2.0 * 255.0;
  result.convertTo(result, CV_8UC3);
  Mat result_ori;
  resize(result, result_ori, Size(this->ori_h, this->ori_w));
  return result_ori;
}

Mat AutoRetouch::retouch(const char* img_path) {
  Mat img = get_img(img_path);
  Mat img_trans = preprocess(img);
  input_tensor = Ort::Value::CreateTensor<float>(
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
      img_trans.ptr<float>(), img_trans.total(), input_shape.data(),
      input_shape.size());
  //   input_tensor = Ort::Value::CreateTensor<float>(
  //       Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
  //       img_trans.data(), img_trans.size(), input_shape.data(),
  //       input_shape.size());

  output_tensor = Ort::Value::CreateTensor<float>(
      allocator, output_shape.data(), output_shape.size());
  session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor,
              1, output_node_names.data(), &output_tensor, 1);
  cv::Mat result = postprocess(output_tensor);
  cout << "apply done!" << endl;
  return result;
}