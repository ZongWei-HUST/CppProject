#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"


using namespace std;
using namespace cv;


//显示图片的shape,size方法只适用于二维矩阵
void show_shape(const Mat& img){
    for(int i = 0; i < img.dims; ++i) {
        if(i) std::cout << " x ";
        std::cout << img.size[i];
    }
    std::cout << std::endl;
}

int main() {
    // 1. 创建会话
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test"), "/home/duzongwei/CppWorkPlace/CppProject/Deploy/model/srcnn.onnx", session_options);

    // 2. 获取输入和输出的名称
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
    std::cout << "input_name:" << input_name.get() << std::endl;
    std::cout << "input_name:" << output_name.get() << std::endl;


	auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	std::cout << "input_dims:" << input_dims[0] << std::endl;
	std::cout << "output_dims:" << output_dims[0] << std::endl;

    // // 3. 创建输入和输出的张量
    // vector<int64_t> input_shape = { 1, 3, 720, 1280 };
    // vector<int64_t> output_shape = { 1, 3, 1440, 2560 };
    // Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_shape.data(), input_shape.size());
    // auto output_tensor = Ort::Value::CreateTensor<float>(memory_info, output_shape.data(), output_shape.size());

    // // 4. 使用OpenCV读取图片并进行预处理
    Mat img = imread("/home/duzongwei/CppWorkPlace/CppProject/Deploy/model/face.png");
	Mat blob = dnn::blobFromImage(img, 1., Size(256, 256), Scalar(0, 0, 0), true, false);
    show_shape(img);
    show_shape(blob);


    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    // 这里好像需要转换成array.data()形式
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size());
    cout << input_tensor_ << endl;

    // return 0;
    assert(input_tensor_.IsTensor());

    std::vector<const char*> input_node_names;
    input_node_names.push_back(input_name.get());
    std::vector<const char*> output_node_names{"output"};

    auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor_, 1, output_node_names.data(), 1);

    cout << output_tensor.front() << endl;
    cout << input_name.get() << endl;

    // // 5. 进行推理
    // session.Run(Ort::RunOptions{ nullptr }, input_name, &input_tensor, 1, output_name, &output_tensor, 1);

    // // 6. 将输出的张量转换为OpenCV Mat
    // float* output_data = output_tensor.GetTensorMutableData<float>();
    // cv::Mat output_image(1440, 2560, CV_32FC3, output_data);
    // output_image.convertTo(output_image, CV_8UC3);
    // cv::cvtColor(output_image, output_image, cv::COLOR_RGB2BGR);

    // // 7. 显示输出图片
    // cv::imwrite("/home/duzongwei/CppWorkPlace/CppProject/Deploy/model/face_sr.png", output_image);

  return 0;
}

