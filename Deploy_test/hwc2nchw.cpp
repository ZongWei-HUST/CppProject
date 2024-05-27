#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"


int main(){
    cv::Mat image = cv::imread("face.png");
    if (image.empty()) {
        std::cerr << "Failed to read image file." << std::endl;
        return -1;
    }

    // Convert image from OpenCV's default BGR format to RGB format
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Resize image to the desired size
    cv::Size size(256, 256);
    cv::resize(image, image, size);

    // Normalize pixel values to range [0, 1]
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);

    // HWC to NCHW
    std::vector<float> input_data(image.total() * image.channels());
    int channels = image.channels();
    int height = image.rows;
    int width = image.cols;
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                input_data[c * height * width + h * width + w] = image.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    // Create input tensor
    std::vector<int64_t> input_shape = {1, channels, height, width};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault), input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

}


