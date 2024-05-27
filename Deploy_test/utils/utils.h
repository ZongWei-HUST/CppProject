#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;

// print like python.
// `inline` : use with multi cpp files include the same .h.
// `#pragma once` : use with one cpp include multi .h files.
inline void printAsPython() { std::cout << std::endl; };

template <typename T, typename... Types>
void printAsPython(const T& fisrtArg, const Types&... args) {
  std::cout << fisrtArg << " ";
  printAsPython(args...);
}

// print cv::Mat as format `mode`.
void printMat(cv::Mat& img, const char* mode = "numpy") {
  if (strcmp(mode, "python") == 0) {
    std::cout << cv::format(img, cv::Formatter::FMT_PYTHON) << ";" << std::endl
              << std::endl;

  } else if (strcmp(mode, "numpy") == 0) {
    std::cout << cv::format(img, cv::Formatter::FMT_NUMPY) << ";" << std::endl
              << std::endl;
  } else {
    std::cout << "mode is incorrect" << std::endl;
  }
}

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
  std::cout << std::endl;
}
