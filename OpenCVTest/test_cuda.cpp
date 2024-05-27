#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::cuda;

int main(int argc, char* argv[]) {
  cv::Mat h_img1 = cv::imread("../images/face.png");
  // Define device variables
  cv::cuda::GpuMat d_result1, d_img1;
  // Upload Image to device
  d_img1.upload(h_img1);

  // Convert image to different color spaces
  cv::cuda::cvtColor(d_img1, d_result1, cv::COLOR_BGR2GRAY);

  cv::Mat h_result1;
  // Download results back to host
  d_result1.download(h_result1);

  cv::imwrite("../images/face_gray.png", h_result1);

  cv::waitKey();
  return 0;
}