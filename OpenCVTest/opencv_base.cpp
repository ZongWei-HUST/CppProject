#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

int main(){
    Mat a(5, 5, CV_8UC1, Scalar(4, 5, 6));
    Mat b(5, 5, CV_8UC2, Scalar(4, 5, 6));
    Mat c(5, 5, CV_8UC3, Scalar(4, 5, 6));
    std::cout << c << std::endl;
    std::cout << c.cols << std::endl;
    std::cout << c.rows << std::endl;
    std::cout << c.step[0] << std::endl;
    std::cout << c.step[1] << std::endl;
    std::cout << c.step[2] << std::endl;
    std::cout << c.step << std::endl;
    cout << (double) *(c.data + c.step[0] * 2 + c.step[1] * 2 + 3) << endl;
    cout << "-----------------------" << endl;

    std::cout << c.elemSize() << std::endl;
    std::cout << c.total() << std::endl;
    std::cout << c.channels() << std::endl;

    cout << "-----------------------" << endl;
    // std::cout << b << std::endl;
    // std::cout << c << std::endl;
    auto va1 = (int) a.at<uchar>(0, 0);
    auto vc3 = c.at<cv::Vec3b>(0, 0);
    int first = (int) vc3.val[0];
    cout << va1 << endl;
    cout << vc3 << endl;
    cout << first << endl;
}
