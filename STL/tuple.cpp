#include <tuple>
#include <string>
#include <iostream>
#include <algorithm>
// using namespace std;


int main() {
    // 初始化
    std::tuple<int, int, std::string> t1;
    t1 = std::make_tuple(1, 1, "dzw");
    std::tuple<int, int> t2(1, 2);
    std::tuple<int, int> t3{3, 4};
    auto p = std::make_pair("str", 1);
    std::tuple<std::string, int> t4 {p};

    // 访问元素
    int first = std::get<0>(t3);
    std::cout << first << std::endl;
    std::get<0>(t3) = 1;
    std::cout << std::get<0>(t3) << std::endl;

    // 获得元素个数
    std::cout << std::tuple_size<decltype(t3)>::value << "\n";

    // tile解包获取元素值
    int one, two;
    std::tie(one, two) = t3;
    std::cout << one << " " << two << "\n";
}
