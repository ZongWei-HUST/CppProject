#include <array>
#include <iostream>
#include <algorithm>
using namespace std;

int main(){
    // 初始化
    array<int, 100> a0;  // 值不确定
    array<int, 100> a1{};  // 值=0
    array<int, 100> a2{1, 2, 3};
    array<int, 100> a3 = {1, 2, 3};

    // 访问
    // 1. 智能指针
    for (auto i : a0) cout  << i << endl;
    // 2. 下标访问
    for (int i = 0; i < 4; ++i) cout << a1[i] << endl;
    // 3. 迭代器访问
    auto it = a2.begin();
    for (it; it != a2.end(); ++it) cout << *it << endl;
    // 4. at()函数
    cout << a3.at(0) << endl;
    // 5. get方法
    cout << get<1>(a3) << endl;  // <>中必须为数字,不能为变量

    // 成员函数
    cout << *a3.data() << endl;  // 首个元素的指针
    a0.fill(321);
    for (auto i : a0) cout << i << endl;

    cout << a3.data() << &a3[0] << endl; // 首元素地址相等

    sort(a3.begin(), a3.end());

}