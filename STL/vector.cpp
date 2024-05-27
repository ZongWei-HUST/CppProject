// 头文件
// 定义在命名空间std中
#include "utils.h"
#include <vector>
#include <algorithm>
using namespace std;

class Node{};
int n = 10, m = 5;


// 一维初始化
vector<int> a;  //
vector<Node> b;

vector<int> v(n);  // 长度为n,默认为0
vector<int> v2(n, 1);

vector<int> a1{1, 2, 3};  // 列表初始化
vector<int> b1(a); // 拷贝初始化

// 二维初始化
vector<int> v3[5]; // 一个数组,每个数组保存一个vector类型,行固定
vector<vector<int>> v4;  // 二维数组,行列均可变
vector<vector<int> > a2(n, vector<int>(m, 0));  // n行m列


int main(){
    v.begin(); // 第一个元素的迭代器,或者说地址
    v.end();  // 最后一个元素的后一个位置的迭代器(所有STL容器都是后一个位置)
    v.empty(); // 判断是否为空,空返回true
    v.front(); // 第一个数据
    v.back();  // 最后一个数据
    v.pop_back();  // 删除最后一个数据
    v.push_back(2);  // 尾部增加元素2
    v.size(); // 实际数据个数(unsigned int类型)
    v.clear();  // 清除元素
    v.resize(10, 1);  // 改变数组大小为n,n个空间数值赋为v
    v.insert(v.begin() + 2, 0);  // index=2处插入0
    v.insert(v.end(), 0); // 尾部添加一个0
    v.insert(v.end() - 1, 2);  // index=-1处插入一个2

    vector<int> v1{1,2,3,4,5};
    v1.erase(v1.begin() + 1, v1.end() - 1); // 删除[a, b)的元素, {1,5}
    
    vector<int> v2{2, 5, 4, 3};
    sort(v2.begin(), v2.end());  // {2,3,4,5}
    sort(v2.begin() + 1, v2.end() - 1);  // {2, 4, 5, 3}

    vector<int> vi{1, 2, 3, 4, 5};
    for (int i = 0; i < vi.size(); ++i){
        cout << vi[i];
    }

    // 第一种
    vector<int>::iterator it = vi.begin();
    for (int i = 0; i < vi.size(); ++i){
        cout << *(it + i);
    }
    // 第二种
    vector<int>::iterator it_;
    for (it_ = vi.begin(); it_ != vi.end(); ++it_){
        cout << *it_;
    }
    
    for (auto iter = vi.begin(); iter != vi.end(); ++iter){
        cout << *iter;
    }

    cout << endl << "=======" << endl;
    typedef pair<int, int> pa;

    struct my_cmp
    {
        bool operator()(pa& a, pa& b){
            if (a.first == b.first) return a.second > b.second;
            else return a.first < b.first;
        }
    };

    vector<pa> vec{{1, 2}, {1, 1}, {2, 1}};
    sort(vec.begin(), vec.end(), my_cmp());
    for (auto i : vec) cout << i.first <<" "<< i.second << endl;


    return 0;

}