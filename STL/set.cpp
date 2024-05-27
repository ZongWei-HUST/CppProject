#include <iostream>
#include <set>
using namespace std;

int main(){

    set<pair<int, int> > s;
    s.insert({1, 2});
    s.insert({1, 1});
    s.insert({2, 1});
    s.insert({1, 1});

    // 迭代器访问
    auto it = s.begin();
    for (it; it != s.end(); ++it){
        cout << (*it).first << " " << it->second << endl;
    }
    // 智能指针
    for (auto i : s) cout << i.first << endl;
    // 访问第一个和最后一个元素
    // 注意, set的迭代器没有重载+/-运算符
    // 要访问单个元素需要循环++/--
    cout << (*s.begin()).first << endl; // 第0个
    auto ii = s.begin();
    int N = 1; // 第N个
    for(int i = 0; i < N; ++i) ++ii;
    cout << (*ii).first << " " << (*ii).second << endl;
    cout << (*s.rbegin()).first << " " << endl; // 最后一个元素

    cout << "=========" << endl;
    for (auto i : s) cout << i.first << " " << i.second << endl;
    cout << "=========" << endl;

    struct cmp
    {
        bool operator() (const pair<int, int>& a, const pair<int, int>& b){
            if (a.first == b.first) return a.second > b.second;
            else return a.first < b.first;
        }
    };
    set<pair<int, int>, cmp> s1;  // 注意这里是尖括号<>,所以传递类型
    s1.insert({1, 2});
    s1.insert({1, 1});
    s1.insert({2, 1});
    s1.insert({1, 1});
    for (auto i:s1) cout << i.first << " " << i.second << endl;


}