#include <map>
#include <string>
#include <iostream>
#include <unordered_map>

using namespace std;


int main(){
    // 初始化
    map<string, int> mp;
    // 添加元素
    string s1("1");
    string s2("2");
    string s3("3");
    mp[s1] = 10;
    mp[s2] = 20;
    mp[s3] = 30;

    // 方法函数
    // mp.find返回键为key的迭代器,O(logN),数据不存在返回mp.end()
    map<string, int>::iterator it = mp.find(s1);
    cout << (*it).second << endl;
    cout << it->second << endl;
    mp.erase(mp.begin());

    // 正向遍历
    auto it1 = mp.begin();
    while (it1 != mp.end()){
        cout << it1->first << " " << it1->second << "\n";
        ++it1;
    }
    // 反向遍历
    auto it2 = mp.rbegin();
    while (it2 != mp.rend() ){
        cout << it2->first << " " << it2->second << "\n";
        ++it2;  // 这里也是++,并且红黑树的迭代器没有重载+/-运算符
    }
    cout << "==============" << endl;

    // 二分查找
    auto it3 = mp.lower_bound(string("2"));
    cout << it3->first << " "<< it3->second << endl;
    auto it4 = mp.upper_bound(string("2"));
    cout << it4->first << " "<< it4->second << endl;
    cout << "==============" << endl;

    // 添加元素
    map<int, int> mp1;
    mp1[1] = 2;
    mp1.insert(make_pair(2, 3));
    mp1.insert(pair<int, int>(2, 3));  // 自动删除重复的元素
    mp1.insert({3,4});
    // 访问元素
    cout << mp1[1] << endl;
    auto it_ = mp1.find(4);
    cout << bool(it_ == mp1.end());
    cout << it_->first << " " << it_->second << endl;
    
    unordered_map<int, int> ump;


    return 0;
}