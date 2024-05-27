#include <deque>
#include <iostream>
#include <bits/stdc++.h>
using namespace std;

int main(){
    deque<int> dq;
    dq.push_back(1);  // 队尾插入
    dq.push_back(2);
    dq.push_front(3);  // 队首插入
    dq.push_front(4);
    dq.push_front(5);

    cout << dq.front() << endl;
    cout << dq.back() << endl;

    dq.pop_back();  // 删除队尾
    dq.pop_front(); // 删除队首

    cout << dq.front() << endl;
    cout << dq.back() << endl;

    deque<int>::iterator it = dq.begin();
    dq.erase(it, dq.end() - 1);  // 删除[first, last)中的元素

    cout << dq.front() << endl;
    cout << dq.back() << endl;
    cout << dq.size() << endl;

    dq.clear();

    dq.push_back(2);
    dq.push_back(1);

    sort(dq.begin(), dq.end());
    cout << dq.front() << endl;
    cout << dq.back() << endl;

    return 0;
}