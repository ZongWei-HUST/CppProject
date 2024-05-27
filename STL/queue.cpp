#include <queue>
#include <iostream>
#include <bits/stdc++.h>
using namespace std;


void queue_simu(){
    // 使用q[]数组模拟队列
    // hh 模拟队首元素的下标, 初始值为0
    // tt 模拟队尾元素的下标, 初始值为-1
    int q[100];
    int hh = 0, tt = -1;
    // 入队
    q[++tt] = 1;
    q[++tt] = 2;
    // 出队
    while (hh <= tt){
        int ele = q[hh++];
        printf("%d", ele);
    }
}

int main(){
    // 初始化
    queue<int> q;

    // 方法函数
    q.push(1);
    q.push(2);
    q.push(5);

    cout << q.front() << endl;  // 返回队首元素, 数组最左, 最先进队列
    cout << q.back() << endl;  // 返回队尾元素, 数组最右, 最后进队列

    cout << q.size() << endl;

    q.pop();  // 弹出队首元素, 返回类型void
    cout << q.front() << endl;
    cout << q.back() << endl;

    cout << q.empty() << endl;

    cout << "-----" << endl;
    queue_simu();
    return 0;
}