#include <queue>
#include <iostream>
#include <algorithm>
#include <bits/stdc++.h>
#include "utils.h"
using namespace std;

int main(){
    // 初始化
    // priority_queue<int> pq;

    // pq.top(); // 访问队首元素(优先级最高)
    // pq.push(1); // 入队
    // pq.pop();  // 堆顶(队首)元素出队
    // pq.size(); // 队列元素个数
    // pq.empty();  // 是否为空
    // // 没有clear方法

    // 设置优先级
    priority_queue<int> pq;  // 默认大根堆, 每次取出队列最大值
    priority_queue<int, vector<int>, greater<int>> q; // 小根堆

    struct cmp1
    {
        bool operator()(int x, int y){
            return x > y;
        }
    };

    struct cmp2
    {
        bool operator()(int x, int y){
            return x < y;
        }
    };
    
    priority_queue<int, vector<int>, cmp1> pq1; // 小根堆
    priority_queue<int, vector<int>, cmp2> pq2; // 大根堆
    
    struct Point
    {
        int x, y;
    };
    // 结构体外定义比较规则
    struct cmp
    {
        bool operator()(const Point& a, const Point& b){
            return a.x < b.x;
        }
    };
    priority_queue<Point, vector<Point>, cmp> Point_pq; // a.x大的在前面
    
    // 结构体内自定义比较规则,需要比较时自动调用结构体内部重载运算符
    struct node {
        int x, y;
        bool operator < (const Point &a) const {//直接传入一个参数，不必要写friend
            return x < a.x;//按x升序排列，x大的在堆顶
        }
    };
    priority_queue<Point> Point_pq2;

    struct my_cmp2
    {
        bool operator()(const pair<int, int>& a, const pair<int, int>& b){
            if (a.first == b.first) return a.second > b.second;
            else return a.first < b.first;
        }
    };
    
    auto my_cmp3 = [](const pair<int, int>& a, const pair<int, int>& b){
        if (a.first == b.first) return a.second > b.second;
        else return a.first < b.first;        
    };

    // 特殊类型优先级设置
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(my_cmp3)> pq4(my_cmp3);
    pq4.push({7, 8});
    pq4.push({7, 9});
    pq4.push(make_pair(8, 6));
    while(!pq4.empty()){
        cout << pq4.top().first << " " << pq4.top().second << "\n";
        pq4.pop();
    }

}