#include <stack>
#include <string>
#include <algorithm>
#include <iostream>
#include "utils.h"
using namespace std;

stack<int> s;
stack<string> s2;

void stack_simu(){
    int s[100];  // 数组模拟栈,从左往右为栈底到栈顶
    int tt = -1;  // 栈顶指针, 初始时栈内无元素,tt=-1

    // 入栈
    for (int i = 0; i <= 5; ++i){
        s[++tt] = i;
    }
    // 出栈
    int top_ele = s[tt--];
    cout << "top ele is " << top_ele << endl;
    cout << "tt now is " << tt << endl;
    show_array(s, tt + 1);
}


int main(){
    s.push(1);
    s.push(2);  // 入栈
    s.pop();  // 移除栈顶元素, 但要注意返回类型为void
    // show_stack(s);
    s.top(); // 返回栈顶元素,但不删除
    s.size();  // 返回栈内元素个数
    s.empty();  // 空返回true
    stack_simu();
    return 0;
}