#include <numeric>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <random>
#include <string>
using namespace std;


char op(char ch){
    if (ch >= 'A' && ch <= 'Z') return ch + 32;
    else return ch;
}

int main(){
    // 1.accumulate
    int a[] = {1, 3, 5, 7};
    int res = accumulate(a, a + 3, 0);
    cout << res << endl;

    int n = 10;
    typedef long long ll;
    struct node
    {
        ll num;
    }st[n];

    for (int i = 1; i < n; ++i){
        st[i].num = i + 100ll;
    }
    ll sum = accumulate(st + 1, st + 4, 1ll, [](ll s, node b){
        return s + b.num;
    });
    cout << sum << endl;  // 101,102,103,1
    
    // 2. atoi/stoi
    char s1[] = "123";
    string s = "123";
    cout << atoi(s1) << " " << atoi(s.c_str()) << " " << stoi(s) << " " << endl;

    // 3. fill, array和数组类型
    int a1[5];
    int a1_len = sizeof(a1) / sizeof(a1[0]);
    // 如果是字符数组char s1[];可以使用strlen(s1)
    fill(a1, a1 + a1_len, 1);
    for (int i = 0; i < 5; ++i) cout << a1[i] << " " << endl; 
    char str[10];
    memset(str, 97, sizeof(char) * 10);
    for (int i = 0; i < 10; ++i) cout << str[i] << endl;

    // 4. is_sorted
    int aa[] = {1, 2, 3};
    cout << is_sorted(aa, aa + 3) << endl;

    // 5. iota
    vector<int> v{1,2,3,4,9,10};
    // iota(v.begin(), v.end(), 0);  // 让序列递增赋值
    // for (auto i : v) cout << i << " ";

    // 6. lower_bound + upper_bound
    // v = {0, 1, ..., 9}; vector
    cout << *lower_bound(v.begin(), v.end(), 9) << endl;
    cout << v[lower_bound(v.begin(), v.end(), 9) - v.begin()] << endl;
    cout << *lower_bound(v.begin(), v.end(), 11) << endl;
    cout << *upper_bound(v.begin(), v.end(), 8) << endl;
    cout << *upper_bound(v.begin(), v.end(), 11) << endl;
    // a = {1, 3, 5 ,7}; []
    cout << *lower_bound(a, a + 4, 5) << endl;

    // 7. max_element + min_element + nth_element
    vector<int>::iterator mx = max_element(v.begin(), v.begin() + v.size());
    int mn = *min_element(v.begin(), v.begin() + v.size());
    cout << *mx << " " << " " << mn << endl;
    cout << *max_element(a, a + 4) << endl;

    auto t = minmax_element(v.begin(), v.end());
    cout << *t.first << " " << *t.second << endl;  // 是迭代器，不是指针，只能*t.first不能t->first

    int bb[] = {3, 5, 1, 0};
    nth_element(bb, bb + 2, bb + 4);
    for (int i = 0; i < 4; ++i) cout << bb[i] << endl;

    // 8. next_permutation
    cout << "========" << endl; 
    vector<int> vvv{1, 2, 4};
    int aaa[] = {1, 2, 4};
    do{
        for (int i = 0; i < 3; ++i) cout << aaa[i] << endl;
    }
    while(next_permutation(aaa, aaa + 3));

    // 9. shuffle
    cout << "========" << endl; 
    vector<int> b(5);
    iota(b.begin(), b.end(), 1);  // 递增赋值,从1开始
    // 随机数种子
    mt19937 gen(random_device{}());
    shuffle(b.begin(), b.end(), gen);
    for (auto bi : b) cout << bi << endl;

    // 10. reverse
    string ss1 = "abcd";
    reverse(ss1.begin(), ss1.end());
    cout << ss1 << "\n";

    int arr1[] = {1, 2, 3};
    reverse(arr1, arr1 + 3);
    cout << arr1[0] << arr1[1] << arr1[2] << endl;

    // 11. transform
    string first = "AaBbZz", second;
    second.resize(first.size());
    transform(first.begin(), first.end(), second.begin(), op);
    cout << second << endl;

    // 12 . to_string
    cout << to_string(123.34) << "\n";

    // 13. unique
    cout << "========" << endl; 
    int aaa1[] = {3, 2, 3, 1, 6};
    int bbb[] = {1, 2, 3, 3, 6};

    // 排序后 b：{1, 2, 3, 3, 6}
    sort(bbb, bbb + 5); //对b数组排序
    // 消除重复元素b：{1, 2, 3, 6, 3} 返回的地址为最后一个元素3的地址 
    int len = unique(bbb, bbb + 5) - bbb;//消除 b 的重复元素，并获取长度
    for(int i = 0; i < 5; i++) {
        //因为b有序，查找到的下标就是对应的 相对大小（离散化后的值）
        int pos = lower_bound(bbb, bbb + len, aaa1[i]) - bbb;//在b数组中二分查找第一个大于等于a[i]的下标
        aaa1[i] = pos; // 离散化赋值
    }
    for (int i = 0; i < 5; ++i) cout << aaa1[i] << endl;

}

