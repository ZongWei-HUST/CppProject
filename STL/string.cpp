#include <string>
#include <iostream>
using namespace std;

int main(){
    // 初始化的几种方式
    string s1; // 空字符串
    string s2("123456");  // "123456"的复制品
    string s3("123456", 0, 3); // 左闭右开区间取str, "123"
    string s4(5, '2'); // "22222"    
    string s5(s2, 2);  // 从index=2开始取, "3456"
    string s6("123456", 2);  // 类似于s3,省略,取到index=2, "12"

    cout << s5 << " " << s6 << endl;
    
    // 访问
    cout << s2[1] << endl;

    cout << (s2 < "6") << endl;

    // ios::sync_with_stdio(false);
    // cin.tie(0), cout.tie(0);

    // int n;
    // string s;
    // cin >> n;
    // cin.get();
    // getline(cin, s);
    // cout << s;

    s2.insert(s2.begin(), '0');
    s2.erase(s2.end() - 1);
    s2.replace(1, 1, "0");
    s2.replace(s2.begin() + 2, s2.end() - 2, "1");
    cout << s2 << endl;
    cout << s2.substr(2, 3) << endl;

    string ss("dog bird chicken bird cat");
    cout << ss.find_first_of("13br98") << endl;
}