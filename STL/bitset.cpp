#include <bitset>
#include <iostream>
using namespace std;


int main() {
    bitset<4> bitset1;  //无参构造，长度为４，默认每一位为０

    bitset<9> bitset2(12); //长度为9，二进制保存，前面用０补充

    string s = "100101";
    bitset<10> bitset3(s); //长度为10，前面用０补充

    char s2[] = "10101";
    bitset<13> bitset4(s2); //长度为13，前面用０补充

    cout << bitset1 << endl; //0000
    cout << bitset2 << endl; //000001100
    cout << bitset3 << endl; //0000100101
    cout << bitset4 << endl; //0000000010101

    cout << bitset2.to_ulong() << endl;
    return 0;
}
