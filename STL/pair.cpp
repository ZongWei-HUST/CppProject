#include <utility>
#include <iostream>
using namespace std;

int main(){
    pair<int, int> p[20];  // pair数组
    for (int i = 0; i < 20; ++i){
        cout << p[i].first << " " << p[i].second << endl;
    }
    return 0;
}