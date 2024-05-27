// #pragma once
#include <vector>
#include <stack>
#include <iostream>

// print like python.
// `inline` : use with multi cpp files include the same .h.
// `#pragma once` : use with one cpp include multi .h files.
inline void printAsPython() { std::cout << std::endl; };

template <typename T, typename... Types>
void printAsPython(const T& fisrtArg, const Types&... args) {
  std::cout << fisrtArg << " ";
  printAsPython(args...);
}
 

template <class T>
void show_vector(std::vector<T>& v) {
  for (auto&& item : v) {
    std::cout << item << " ";
  }
  std::cout << std::endl;
}

template <class T>
void show_stack(std::stack<T>& s){
  while (!s.empty())
  {
    std::cout << s.top() << " ";
    s.pop();
  }
  std::cout << std::endl;
}

template <class T>
void show_array(T arr, int n){
  std::cout << "n is " << n << std::endl;
  for (int i = 0; i < n; ++i){
    std::cout << arr[i] << " ";
  }
  std::cout << std::endl;
}