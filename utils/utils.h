#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

template <typename T>
std::string Dims2String(std::vector<T>& dims) {
  std::stringstream ss;
  ss << "[";
  if (dims.size() > 0) {
    ss << dims[0];
  }
  for (size_t i = 1; i < dims.size(); ++i) {
    ss << ", " << dims[i];
  }
  ss << "]";
  return ss.str();
}

template <typename T>
void Rand(T* data, size_t length, T lower, T upper) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  for (size_t i = 0; i < length; ++i) {
    data[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
  }
}

template <typename T>
void Print(T* data, size_t length, std::string discription) {
  std::cout << discription << " [";
  if (length > 0) {
    std::cout << data[0];
  }
  for (size_t i = 1; i < length; ++i) {
    std::cout << ", " << data[i];
  }
  std::cout << "]" << std::endl;
}
