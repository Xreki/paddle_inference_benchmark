#pragma once

#include <paddle/capi.h>
#include <string>
#include <vector>

class InferenceHelper {
 public:
  InferenceHelper(bool use_gpu = false);

  void Init(const std::string& merged_model_path);

  void Init(const std::string& config_path, const std::string& params_dirname);

  void Infer(std::vector<int>& dims, int repeat);

  void Release();

 private:
  paddle_gradient_machine machine_;
  bool use_gpu_{false};
};
