/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <paddle/capi.h>
#include <iostream>
#include <string>

class InferenceHelper {
 public:
  InferenceHelper(bool use_gpu = false);

  void Init(const std::string& merged_model_path);

  void Init(const std::string& config_path, const std::string& params_dirname);

  void Infer(int repeat);

  void Release();

 private:
  paddle_gradient_machine machine_;
  bool use_gpu_;
};
