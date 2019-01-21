// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "common.h"

namespace paddle {
namespace inference {

void SetInputs(std::vector<paddle::PaddleTensor> &input_tensors, std::string &input_path, std::string &input_dims) {
  // q_ids
  // paddle::PaddleTensor q_ids_tensor;
  // 
  // std::vector<int> q_ids_shape = {}

  // q_ids_tensor.name = "q_ids";
  // q_ids_tensor.dtype = paddle::PaddleDType::INT64;
  //
  // "p_ids", "q_id0"
}

void profile(std::string model_dir, bool use_gpu, bool use_analysis, bool use_tensorrt) {
  std::vector<paddle::PaddleTensor> outputs;

  contrib::AnalysisConfig config;
  SetConfig<contrib::AnalysisConfig>(&config, model_dir, use_gpu, use_tensorrt,
                                     FLAGS_batch_size);
  TestImpl(reinterpret_cast<PaddlePredictor::Config *>(&config), &outputs, use_gpu && (use_analysis || use_tensorrt));
  
  for (size_t i = 0; i < outputs.size(); ++i) {
    LOG(INFO) << "<<< output: " << i << " >>>";
    PrintTensor(outputs[i], 4);
  }
}

TEST(machine_reading_comprehension, profile) {
  std::string model_dir = FLAGS_infer_model;
  profile(model_dir, FLAGS_use_gpu, FLAGS_use_analysis, FLAGS_use_tensorrt);
}

}  // namespace inference
}  // namespace paddle