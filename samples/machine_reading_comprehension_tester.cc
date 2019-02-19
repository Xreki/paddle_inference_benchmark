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
  paddle::PaddleTensor q_ids_tensor;
  
  std::vector<int> q_ids_shape = {8, 1};
  std::vector<int64_t> q_ids = {3, 1, 4, 7, 10, 8, 2, 5};

  q_ids_tensor.name = "q_ids";
  q_ids_tensor.shape = q_ids_shape;
  q_ids_tensor.dtype = paddle::PaddleDType::INT64;
  q_ids_tensor.data.Resize(sizeof(int64_t) * 8);
  std::copy(q_ids.begin(), q_ids.end(),
            static_cast<int64_t *>(q_ids_tensor.data.data()));
  
  std::vector<std::vector<size_t>> q_ids_lod = {{0, 2, 4}, {0, 1, 3, 7, 8}};
  q_ids_tensor.lod = q_ids_lod;

  // p_ids
  paddle::PaddleTensor p_ids_tensor;

  std::vector<int> p_ids_shape = {5, 1};
  std::vector<int64_t> p_ids = {3, 5, 4, 8, 1};

  p_ids_tensor.name = "p_ids";
  p_ids_tensor.shape = p_ids_shape;
  p_ids_tensor.dtype = paddle::PaddleDType::INT64;
  p_ids_tensor.data.Resize(sizeof(int64_t) * 5);
  std::copy(p_ids.begin(), p_ids.end(),
            static_cast<int64_t *>(p_ids_tensor.data.data()));

  std::vector<std::vector<size_t>> p_ids_lod = {{0, 2, 4}, {0, 1, 2, 3, 5}};
  p_ids_tensor.lod = p_ids_lod;

  // q_id0
  paddle::PaddleTensor q_id0_tensor;

  std::vector<int> q_id0_shape = {3, 1};
  std::vector<int64_t> q_id0 = {5, 4, 1};

  q_id0_tensor.name = "q_id0";
  q_id0_tensor.shape = q_id0_shape;
  q_id0_tensor.dtype = paddle::PaddleDType::INT64;
  q_id0_tensor.data.Resize(sizeof(int64_t) * 3);
  std::copy(q_id0.begin(), q_id0.end(),
            static_cast<int64_t *>(q_id0_tensor.data.data()));

  std::vector<std::vector<size_t>> q_id0_lod = {{0, 2, 3}};
  q_id0_tensor.lod = q_id0_lod;

  // input_tensors
  input_tensors.push_back(q_ids_tensor);
  input_tensors.push_back(p_ids_tensor);
  input_tensors.push_back(q_id0_tensor);
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
