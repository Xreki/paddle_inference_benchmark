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

void SetInputs(std::vector<paddle::PaddleTensor> &input_tensors,
               std::string &input_dir, std::string &input_dims) {
  // q_ids
  paddle::PaddleTensor q_ids_tensor;

  q_ids_tensor.name = "q_ids";
  q_ids_tensor.dtype = paddle::PaddleDType::INT64;

  if (input_dir.empty()) {
    std::vector<int> q_ids_shape = {8, 1};
    std::vector<int64_t> q_ids = {3, 1, 4, 7, 10, 8, 2, 5};
    std::vector<std::vector<size_t>> q_ids_lod = {{0, 2, 4}, {0, 1, 3, 7, 8}};

    SetupTensor<int64_t>(&q_ids_tensor, q_ids_shape, q_ids);
    q_ids_tensor.lod = q_ids_lod;
  } else {
    LOG(INFO) << "path of q_ids: " << input_dir + "/q_ids.txt";
    SetupLoDTensor<int64_t>(input_dir + "/q_ids.txt", &q_ids_tensor);
  }

  // start_lables
  paddle::PaddleTensor start_labels_tensor;

  start_labels_tensor.name = "start_lables";
  start_labels_tensor.dtype = paddle::PaddleDType::INT64;

  if (input_dir.empty()) {
    std::vector<int> start_labels_shape = {5, 1};
    std::vector<int64_t> start_labels = {3, 5, 2, 1, 7};
    std::vector<std::vector<size_t>> start_labels_lod = {{0, 2, 5}};

    SetupTensor<int64_t>(&start_labels_tensor, start_labels_shape,
                         start_labels);
    start_labels_tensor.lod = start_labels_lod;
  } else {
    LOG(INFO) << "path of start_lables: " << input_dir + "/start_lables.txt";
    SetupLoDTensor<int64_t>(input_dir + "/start_lables.txt",
                            &start_labels_tensor);
  }

  // p_ids
  paddle::PaddleTensor p_ids_tensor;

  p_ids_tensor.name = "p_ids";
  p_ids_tensor.dtype = paddle::PaddleDType::INT64;

  if (input_dir.empty()) {
    std::vector<int> p_ids_shape = {5, 1};
    std::vector<int64_t> p_ids = {3, 5, 4, 8, 1};
    std::vector<std::vector<size_t>> p_ids_lod = {{0, 2, 4}, {0, 1, 2, 3, 5}};

    SetupTensor<int64_t>(&p_ids_tensor, p_ids_shape, p_ids);
    p_ids_tensor.lod = p_ids_lod;
  } else {
    LOG(INFO) << "path of p_ids: " << input_dir + "/p_ids.txt";
    SetupLoDTensor<int64_t>(input_dir + "/p_ids.txt", &p_ids_tensor);
  }

  // q_id0
  paddle::PaddleTensor q_id0_tensor;

  q_id0_tensor.name = "q_id0";
  q_id0_tensor.dtype = paddle::PaddleDType::INT64;

  if (input_dir.empty()) {
    std::vector<int> q_id0_shape = {8, 1};
    std::vector<int64_t> q_id0 = {5, 4, 1, 8, 9, 2, 3, 7};
    std::vector<std::vector<size_t>> q_id0_lod = {{0, 3, 8}};

    SetupTensor<int64_t>(&q_id0_tensor, q_id0_shape, q_id0);
    q_id0_tensor.lod = q_id0_lod;
  } else {
    LOG(INFO) << "path of q_id0: " << input_dir + "/q_id0.txt";
    SetupLoDTensor<int64_t>(input_dir + "/q_id0.txt", &q_id0_tensor);
  }

  // input_tensors
  input_tensors.push_back(q_ids_tensor);
  input_tensors.push_back(start_labels_tensor);
  input_tensors.push_back(p_ids_tensor);
  input_tensors.push_back(q_id0_tensor);
}

void profile(std::string model_dir, bool use_gpu, bool use_analysis,
             bool use_tensorrt) {
  std::vector<paddle::PaddleTensor> outputs;

  AnalysisConfig config;
  SetConfig<AnalysisConfig>(&config, model_dir, use_gpu, use_tensorrt,
                            FLAGS_batch_size);
  TestImpl(reinterpret_cast<PaddlePredictor::Config *>(&config), &outputs,
           use_gpu && (use_analysis || use_tensorrt), false);

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
