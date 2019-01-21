/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "common.h"

namespace paddle {
namespace inference {

void SetInputs(std::vector<paddle::PaddleTensor> &input_tensors, std::string &image_path, std::string &image_dims) {
  //
  // input image
  //
  paddle::PaddleTensor image_tensor;

  int batch_size = 1;
  int channel = 3;
  int height = 1333;
  int width = 1333;
  if (!image_dims.empty()) {
    std::vector<int> image_shape = ParseDims(image_dims);
    size_t length = image_shape.size();
    if (length >= 4) {
      batch_size = image_shape[length - 4];
    }
    if (length >= 3) {
      channel = image_shape[length - 3];
    }
    if (length >= 2) {
      height = image_shape[length - 2];
    }
    if (length >= 1) {
      width = image_shape[length - 1];
    }
  }

  image_tensor.name = "image";
  image_tensor.dtype = paddle::PaddleDType::FLOAT32;
  std::vector<int> image_shape = {batch_size, channel, height, width};
  SetupTensor<float>(image_tensor, image_shape, static_cast<float>(0),
                     static_cast<float>(255));

  //
  // input im_info
  //
  paddle::PaddleTensor im_info_tensor;

  im_info_tensor.name = "im_info";
  im_info_tensor.dtype = paddle::PaddleDType::FLOAT32;
  SetupTensor<float>(im_info_tensor, {1, 3}, static_cast<float>(0),
                     static_cast<float>(1));

  // input_tensors
  input_tensors.push_back(image_tensor);
  input_tensors.push_back(im_info_tensor);
}

void profile(std::string model_dir, bool use_gpu, bool use_analysis, bool use_tensorrt) {
  std::vector<paddle::PaddleTensor> inputs;
  SetInputs(inputs, FLAGS_image_dir, FLAGS_image_dims);

  std::vector<std::vector<PaddleTensor>> inputs_all;
  inputs_all.push_back(inputs);

  std::vector<paddle::PaddleTensor> outputs;
    
  contrib::AnalysisConfig config;
  SetConfig<contrib::AnalysisConfig>(&config, model_dir, use_gpu, use_tensorrt,
                                     FLAGS_batch_size);
  TestPrediction(reinterpret_cast<PaddlePredictor::Config *>(&config),
                 inputs_all, &outputs, FLAGS_num_threads,
                 use_gpu && (use_analysis || use_tensorrt));
}

TEST(video, profile) {
  std::string model_dir = FLAGS_infer_model;
  profile(model_dir, FLAGS_use_gpu, FLAGS_use_analysis, FLAGS_use_tensorrt);
}

}  // namespace inference
}  // namespace paddle
