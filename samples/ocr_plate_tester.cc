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

void ConvertOutput(const std::vector<paddle::PaddleTensor> &tensors) {
  std::vector<std::vector<float>> data_alls;
  std::vector<std::vector<int>> shape_alls;

  // use reference to avoid double free
  for (auto &t : tensors) {
    shape_alls.push_back(t.shape);
    const size_t num_elements = t.data.length() / sizeof(float);
    float *t_data = static_cast<float *>(t.data.data());
    std::vector<float> data(num_elements, 0);
    std::copy(t_data, t_data + num_elements, data.data());
    data_alls.push_back(data);
  }

  // std::string plate_str = "";
  // for (float k : data_alls[0]) {
  //   std::cerr << "text\t" << k << std::endl;
  //   if (k == 0 || k == 1 || k == 2) {
  //     continue;
  //   }
  //   // plate_str += table_dict[k];
  // }

  // for (float k : data_alls[1]) {
  //   std::cerr << "scores\t" << k << std::endl;
  // }
}

void SetInputs(std::vector<paddle::PaddleTensor> &input_tensors, std::string &image_path, std::string &image_dims) {
  //
  // image tensor -> pixel
  //
  paddle::PaddleTensor image_tensor;

  int batch_size = 1;
  int channel = 1;
  int height = 48;
  int width = 512;
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

  image_tensor.name = "pixel";
  image_tensor.dtype = paddle::PaddleDType::FLOAT32;
  std::vector<int> image_shape = {batch_size, channel, height, width};
  if (image_path.empty()) {
    SetupTensor<float>(image_tensor, image_shape,
                       static_cast<float>(-1), static_cast<float>(1));
  } else {
    LOG(INFO) << "image_path: " << image_path;
    SetupTensor<float>(image_path, image_tensor, &image_shape, 127.5);
  }

  //
  // init_ids_tensor -> init_ids
  //
  paddle::PaddleTensor init_ids_tensor;

  std::vector<int> ids_shape = {batch_size, 1};
  std::vector<int64_t> init_ids = {0};

  init_ids_tensor.name = "init_ids";
  init_ids_tensor.shape = ids_shape;
  init_ids_tensor.dtype = paddle::PaddleDType::INT64;
  init_ids_tensor.data.Resize(sizeof(int64_t) * 1);
  std::copy(init_ids.begin(), init_ids.end(),
            static_cast<int64_t *>(init_ids_tensor.data.data()));

  std::vector<size_t> lod_1 = {0, 1};
  std::vector<size_t> lod_2 = {0, 1};
  std::vector<std::vector<size_t>> lod = {{0, 1}, {0, 1}};
  init_ids_tensor.lod = lod;

  //
  // init scores
  //
  paddle::PaddleTensor init_scores_tensor;

  std::vector<int> scores_shape;
  scores_shape.push_back(1);
  scores_shape.push_back(1);

  std::vector<float> init_scores;
  init_scores.push_back(1.0);

  init_scores_tensor.name = "init_scores";
  init_scores_tensor.shape = scores_shape;
  init_scores_tensor.dtype = paddle::PaddleDType::FLOAT32;
  init_scores_tensor.data.Resize(sizeof(float) * 1);
  std::copy(init_scores.begin(), init_scores.end(),
            static_cast<float *>(init_scores_tensor.data.data()));
  init_scores_tensor.lod = lod;

  // input_tensors
  input_tensors.push_back(image_tensor);
  input_tensors.push_back(init_ids_tensor);
  input_tensors.push_back(init_scores_tensor);

#if 0
  //
  // position encoding
  //
  paddle::PaddleTensor position_encoding_tensor;

  std::vector<int> position_encoding_shape;
  position_encoding_shape.push_back(1);
  position_encoding_shape.push_back(33);
  position_encoding_shape.push_back(10);
  position_encoding_shape.push_back(23);

  std::vector<int> pos_data;
  for (int i = 0; i < 10; i++) {
    for (int row = 0; row < 10; row++) {
      for (int col = 0; col < 23; col++) {
        if (i == row) {
          pos_data.push_back(1);
        } else {
          pos_data.push_back(0);
        }
      }
    }
  }
  for (int i = 0; i < 23; i++) {
    for (int row = 0; row < 10; row++) {
      for (int col = 0; col < 23; col++) {
        if (i == col) {
          pos_data.push_back(1);
        } else {
          pos_data.push_back(0);
        }
      }
    }
  }

  position_encoding_tensor.name = "position_encoding";
  position_encoding_tensor.shape = position_encoding_shape;
  position_encoding_tensor.dtype = paddle::PaddleDType::FLOAT32;
  position_encoding_tensor.data.Resize(sizeof(float) * 33 * 10 * 23);
  std::copy(pos_data.begin(), pos_data.end(),
            static_cast<float *>(position_encoding_tensor.data.data()));
  input_tensors.push_back(position_encoding_tensor);
#endif
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

TEST(attention_ocr, profile) {
  std::string model_dir = FLAGS_infer_model;
  profile(model_dir, FLAGS_use_gpu, FLAGS_use_analysis, FLAGS_use_tensorrt);
}

}  // namespace inference
}  // namespace paddle
