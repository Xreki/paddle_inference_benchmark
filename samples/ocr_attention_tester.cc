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

std::vector<float> GeneratePositionData(std::vector<int> shape) {
  PADDLE_ENFORCE_EQ(shape.size(), 4UL);
  PADDLE_ENFORCE_EQ(shape[0], 1UL);
  PADDLE_ENFORCE_EQ(shape[1], 33UL);

  std::vector<float> position_data;
  for (int i = 0; i < 10; i++) {
    for (int row = 0; row < shape[2]; row++) {
      for (int col = 0; col < shape[3]; col++) {
        if (i == row) {
          position_data.push_back(1.);
        } else {
          position_data.push_back(0.);
        }
      }
    }
  }
  for (int i = 0; i < 23; i++) {
    for (int row = 0; row < shape[2]; row++) {
      for (int col = 0; col < shape[3]; col++) {
        if (i == col) {
          position_data.push_back(1.);
        } else {
          position_data.push_back(0.);
        }
      }
    }
  }
  return position_data;
}

void SetInputs(std::vector<paddle::PaddleTensor> &input_tensors,
               std::string &image_path, std::string &image_dims) {
  //
  // image tensor -> pixel
  //
  paddle::PaddleTensor image_tensor;

  int batch_size = 1;
  int channel = 1;
  int height = 48;
  int width = 214;
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
    SetupTensor<float>(&image_tensor, image_shape, static_cast<float>(-1),
                       static_cast<float>(1));
  } else {
    LOG(INFO) << "image_path: " << image_path;
    SetupTensor<float>(image_path, &image_tensor, &image_shape, 127.5);
  }

  //
  // init_ids_tensor -> init_ids
  //
  paddle::PaddleTensor init_ids_tensor;

  std::vector<int> ids_shape = {batch_size, 1};
  std::vector<int64_t> init_ids = {0};
  std::vector<std::vector<size_t>> lod = {{0, 1}, {0, 1}};

  init_ids_tensor.name = "init_ids";
  init_ids_tensor.dtype = paddle::PaddleDType::INT64;
  init_ids_tensor.lod = lod;
  SetupTensor<int64_t>(&init_ids_tensor, ids_shape, init_ids);

  //
  // init scores
  //
  paddle::PaddleTensor init_scores_tensor;

  std::vector<int> scores_shape = {1, 1};
  std::vector<float> init_scores = {1.0};

  init_scores_tensor.name = "init_scores";
  init_scores_tensor.dtype = paddle::PaddleDType::FLOAT32;
  init_scores_tensor.lod = lod;
  SetupTensor<float>(&init_scores_tensor, scores_shape, init_scores);

  //
  // parent_idx
  //
  // paddle::PaddleTensor parent_idx_tensor;

  // std::vector<int> parent_idx_shape = {1};
  // std::vector<int64_t> parent_idx = {0};

  // parent_idx_tensor.name = "parent_idx";
  // parent_idx_tensor.dtype = paddle::PaddleDType::INT64;
  // SetupTensor<int64_t>(&parent_idx_tensor, parent_idx_shape, parent_idx);

  // input_tensors
  input_tensors.push_back(image_tensor);
  input_tensors.push_back(init_ids_tensor);
  input_tensors.push_back(init_scores_tensor);
// input_tensors.push_back(parent_idx_tensor);

#if 0
  //
  // position encoding
  //
  paddle::PaddleTensor position_encoding_tensor;

  std::vector<int> position_encoding_shape = {1, 33, 10, 23};
  std::vector<float> position_encoding = GeneratePositionData(position_encoding_shape);

  position_encoding_tensor.name = "position_encoding";
  position_encoding_tensor.shape = position_encoding_shape;
  position_encoding_tensor.dtype = paddle::PaddleDType::FLOAT32;
  SetupTensor<float>(&position_encoding_tensor, position_encoding_shape, position_encoding);

  input_tensors.push_back(position_encoding_tensor);
#endif
}

void SetZeroCopyInputs(
    std::vector<std::unique_ptr<paddle::ZeroCopyTensor>> &input_tensors,
    std::string &image_path, std::string &image_dims) {
  //
  // image tensor -> pixel
  //
  auto &image_tensor = input_tensors[0];

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

  std::vector<int> image_shape = {batch_size, channel, height, width};
  if (image_path.empty()) {
    SetupZeroCopyTensor<float>(image_tensor.get(), image_shape,
                               static_cast<float>(-1), static_cast<float>(1));
  } else {
    LOG(INFO) << "image_path: " << image_path;
    SetupZeroCopyTensor<float>(image_path, image_tensor.get(), &image_shape,
                               127.5);
  }

  //
  // init_ids_tensor -> init_ids
  //
  auto &init_ids_tensor = input_tensors[1];

  std::vector<int> ids_shape = {batch_size, 1};
  std::vector<int64_t> init_ids = {0};
  std::vector<std::vector<size_t>> lod = {{0, 1}, {0, 1}};

  SetupZeroCopyTensor<int64_t>(init_ids_tensor.get(), ids_shape, init_ids);
  init_ids_tensor->SetLoD(lod);

  //
  // init scores
  //
  auto &init_scores_tensor = input_tensors[2];

  std::vector<int> scores_shape = {1, 1};
  std::vector<float> init_scores = {1.0};

  SetupZeroCopyTensor<float>(init_scores_tensor.get(), scores_shape,
                             init_scores);
  init_scores_tensor->SetLoD(lod);
}

void profile(std::string model_dir, bool use_gpu, bool use_analysis,
             bool use_tensorrt) {
  AnalysisConfig config;
  SetConfig<AnalysisConfig>(&config, model_dir, use_gpu, use_tensorrt,
                            FLAGS_batch_size);
  if (FLAGS_use_analysis && FLAGS_use_zerocopy) {
    // Zero-copy is not beneficial for inference on GPU.
    auto predictor = CreateTestPredictor(
        reinterpret_cast<PaddlePredictor::Config *>(&config), use_analysis);

    std::vector<std::unique_ptr<paddle::ZeroCopyTensor>> input_tensors;
    input_tensors.push_back(predictor->GetInputTensor("pixel"));
    input_tensors.push_back(predictor->GetInputTensor("init_ids"));
    input_tensors.push_back(predictor->GetInputTensor("init_scores"));

    std::vector<std::unique_ptr<paddle::ZeroCopyTensor>> output_tensors;
    output_tensors.push_back(
        predictor->GetOutputTensor("cast_1.tmp_0"));  // label
    output_tensors.push_back(predictor->GetOutputTensor(
        "tensor_array_to_tensor_0.tmp_0"));  // weight

    std::vector<std::string> input_list;
    if (GenerateInputList(&input_list, FLAGS_input_dir)) {
      LOG(WARNING) << "Get no inputs in input_dir (" << FLAGS_input_dir
                   << "), use fake inputs instead.";
      input_list.push_back("dummpy");
    }

    if (input_list[0] != "dummpy") {
      std::string input_path = input_list[0];
      SetZeroCopyInputs(input_tensors, input_path, FLAGS_image_dims);
    } else {
      std::string input_path = "";
      SetZeroCopyInputs(input_tensors, input_path, FLAGS_image_dims);
    }

    int batch_size = FLAGS_batch_size;

    // warmup run
    LOG(INFO) << "Warm up run...";
    {
      Timer warmup_timer;
      warmup_timer.tic();
      predictor->ZeroCopyRun();
      PrintTime(batch_size, 1, 1, 0, warmup_timer.toc(), 1);
      if (FLAGS_profile) {
        paddle::platform::ResetProfiler();
      }
    }

    int num_times = FLAGS_repeat;
    LOG(INFO) << "Run " << num_times << " times...";
    Timer run_timer;
    run_timer.tic();
    for (int r = 0; r < num_times; r++) {
      predictor->ZeroCopyRun();
    }
    double latency = run_timer.toc() / num_times;
    PrintTime(batch_size, num_times, 1, 0, latency, 1);
  } else {
    std::vector<paddle::PaddleTensor> outputs;

    TestImpl(reinterpret_cast<PaddlePredictor::Config *>(&config), &outputs,
             use_gpu && (use_analysis || use_tensorrt));

    for (size_t i = 0; i < outputs.size(); ++i) {
      LOG(INFO) << "<<< output: " << i << " >>>";
      PrintTensor(outputs[i], 4);
    }
  }
}

void profile_encoder_decoder(std::string model_dir) {
  std::vector<paddle::PaddleTensor> input_tensors;

  std::vector<std::string> input_list;
  if (GenerateInputList(&input_list, FLAGS_input_dir)) {
    LOG(WARNING) << "Get no inputs in input_dir (" << FLAGS_input_dir
                 << "), use fake inputs instead.";
    input_list.push_back("dummpy");
  }

  std::string image_path = input_list[0];
  std::string image_dims = "";
  SetInputs(input_tensors, image_path, image_dims);
  PADDLE_ENFORCE_EQ(input_tensors.size(), 3UL);

  std::vector<paddle::PaddleTensor> encoder_inputs;
  std::vector<paddle::PaddleTensor> encoder_outputs;

  // encoder
  {
    encoder_inputs.push_back(input_tensors[0]);
    AnalysisConfig encoder_config;
    std::string encoder_model_dir = model_dir + "/encoder";
    SetConfig<AnalysisConfig>(&encoder_config, encoder_model_dir, true, false,
                              1);

    auto encoder_predictor =
        CreatePaddlePredictor<AnalysisConfig>(encoder_config);

    encoder_predictor->Run(encoder_inputs, &encoder_outputs, 1);
  }

  std::vector<paddle::PaddleTensor> decoder_inputs;
  std::vector<paddle::PaddleTensor> decoder_outputs;

  // decoder
  {
    std::string decoder_model_dir = model_dir + "/decoder_change_int64";
#if 1
    AnalysisConfig decoder_config;
    SetConfig<AnalysisConfig>(&decoder_config, decoder_model_dir, false, false,
                              1);

    auto decoder_predictor =
        CreatePaddlePredictor<AnalysisConfig>(decoder_config);
#else
    NativeConfig decoder_config;
    decoder_config.prog_file = decoder_model_dir + "/model";
    decoder_config.param_file = decoder_model_dir + "/params";

    auto decoder_predictor =
        CreatePaddlePredictor<NativeConfig>(decoder_config);
#endif
    for (size_t i = 0; i < encoder_outputs.size(); ++i) {
      decoder_inputs.push_back(encoder_outputs[i]);
    }
    decoder_inputs.push_back(input_tensors[1]);
    decoder_inputs.push_back(input_tensors[2]);
    decoder_predictor->Run(decoder_inputs, &decoder_outputs, 1);
  }

  for (size_t i = 0; i < decoder_outputs.size(); ++i) {
    LOG(INFO) << "<<< output: " << i << " >>>";
    PrintTensor(decoder_outputs[i], 4);
  }
}

TEST(attention_ocr, profile) {
  std::string model_dir = FLAGS_infer_model;
  // profile(model_dir, FLAGS_use_gpu, FLAGS_use_analysis, FLAGS_use_tensorrt);
  profile_encoder_decoder(model_dir);
}

}  // namespace inference
}  // namespace paddle
