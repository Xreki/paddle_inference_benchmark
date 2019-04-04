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
               std::string &image_path, std::string &image_dims) {
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
    SetupTensor<float>(&image_tensor, image_shape, static_cast<float>(-1),
                       static_cast<float>(1));
  } else {
    LOG(INFO) << "image_path: " << image_path;
    SetupTensor<float>(image_path, &image_tensor, &image_shape, 127.5);
  }

  // input_tensors
  input_tensors.push_back(image_tensor);
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

    std::vector<std::unique_ptr<paddle::ZeroCopyTensor>> output_tensors;
    // output_tensors.push_back(predictor->GetOutputTensor("cast_1.tmp_0")); //
    // label

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

TEST(attention_ocr, profile) {
  std::string model_dir = FLAGS_infer_model;
  profile(model_dir, FLAGS_use_gpu, FLAGS_use_analysis, FLAGS_use_tensorrt);
}

}  // namespace inference
}  // namespace paddle
