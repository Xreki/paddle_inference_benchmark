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

#pragma once

#include <dirent.h>
#include <glog/logging.h>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/fluid/inference/tests/api/tester_helper.h"

DEFINE_bool(use_gpu, true, "Whether use gpu.");
DEFINE_bool(use_tensorrt, true, "Test the performance of TensorRT engine.");
DEFINE_string(prog_filename, "", "Name of model file.");
DEFINE_string(param_filename, "", "Name of parameters file.");
DEFINE_string(image_dir, "", "Path of input data.");
DEFINE_string(image_dims, "", "Dims of input data.");

namespace paddle {
namespace inference {

void SetInputs(std::vector<paddle::PaddleTensor>& input_tensors,
               std::string& input_path, std::string& input_dims);

std::vector<int> ParseDims(std::string dims_str) {
  std::vector<int> dims;
  std::string token;
  std::istringstream token_stream(dims_str);
  while (std::getline(token_stream, token, 'x')) {
    dims.push_back(std::stoi(token));
  }
  return dims;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& value) {
  os << "{";
  if (value.size() > 0) {
    os << value[0];
  }
  for (size_t i = 1; i < value.size(); ++i) {
    os << ", " << value[i];
  }
  os << "}";
  return os;
}

template <typename T>
void SetupTensor(const std::string filename, paddle::PaddleTensor& tensor,
                 std::vector<int>* shape, T mean = 0) {
  std::ifstream is(filename);

  size_t size = 1;
  for (size_t i = 0; i < shape->size(); ++i) {
    if (shape->at(i) <= 0) {
      int64_t l = 0;
      is >> l;
      shape->at(i) = l;
    }
    size *= shape->at(i);
  }
  LOG(INFO) << "shape: " << *shape;

  std::vector<T> data;
  for (size_t i = 0; i < size; ++i) {
    T value;
    is >> value;
    data.push_back(static_cast<T>(value - mean));
  }
  is.close();

  tensor.shape = *shape;
  tensor.data.Resize(sizeof(T) * size);
  std::copy(data.begin(), data.end(), static_cast<T*>(tensor.data.data()));
}

template <typename T>
void SetupTensor(paddle::PaddleTensor& tensor, std::vector<int> shape, T lower,
                 T upper) {
  size_t size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    size *= shape[i];
  }
  LOG(INFO) << "shape: " << shape;

  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  std::vector<T> data;
  for (size_t i = 0; i < size; ++i) {
    data.push_back(static_cast<T>(uniform_dist(rng) * (upper - lower) + lower));
  }

  tensor.shape = shape;
  tensor.data.Resize(sizeof(T) * size);
  std::copy(data.begin(), data.end(), static_cast<T*>(tensor.data.data()));
}

template <typename ConfigType>
void SetConfig(ConfigType* config, std::string model_dir, bool use_gpu,
               bool use_tensorrt = false, int batch_size = -1) {
  if (!FLAGS_prog_filename.empty() && !FLAGS_param_filename.empty()) {
    config->prog_file = model_dir + "/" + FLAGS_prog_filename;
    config->param_file = model_dir + "/" + FLAGS_param_filename;
  } else {
    config->model_dir = model_dir;
  }
  if (use_gpu) {
    config->use_gpu = true;
    config->device = 0;
    config->fraction_of_gpu_memory = 0.15;
  }
}

template <>
void SetConfig<AnalysisConfig>(AnalysisConfig* config, std::string model_dir,
                               bool use_gpu, bool use_tensorrt,
                               int batch_size) {
  if (!FLAGS_prog_filename.empty() && !FLAGS_param_filename.empty()) {
    config->SetModel(model_dir + "/" + FLAGS_prog_filename,
                     model_dir + "/" + FLAGS_param_filename);
  } else {
    config->SetModel(model_dir);
  }
  if (use_gpu) {
    config->EnableUseGpu(100, 0);
    if (use_tensorrt) {
      config->EnableTensorRtEngine(1 << 10, batch_size);
      config->pass_builder()->DeletePass("conv_bn_fuse_pass");
      config->pass_builder()->DeletePass("fc_fuse_pass");
      config->pass_builder()->TurnOnDebug();
    } else {
      config->SwitchIrOptim();
    }
  }
}

int GenerateInputList(std::vector<std::string>* input_list,
                      std::string& input_dir) {
  DIR* dir = NULL;
  if ((dir = opendir(input_dir.c_str())) == NULL) {
    std::cerr << "Open image dir failed. Dir: " << input_dir << std::endl;
    return -1;
  }

  input_list->clear();
  struct dirent* ptr = NULL;
  while ((ptr = readdir(dir)) != NULL) {
    if (ptr->d_name[0] == '.') {
      continue;
    }
    std::string filename = std::string(ptr->d_name);
    if (filename.length() < 5) {
      std::cerr << "Wrong image file format: " << filename << std::endl;
      continue;
    } else {
      std::string suffix = filename.substr(filename.length() - 4, 4);
      if (suffix.compare(".txt") != 0) {
        std::cerr << "Wrong image file format: " << filename << std::endl;
        continue;
      }
    }
    input_list->push_back(filename);
  }
  closedir(dir);
  return 0;
}

void TestImpl(const PaddlePredictor::Config* config,
              std::vector<paddle::PaddleTensor>* outputs,
              bool use_analysis = true) {
  PrintConfig(config, use_analysis);

  int batch_size = FLAGS_batch_size;
  int num_times = FLAGS_repeat;
  auto predictor = CreateTestPredictor(config, use_analysis);

  if (FLAGS_profile) {
    paddle::platform::ResetProfiler();
  }

  std::vector<std::string> input_list;
  if (GenerateInputList(&input_list, FLAGS_image_dir)) {
    LOG(WARNING) << "Get no inputs in image_dir (" << FLAGS_image_dir
                 << "), use fake inputs instead.";
    input_list.push_back("dummpy");
  }

  double total_latency = 0.0f;
  for (size_t i = 0; i < input_list.size(); ++i) {
    std::vector<paddle::PaddleTensor> inputs;
    outputs->clear();

    if (input_list[i] != "dummpy") {
      std::string input_path = FLAGS_image_dir + "/" + input_list[i];
      SetInputs(inputs, input_path, FLAGS_image_dims);
    } else {
      std::string input_path = "";
      SetInputs(inputs, input_path, FLAGS_image_dims);
    }

    if (input_list.size() <= 1) {
      // warmup run
      LOG(INFO) << "Warm up run...";
      {
        Timer warmup_timer;
        warmup_timer.tic();
        predictor->Run(inputs, outputs, batch_size);
        PrintTime(batch_size, 1, 1, 0, warmup_timer.toc(), 1);
        if (FLAGS_profile) {
          paddle::platform::ResetProfiler();
        }
      }

      LOG(INFO) << "Run " << num_times << " times...";
    }

    {
      Timer run_timer;
      run_timer.tic();
      for (int r = 0; r < num_times; r++) {
        predictor->Run(inputs, outputs, batch_size);
      }
      double latency = run_timer.toc() / num_times;
      total_latency += latency;
    }
  }
  PrintTime(batch_size, num_times, 1, 0, total_latency, input_list.size());
}

template <typename T>
std::string DataToString1D(T* data, size_t length, size_t print_elements) {
  if (length < 1) {
    LOG(FATAL) << "Invalid data.";
  }

  std::ostringstream os;
  os << "[" << data[0];
  if (length <= 2 * print_elements) {
    for (size_t i = 1; i < length; ++i) {
      os << ", " << data[i];
    }
  } else {
    for (size_t i = 1; i < print_elements; ++i) {
      os << ", " << data[i];
    }
    os << " ... " << data[length - print_elements];
    for (size_t i = length - print_elements + 1; i < length; ++i) {
      os << ", " << data[i];
    }
  }
  os << "]";
  return os.str();
}

template <typename T>
std::string PrintData(T* data, const std::vector<int>& shape,
                      size_t print_elements) {
  size_t num_dims = shape.size();
  if (num_dims <= 0) {
    return "";
  }

  size_t num_cols = shape[num_dims - 1];

  if (num_dims == 1) {
    return DataToString1D(data, num_cols, print_elements);
  } else {
    size_t num_rows = 1;
    std::string spaces = "";
    std::ostringstream os;
    for (size_t i = 0; i < num_dims - 1; ++i) {
      os << "[";
      spaces += " ";
      num_rows *= shape[i];
    }

    os << DataToString1D(data, num_cols, print_elements);
    if (num_rows <= 2 * print_elements) {
      for (size_t i = 1; i < num_rows; ++i) {
        os << "\n"
           << spaces
           << DataToString1D(data + i * num_cols, num_cols, print_elements);
      }
    } else {
      for (size_t i = 1; i < print_elements; ++i) {
        os << "\n"
           << spaces
           << DataToString1D(data + i * num_cols, num_cols, print_elements);
      }
      os << "\n" << spaces << "...\n" << spaces << "...\n" << spaces << "...";
      for (size_t i = num_rows - print_elements; i < num_rows; ++i) {
        os << "\n"
           << spaces
           << DataToString1D(data + i * num_cols, num_cols, print_elements);
      }
    }

    for (size_t i = 0; i < num_dims - 1; ++i) {
      os << "]";
    }
    return os.str();
  }
}

void PrintTensor(const PaddleTensor& tensor, size_t print_elements = 0) {
  LOG(INFO) << "name: " << tensor.name;
  LOG(INFO) << "shape: " << tensor.shape;
  LOG(INFO) << "lod: " << tensor.lod;
  if (tensor.dtype == PaddleDType::FLOAT32) {
    LOG(INFO) << "dtype: PaddleDType::FLOAT32";
    LOG(INFO) << "data:\n"
              << PrintData<float>(static_cast<float*>(tensor.data.data()),
                                  tensor.shape, print_elements);
  } else if (tensor.dtype == PaddleDType::INT64) {
    LOG(INFO) << "datype: PaddleDType::INT64";
    LOG(INFO) << "data:\n"
              << PrintData<int64_t>(static_cast<int64_t*>(tensor.data.data()),
                                    tensor.shape, print_elements);
  } else {
    LOG(FATAL) << "Unsupported dtype.";
  }
}

}  // namespace inference
}  // namespace paddle
