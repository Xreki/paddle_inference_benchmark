#pragma once

#include "paddle/fluid/inference/io.h"

class InferenceHelper {
 public:
  InferenceHelper(bool use_gpu = false);
  ~InferenceHelper() {
    delete executor_;
    delete scope_;
  }

  void Init(const std::string& dirname);

  void Init(const std::string& model_path, const std::string& params_path);

  void Infer(int repeat);

  void Release();

  void PrintResults();

 private:
  std::vector<std::vector<int64_t>> GetTargetShapes(
      const std::vector<std::string>& target_names);

  void InitFeedFetchInfo();

  std::unique_ptr<paddle::framework::ProgramDesc> program_;
  std::vector<std::string> feed_target_names_;
  std::vector<std::string> fetch_target_names_;
  std::vector<std::vector<int64_t>> feed_target_shapes_;
  std::vector<std::vector<int64_t>> fetch_target_shapes_;
  std::map<std::string, const paddle::framework::LoDTensor*> feed_targets_;
  std::map<std::string, paddle::framework::LoDTensor*> fetch_targets_;

  paddle::platform::Place place_;
  paddle::framework::Executor* executor_{nullptr};
  paddle::framework::Scope* scope_{nullptr};

  bool use_gpu_{false};
};
