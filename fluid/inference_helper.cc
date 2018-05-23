#include "fluid/inference_helper.h"
#include <iostream>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/profiler.h"
#include "utils/timer.h"
#include "utils/utils.h"

InferenceHelper::InferenceHelper(bool use_gpu, bool enable_profiler)
    : use_gpu_(use_gpu), enable_profiler_(enable_profiler) {
  std::vector<std::string> argvs;
  argvs.push_back("--fraction_of_gpu_memory_to_use=0.1");
  if (use_gpu_) {
    argvs.push_back("--devices=0");
  }
  paddle::inference::Init(argvs);

  // 1. Define place, executor, scope
  if (use_gpu_) {
    place_ = paddle::platform::CUDAPlace(0);
  } else {
    place_ = paddle::platform::CPUPlace();
  }
  executor_ = new paddle::framework::Executor(place_);
}

void InferenceHelper::Init(const std::string& dirname) {
  Release();

  scope_ = new paddle::framework::Scope();

  // 2. Initialize the inference program
  {
    Timer time("load_program");
    program_ = paddle::inference::Load(executor_, scope_, dirname);
  }

  // 3. Optional: perform optimization on the inference program

  InitFeedFetchInfo();
}

void InferenceHelper::Init(const std::string& model_path,
                           const std::string& params_path) {
  Release();

  scope_ = new paddle::framework::Scope();

  // 2. Initialize the inference program
  {
    Timer time("load_program");
    program_ =
        paddle::inference::Load(executor_, scope_, model_path, params_path);
  }

  // 3. Optional: perform optimization on the inference program

  InitFeedFetchInfo();
}

void InferenceHelper::Infer(int repeat) {
  feed_targets_.clear();
  fetch_targets_.clear();

  // 5. Set up maps for feed and fetch targets
  // set_feed_variable
  paddle::framework::LoDTensor input;
  input.set_lod({{0, 1024}});
  paddle::framework::DDim input_dims = paddle::framework::make_ddim({1024, 1});
  int64_t* input_ptr =
      input.mutable_data<int64_t>(input_dims, paddle::platform::CPUPlace());
  Rand<int64_t>(input_ptr, input.numel(), 0, 5148);

  feed_targets_[feed_target_names_[0]] = &input;

  // get_fetch_variable
  paddle::framework::LoDTensor output;
  fetch_targets_[fetch_target_names_[0]] = &output;

  // Run the inference program
  executor_->Run(*program_, scope_, &feed_targets_, &fetch_targets_);

  if (enable_profiler_) {
    paddle::platform::ProfilerState state;
    if (use_gpu_) {
      state = paddle::platform::ProfilerState::kCUDA;
      // paddle::platform::SetDeviceId(0);
    } else {
      state = paddle::platform::ProfilerState::kCPU;
    }
    paddle::platform::EnableProfiler(state);
  }

  {
    Timer time("run_inference");
    for (int i = 0; i < repeat; ++i) {
      paddle::platform::RecordEvent record_event(
          "run_inference",
          paddle::platform::DeviceContextPool::Instance().Get(place_));
      executor_->Run(*program_, scope_, &feed_targets_, &fetch_targets_);
    }
  }
  if (enable_profiler_) {
    paddle::platform::DisableProfiler(
        paddle::platform::EventSortingKey::kDefault, "run_inference_profiler");
    paddle::platform::ResetProfiler();
  }
}

void InferenceHelper::Release() {
  delete scope_;
  scope_ = nullptr;
}

void InferenceHelper::PrintResults() {
  for (size_t i = 0; i < fetch_target_names_.size(); ++i) {
    paddle::framework::LoDTensor* output =
        fetch_targets_[fetch_target_names_[i]];
    std::cout << "dims " << i << ": " << output->dims() << std::endl;
    if (output->type().hash_code() == typeid(float).hash_code()) {
      float* output_ptr = output->data<float>();
      Print<float>(output_ptr, output->numel(),
                   "results " + std::to_string(i) + ":");
    } else if (output->type().hash_code() == typeid(int64_t).hash_code()) {
      int64_t* output_ptr = output->data<int64_t>();
      Print<int64_t>(output_ptr, output->numel(),
                     "results " + std::to_string(i) + ":");
    }
  }
}

std::vector<std::vector<int64_t>> InferenceHelper::GetTargetShapes(
    const std::vector<std::string>& target_names) {
  auto& global_block = program_->Block(0);

  std::vector<std::vector<int64_t>> target_shapes;
  for (size_t i = 0; i < target_names.size(); ++i) {
    auto* var = global_block.FindVar(target_names[i]);
    std::vector<int64_t> var_shape = var->GetShape();
    target_shapes.push_back(var_shape);
  }

  return target_shapes;
}

void InferenceHelper::InitFeedFetchInfo() {
  // 4. Get the feed_target_names and fetch_target_names
  feed_target_names_ = program_->GetFeedTargetNames();
  feed_target_shapes_ = GetTargetShapes(feed_target_names_);
  for (size_t i = 0; i < feed_target_names_.size(); ++i) {
    std::cout << "feed targets names " << i << ": " << feed_target_names_[i]
              << " " << Dims2String<int64_t>(feed_target_shapes_[i])
              << std::endl;
  }
  fetch_target_names_ = program_->GetFetchTargetNames();
  fetch_target_shapes_ = GetTargetShapes(fetch_target_names_);
  for (size_t i = 0; i < fetch_target_names_.size(); ++i) {
    std::cout << "fetch targets names " << i << ": " << fetch_target_names_[i]
              << " " << Dims2String<int64_t>(fetch_target_shapes_[i])
              << std::endl;
  }
}
