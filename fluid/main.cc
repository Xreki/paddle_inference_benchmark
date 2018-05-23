#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "fluid/inference_helper.h"

static std::string g_dirname = "";
static std::string g_model_path = "";
static std::string g_params_path = "";
static std::vector<int> g_dims = {};
static int g_repeat = 1;
static bool g_use_gpu = false;
static bool g_enable_profiler = false;

/*
 * \brief parse command line arguments
 **/
void ParseArgs(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--dirname") {
      g_dirname = std::string(argv[++i]);
    } else if (std::string(argv[i]) == "--model_path") {
      g_model_path = std::string(argv[++i]);
    } else if (std::string(argv[i]) == "--params_path") {
      g_params_path = std::string(argv[++i]);
    } else if (std::string(argv[i]) == "--repeat") {
      g_repeat = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--dims") {
      std::string token;
      std::istringstream stream(std::string(argv[++i]));
      while (std::getline(stream, token, 'x')) {
        g_dims.push_back(std::stoi(token));
      }
    } else if (std::string(argv[i]) == "--use_gpu") {
      g_use_gpu = std::stoi(argv[++i]) == 0 ? false : true;
    } else if (std::string(argv[i]) == "--enable_profiler") {
      g_enable_profiler = std::stoi(argv[++i]) == 0 ? false : true;
    }
  }
}

int main(int argc, char* argv[]) {
  ParseArgs(argc, argv);

  InferenceHelper helper(g_use_gpu, g_enable_profiler);
  if (!g_model_path.empty() && !g_params_path.empty()) {
    helper.Init(g_model_path, g_params_path);
  } else if (!g_dirname.empty()) {
    helper.Init(g_dirname);
  } else {
    std::cout << "Please specify the model path, use --dirname or "
                 "--config_path and --params_path.";
    exit(1);
  }
  helper.Infer(g_repeat);
  helper.PrintResults();
  helper.Release();
  return 0;
}
