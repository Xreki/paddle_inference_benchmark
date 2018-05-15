#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "v2/inference_helper.h"

static std::string g_merged_model_path = "";
static std::string g_config_path = "";
static std::string g_params_dirname = "";
static std::vector<int> g_dims = {};
static int g_repeat = 1;

/*
 * \brief parse command line arguments
 **/
void ParseArgs(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--merged_model_path") {
      g_merged_model_path = std::string(argv[++i]);
    } else if (std::string(argv[i]) == "--config_path") {
      g_config_path = std::string(argv[++i]);
    } else if (std::string(argv[i]) == "--params_dirname") {
      g_params_dirname = std::string(argv[++i]);
    } else if (std::string(argv[i]) == "--repeat") {
      g_repeat = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--dims") {
      std::string token;
      std::istringstream stream(std::string(argv[++i]));
      while (std::getline(stream, token, 'x')) {
        g_dims.push_back(std::stoi(token));
      }
    }
  }
}

int main(int argc, char* argv[]) {
  ParseArgs(argc, argv);

  InferenceHelper helper(true);
  if (!g_merged_model_path.empty()) {
    helper.Init(g_merged_model_path);
  } else if (!g_config_path.empty()) {
    helper.Init(g_config_path, g_params_dirname);
  } else {
    std::cout << "Please specify the model path, use --merged_model_path or "
                 "--config_path and --params_dirname.";
    exit(1);
  }
  helper.Infer(g_dims, g_repeat);
  helper.Release();
  return 0;
}
