#include <string>
#include "v2/inference_helper.h"

static std::string g_merged_model_path = "";
static std::string g_config_path = "";
static std::string g_params_dirname = "";
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
      g_repeat = atoi(argv[++i]);
    }
  }
}

int main(int argc, char* argv[]) {
  ParseArgs(argc, argv);

  InferenceHelper helper(true);
  if (!g_merged_model_path.empty()) {
    helper.Init(g_merged_model_path);
  } else {
    helper.Init(g_config_path, g_params_dirname);
  }
  helper.Infer(g_repeat);
  helper.Release();
}
