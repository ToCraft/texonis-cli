#ifndef TEXONIS_H
#define TEXONIS_H
#import <string>
#import "llama.h"

namespace texonis {
  void init();
  void deInit();
  llama_model* loadModel(std::string modelPath, llama_model_params& params);
}
#endif
