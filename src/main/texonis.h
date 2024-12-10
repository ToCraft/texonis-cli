#ifndef TEXONIS_H
#define TEXONIS_H
#include <string>
#include <vector>
#include "llama.h"

namespace texonis {
  void init();
  void deInit(llama_sampler* smpl, llama_context* ctx, llama_model* model);
  llama_model_params modelParams(int n_gpu_layers);
  llama_context_params contextParams(int n_ctx);
  llama_sampler* createSampler(float minP, size_t minK, float temp, long seed);
  llama_model* loadModel(std::string modelPath, llama_model_params& params);
  std::vector<llama_token> tokenize(llama_model* model, llama_context* ctx, std::string prompt);
  std::string generate(llama_model* model, llama_context* ctx, llama_sampler* smpl, std::vector<llama_token> prompt_tokens);
}
#endif
