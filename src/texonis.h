#ifndef TEXONIS_H
#define TEXONIS_H
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <functional>
#include "llama.h"

namespace texonis {
  void init();
  void deInit();
  llama_model_params modelParams(int n_gpu_layers);
  llama_context_params contextParams(int n_ctx);
  llama_sampler* createSampler(float minP, size_t minK, float temp, long seed);
  llama_model* loadModel(std::string modelPath, llama_model_params& params);
  std::vector<llama_token> tokenize(llama_model* model, llama_context* ctx, std::string prompt);
  std::string generate(llama_model* model, llama_context* ctx, llama_sampler* smpl, std::string prompt, std::function<bool(std::string)> func);
  std::string generate(llama_model* model, llama_context* ctx, llama_sampler* smpl, std::vector<llama_token> prompt_tokens, std::function<bool(std::string)> func);
  
	class Texonis {
		private:
			// generic
			llama_model* model;
			llama_context* ctx;
			llama_sampler* smpl;
			// chat-only
			int prev_len = 0;
			std::vector<llama_chat_message> messages;
			std::vector<char> formatted;
			
		public:			
			Texonis(std::string model_path, llama_model_params model_params, llama_context_params ctx_params, llama_sampler* smpl);
			
			std::string generateText(std::string prompt, std::function<bool(std::string)> func);
			
			void sendMessage(std::string role, std::string message);
			std::string generateMessage(std::string role, std::function<bool(std::string)> func);
			
			void free();
    };
    
    Texonis createLlm(std::string model_path, int ngl, int n_ctx, long seed);
    Texonis createLlm(std::string model_path, long seed);
}
#endif
