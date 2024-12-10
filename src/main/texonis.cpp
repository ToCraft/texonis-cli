#ifndef TEXONIS_CPP
#define TEXONIS_CPP
#include "texonis.h"

namespace texonis {
   void init() {
    // only print errors
    llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    // load dynamic backends
    ggml_backend_load_all();
   }

  void deInit(llama_sampler* smpl, llama_context* ctx, llama_model* model) {
	  llama_sampler_free(smpl);
	  llama_free(ctx);
	  llama_free_model(model);
	  
	  llama_backend_free();
   }

   llama_model_params modelParams(int n_gpu_layers) {
	   llama_model_params model_params = llama_model_default_params();
	   model_params.n_gpu_layers = n_gpu_layers;
	   return model_params;
   }
   
   llama_context_params contextParams(int n_ctx) {
	   llama_context_params ctx_params = llama_context_default_params();
	   ctx_params.n_ctx = n_ctx;
	   ctx_params.n_batch = n_ctx;
	   return ctx_params;
   }
   
   llama_sampler* createSampler(float minP, size_t minK, float temp, long seed) {
	   // initialize the sampler
	   llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
	   llama_sampler_chain_add(smpl, llama_sampler_init_min_p(minP, minK));
	   llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
	   llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
	   return smpl;
   }

   llama_model* loadModel(std::string modelPath, llama_model_params& params) {
     return llama_load_model_from_file(modelPath.c_str(), params);
   }
   
      
   std::vector<llama_token> tokenize(llama_model* model, llama_context* ctx, std::string prompt) {
        const int n_prompt_tokens = -llama_tokenize(model, prompt.c_str(), prompt.size(), NULL, 0, true, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(model, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), llama_get_kv_cache_used_cells(ctx) == 0, true) < 0) {
            GGML_ABORT("failed to tokenize the prompt\n");
        }
        return prompt_tokens;
   }
   
   std::string generate(llama_model* model, llama_context* ctx, llama_sampler* smpl, std::vector<llama_token> prompt_tokens) {
	   llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
	   std::string response;
	   llama_token new_token_id;
        while (true) {
            // check if we have enough space in the context to evaluate this batch
            int n_ctx = llama_n_ctx(ctx);
            int n_ctx_used = llama_get_kv_cache_used_cells(ctx);
            if (n_ctx_used + batch.n_tokens > n_ctx) {
                printf("\033[0m\n");
                fprintf(stderr, "context size exceeded\n");
                exit(0);
            }

            if (llama_decode(ctx, batch)) {
                GGML_ABORT("failed to decode\n");
            }

            // sample the next token
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_token_is_eog(model, new_token_id)) {
                break;
            }

            // convert the token to a string, print it and add it to the response
            char buf[256];
            int n = llama_token_to_piece(model, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                GGML_ABORT("failed to convert token to piece\n");
            }
            std::string piece(buf, n);
            printf("%s", piece.c_str());
            fflush(stdout);
            response += piece;

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);
        }
        
        return response;
   }
}
#endif
