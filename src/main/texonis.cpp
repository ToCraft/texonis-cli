#ifndef TEXONIS_CPP
#define TEXONIS_CPP
#include "texonis.h"

namespace texonis {
   void init() {
     llama_backend_init();
   }

   void deInit() {
     llama_backend_free();
   }

   llama_model* loadModel(std::string modelPath, llama_model_params& params) {
     return llama_load_model_from_file(modelPath.c_str(), params);
   }

  llama_batch prompt(llama_model* model, std::string input, bool bos, bool special) {
    
  }
}
#endif
