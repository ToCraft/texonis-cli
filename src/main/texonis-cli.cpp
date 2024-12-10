#include <iostream>
#include "llama.h"

int main() {
  llama_context_params contextParams = llama_context_default_params();
  std::cout << "Hello World!" << contextParams.n_ctx;
  return 0;
}
