#include "texonis.h"
#include "llama.h"
#include <iostream>
#include <string>

using namespace texonis;

int main() {
  llama_model_params modelP = llama_model_default_params();
  llama_model* model = loadModel("/data/data/com.termux/files/home/texonis-cli/build/test", modelP);
  std::cout << "LOL: " << model;
  return 0;
}
