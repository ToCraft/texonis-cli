#include "texonis.h"
#include "llama.h"
#include <iostream>
#include <string>

using namespace texonis;

int main(int argc, char* argv[]) {
	std::string model_path;
	if (argc >= 2) {
		model_path = argv[1];
	} else {
		std::cout << "LOL: " << argv[0];
		model_path = "/usr/local/texonis-cli/model.gguf";
	}

	std::string input;
	if (argc > 2) {
		input = argv[2];
	} else {
		input = "";
	}
	long seed;
	if (argc > 3) {
		seed = std::stol(argv[3]);
	} else {
		seed = time(NULL);
	}
	std::cout << " Using seed " << seed << "\n";
    int ngl = 99;
    int n_ctx = 2048;

	texonis::init();

	// load model
	llama_model_params model_params = modelParams(ngl);
	llama_model*  model = loadModel(model_path, model_params);
	if (!model) {
		std::cout << "Couldn't load model at " << model_path;
		return 1;
	}

	// initialize the context
    llama_context_params ctx_params = contextParams(n_ctx);
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
		std::cout << "Couldn't create the llama context!";
        return 1;
    }

    // initialize the sampler
    llama_sampler* smpl = createSampler(0.05f, 1, 0.8f, seed);

    // Generate
    std::vector<llama_token> tokens = tokenize(model, ctx, input);
    std::string output = generate(model, ctx, smpl, tokens);
    std::cout << " You got: " << output;

	texonis::deInit(smpl, ctx, model);
	return 0;
}
