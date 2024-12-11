#include "texonis.h"
#include <iostream>
#include <string>

using namespace texonis;

int main(int argc, char* argv[]) {
	// parse args
	std::string model_path;
	if (argc >= 2) {
		model_path = argv[1];
	} else {
		model_path = "/usr/local/texonis-cli/model.gguf";
	}

	long seed;
	if (argc > 2) {
		seed = std::stol(argv[2]);
	} else {
		seed = time(NULL);
	}
	std::cout << " Using seed " << seed << "\n";

	// initialize
	init();

	Texonis llm = createLlm(model_path, 99, 2048, seed);

	// set system prompt
	llm.sendMessage("system", "You are an adventure game the user is playing. The game only consists of text. The user is able to say what the main character will do. You will obey the user but still be realistic. For example, when the user wants to fly away but isn't a magician, they are most likely to fail. Briefly describe the environment of the user. Do not list possible actions.");
	// Generate
	auto output = [](std::string piece) -> bool {std::cout << piece << std::flush; return true;};
	llm.generateMessage("assistent", output);
    while (true) {
        // get user input
        printf("\n> ");
        std::string user;
        std::getline(std::cin, user);

        if (user.empty()) {
            break;
        }

        llm.sendMessage("user", user.c_str());

        try {
			llm.generateMessage("assistent", output);
		} catch (char* e) {
			std::cout << e;
			return 1;
		}
    }

	llm.free();
	deInit();
	return 0;
}
