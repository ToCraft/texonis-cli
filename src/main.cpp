#include "texonis.h"
#include <algorithm>
#include <iostream>
#include <ctime>

using namespace texonis;

	char* getCmdOption(char ** begin, char ** end, const std::string & option) {
		char ** itr = std::find(begin, end, option);
		if (itr != end && ++itr != end)
		{
			return *itr;
		}
		return nullptr;
	}

	bool cmdOptionExists(char** begin, char** end, const std::string& option) {
		return std::find(begin, end, option) != end;
	}


int main(int argc, char* argv[]) {
	const std::string usage = "-m <model_path> [-s <seed>] [-a <assistent_name>] [-u <user_name>] [-sys <system_name>] [-msg <system>] [-text]|[-i]\n";
	
	// parse args
	std::string model_path;
	bool model_path_init = false;
	if (cmdOptionExists(argv, argv+argc, "-m")) {
		char* m = getCmdOption(argv, argv+argc, "-m");
		if (m) {
			model_path = m;
			model_path_init = true;
		}
	}
	if (!model_path_init) {
		std::cout << usage << std::flush;
		return 1;
	}

	long seed = time(NULL);
	if (cmdOptionExists(argv, argv+argc, "-s")) {
		char* s = getCmdOption(argv, argv+argc, "-s");
		if (s) {
			try {
				seed = std::stol(s);
			} catch(...) {
				seed = time(NULL);
			}
		}
	}
	
	std::string system_name = "system";
	if (cmdOptionExists(argv, argv+argc, "-sys")) {
		char* a = getCmdOption(argv, argv+argc, "-sys");
		if (a) {
			system_name = a;
		}
	}
	
	std::string assistent_name = "assistent";
	if (cmdOptionExists(argv, argv+argc, "-a")) {
		char* s = getCmdOption(argv, argv+argc, "-a");
		if (s) {
			assistent_name = s;
		}
	}
	
	std::string user_name = "user";
	if (cmdOptionExists(argv, argv+argc, "-u")) {
		char* u = getCmdOption(argv, argv+argc, "-u");
		if (u) {
			user_name = u;
		}
	}
	
	std::string system_prompt = "You are an adventure game the user is playing. The game only consists of text. The user is able to say what the main character will do. You will obey the user but still be realistic. For example, when the user wants to fly away but isn't a magician, they are most likely to fail. Briefly describe the environment of the user. Do not list possible actions.";
	if (cmdOptionExists(argv, argv+argc, "-msg")) {
		char* p = getCmdOption(argv, argv+argc, "-msg");
		if (p) {
			system_prompt = p;
		}
	}
	
	bool interactive = cmdOptionExists(argv, argv+argc, "-i");
	bool text_only = interactive || cmdOptionExists(argv, argv+argc, "-text");

	std::cout << "Using seed " << seed << "\n";
	std::cout << "Enter a blank message to exit\n" << std::flush;
	
	// initialize
	init();

	Texonis llm = createLlm(model_path, 99, 2048, seed);

	// Generate
	auto output = [](std::string piece) -> bool {std::cout << piece << std::flush; return true;};
	if (!text_only) {
		std::cout << "Using chat mode\n\n";
		// set system prompt
		llm.sendMessage(system_name, system_prompt); 
		llm.generateMessage(assistent_name, output);
		while (true) {
			// get user input
			printf("\n> ");
			std::string user;
			std::getline(std::cin, user);

			if (user.empty()) {
				break;
			}

			llm.sendMessage(user_name, user.c_str());

			try {
				llm.generateMessage(assistent_name, output);
			} catch (char* e) {
				std::cout << e;
				return 1;
			}
		}
	} else {
		if (interactive) {
			std::cout << "Using interactive text mode\n\n";
			std::string text = system_prompt;
			while (true) {
				try {
					text += llm.generateText(text, output);
				} catch (char* e) {
					std::cout << e;
					return 1;
				}
				
				// get user input
				printf("\n> ");
				std::string user;
				std::getline(std::cin, user);

				if (user.empty()) {
					break;
				}

				text += user;
				std::cout << std::endl << std::flush;
			}
		} else {
			std::cout << "Using text mode\n\n";
			// Generate
			llm.generateText(system_prompt, output);
		}
		std::cout << std::endl << std::flush;
	}

	llm.free();
	deInit();
	return 0;
}
