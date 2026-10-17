// nnc app — command-line options.
//
// One parser recognises every flag and mode; main() dispatches on the
// result. Adding a flag means adding it here (parse + usage) and nowhere
// else.

#pragma once

#include "nnc.h"

#include <cstdint>
#include <string>
#include <vector>

enum class app_mode
{
	chat, // the default: pick or download a model, then chat (or answer -p once)
	help,
	version,
	test,
	list_models,
	download,
	gguf_info,
	gguf_stats,
	inspect_model,
	inspect_all,
	model_info,
	probe,
	forward,
	tokenize,
	prompt,
	gen,
	perplexity,
};

struct app_options
{
	app_mode mode = app_mode::chat;
	std::vector<std::string> args; // the mode's positional arguments

	std::string model; // -m: a path, or a name from the download catalogue
	std::string prompt; // -p: answer this one message and exit
	std::string system_prompt; // --system
	std::string translate_from, translate_to; // --translate FROM:TO: translate instead of chatting
	std::string task; // --task: embedding models' task prefix (embed.h); "" = chosen by mode
	int dim = 0; // --dim: embedding values to keep; 0 = all
	std::string draft; // --draft: an MTP drafter; "" = look for one (Gemma 4)
	bool no_draft = false; // --no-draft
	bool kv_f32 = false; // --kv-f32: an F32 K/V cache
	bool think = false; // --think: let the model reason before it answers
	int n_draft = 0; // --draft-n: most tokens drafted per step; 0 = library default
	nnc_weight_format weights = NNC_WEIGHTS_Q8_0;
	int n_predict = 0; // -n: most tokens per reply; 0 = until the turn ends
	int n_ctx = 0; // -c: 0 = min(4096, the model's training context)
	int n_threads = 0; // -t: 0 = library default
	nnc_sampler_params sampler = nnc_sampler_default_params();
	uint64_t seed = 0; // 0 = random
	bool perf = false;
	bool verbose = false;
	bool yes = false; // answer yes to download questions
};

// Parse argv. On failure prints the problem to stderr and returns false.
bool parse_options(int argc, char** argv, app_options& out);

// "en:de" -> "en", "de". False unless both sides are non-empty.
bool parse_language_pair(const std::string& pair, std::string& from, std::string& to);

void print_usage(const char* argv0);
