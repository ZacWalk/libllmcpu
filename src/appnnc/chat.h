// nnc app — the chat REPL and one-shot prompts, built on the public API.

#pragma once

#include "cli.h"

#include <string>

// Pick, download or open the model, then chat — interactively, or answer
// options.prompt once.
int run_chat(const app_options& options);

// Load `path` with a progress bar; Ctrl-C cancels. Returns null after
// reporting the failure.
nnc_model* load_model_ui(const std::string& path, nnc_weight_format weights);
