// nnc app — text-embedding models (EmbeddingGemma): vectors for scripts,
// and an interactive similarity explorer.

#pragma once

#include "cli.h"

#include <string>

// The prefix EmbeddingGemma expects for a task ("query", "document",
// "similarity", ...), "" for "none", or null for an unknown name.
const char* embed_task_prefix(const std::string& name);

// The task names, comma separated, for messages.
std::string embed_task_names();

// Run an embedding model that run_chat() has loaded; it frees the model.
// -p TEXT or piped lines print one JSON array per line; typed lines are
// compared with the lines before them.
int run_embed(const app_options& options, nnc_model* model);
