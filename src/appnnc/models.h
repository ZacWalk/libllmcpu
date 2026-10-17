// nnc app — finding, choosing and downloading models.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct catalog_entry
{
	const char* name;
	const char* file;
	const char* url;
	uint64_t bytes;
	const char* description;
};

// The models nnc can download (model_catalog.def), recommended one first.
const std::vector<catalog_entry>& model_catalog();

// The entry whose name or file name is `name`, or null.
const catalog_entry* find_catalog_entry(const std::string& name);

// .gguf files under ./models, ~/.lmstudio/models and ~/models that can be
// chatted with or embedded with: without vision projectors (mmproj-*) and
// drafters (mtp-*, dflash-*), each found once.
std::vector<std::string> find_models();

// A multi-token-prediction drafter for the model at `model_path`: a file
// mtp-<name>[-<quant>].gguf, next to the model or in the folders above,
// where the model's file name starts with <name> ("gemma-4-E2B-it" for
// gemma-4-E2B-it-Q4_K_M.gguf). "" when there is none.
std::string find_draft_model(const std::string& model_path);

// The catalogue entry of a drafter for the model at `model_path`, or null.
const catalog_entry* find_catalog_drafter(const std::string& model_path);

// Where downloads go: ./models if that folder exists, else ~/models.
std::string download_dir();

// The model to run: `requested` (-m) if given — a path, or a catalogue name
// that is downloaded if missing — else the one model found, else a choice
// between the models found, else an offer to download one. `yes` answers
// download questions. Returns "" after telling the user why.
std::string resolve_model(const std::string& requested, bool yes);

int run_list_models();
int run_download(const std::string& name, bool yes);
