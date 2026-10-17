// nnc app — command-line options. See cli.h.

#include "cli.h"

#include "embed.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace
{
	struct mode_flag
	{
		const char* name;
		app_mode mode;
		int min_args;
		int max_args;
	};

	const mode_flag k_modes[] = {
		{"--help", app_mode::help, 0, 0},
		{"-h", app_mode::help, 0, 0},
		{"--version", app_mode::version, 0, 0},
		{"--test", app_mode::test, 0, 0},
		{"--list-models", app_mode::list_models, 0, 0},
		{"--download", app_mode::download, 0, 1},
		{"--gguf-info", app_mode::gguf_info, 1, 1},
		{"--gguf-stats", app_mode::gguf_stats, 1, 2},
		{"--inspect-model", app_mode::inspect_model, 1, 1},
		{"--inspect-all", app_mode::inspect_all, 0, 0},
		{"--gemma-info", app_mode::model_info, 1, 1},
		{"--gemma-probe", app_mode::probe, 1, 2},
		{"--gemma-forward", app_mode::forward, 1, 2},
		{"--gemma-tokenize", app_mode::tokenize, 2, 2},
		{"--gemma-prompt", app_mode::prompt, 2, 4},
		{"--gemma-gen", app_mode::gen, 2, 4},
		{"--perplexity", app_mode::perplexity, 2, 3},
	};

	// Options that take a value, short and long spellings.
	const char* const k_value_options[] = {
		"-m", "--model", "-p", "--prompt", "-sys", "--system", "-n", "--n_predict", "--n-predict",
		"-c", "--ctx-size", "--ctx_size", "-t", "--threads", "--temp", "--top_k", "--top-k",
		"--top_p", "--top-p", "--min_p", "--min-p", "-s", "--seed", "-b", "--batch_size",
		"--translate", "--task", "--dim", "--draft", "-md", "--model-draft", "--draft-n",
	};

	const char* const k_flags[] = {
		"--bf16", "--q8", "--q4", "--perf", "-v", "--verbose", "-y", "--yes", "--no-draft", "--kv-f32", "--think",
	};

	template <size_t N>
	bool in(const char* const (&list)[N], const std::string_view a)
	{
		for (const char* s : list)
			if (a == s) return true;
		return false;
	}

	const mode_flag* find_mode(const std::string_view a)
	{
		for (const auto& m : k_modes)
			if (a == m.name) return &m;
		return nullptr;
	}

	bool known(const std::string& a)
	{
		return find_mode(a) || in(k_value_options, a) || in(k_flags, a);
	}

	// Older spellings: "-bf16", "-test", "/test" mean "--bf16", "--test".
	std::string normalise(const std::string& a)
	{
		if (a == "/test") return "--test";
		if (a.size() > 2 && a[0] == '-' && a[1] != '-' && !known(a) && known("-" + a)) return "-" + a;
		return a;
	}

	bool parse_int(const char* flag, const char* s, const int lo, const int hi, int& out)
	{
		errno = 0;
		char* end = nullptr;
		const long long v = std::strtoll(s, &end, 10);
		if (errno != 0 || end == s || *end != '\0' || v < lo || v > hi)
		{
			std::fprintf(stderr, "nnc: %s needs a whole number from %d to %d, not '%s'\n", flag, lo, hi, s);
			return false;
		}
		out = static_cast<int>(v);
		return true;
	}

	bool parse_float(const char* flag, const char* s, const float lo, const float hi, float& out)
	{
		errno = 0;
		char* end = nullptr;
		const double v = std::strtod(s, &end);
		if (errno != 0 || end == s || *end != '\0' || !std::isfinite(v) || v < lo || v > hi)
		{
			std::fprintf(stderr, "nnc: %s needs a number from %g to %g, not '%s'\n", flag, lo, hi, s);
			return false;
		}
		out = static_cast<float>(v);
		return true;
	}
}

bool parse_language_pair(const std::string& pair, std::string& from, std::string& to)
{
	const size_t colon = pair.find(':');
	if (colon == std::string::npos || colon == 0 || colon + 1 >= pair.size()
		|| pair.find(':', colon + 1) != std::string::npos)
		return false;
	from = pair.substr(0, colon);
	to = pair.substr(colon + 1);
	return true;
}

bool parse_options(const int argc, char** argv, app_options& o)
{
	bool have_mode = false;
	for (int i = 1; i < argc; ++i)
	{
		const std::string arg = normalise(argv[i]);
		const char* flag = argv[i];

		if (const mode_flag* m = find_mode(arg))
		{
			if (have_mode && o.mode != m->mode)
			{
				std::fprintf(stderr, "nnc: %s cannot be combined with another mode\n", flag);
				return false;
			}
			have_mode = true;
			o.mode = m->mode;
			// Required arguments are taken as they come; optional ones only
			// when they do not look like an option.
			int taken = 0;
			while (taken < m->max_args && i + 1 < argc
				&& (taken < m->min_args || argv[i + 1][0] != '-'))
			{
				o.args.emplace_back(argv[++i]);
				++taken;
			}
			if (taken < m->min_args)
			{
				std::fprintf(stderr, "nnc: %s needs %d argument%s (see --help)\n", flag, m->min_args,
				             m->min_args == 1 ? "" : "s");
				return false;
			}
			continue;
		}

		if (in(k_flags, arg))
		{
			if (arg == "--bf16") o.weights = NNC_WEIGHTS_BF16;
			else if (arg == "--q8") o.weights = NNC_WEIGHTS_Q8_0;
			else if (arg == "--q4") o.weights = NNC_WEIGHTS_Q4;
			else if (arg == "--perf") o.perf = true;
			else if (arg == "-v" || arg == "--verbose") o.verbose = true;
			else if (arg == "-y" || arg == "--yes") o.yes = true;
			else if (arg == "--no-draft") o.no_draft = true;
			else if (arg == "--kv-f32") o.kv_f32 = true;
			else if (arg == "--think") o.think = true;
			continue;
		}

		if (in(k_value_options, arg))
		{
			if (i + 1 >= argc)
			{
				std::fprintf(stderr, "nnc: %s needs a value\n", flag);
				return false;
			}
			const char* v = argv[++i];
			bool ok = true;
			if (arg == "-m" || arg == "--model") o.model = v;
			else if (arg == "-p" || arg == "--prompt") o.prompt = v;
			else if (arg == "-sys" || arg == "--system") o.system_prompt = v;
			else if (arg == "-n" || arg == "--n_predict" || arg == "--n-predict")
				ok = parse_int(flag, v, 0, INT_MAX, o.n_predict);
			else if (arg == "-c" || arg == "--ctx-size" || arg == "--ctx_size")
				ok = parse_int(flag, v, 0, 1 << 20, o.n_ctx);
			else if (arg == "-t" || arg == "--threads") ok = parse_int(flag, v, 0, 64, o.n_threads);
			else if (arg == "--temp") ok = parse_float(flag, v, 0.0f, 100.0f, o.sampler.temperature);
			else if (arg == "--top_k" || arg == "--top-k") ok = parse_int(flag, v, 0, INT_MAX, o.sampler.top_k);
			else if (arg == "--top_p" || arg == "--top-p") ok = parse_float(flag, v, 0.0f, 1.0f, o.sampler.top_p);
			else if (arg == "--min_p" || arg == "--min-p") ok = parse_float(flag, v, 0.0f, 1.0f, o.sampler.min_p);
			else if (arg == "--translate")
			{
				ok = parse_language_pair(v, o.translate_from, o.translate_to);
				if (!ok) std::fprintf(stderr, "nnc: %s needs two languages as FROM:TO, e.g. en:de, not '%s'\n", flag, v);
			}
			else if (arg == "--task")
			{
				ok = embed_task_prefix(v) != nullptr;
				if (ok) o.task = v;
				else std::fprintf(stderr, "nnc: %s needs one of %s, not '%s'\n", flag, embed_task_names().c_str(), v);
			}
			else if (arg == "--dim") ok = parse_int(flag, v, 1, 1 << 16, o.dim);
			else if (arg == "--draft" || arg == "-md" || arg == "--model-draft") o.draft = v;
			else if (arg == "--draft-n") ok = parse_int(flag, v, 1, 15, o.n_draft);
			else if (arg == "-s" || arg == "--seed")
			{
				// Negative (the old "-1") means "pick one at random", as does 0.
				errno = 0;
				char* end = nullptr;
				const long long s = std::strtoll(v, &end, 10);
				ok = errno == 0 && end != v && *end == '\0';
				if (!ok) std::fprintf(stderr, "nnc: %s needs a whole number, not '%s'\n", flag, v);
				o.seed = (ok && s > 0) ? static_cast<uint64_t>(s) : 0;
			}
			// -b is accepted for old scripts; prefill batching is fixed.
			if (!ok) return false;
			continue;
		}

		std::fprintf(stderr, "nnc: unknown argument '%s' (see --help)\n", flag);
		return false;
	}
	return true;
}

void print_usage(const char* argv0)
{
	std::printf(
		"nnc %s - run Gemma, Llama and Qwen GGUF models on the CPU\n"
		"\n"
		"usage: %s [options]            pick a model (or download one) and chat\n"
		"       %s -m MODEL -p TEXT     answer one message and exit\n"
		"\n"
		"model:\n"
		"  -m, --model FILE|NAME    a .gguf file, or a name from --download's list\n"
		"                           (default: choose from the models found)\n"
		"  --q8                     8-bit weights in memory (default)\n"
		"  --q4                     4-bit weights: ~1.6x faster, lower quality\n"
		"  --bf16                   keep 16-bit weights, used straight from the file\n"
		"  -c, --ctx-size N         context window in tokens (default: up to 4096)\n"
		"  --kv-f32                 keep the K/V cache in F32 (default BF16: half the memory)\n"
		"  -t, --threads N          worker threads (default: $NNC_THREADS, else up to 8)\n"
		"  --draft FILE             an MTP drafter, for faster replies (default: Gemma 4 models use\n"
		"                           an mtp-<model>.gguf found beside them or in the model folders)\n"
		"  --no-draft               do not use a drafter\n"
		"  --draft-n N              most tokens the drafter guesses at a time (default 4; 2 for 26B A4B)\n"
		"\n"
		"chat:\n"
		"  -p, --prompt TEXT        answer TEXT and exit; the reply goes to stdout\n"
		"  -sys, --system TEXT      instructions for the model, given before the first message\n"
		"  --think                  let the model reason before it answers (Gemma 4; Qwen 3, 3.5, 3.6\n"
		"                           and 3.8): better at hard problems, much longer replies; the\n"
		"                           reasoning is shown dimmed on stderr and fills the context too\n"
		"  -n, --n_predict N        most tokens per reply, reasoning included (default: until the\n"
		"                           reply ends)\n"
		"  --temp X                 sampling temperature; 0 = always the likeliest token (default 0.7)\n"
		"  --top-k N                sample from the N likeliest tokens; 0 = all (default 40)\n"
		"  --top-p X                ...that make up probability X (default 0.95)\n"
		"  --min-p X                ...each at least X times the likeliest (default 0.05)\n"
		"  -s, --seed N             random seed, for repeatable replies (default: random)\n"
		"  --perf                   print a per-phase timing breakdown after each reply\n"
		"  -v, --verbose            show what the library is doing\n"
		"  -y, --yes                download without asking\n"
		"\n"
		"  In the chat, /help lists commands; Ctrl-C stops a reply.\n"
		"\n"
		"translation:\n"
		"  --translate FROM:TO      translate each message instead of chatting, e.g. en:de\n"
		"                           (made for TranslateGemma; works with any chat model)\n"
		"\n"
		"embedding models (EmbeddingGemma):\n"
		"  Typed lines are compared with the ones before; piped lines, or -p TEXT,\n"
		"  print one JSON array of values per line.\n"
		"  --task NAME              prefix for the task: %s\n"
		"                           (default: similarity when typing, none otherwise;\n"
		"                           query and document for search)\n"
		"  --dim N                  keep the first N values (EmbeddingGemma: 768, 512, 256 or 128)\n"
		"\n"
		"modes (each runs instead of the chat):\n"
		"  --download [NAME]        download a model (no NAME: choose from a list)\n"
		"  --list-models            list the .gguf files nnc finds\n"
		"  --test                   run the self-tests\n"
		"  --version                print the version\n"
		"  --gguf-info FILE         a GGUF file's metadata and tensor table\n"
		"  --gguf-stats FILE [NAME] statistics of the F32/F16/BF16 tensors whose name holds NAME\n"
		"  --inspect-model FILE     whether nnc can load FILE; --inspect-all checks every model found\n"
		"  --gemma-info FILE, --gemma-probe FILE [ID], --gemma-forward FILE [ID],\n"
		"  --gemma-tokenize FILE \"TEXT\", --gemma-prompt FILE \"TEXT\" [N] [CTX],\n"
		"  --gemma-gen FILE \"ID ID ...\" [N] [CTX]\n"
		"                           loader, tokenizer and forward-pass diagnostics (greedy)\n"
		"  --perplexity FILE TEXTFILE [N]\n"
		"                           how well the model predicts the first N tokens of TEXTFILE (default 512)\n"
		"\n"
		"Models are looked for in ./models, ~/.lmstudio/models and ~/models.\n",
		nnc_version(), argv0, argv0, embed_task_names().c_str());
}
