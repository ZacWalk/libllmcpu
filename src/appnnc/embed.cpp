// nnc app — text-embedding models. See embed.h.

#include "embed.h"

#include "platform.h"
#include "ui.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace
{
	struct task
	{
		const char* name;
		const char* prefix;
	};

	// EmbeddingGemma's prompts, from its model card.
	const task k_tasks[] = {
		{"none", ""},
		{"query", "task: search result | query: "},
		{"document", "title: none | text: "},
		{"similarity", "task: sentence similarity | query: "},
		{"classification", "task: classification | query: "},
		{"clustering", "task: clustering | query: "},
		{"qa", "task: question answering | query: "},
		{"fact", "task: fact checking | query: "},
		{"code", "task: code retrieval | query: "},
	};

	std::string trim(const std::string& s)
	{
		const size_t a = s.find_first_not_of(" \t");
		if (a == std::string::npos) return {};
		const size_t b = s.find_last_not_of(" \t");
		return s.substr(a, b - a + 1);
	}

	// At most `max_bytes` of `s`, cut at a character boundary, with "..."
	// when something was cut.
	std::string clip(const std::string& s, const size_t max_bytes)
	{
		if (s.size() <= max_bytes) return s;
		size_t n = max_bytes;
		while (n > 0 && (static_cast<unsigned char>(s[n]) & 0xC0) == 0x80) --n;
		return s.substr(0, n) + "...";
	}

	void print_json(const std::vector<float>& v)
	{
		std::string out = "[";
		char buf[32];
		for (size_t i = 0; i < v.size(); ++i)
		{
			std::snprintf(buf, sizeof(buf), i ? ",%.7g" : "%.7g", static_cast<double>(v[i]));
			out += buf;
		}
		out += "]\n";
		std::fwrite(out.data(), 1, out.size(), stdout);
	}

	// The vectors have unit length, so this is their cosine similarity.
	float dot(const std::vector<float>& a, const std::vector<float>& b)
	{
		double s = 0.0;
		for (size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * b[i];
		return static_cast<float>(s);
	}

	void print_help()
	{
		ui_info("Type a line to see how close in meaning it is to each line before it\n"
			"(1 = the same meaning, around 0 = unrelated).\n"
			"Commands:\n"
			"  /list     the lines so far\n"
			"  /reset    forget them\n"
			"  /exit     quit (so do Ctrl-D, or Ctrl-Z then Enter on Windows)\n"
			"  /help     this list");
	}
}

const char* embed_task_prefix(const std::string& name)
{
	for (const task& t : k_tasks)
		if (name == t.name) return t.prefix;
	return nullptr;
}

std::string embed_task_names()
{
	std::string s;
	for (const task& t : k_tasks)
	{
		if (!s.empty()) s += ", ";
		s += t.name;
	}
	return s;
}

int run_embed(const app_options& o, nnc_model* model)
{
	const nnc_model_info info = nnc_model_get_info(model);
	const int dim = o.dim > 0 ? o.dim : info.n_embd_out;
	if (dim > info.n_embd_out)
	{
		ui_error("%s makes vectors of %d values; --dim %d asks for more", info.name, info.n_embd_out, o.dim);
		nnc_model_free(model);
		return 1;
	}

	// Scripts get one vector per line of input, in order; a person typing
	// gets similarities, for which the "similarity" prompt suits best.
	const bool interactive = o.prompt.empty() && platform_stdin_is_terminal();
	const std::string task = !o.task.empty() ? o.task : (interactive ? "similarity" : "none");
	const std::string prefix = embed_task_prefix(task);

	std::vector<float> vec(static_cast<size_t>(dim));
	auto embed = [&](const std::string& text)
	{
		const std::string full = prefix + text;
		if (nnc_embed(model, full.data(), full.size(), vec.data(), dim) == NNC_OK) return true;
		ui_error("%s", nnc_last_error());
		return false;
	};

	int rc = 0;
	if (!interactive)
	{
		if (!o.prompt.empty())
		{
			rc = embed(o.prompt) ? 0 : 1;
			if (rc == 0) print_json(vec);
		}
		else
		{
			std::string line;
			while (rc == 0 && platform_read_line(line))
			{
				if (!embed(line)) rc = 1;
				else print_json(vec);
			}
		}
		std::fflush(stdout);
		nnc_model_free(model);
		return rc;
	}

	ui_info("%s%s%s (%s, %d layers) - a text-embedding model: %d values per text, task \"%s\"", ui_bold(), info.name,
	        ui_reset(), info.architecture, info.n_layer, dim, task.c_str());
	ui_info("%sType a line to see how close in meaning it is to the lines before it. /help lists commands.%s",
	        ui_dim(), ui_reset());

	std::vector<std::pair<std::string, std::vector<float>>> seen;
	std::string line;
	for (;;)
	{
		std::fflush(stdout);
		std::fprintf(stderr, "\n%s>%s ", ui_bold(), ui_reset());
		std::fflush(stderr);
		if (!platform_read_line(line))
		{
			std::fputc('\n', stderr);
			break;
		}
		const std::string msg = trim(line);
		if (msg.empty()) continue;
		if (msg == "/exit" || msg == "/quit" || msg == "/bye") break;
		if (msg == "/help" || msg == "/?")
		{
			print_help();
			continue;
		}
		if (msg == "/reset" || msg == "/clear")
		{
			seen.clear();
			ui_info("%s(forgotten)%s", ui_dim(), ui_reset());
			continue;
		}
		if (msg == "/list")
		{
			for (size_t i = 0; i < seen.size(); ++i) std::printf("  %zu. %s\n", i + 1, seen[i].first.c_str());
			if (seen.empty()) ui_info("%s(no lines yet)%s", ui_dim(), ui_reset());
			continue;
		}
		if (msg[0] == '/' && msg.find(' ') == std::string::npos)
		{
			ui_info("Unknown command %s; /help lists the commands.", msg.c_str());
			continue;
		}

		const auto t0 = std::chrono::steady_clock::now();
		if (!embed(msg)) continue;
		const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

		std::vector<std::pair<float, size_t>> ranked;
		for (size_t i = 0; i < seen.size(); ++i) ranked.emplace_back(dot(vec, seen[i].second), i);
		std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
		const size_t shown = std::min<size_t>(ranked.size(), 5);
		for (size_t k = 0; k < shown; ++k)
		{
			const float sim = ranked[k].first;
			const int bar = static_cast<int>(std::clamp(sim, 0.0f, 1.0f) * 20.0f + 0.5f);
			std::string blocks;
			for (int b = 0; b < bar; ++b) blocks += "\xE2\x96\x88"; // U+2588 full block
			blocks.append(static_cast<size_t>(20 - bar), ' ');
			std::printf("  %5.2f  %s  %s\n", static_cast<double>(sim), blocks.c_str(),
			            clip(seen[ranked[k].second].first, 70).c_str());
		}
		if (ranked.size() > shown) ui_info("%s  (%zu more)%s", ui_dim(), ranked.size() - shown, ui_reset());
		ui_info("%s[%d values in %.0f ms%s]%s", ui_dim(), dim, ms,
		        seen.empty() ? "; type another line to compare it with this one" : "", ui_reset());
		if (o.perf)
		{
			std::fputs(nnc_perf_report("embedding", 0), stderr);
			nnc_perf_reset();
		}
		seen.emplace_back(msg, vec);
	}
	nnc_model_free(model);
	return 0;
}
