// nnc app — finding, choosing and downloading models. See models.h.

#include "models.h"

#include "platform.h"
#include "ui.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>

namespace fs = std::filesystem;

namespace
{
	bool iequals_suffix(const std::string& s, const char* suffix)
	{
		const size_t n = std::strlen(suffix);
		if (s.size() < n) return false;
		for (size_t i = 0; i < n; ++i)
		{
			char a = s[s.size() - n + i];
			char b = suffix[i];
			if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
			if (b >= 'A' && b <= 'Z') b = static_cast<char>(b - 'A' + 'a');
			if (a != b) return false;
		}
		return true;
	}

	bool icontains(std::string s, const char* needle)
	{
		for (char& c : s)
			if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
		return s.find(needle) != std::string::npos;
	}

	std::string lower(std::string s)
	{
		for (char& c : s)
			if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
		return s;
	}

	// Drafters only help another model generate: mtp-* (which nnc uses with
	// Gemma 4) and dflash-* (which it does not run).
	bool is_drafter_file(const std::string& name)
	{
		const std::string n = lower(name);
		return n.rfind("mtp-", 0) == 0 || n.rfind("dflash-", 0) == 0;
	}

	// "q8_0", "q4_k_m", "iq4_xs", "bf16", "f16", "f32", "ud": a file name's
	// quantization part (lower case).
	bool is_quant_label(const std::string& s)
	{
		if (s == "bf16" || s == "f16" || s == "f32" || s == "ud") return true;
		const size_t i = s.rfind("iq", 0) == 0 ? 2 : (s.rfind('q', 0) == 0 ? 1 : std::string::npos);
		return i != std::string::npos && i < s.size() && s[i] >= '0' && s[i] <= '9';
	}

	// Whether `drafter` (a file name, mtp-<name>[-<quant>].gguf) is made for
	// the model file `model`: its file name starts with <name>, then '-' or '.'.
	bool drafter_fits(const std::string& drafter, const std::string& model)
	{
		const std::string d = lower(drafter), target = lower(model);
		if (d.rfind("mtp-", 0) != 0 || !iequals_suffix(d, ".gguf") || d.size() <= 9) return false;
		std::string stem = d.substr(4, d.size() - 4 - 5);
		for (size_t dash; (dash = stem.find_last_of('-')) != std::string::npos && is_quant_label(stem.substr(dash + 1));)
			stem.resize(dash);
		if (stem.empty() || target.size() <= stem.size() || target.compare(0, stem.size(), stem) != 0) return false;
		const char after = target[stem.size()];
		return after == '-' || after == '.';
	}

	std::vector<std::string> search_dirs()
	{
		std::vector<std::string> dirs{"models"};
		const std::string home = platform_home_dir();
		if (!home.empty())
		{
			dirs.push_back(path_u8(u8_path(home) / ".lmstudio" / "models"));
			dirs.push_back(path_u8(u8_path(home) / "models"));
		}
		return dirs;
	}

	uint64_t file_size(const std::string& path)
	{
		std::error_code ec;
		const uint64_t n = fs::file_size(u8_path(path), ec);
		return ec ? 0 : n;
	}

	bool is_gguf(const std::string& path)
	{
		std::ifstream f(u8_path(path), std::ios::binary);
		char magic[4] = {};
		return f.read(magic, 4) && std::memcmp(magic, "GGUF", 4) == 0;
	}

	// Read a menu choice: a number in [1, count], "" for `fallback`, or one
	// of the letters in `letters`. Returns the number, or the letter as a
	// negative char code, or 0 at end of input.
	int read_choice(const char* prompt, const int count, const int fallback, const char* letters)
	{
		for (;;)
		{
			std::fprintf(stderr, "%s", prompt);
			std::fflush(stderr);
			std::string line;
			if (!platform_read_line(line)) return 0;
			const size_t a = line.find_first_not_of(" \t");
			if (a == std::string::npos) return fallback;
			const size_t b = line.find_last_not_of(" \t");
			const std::string s = line.substr(a, b - a + 1);
			if (s.size() == 1 && std::strchr(letters, s[0] | 0x20)) return -(s[0] | 0x20);
			char* end = nullptr;
			const long v = std::strtol(s.c_str(), &end, 10);
			if (*end == '\0' && v >= 1 && v <= count) return static_cast<int>(v);
			std::fprintf(stderr, "Please type a number from 1 to %d%s.\n", count, *letters ? ", or a letter shown" : "");
		}
	}

	struct download_state
	{
		progress_bar* bar;
		std::chrono::steady_clock::time_point start;
		uint64_t start_bytes = 0;
		bool started = false;
	};

	bool on_download_progress(const uint64_t done, const uint64_t total, void* user)
	{
		auto& st = *static_cast<download_state*>(user);
		if (platform_interrupted()) return false;
		const auto now = std::chrono::steady_clock::now();
		if (!st.started)
		{
			st.started = true;
			st.start = now;
			st.start_bytes = done;
		}
		const double secs = std::chrono::duration<double>(now - st.start).count();
		const double rate = secs > 0.5 ? static_cast<double>(done - st.start_bytes) / secs : 0.0;
		std::string detail = format_bytes(done) + " / " + (total ? format_bytes(total) : std::string("?"));
		if (rate > 0)
		{
			detail += "  " + format_bytes(static_cast<uint64_t>(rate)) + "/s";
			if (total > done) detail += "  " + format_duration(static_cast<double>(total - done) / rate) + " left";
		}
		st.bar->update(total ? static_cast<double>(done) / static_cast<double>(total) : 0.0, detail);
		return true;
	}

	// Download `e` to download_dir(). Asks first unless `yes`.
	std::string download_model(const catalog_entry& e, const bool yes)
	{
		const std::string dir = download_dir();
		const std::string path = path_u8(u8_path(dir) / e.file);
		if (file_size(path) == e.bytes) return path;

		if (!yes)
		{
			if (!platform_stdin_is_terminal())
			{
				ui_error("%s is not downloaded; run nnc --download %s, or add --yes", e.name, e.name);
				return {};
			}
			const std::string q = "Download " + std::string(e.file) + " (" + format_bytes(e.bytes) + ") to "
				+ tidy_path(path_u8(fs::absolute(u8_path(dir)))) + "?";
			if (!ui_confirm(q.c_str(), true)) return {};
		}

		std::error_code ec;
		fs::create_directories(u8_path(dir), ec);
		if (ec)
		{
			ui_error("cannot create '%s': %s", dir.c_str(), ec.message().c_str());
			return {};
		}

		const std::string part = path + ".part";
		if (file_size(part) > e.bytes) fs::remove(u8_path(part), ec); // not ours to resume
		if (file_size(part) != e.bytes)
		{
			progress_bar bar("Downloading " + std::string(e.file));
			download_state st{&bar, {}};
			std::string error;
			bool ok;
			{
				interrupt_scope scope; // Ctrl-C stops the download instead of ending nnc
				ok = platform_download(e.url, part, e.bytes, on_download_progress, &st, error);
			}
			if (!ok)
			{
				bar.finish("stopped");
				if (error == "cancelled")
					ui_info("Download stopped. Run the same command again to resume it.");
				else
					ui_error("%s. Run the same command again to resume.", error.c_str());
				return {};
			}
			bar.finish("done");
		}

		if (file_size(part) != e.bytes || !is_gguf(part))
		{
			ui_error("%s did not arrive intact (expected %llu bytes); delete '%s' and try again", e.file,
			         static_cast<unsigned long long>(e.bytes), tidy_path(part).c_str());
			return {};
		}
		std::string replace_error;
		if (!platform_replace_file(part, path, replace_error))
		{
			ui_error("cannot replace '%s': %s", path.c_str(), replace_error.c_str());
			return {};
		}
		ui_info("Saved %s", tidy_path(path_u8(fs::absolute(u8_path(path)))).c_str());
		return path;
	}

	// Offer the catalogue (without drafters, which cannot run on their own).
	// With `yes`, takes the recommended model.
	std::string choose_download(const bool yes)
	{
		const auto& all = model_catalog();
		if (yes) return download_model(all.front(), true);
		std::vector<const catalog_entry*> cat;
		for (const auto& e : all)
			if (!is_drafter_file(e.file)) cat.push_back(&e);

		std::fprintf(stderr, "\nModels nnc can download:\n");
		for (size_t i = 0; i < cat.size(); ++i)
			std::fprintf(stderr, "  %zu) %-34s %8s  %s%s%s\n", i + 1, cat[i]->file, format_bytes(cat[i]->bytes).c_str(),
			             ui_dim(), cat[i]->description, ui_reset());
		std::fprintf(stderr, "Downloads are saved to %s.\n",
		             tidy_path(path_u8(fs::absolute(u8_path(download_dir())))).c_str());
		const int c = read_choice("Choose a model [1], or q to quit: ", static_cast<int>(cat.size()), 1, "q");
		if (c <= 0) return {};
		return download_model(*cat[static_cast<size_t>(c - 1)], true);
	}
}

const std::vector<catalog_entry>& model_catalog()
{
	static const std::vector<catalog_entry> entries = {
#define NNC_MODEL(name, file, url, bytes, description) {name, file, url, bytes, description},
#include "model_catalog.def"
#undef NNC_MODEL
	};
	return entries;
}

const catalog_entry* find_catalog_entry(const std::string& name)
{
	for (const auto& e : model_catalog())
		if (name == e.name || name == e.file) return &e;
	return nullptr;
}

std::vector<std::string> find_models()
{
	std::vector<std::string> out;
	std::set<fs::path> seen;
	for (const std::string& dir : search_dirs())
	{
		std::error_code ec;
		const fs::path root = u8_path(dir);
		if (!fs::is_directory(root, ec)) continue;
		std::vector<std::string> found;
		for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
		     !ec && it != end; it.increment(ec))
		{
			if (!it->is_regular_file(ec)) continue;
			const std::string name = path_u8(it->path().filename());
			if (!iequals_suffix(name, ".gguf") || icontains(name, "mmproj") || is_drafter_file(name)) continue;
			const fs::path canon = fs::weakly_canonical(it->path(), ec);
			if (!seen.insert(ec ? it->path() : canon).second) continue;
			found.push_back(path_u8(it->path()));
		}
		std::sort(found.begin(), found.end());
		out.insert(out.end(), found.begin(), found.end());
	}
	return out;
}

std::string find_draft_model(const std::string& model_path)
{
	const fs::path model = u8_path(model_path);
	const std::string target = path_u8(model.filename());
	std::vector<std::string> dirs{path_u8(model.parent_path().empty() ? fs::path(".") : model.parent_path())};
	for (const std::string& d : search_dirs()) dirs.push_back(d);
	for (const std::string& dir : dirs)
	{
		std::error_code ec;
		const fs::path root = u8_path(dir);
		if (!fs::is_directory(root, ec)) continue;
		for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
		     !ec && it != end; it.increment(ec))
			if (it->is_regular_file(ec) && drafter_fits(path_u8(it->path().filename()), target))
				return path_u8(it->path());
	}
	return {};
}

const catalog_entry* find_catalog_drafter(const std::string& model_path)
{
	const std::string target = path_u8(u8_path(model_path).filename());
	for (const auto& e : model_catalog())
		if (drafter_fits(e.file, target)) return &e;
	return nullptr;
}

std::string download_dir()
{
	std::error_code ec;
	if (fs::is_directory(u8_path("models"), ec)) return "models";
	const std::string home = platform_home_dir();
	return home.empty() ? std::string("models") : path_u8(u8_path(home) / "models");
}

std::string resolve_model(const std::string& requested, const bool yes)
{
	if (!requested.empty())
	{
		std::error_code ec;
		if (fs::is_regular_file(u8_path(requested), ec)) return requested;
		if (const catalog_entry* e = find_catalog_entry(requested))
		{
			for (const std::string& p : find_models())
				if (path_u8(u8_path(p).filename()) == e->file && file_size(p) == e->bytes) return p;
			return download_model(*e, yes);
		}
		ui_error("there is no model file '%s'. Run nnc --download to see the models nnc can fetch.",
		         requested.c_str());
		return {};
	}

	const std::vector<std::string> models = find_models();
	const bool interactive = platform_stdin_is_terminal();
	if (models.empty())
	{
		ui_info("No models found in ./models, ~/.lmstudio/models or ~/models.");
		if (!interactive && !yes)
		{
			ui_error("pass -m FILE, or run nnc --download to fetch a model");
			return {};
		}
		return choose_download(yes);
	}
	if (models.size() == 1) return models[0];
	if (!interactive)
	{
		ui_error("%zu models found; choose one with -m FILE (nnc --list-models shows them)", models.size());
		return {};
	}

	std::fprintf(stderr, "Models found:\n");
	for (size_t i = 0; i < models.size(); ++i)
	{
		const fs::path p = u8_path(models[i]);
		std::fprintf(stderr, "  %zu) %-40s %8s  %s%s%s\n", i + 1, path_u8(p.filename()).c_str(),
		             format_bytes(file_size(models[i])).c_str(), ui_dim(), tidy_path(path_u8(p.parent_path())).c_str(),
		             ui_reset());
	}
	std::fprintf(stderr, "  d) download another model\n");
	const int c = read_choice("Choose a model [1]: ", static_cast<int>(models.size()), 1, "dq");
	if (c == -'d') return choose_download(false);
	if (c <= 0) return {};
	return models[static_cast<size_t>(c - 1)];
}

int run_list_models()
{
	const std::vector<std::string> models = find_models();
	if (models.empty())
	{
		ui_info("No models found in ./models, ~/.lmstudio/models or ~/models. Run nnc --download to fetch one.");
		return 1;
	}
	for (const std::string& m : models)
		std::printf("%8s  %s\n", format_bytes(file_size(m)).c_str(), m.c_str());
	return 0;
}

int run_download(const std::string& name, const bool yes)
{
	std::string path;
	if (name.empty())
	{
		path = choose_download(yes);
	}
	else
	{
		const catalog_entry* e = find_catalog_entry(name);
		if (!e)
		{
			ui_error("nnc does not know a model called '%s'. These are the ones it can download:", name.c_str());
			for (const auto& c : model_catalog()) ui_info("  %-18s %s", c.name, c.description);
			return 1;
		}
		path = download_model(*e, yes);
	}
	if (path.empty()) return 1;
	std::printf("%s\n", path.c_str());
	return 0;
}
