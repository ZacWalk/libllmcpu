// nnc app — the chat REPL and one-shot prompts. See chat.h.

#include "chat.h"

#include "embed.h"
#include "models.h"
#include "platform.h"
#include "ui.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

namespace
{
	bool on_load_progress(const float p, void* user)
	{
		static_cast<progress_bar*>(user)->update(p);
		return !platform_interrupted();
	}

	struct stream
	{
		bool flush_each = false; // a person is watching: show every token at once
		bool line_open = false; // the reply so far does not end in a newline
		bool thinking = false; // reasoning is on screen, and the answer has not started
		bool thought_open = false; // ...and the reasoning does not end in a newline
	};

	// The model's reasoning goes to stderr, dimmed, so that stdout carries
	// only the answer.
	bool on_thinking(const char* text, const size_t length, void* user)
	{
		auto& s = *static_cast<stream*>(user);
		s.thinking = true;
		std::fputs(ui_dim(), stderr);
		std::fwrite(text, 1, length, stderr);
		std::fputs(ui_reset(), stderr);
		std::fflush(stderr);
		s.thought_open = length > 0 && text[length - 1] != '\n';
		return true;
	}

	// End the reasoning's last line, and leave a blank line before an answer.
	void end_thinking(stream& s, const bool answer_follows)
	{
		if (!s.thinking) return;
		if (s.thought_open) std::fputc('\n', stderr);
		if (answer_follows) std::fputc('\n', stderr);
		std::fflush(stderr);
		s.thinking = false;
		s.thought_open = false;
	}

	bool on_text(const char* text, const size_t length, void* user)
	{
		auto& s = *static_cast<stream*>(user);
		end_thinking(s, true);
		std::fwrite(text, 1, length, stdout);
		if (s.flush_each) std::fflush(stdout);
		s.line_open = length > 0 && text[length - 1] != '\n';
		return true;
	}

	void cancel(void* session) { nnc_session_cancel(static_cast<nnc_session*>(session)); }

	// What a message becomes: a chat turn, or with both languages set, a
	// request to translate it.
	struct turn_mode
	{
		std::string from, to;
		bool translating() const { return !from.empty(); }
	};

	std::string file_name(const std::string& path)
	{
		const size_t slash = path.find_last_of("/\\");
		return slash == std::string::npos ? path : path.substr(slash + 1);
	}

	const char* weights_name(const nnc_weight_format w)
	{
		switch (w)
		{
		case NNC_WEIGHTS_Q4: return "Q4";
		case NNC_WEIGHTS_BF16: return "BF16";
		default: return "Q8_0";
		}
	}

	// One chat (or translation) turn with Ctrl-C wired to stop it. When the
	// conversation has filled the context, it starts a fresh one and tries
	// again.
	nnc_status chat_turn(nnc_session* s, const std::string& message, const turn_mode& mode,
	                     const nnc_generate_params& gp, nnc_generate_result& r)
	{
		interrupt_scope scope(cancel, s);
		auto once = [&]
		{
			return mode.translating()
				       ? nnc_session_translate(s, message.c_str(), mode.from.c_str(), mode.to.c_str(), &gp, &r)
				       : nnc_session_chat(s, message.c_str(), &gp, &r);
		};
		nnc_status st = once();
		if (st == NNC_ERR_CONTEXT_FULL && nnc_session_n_past(s) > 0)
		{
			ui_info("%s(the conversation filled the %d-token context, so it starts afresh)%s", ui_dim(),
			        nnc_session_n_ctx(s), ui_reset());
			nnc_session_reset(s);
			st = once();
		}
		return st;
	}

	// The line after a reply: tokens, speed, and why it stopped if that
	// was not the model's own choice.
	void report_turn(const nnc_status st, const nnc_generate_result& r, stream& out, const app_options& o,
	                 const bool show_stats)
	{
		end_thinking(out, false);
		if (out.line_open) std::fputc('\n', stdout);
		out.line_open = false;
		std::fflush(stdout);
		if (st == NNC_ERR_CANCELLED)
		{
			ui_info("%s[stopped]%s", ui_dim(), ui_reset());
			return;
		}
		if (st != NNC_OK)
		{
			ui_error("%s", nnc_last_error());
			return;
		}

		std::string note;
		if (r.stop == NNC_STOP_MAX_TOKENS) note = ", cut at -n " + std::to_string(o.n_predict);
		else if (r.stop == NNC_STOP_CANCELLED) note = ", stopped";
		else if (r.stop == NNC_STOP_CONTEXT_FULL) note = ", context full: the next message starts afresh";
		if (show_stats || !note.empty())
		{
			const double tps = r.decode_ms > 0 ? 1000.0 * r.n_generated / r.decode_ms : 0.0;
			char thought[48] = "";
			if (r.n_thinking > 0) std::snprintf(thought, sizeof(thought), " (%d of them thinking)", r.n_thinking);
			if (o.verbose)
			{
				char drafts[96] = "";
				if (r.n_drafted > 0)
					std::snprintf(drafts, sizeof(drafts), "; %d of %d drafted tokens kept", r.n_draft_accepted,
					              r.n_drafted);
				ui_info("%s[%d tokens%s, %.1f tok/s; prompt %d tokens in %.2f s%s%s]%s", ui_dim(), r.n_generated,
				        thought, tps, r.n_prompt, r.prefill_ms / 1000.0, drafts, note.c_str(), ui_reset());
			}
			else
				ui_info("%s[%d tokens%s, %.1f tok/s%s]%s", ui_dim(), r.n_generated, thought, tps, note.c_str(),
				        ui_reset());
		}
		if (o.perf)
		{
			std::fputs(nnc_perf_report("reply", r.n_prompt + r.n_generated), stderr);
			nnc_perf_reset();
		}
	}

	void print_help()
	{
		ui_info("Commands:\n"
			"  /reset            forget the conversation and start a new one\n"
			"  /translate EN:DE  translate each message from one language to another\n"
			"  /translate off    chat again\n"
			"  /exit             quit (so do Ctrl-D, or Ctrl-Z then Enter on Windows)\n"
			"  /help             this list\n"
			"Ctrl-C stops a reply that is being written.");
	}

	// Strip surrounding blanks.
	std::string trim(const std::string& s)
	{
		const size_t a = s.find_first_not_of(" \t");
		if (a == std::string::npos) return {};
		const size_t b = s.find_last_not_of(" \t");
		return s.substr(a, b - a + 1);
	}
}

nnc_model* load_model_ui(const std::string& path, const nnc_weight_format weights)
{
	progress_bar bar("Loading " + file_name(path));
	nnc_model_params mp = nnc_model_default_params();
	mp.weights = weights;
	mp.progress = on_load_progress;
	mp.progress_user = &bar;

	const auto t0 = std::chrono::steady_clock::now();
	nnc_model* model = nullptr;
	nnc_status st;
	{
		interrupt_scope scope; // Ctrl-C cancels the load
		st = nnc_model_load(path.c_str(), &mp, &model);
	}
	if (st != NNC_OK)
	{
		bar.finish(st == NNC_ERR_CANCELLED ? "cancelled" : "failed");
		if (st != NNC_ERR_CANCELLED) ui_error("%s", nnc_last_error());
		return nullptr;
	}
	const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	const nnc_model_info info = nnc_model_get_info(model);
	char summary[160];
	std::snprintf(summary, sizeof(summary), "done in %.1f s (%s in memory)", secs,
	              format_bytes(info.resident_bytes ? info.resident_bytes : info.file_bytes).c_str());
	bar.finish(summary);
	return model;
}

int run_chat(const app_options& o)
{
	const std::string path = resolve_model(o.model, o.yes);
	if (path.empty()) return 1;
	nnc_model* model = load_model_ui(path, o.weights);
	if (!model) return 1;
	const nnc_model_info info = nnc_model_get_info(model);
	if (info.n_embd_out > 0) return run_embed(o, model);

	// Gemma 4 drafts with its MTP "assistant" model when one is around.
	nnc_model* draft = nullptr;
	std::string draft_path = o.draft;
	if (!o.no_draft && draft_path.empty() && std::string(info.architecture) == "gemma4")
		draft_path = find_draft_model(path);
	if (!o.no_draft && !draft_path.empty())
	{
		draft = load_model_ui(draft_path, NNC_WEIGHTS_Q8_0);
		if (!draft && !o.draft.empty())
		{
			nnc_model_free(model);
			return 1;
		}
	}

	nnc_session_params sp = nnc_session_default_params();
	sp.n_ctx = o.n_ctx;
	sp.seed = o.seed;
	sp.system_prompt = o.system_prompt.empty() ? nullptr : o.system_prompt.c_str();
	sp.draft_model = draft;
	sp.n_draft = o.n_draft;
	sp.kv_type = o.kv_f32 ? NNC_KV_F32 : NNC_KV_BF16;
	sp.thinking = o.think;
	nnc_session* session = nullptr;
	nnc_status created = nnc_session_create(model, &sp, &session);
	if (created != NNC_OK && draft && o.draft.empty())
	{
		// A drafter that was only found, not asked for, is not worth failing over.
		ui_info("%s(not drafting with %s: %s)%s", ui_dim(), file_name(draft_path).c_str(), nnc_last_error(),
		        ui_reset());
		nnc_model_free(draft);
		draft = nullptr;
		sp.draft_model = nullptr;
		created = nnc_session_create(model, &sp, &session);
	}
	if (created != NNC_OK)
	{
		ui_error("%s", nnc_last_error());
		nnc_model_free(draft);
		nnc_model_free(model);
		return 1;
	}

	stream out;
	out.flush_each = platform_stdout_is_terminal();
	nnc_generate_params gp = nnc_generate_default_params();
	gp.max_tokens = o.n_predict;
	gp.sampler = o.sampler;
	gp.on_text = on_text;
	gp.on_thinking = on_thinking;
	gp.user = &out;
	if (o.think && !info.has_thinking)
		ui_info("%s(%s has no thinking mode, so --think changes nothing)%s", ui_dim(), info.name, ui_reset());
	if (o.perf)
	{
		nnc_perf_enable(true);
		nnc_perf_reset();
	}

	turn_mode mode{o.translate_from, o.translate_to};
	int rc = 0;
	if (!o.prompt.empty())
	{
		nnc_generate_result r{};
		const nnc_status st = chat_turn(session, o.prompt, mode, gp, r);
		report_turn(st, r, out, o, o.verbose);
		rc = (st == NNC_OK) ? 0 : 1;
	}
	else
	{
		ui_info("%s%s%s (%s, %d layers, %s weights) - %d threads, %d-token context%s%s%s",
		        ui_bold(), info.name, ui_reset(), info.architecture, info.n_layer, weights_name(info.weights),
		        nnc_num_threads(), nnc_session_n_ctx(session), o.think && info.has_thinking ? ", thinking" : "",
		        draft ? ", drafting with " : "", draft ? file_name(draft_path).c_str() : "");
		if (mode.translating())
			ui_info("Each message is translated from %s to %s; /translate off to chat instead.", mode.from.c_str(),
			        mode.to.c_str());
		else if (info.translator)
			ui_info("This is a translation model: /translate en:de (or start with --translate en:de) translates "
			        "each message from English to German, and so on.");
		else if (!info.has_chat_template)
			ui_info("nnc does not know this model's chat format, so your text goes to it as it is.");
		if (!draft && !o.no_draft)
			if (const catalog_entry* e = find_catalog_drafter(path))
				ui_info("%sTip: nnc --download %s fetches a drafter that makes this model reply faster.%s", ui_dim(),
				        e->name, ui_reset());
		if (info.has_thinking && !o.think && !mode.translating())
			ui_info("%sTip: --think lets this model reason before it answers: slower, better at hard problems.%s",
			        ui_dim(), ui_reset());
		ui_info("%sType a message and press Enter. /help lists commands; Ctrl-C stops a reply.%s", ui_dim(),
		        ui_reset());

		const bool tty = platform_stdin_is_terminal();
		std::string line;
		for (;;)
		{
			std::fflush(stdout);
			if (tty)
			{
				std::fprintf(stderr, "\n%s>%s ", ui_bold(), ui_reset());
				std::fflush(stderr);
			}
			if (!platform_read_line(line))
			{
				if (tty) std::fputc('\n', stderr);
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
				nnc_session_reset(session);
				ui_info("%s(new conversation)%s", ui_dim(), ui_reset());
				continue;
			}
			if (msg.rfind("/translate", 0) == 0 && (msg.size() == 10 || msg[10] == ' '))
			{
				const std::string arg = trim(msg.substr(10));
				if (arg == "off")
				{
					mode = {};
					ui_info("%s(chatting)%s", ui_dim(), ui_reset());
				}
				else if (parse_language_pair(arg, mode.from, mode.to))
					ui_info("%s(translating from %s to %s)%s", ui_dim(), mode.from.c_str(), mode.to.c_str(),
					        ui_reset());
				else
					ui_info("Use /translate FROM:TO with language codes, e.g. /translate en:de, or /translate off.");
				continue;
			}
			if (msg[0] == '/' && msg.find(' ') == std::string::npos)
			{
				ui_info("Unknown command %s; /help lists the commands.", msg.c_str());
				continue;
			}

			nnc_generate_result r{};
			const nnc_status st = chat_turn(session, msg, mode, gp, r);
			report_turn(st, r, out, o, true);
		}
	}

	nnc_session_free(session);
	nnc_model_free(draft);
	nnc_model_free(model);
	return rc;
}
