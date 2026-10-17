// nnc — sessions. See session.h.

#include "session.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace
{
	double ms_since(const int64_t t0_us)
	{
		return static_cast<double>(nnc_time_us() - t0_us) / 1000.0;
	}

	// Evaluate every pending token, in prefill-sized batches. Stops between
	// batches if cancelled; whatever was evaluated stays valid.
	nnc_status eval_pending(nnc_session& s)
	{
		const nnc_model& m = *s.model;
		const int size = static_cast<int>(s.tokens.size());
		while (s.cache.cur_pos < size)
		{
			if (s.cancel.load(std::memory_order_relaxed))
			{
				nnc_fail(NNC_ERR_CANCELLED, "cancelled");
				return NNC_ERR_CANCELLED;
			}
			const int pos = s.cache.cur_pos;
			const int n = std::min(MODEL_PREFILL_BATCH, size - pos);
			if (!model_forward(m, s.cache, s.tokens.data() + pos, n, pos))
			{
				s.have_hidden = false;
				s.logits_valid = false;
				return nnc_last_status();
			}
			std::memcpy(s.hidden.data(), model_last_hidden(m, s.cache, n),
			            sizeof(float) * static_cast<size_t>(m.hparams.n_embd));
			s.have_hidden = true;
			s.logits_valid = false;
		}
		return NNC_OK;
	}

	// Recurrent models: before a call evaluates `pending` tokens, note where
	// the context stands, so that a cancel between its batches can go back.
	// A single batch needs no copy: a cancel lands before it or not at all,
	// and model_forward fails, if it does, before it changes anything.
	void save_recurrent(nnc_session& s, const size_t pending)
	{
		if (!model_is_recurrent(*s.model) || pending <= static_cast<size_t>(MODEL_PREFILL_BATCH)) return;
		model_state_save(s.cache);
		s.saved_hidden = s.hidden;
		s.saved_have_hidden = s.have_hidden;
	}

	// Recurrent models: go back to at most `target` evaluated tokens - to
	// the last saved state when it lies within them, else to the start, so
	// that the context is evaluated again. Either way, the tokens past the
	// new position become pending.
	void rewind_recurrent(nnc_session& s, const int target)
	{
		if (s.cache.saved_pos >= 0 && s.cache.saved_pos <= target && model_state_restore(s.cache))
		{
			s.hidden = s.saved_hidden;
			s.have_hidden = s.saved_have_hidden;
		}
		else
		{
			s.cache.cur_pos = 0;
			s.have_hidden = false;
		}
		s.logits_valid = false;
	}

	// Make sure `hidden` describes the token after the whole context.
	nnc_status refresh(nnc_session& s)
	{
		const int size = static_cast<int>(s.tokens.size());
		// A prediction that was lost (to a cancel) is rebuilt by
		// re-evaluating the last token; its K/V are rewritten unchanged. A
		// recurrent state has already taken the token in, so it goes back.
		if (!s.have_hidden && size > 0 && s.cache.cur_pos == size)
		{
			if (model_is_recurrent(*s.model)) rewind_recurrent(s, size - 1);
			else if (!model_kv_rewind(s.cache, size - 1)) model_kv_rewind(s.cache, 0);
		}
		return eval_pending(s);
	}

	// Put the context back to its first `size` tokens after a failed or
	// cancelled call. Evaluated K/V for those tokens stay valid, but the
	// prediction may now describe a dropped token, so it is rebuilt later.
	// A recurrent state that took in dropped tokens goes back instead, and
	// so does a ring buffer that has overwritten keys still needed.
	void rollback(nnc_session& s, const size_t size)
	{
		s.tokens.resize(size);
		s.logits_valid = false;
		if (model_is_recurrent(*s.model))
		{
			if (s.cache.cur_pos > static_cast<int>(size)) rewind_recurrent(s, static_cast<int>(size));
			return;
		}
		if (s.cache.cur_pos > static_cast<int>(size) && !model_kv_rewind(s.cache, static_cast<int>(size)))
			model_kv_rewind(s.cache, 0);
		s.have_hidden = false;
	}

	bool check_tokens(const nnc_model& m, const int* tokens, const int n)
	{
		if (n < 0 || (n > 0 && !tokens)) return nnc_fail(NNC_ERR_INVALID_ARGUMENT, "invalid token array");
		for (int i = 0; i < n; ++i)
			if (!m.vocab.valid(tokens[i]))
				return nnc_fail(NNC_ERR_INVALID_ARGUMENT, "token %d is outside the vocabulary [0, %d)", tokens[i],
				                m.vocab.n_vocab());
		return true;
	}

	bool ensure_logits(nnc_session& s)
	{
		if (s.logits_valid) return true;
		model_logits(*s.model, s.hidden.data(), s.logits.data());
		s.logits_valid = true;
		return true;
	}

	int argmax(const float* v, const int n)
	{
		int best = 0;
		for (int i = 1; i < n; ++i)
			if (v[i] > v[best]) best = i;
		return best;
	}

	// top-k -> top-p -> min-p on the model's own distribution, then
	// temperature over what survives, then a draw. The same order as
	// llama.cpp's default chain, so settings carry over.
	int sample_from_logits(nnc_session& s, const float* logits, const nnc_sampler_params& p)
	{
		const int n_vocab = s.model->hparams.n_vocab;
		auto& cand = s.candidates;
		cand.clear();

		const int k = (p.top_k > 0 && p.top_k < n_vocab) ? p.top_k : n_vocab;
		auto greater = [](const std::pair<float, int>& a, const std::pair<float, int>& b)
		{
			return a.first > b.first || (a.first == b.first && a.second < b.second);
		};
		if (k < n_vocab)
		{
			// One pass with a k-element min-heap: almost every logit loses to
			// the heap's floor on a single compare.
			for (int i = 0; i < n_vocab; ++i)
			{
				if (static_cast<int>(cand.size()) < k)
				{
					cand.emplace_back(logits[i], i);
					std::push_heap(cand.begin(), cand.end(), greater);
				}
				else if (logits[i] > cand.front().first)
				{
					std::pop_heap(cand.begin(), cand.end(), greater);
					cand.back() = {logits[i], i};
					std::push_heap(cand.begin(), cand.end(), greater);
				}
			}
		}
		else
		{
			cand.reserve(static_cast<size_t>(n_vocab));
			for (int i = 0; i < n_vocab; ++i) cand.emplace_back(logits[i], i);
		}
		std::sort(cand.begin(), cand.end(), greater);
		if (cand.empty() || !std::isfinite(cand[0].first)) return cand.empty() ? 0 : cand[0].second;

		// Probabilities at temperature 1, for the truncation rules.
		auto& prob = s.probs;
		prob.resize(cand.size());
		const float top = cand[0].first;
		double sum = 0;
		for (size_t i = 0; i < cand.size(); ++i)
		{
			prob[i] = std::exp(cand[i].first - top);
			sum += prob[i];
		}
		size_t keep = cand.size();
		if (p.top_p > 0.0f && p.top_p < 1.0f)
		{
			double cum = 0;
			for (size_t i = 0; i < keep; ++i)
			{
				cum += prob[i] / sum;
				if (cum >= p.top_p)
				{
					keep = i + 1;
					break;
				}
			}
		}
		if (p.min_p > 0.0f)
		{
			const float floor = p.min_p * prob[0];
			size_t i = 1;
			while (i < keep && prob[i] >= floor) ++i;
			keep = i;
		}

		// Temperature over the survivors, then one uniform draw.
		const double inv_t = 1.0 / p.temperature;
		double total = 0;
		for (size_t i = 0; i < keep; ++i)
		{
			prob[i] = static_cast<float>(std::exp((cand[i].first - top) * inv_t));
			total += prob[i];
		}
		std::uniform_real_distribution<double> uni(0.0, total);
		double r = uni(s.rng);
		for (size_t i = 0; i < keep; ++i)
		{
			r -= prob[i];
			if (r <= 0) return cand[i].second;
		}
		return cand[keep - 1].second;
	}

	// Streams generated bytes to a callback as valid UTF-8 in whole
	// characters. The start of a character that has not fully arrived waits
	// in `pending` (the session's text_pending or thought_pending).
	struct text_stream
	{
		nnc_text_callback callback;
		void* user;
		std::string& pending;
		std::string ready;
		bool stopped = false; // the callback asked to stop; it is not called again

		// False once the callback has asked to stop.
		bool write(const std::string& piece, const bool final = false)
		{
			if (!callback || stopped) return !stopped;
			pending += piece;
			ready.clear();
			utf8_take(pending, ready, final);
			if (!ready.empty() && !callback(ready.data(), ready.size(), user)) stopped = true;
			return !stopped;
		}
	};
}

void utf8_take(std::string& pending, std::string& out, const bool final)
{
	static constexpr char replacement[] = "\xEF\xBF\xBD"; // U+FFFD
	const size_t n = pending.size();
	size_t i = 0;
	while (i < n)
	{
		const auto c = static_cast<unsigned char>(pending[i]);
		// Length from the lead byte, and the range of the second byte, per
		// RFC 3629: no overlong forms, surrogates or code points past U+10FFFF.
		size_t len = 0;
		unsigned char lo = 0x80, hi = 0xBF;
		if (c < 0x80)
		{
			len = 1;
		}
		else if (c >= 0xC2 && c <= 0xDF)
		{
			len = 2;
		}
		else if (c == 0xE0)
		{
			len = 3;
			lo = 0xA0;
		}
		else if (c == 0xED)
		{
			len = 3;
			hi = 0x9F;
		}
		else if (c >= 0xE1 && c <= 0xEF)
		{
			len = 3;
		}
		else if (c == 0xF0)
		{
			len = 4;
			lo = 0x90;
		}
		else if (c == 0xF4)
		{
			len = 4;
			hi = 0x8F;
		}
		else if (c >= 0xF1 && c <= 0xF3)
		{
			len = 4;
		}
		if (len == 0)
		{
			out += replacement;
			++i;
			continue;
		}
		size_t k = 1;
		bool broken = false;
		while (k < len && i + k < n)
		{
			const auto b = static_cast<unsigned char>(pending[i + k]);
			if (b < (k == 1 ? lo : 0x80) || b > (k == 1 ? hi : 0xBF))
			{
				broken = true;
				break;
			}
			++k;
		}
		if (k == len)
			out.append(pending, i, len);
		else if (!broken && !final)
			break; // the rest of this character has not arrived yet
		else
			out += replacement;
		i += k;
	}
	pending.erase(0, i);
}

bool session_init(nnc_session& s, const nnc_model& m, const nnc_session_params& p)
{
	if (m.hparams.n_embd_out > 0)
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
		                "%s is a text-embedding model: it turns text into vectors (nnc_embed) rather than "
		                "generating text", m.name.c_str());
	if (m.hparams.n_embd_target > 0)
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
		                "%s is an MTP drafter: it speeds up the model it was made for (draft_model), and cannot "
		                "generate on its own", m.name.c_str());
	const int n_ctx = (p.n_ctx > 0) ? p.n_ctx : std::min(4096, m.hparams.n_ctx);
	s.model = &m;
	if (!model_kv_init(m, s.cache, n_ctx, MODEL_PREFILL_BATCH, p.kv_type == NNC_KV_F32)) return false;
	nnc_log(NNC_LOG_INFO, "context: %d tokens, %.1f MB of %s K/V cache%s", n_ctx,
	        static_cast<double>(s.cache.bytes) / (1024.0 * 1024.0), s.cache.kv_bf16 ? "BF16" : "F32",
	        model_is_recurrent(m) ? " and recurrent state" : "");
	s.hidden.assign(static_cast<size_t>(m.hparams.n_embd), 0.0f);
	s.logits.assign(static_cast<size_t>(m.hparams.n_vocab), 0.0f);
	s.tokens.reserve(static_cast<size_t>(n_ctx));
	s.system_prompt = p.system_prompt ? p.system_prompt : "";
	s.thinking = p.thinking && m.vocab.can_think;
	const uint64_t seed = p.seed ? p.seed : std::random_device{}() ^ (static_cast<uint64_t>(nnc_time_ns()) << 1);
	s.rng.seed(seed);

	if (p.draft_model)
	{
		if (!model_draft_init(*p.draft_model, m, s.cache, s.dcache)) return false;
		s.draft = p.draft_model;
		const int n_default = m.hparams.n_expert > 0 ? SESSION_DRAFT_N_MOE : SESSION_DRAFT_N;
		s.n_draft = p.n_draft > 0 ? std::min(p.n_draft, MODEL_PREFILL_BATCH - 1) : n_default;
		s.draft_h.assign(static_cast<size_t>(m.hparams.n_embd), 0.0f);
		s.verify_tokens.assign(static_cast<size_t>(s.n_draft) + 1, 0);
		s.verify_logits.assign((static_cast<size_t>(s.n_draft) + 1) * static_cast<size_t>(m.hparams.n_vocab), 0.0f);
	}
	return true;
}

void session_reset(nnc_session& s)
{
	s.tokens.clear();
	s.cache.cur_pos = 0;
	s.cache.saved_pos = -1;
	s.have_hidden = false;
	s.logits_valid = false;
	s.text_pending.clear();
	s.thought_pending.clear();
}

nnc_status session_eval(nnc_session& s, const int* tokens, const int n_tokens)
{
	if (!check_tokens(*s.model, tokens, n_tokens)) return NNC_ERR_INVALID_ARGUMENT;
	const size_t size = s.tokens.size();
	if (size + static_cast<size_t>(n_tokens) > static_cast<size_t>(s.cache.n_ctx))
	{
		nnc_fail(NNC_ERR_CONTEXT_FULL, "%d more tokens do not fit: %zu of the %d-token context are in use",
		         n_tokens, size, s.cache.n_ctx);
		return NNC_ERR_CONTEXT_FULL;
	}
	save_recurrent(s, size + static_cast<size_t>(n_tokens) - static_cast<size_t>(s.cache.cur_pos));
	if (n_tokens > 0) s.tokens.insert(s.tokens.end(), tokens, tokens + n_tokens);
	const nnc_status st = refresh(s);
	if (st != NNC_OK)
		rollback(s, size);
	else if (n_tokens > 0)
	{
		// Generated text no longer runs on from here.
		s.text_pending.clear();
		s.thought_pending.clear();
	}
	return st;
}

const float* session_logits(nnc_session& s)
{
	if (refresh(s) != NNC_OK || !s.have_hidden) return nullptr;
	ensure_logits(s);
	return s.logits.data();
}

int session_sample(nnc_session& s, const nnc_sampler_params& p)
{
	if (refresh(s) != NNC_OK) return NNC_TOKEN_NULL;
	if (!s.have_hidden)
	{
		nnc_fail(NNC_ERR_INVALID_ARGUMENT, "there is nothing to predict from: the context is empty");
		return NNC_TOKEN_NULL;
	}
	if (p.temperature <= 0.0f)
	{
		if (s.logits_valid) return argmax(s.logits.data(), s.model->hparams.n_vocab);
		// The scratch receives raw (not soft-capped) logits, so it does not
		// count as the session's logits afterwards.
		return model_argmax(*s.model, s.hidden.data(), s.logits.data());
	}
	ensure_logits(s);
	return sample_from_logits(s, s.logits.data(), p);
}

nnc_status session_generate(nnc_session& s, const int* prompt, const int n_prompt,
                            const nnc_generate_params& p, nnc_generate_result& r)
{
	r = nnc_generate_result{};
	const nnc_model& m = *s.model;
	if (!check_tokens(m, prompt, n_prompt)) return NNC_ERR_INVALID_ARGUMENT;
	const size_t size0 = s.tokens.size();
	const int n_ctx = s.cache.n_ctx;
	// The prompt plus at least one generated token must fit.
	if (size0 + static_cast<size_t>(n_prompt) + 1 > static_cast<size_t>(n_ctx))
	{
		nnc_fail(NNC_ERR_CONTEXT_FULL, "a %d-token prompt does not fit: %zu of the %d-token context are in use",
		         n_prompt, size0, n_ctx);
		return NNC_ERR_CONTEXT_FULL;
	}
	if (size0 == 0 && n_prompt == 0)
	{
		nnc_fail(NNC_ERR_INVALID_ARGUMENT, "there is nothing to continue: the context and the prompt are empty");
		return NNC_ERR_INVALID_ARGUMENT;
	}
	save_recurrent(s, size0 + static_cast<size_t>(n_prompt) - static_cast<size_t>(s.cache.cur_pos));
	if (n_prompt > 0) s.tokens.insert(s.tokens.end(), prompt, prompt + n_prompt);

	const int64_t t_prefill = nnc_time_us();
	const int before = s.cache.cur_pos;
	nnc_status st = refresh(s);
	if (st != NNC_OK)
	{
		rollback(s, size0);
		return st;
	}
	r.n_prompt = s.cache.cur_pos - before;
	r.prefill_ms = ms_since(t_prefill);
	// A character cut in half by the last call is completed only when this
	// call carries straight on from it and streams. Cleared only now, so a
	// prompt that is cancelled and rolled back keeps it.
	if (n_prompt > 0 || !p.on_text) s.text_pending.clear();
	if (n_prompt > 0 || !p.on_thinking) s.thought_pending.clear();

	const int64_t t_decode = nnc_time_us();
	text_stream out{p.on_text, p.user, s.text_pending};
	text_stream thought{p.on_thinking, p.user, s.thought_pending};
	thought_split split;
	split.start(m.vocab, s.tokens);
	std::string shown;

	// Adds a generated token to the context (unevaluated, unless a draft
	// round already evaluated it) and to the answer or the reasoning. False
	// once generation must stop, with r.stop saying why.
	auto emit = [&](const int tok)
	{
		s.tokens.push_back(tok);
		if (vocab_is_eog(m.vocab, tok))
		{
			r.stop = NNC_STOP_END_OF_TURN;
			return false;
		}
		++r.n_generated;
		const bool reasoning = split.route(tok, vocab_piece(m.vocab, tok), shown);
		if (reasoning) ++r.n_thinking;
		if (!(reasoning ? thought : out).write(shown))
		{
			r.stop = NNC_STOP_CALLBACK;
			return false;
		}
		if (static_cast<int>(s.tokens.size()) >= n_ctx)
		{
			r.stop = NNC_STOP_CONTEXT_FULL;
			return false;
		}
		if (p.max_tokens > 0 && r.n_generated >= p.max_tokens)
		{
			r.stop = NNC_STOP_MAX_TOKENS;
			return false;
		}
		return true;
	};

	// One round of speculative decoding. The context ends with a generated
	// token that is not evaluated yet; the drafter guesses the tokens after
	// it, and one forward pass over the token and its guesses yields the
	// model's own prediction after each. Guesses are kept while they match
	// the token the model samples at that place, so the output is exactly
	// what plain decoding gives. `next` receives the first sampled token
	// that was not a guess. Sets `stopped` when a kept guess ends the reply.
	int next = -1;
	bool stopped = false;
	auto speculate = [&]() -> nnc_status
	{
		const int pos = s.cache.cur_pos;
		int k = std::min({s.n_draft, n_ctx - (pos + 1), s.cache.n_batch - 1});
		if (p.max_tokens > 0) k = std::min(k, p.max_tokens - r.n_generated);
		if (k < 1 || pos == 0 || !s.have_hidden || static_cast<int>(s.tokens.size()) != pos + 1)
			return eval_pending(s);

		int* vt = s.verify_tokens.data();
		vt[0] = s.tokens.back();
		std::copy(s.hidden.begin(), s.hidden.end(), s.draft_h.begin());
		int n = 0;
		while (n < k && !vocab_is_eog(m.vocab, vt[n]))
		{
			float prob = 0.0f;
			const int d = model_draft_step(*s.draft, m, s.dcache, vt[n], s.draft_h.data(), pos, s.draft_h.data(),
			                               &prob);
			if (prob < SESSION_DRAFT_P_MIN) break;
			vt[n + 1] = d;
			++n;
		}
		if (n == 0) return eval_pending(s);
		r.n_drafted += n;

		if (!model_forward(m, s.cache, vt, n + 1, pos))
		{
			s.have_hidden = false;
			s.logits_valid = false;
			return nnc_last_status();
		}
		const bool greedy = p.sampler.temperature <= 0.0f;
		const size_t n_vocab = static_cast<size_t>(m.hparams.n_vocab);
		model_logits_batch(m, s.cache, n + 1, s.verify_logits.data(), !greedy);

		// Row a predicts the token after vt[a].
		int a = 0;
		for (;;)
		{
			const float* row = s.verify_logits.data() + static_cast<size_t>(a) * n_vocab;
			const int t = greedy ? argmax(row, static_cast<int>(n_vocab)) : sample_from_logits(s, row, p.sampler);
			if (a < n && t == vt[a + 1])
			{
				++a;
				++r.n_draft_accepted;
				if (emit(t)) continue;
				stopped = true;
				break;
			}
			next = t;
			break;
		}
		// The kept tokens' K/V stay; those of rejected guesses are dropped.
		// (Should a ring buffer not allow that, the context is evaluated
		// again from the start.)
		const size_t n_embd = static_cast<size_t>(m.hparams.n_embd);
		s.logits_valid = false;
		if (!model_kv_rewind(s.cache, pos + a + 1))
		{
			model_kv_rewind(s.cache, 0);
			s.have_hidden = false;
			return NNC_OK;
		}
		std::memcpy(s.hidden.data(), s.cache.sx.data() + static_cast<size_t>(a) * n_embd, n_embd * sizeof(float));
		s.have_hidden = true;
		return NNC_OK;
	};

	for (;;)
	{
		if (s.cancel.load(std::memory_order_relaxed))
		{
			r.stop = NNC_STOP_CANCELLED;
			break;
		}
		const int tok = next >= 0 ? next : session_sample(s, p.sampler);
		next = -1;
		if (tok < 0) return nnc_last_status();
		// Room for this token was checked before; it joins the context
		// unevaluated, and is evaluated only if generation goes on.
		if (!emit(tok)) break;
		st = s.draft ? speculate() : eval_pending(s);
		if (st == NNC_ERR_CANCELLED)
		{
			r.stop = NNC_STOP_CANCELLED;
			break;
		}
		if (st != NNC_OK) return st;
		if (stopped) break;
	}
	// A reply that cannot go on ends here, so a character still cut in half
	// becomes U+FFFD. One stopped by max_tokens, a cancel or the callback
	// keeps it for a call that carries on.
	if (r.stop == NNC_STOP_END_OF_TURN || r.stop == NNC_STOP_CONTEXT_FULL)
	{
		out.write({}, true);
		thought.write({}, true);
	}
	r.decode_ms = ms_since(t_decode);
	return NNC_OK;
}

namespace
{
	// See session_turn_tokens.
	std::vector<int> turn_tokens(const model_vocab& v, const std::vector<int>& context, const std::string& system,
	                             const std::string& message, const bool thinking)
	{
		const bool first = context.empty();
		const bool think = thinking && v.can_think;
		std::vector<int> turn;
		auto text = [&](const std::string& t)
		{
			const std::vector<int> ids = vocab_tokenize(v, t, false);
			turn.insert(turn.end(), ids.begin(), ids.end());
		};
		auto bos = [&]
		{
			if (v.add_bos && v.bos_id >= 0) turn.push_back(v.bos_id);
		};
		const bool reply_closed = !first && context.back() == v.tok_turn_end;

		switch (v.chat)
		{
		case chat_format::gemma:
			// <start_of_turn>user\n{system\n\n}{message}<end_of_turn>\n<start_of_turn>model\n
			// Gemma has no system role; its template prepends the system
			// text to the first user turn. FunctionGemma gives it a turn of
			// its own: <start_of_turn>developer\n{system}<end_of_turn>\n.
			if (first)
			{
				bos();
				if (v.developer_turn && !system.empty())
				{
					turn.push_back(v.tok_turn_start);
					text("developer");
					turn.push_back(v.tok_newline);
					text(system);
					turn.push_back(v.tok_turn_end);
					turn.push_back(v.tok_newline);
				}
			}
			else
			{
				if (!reply_closed) turn.push_back(v.tok_turn_end);
				turn.push_back(v.tok_newline);
			}
			turn.push_back(v.tok_turn_start);
			text("user");
			turn.push_back(v.tok_newline);
			text(first && !system.empty() && !v.developer_turn ? system + "\n\n" + message : message);
			turn.push_back(v.tok_turn_end);
			turn.push_back(v.tok_newline);
			turn.push_back(v.tok_turn_start);
			text("model");
			turn.push_back(v.tok_newline);
			break;

		case chat_format::gemma4:
			// [<|turn>system\n[<|think|>\n]{system}<turn|>\n]<|turn>user\n{message}<turn|>\n
			// <|turn>model\n[<|channel>thought\n<channel|>]
			// Gemma 4 has a system role, and <|think|> at the top of the
			// system turn lets the model think, so with thinking on there is
			// a system turn even without a system prompt. With thinking off,
			// the 26B and 31B templates also open the model turn with an
			// empty thought channel; without it those models start thinking
			// on their own.
			if (first)
			{
				bos();
				if (!system.empty() || think)
				{
					turn.push_back(v.tok_turn_start);
					text("system");
					turn.push_back(v.tok_newline);
					if (think)
					{
						turn.push_back(v.tok_think);
						turn.push_back(v.tok_newline);
					}
					if (!system.empty()) text(system);
					turn.push_back(v.tok_turn_end);
					turn.push_back(v.tok_newline);
				}
			}
			else
			{
				if (!reply_closed) turn.push_back(v.tok_turn_end);
				turn.push_back(v.tok_newline);
			}
			turn.push_back(v.tok_turn_start);
			text("user");
			turn.push_back(v.tok_newline);
			text(message);
			turn.push_back(v.tok_turn_end);
			turn.push_back(v.tok_newline);
			turn.push_back(v.tok_turn_start);
			text("model");
			turn.push_back(v.tok_newline);
			if (v.empty_thought && !think)
			{
				turn.push_back(v.tok_channel_start);
				text("thought\n");
				turn.push_back(v.tok_channel_end);
			}
			break;

		case chat_format::llama3:
			// <|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>
			// <|start_header_id|>assistant<|end_header_id|>\n\n
			if (first)
			{
				bos();
				if (!system.empty())
				{
					turn.push_back(v.tok_turn_start);
					text("system");
					turn.push_back(v.tok_header_end);
					text("\n\n" + system);
					turn.push_back(v.tok_turn_end);
				}
			}
			else if (!reply_closed)
			{
				turn.push_back(v.tok_turn_end);
			}
			turn.push_back(v.tok_turn_start);
			text("user");
			turn.push_back(v.tok_header_end);
			text("\n\n" + message);
			turn.push_back(v.tok_turn_end);
			turn.push_back(v.tok_turn_start);
			text("assistant");
			turn.push_back(v.tok_header_end);
			text("\n\n");
			break;

		case chat_format::chatml:
			// [<|im_start|>system\n{system}<|im_end|>\n]<|im_start|>user\n{message}<|im_end|>\n
			// <|im_start|>assistant\n[<think>\n\n</think>\n\n | <think>\n]
			// Without a system prompt the template's own is used (Qwen2.5 has
			// one). With thinking on, the reply opens as the template opens
			// it then ("<think>\n", or as it is for Qwen3, which writes the
			// <think> itself), and Qwen3.8's system turn starts with the
			// instruction for its default reasoning effort. Each run of text
			// between markers is encoded whole, as llama.cpp encodes the
			// rendered template.
			if (first)
			{
				bos();
				std::string sys = system.empty() ? v.default_system : system;
				if (think && !v.think_instructions.empty())
					sys = sys.empty() ? v.think_instructions : v.think_instructions + "\n\n" + sys;
				if (!sys.empty())
				{
					turn.push_back(v.tok_turn_start);
					text("system\n" + sys);
					turn.push_back(v.tok_turn_end);
					text("\n");
				}
			}
			else
			{
				if (!reply_closed) turn.push_back(v.tok_turn_end);
				text("\n");
			}
			turn.push_back(v.tok_turn_start);
			text("user\n" + message);
			turn.push_back(v.tok_turn_end);
			text("\n");
			turn.push_back(v.tok_turn_start);
			text("assistant\n" + (think ? v.reply_open_think : v.reply_open));
			break;

		case chat_format::none:
			// No known turn format: the text goes in as it is, one line per turn.
			if (first)
			{
				bos();
				text(system.empty() ? message : system + "\n\n" + message);
			}
			else
			{
				text("\n" + message);
			}
			break;
		}
		return turn;
	}

	// One user turn in the model's chat format, then the reply. The system
	// prompt goes with the first turn when `use_system` is set.
	nnc_status chat_turn(nnc_session& s, const std::string& message, const bool use_system,
	                     const nnc_generate_params& p, nnc_generate_result& r)
	{
		const std::vector<int> turn = turn_tokens(s.model->vocab, s.tokens, use_system ? s.system_prompt : std::string(),
		                                          message, s.thinking);
		return session_generate(s, turn.data(), static_cast<int>(turn.size()), p, r);
	}

	std::string trim(const std::string& s)
	{
		const char* blank = " \t\r\n";
		const size_t a = s.find_first_not_of(blank);
		if (a == std::string::npos) return {};
		return s.substr(a, s.find_last_not_of(blank) - a + 1);
	}

	std::string lower(std::string s)
	{
		for (char& c : s)
			if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
		return s;
	}

	// Languages for models that bring no list of their own.
	struct language_entry
	{
		const char* code;
		const char* name;
	};

	const language_entry k_languages[] = {
		{"ar", "Arabic"}, {"bg", "Bulgarian"}, {"bn", "Bengali"}, {"ca", "Catalan"}, {"cs", "Czech"},
		{"cy", "Welsh"}, {"da", "Danish"}, {"de", "German"}, {"el", "Greek"}, {"en", "English"},
		{"es", "Spanish"}, {"et", "Estonian"}, {"eu", "Basque"}, {"fa", "Persian"}, {"fi", "Finnish"},
		{"fr", "French"}, {"ga", "Irish"}, {"gl", "Galician"}, {"gu", "Gujarati"}, {"he", "Hebrew"},
		{"hi", "Hindi"}, {"hr", "Croatian"}, {"hu", "Hungarian"}, {"hy", "Armenian"}, {"id", "Indonesian"},
		{"is", "Icelandic"}, {"it", "Italian"}, {"ja", "Japanese"}, {"ka", "Georgian"}, {"kk", "Kazakh"},
		{"km", "Khmer"}, {"kn", "Kannada"}, {"ko", "Korean"}, {"lt", "Lithuanian"}, {"lv", "Latvian"},
		{"mk", "Macedonian"}, {"ml", "Malayalam"}, {"mn", "Mongolian"}, {"mr", "Marathi"}, {"ms", "Malay"},
		{"my", "Burmese"}, {"nb", "Norwegian Bokmal"}, {"ne", "Nepali"}, {"nl", "Dutch"}, {"no", "Norwegian"},
		{"pa", "Punjabi"}, {"pl", "Polish"}, {"pt", "Portuguese"}, {"ro", "Romanian"}, {"ru", "Russian"},
		{"sk", "Slovak"}, {"sl", "Slovenian"}, {"sq", "Albanian"}, {"sr", "Serbian"}, {"sv", "Swedish"},
		{"sw", "Swahili"}, {"ta", "Tamil"}, {"te", "Telugu"}, {"th", "Thai"}, {"tl", "Tagalog"},
		{"tr", "Turkish"}, {"uk", "Ukrainian"}, {"ur", "Urdu"}, {"uz", "Uzbek"}, {"vi", "Vietnamese"},
		{"zh", "Chinese"}, {"zu", "Zulu"},
	};

	// "en", "pt_BR" or "German" -> the code (empty for a name) and English
	// name the prompt uses. TranslateGemma accepts only its own codes.
	bool language(const model_vocab& v, const std::string& given, std::string& code, std::string& name)
	{
		std::string c = trim(given);
		std::replace(c.begin(), c.end(), '_', '-');
		if (c.empty()) return nnc_fail(NNC_ERR_INVALID_ARGUMENT, "no language was given");
		if (!v.languages.empty())
		{
			auto it = v.languages.find(c);
			if (it == v.languages.end())
				for (auto i = v.languages.begin(); i != v.languages.end(); ++i)
					if (lower(i->first) == lower(c))
					{
						it = i;
						break;
					}
			if (it == v.languages.end())
				return nnc_fail(NNC_ERR_INVALID_ARGUMENT,
				                "the model does not know the language '%s'; it takes codes such as en, de or pt-BR",
				                given.c_str());
			code = it->first;
			name = it->second;
			return true;
		}
		const std::string base = lower(c.substr(0, c.find('-')));
		for (const auto& e : k_languages)
			if (base == e.code)
			{
				code = c;
				name = e.name;
				return true;
			}
		code.clear();
		name = c;
		return true;
	}
}

std::vector<int> session_turn_tokens(const model_vocab& v, const std::vector<int>& context, const std::string& system,
                                     const std::string& message, const bool thinking)
{
	return turn_tokens(v, context, system, message, thinking);
}

void thought_split::start(const model_vocab& v, const std::vector<int>& context)
{
	*this = thought_split{};
	open = v.tok_thought_open;
	close = v.tok_thought_close;
	label = v.thought_label;
	if (open < 0 || close < 0) return;
	// Back to the current turn's last marker, then forward over what has
	// been shown since, for a name line or newlines still to be dropped.
	size_t at = context.size();
	for (; at > 0; --at)
	{
		const int t = context[at - 1];
		if (t == open || t == close) break;
		if (t == v.tok_turn_start) return;
	}
	if (at == 0) return;
	std::string shown;
	for (size_t i = at - 1; i < context.size(); ++i) route(context[i], vocab_piece(v, context[i]), shown);
}

bool thought_split::route(const int token, const std::string& piece, std::string& text)
{
	text.clear();
	if (open >= 0 && token == open)
	{
		inside = true;
		skip_line = label;
		trim = true;
		return true;
	}
	if (close >= 0 && token == close)
	{
		const bool reasoning = inside;
		inside = false;
		skip_line = false;
		trim = true;
		return reasoning;
	}
	size_t i = 0;
	if (inside && skip_line)
	{
		const size_t nl = piece.find('\n');
		if (nl == std::string::npos) return true;
		skip_line = false;
		i = nl + 1;
	}
	if (trim)
	{
		while (i < piece.size() && (piece[i] == '\n' || piece[i] == '\r')) ++i;
		if (i < piece.size()) trim = false;
	}
	text.assign(piece, i, std::string::npos);
	return inside;
}

nnc_status session_chat(nnc_session& s, const std::string& message,
                        const nnc_generate_params& p, nnc_generate_result& r)
{
	return chat_turn(s, message, true, p, r);
}

nnc_status session_translate(nnc_session& s, const std::string& text, const std::string& source,
                             const std::string& target, const nnc_generate_params& p, nnc_generate_result& r)
{
	std::string src_code, src, tgt_code, tgt;
	if (!language(s.model->vocab, source, src_code, src) || !language(s.model->vocab, target, tgt_code, tgt))
		return NNC_ERR_INVALID_ARGUMENT;
	auto label = [](const std::string& name, const std::string& code)
	{
		return code.empty() ? name : name + " (" + code + ")";
	};
	// TranslateGemma's prompt, word for word from its chat template.
	const std::string message = "You are a professional " + label(src, src_code) + " to " + label(tgt, tgt_code)
		+ " translator. Your goal is to accurately convey the meaning and nuances of the original " + src
		+ " text while adhering to " + tgt + " grammar, vocabulary, and cultural sensitivities.\n"
		+ "Produce only the " + tgt + " translation, without any additional explanations or commentary. "
		+ "Please translate the following " + src + " text into " + tgt + ":\n\n\n" + trim(text);
	return chat_turn(s, message, false, p, r);
}
