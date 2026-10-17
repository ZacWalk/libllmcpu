// nnc — the C API (nnc.h) for models and sessions: argument checks,
// exception containment and handle ownership around the model and session
// engines. The functions that need no model, and the CPU gate in front of
// nnc_model_load(), are in baseline.cpp. Nothing in here computes anything.

#include "nnc.h"

#include "model.h"
#include "session.h"

#include <cstring>
#include <memory>
#include <new>
#include <string>

namespace
{
	nnc_status invalid(const char* what)
	{
		nnc_fail(NNC_ERR_INVALID_ARGUMENT, "%s", what);
		return NNC_ERR_INVALID_ARGUMENT;
	}

	// C callers cannot catch C++ exceptions, so none may cross the API.
	// The library only throws std::bad_alloc in practice.
	template <typename F>
	nnc_status guarded(F&& body)
	{
		try
		{
			return body();
		}
		catch (const std::bad_alloc&)
		{
			nnc_fail(NNC_ERR_OUT_OF_MEMORY, "out of memory");
			return NNC_ERR_OUT_OF_MEMORY;
		}
		catch (const std::exception& e)
		{
			nnc_fail(NNC_ERR_INVALID_ARGUMENT, "internal error: %s", e.what());
			return NNC_ERR_INVALID_ARGUMENT;
		}
	}

	nnc_generate_params generate_params(const nnc_generate_params* p)
	{
		return p ? *p : nnc_generate_default_params();
	}

	bool finite_float(const float value)
	{
		uint32_t bits;
		std::memcpy(&bits, &value, sizeof(bits));
		return (bits & 0x7F800000u) != 0x7F800000u;
	}

	bool valid_sampler(const nnc_sampler_params& p)
	{
		return finite_float(p.temperature) && p.temperature >= 0.0f
			&& p.top_k >= 0
			&& finite_float(p.top_p) && p.top_p >= 0.0f && p.top_p <= 1.0f
			&& finite_float(p.min_p) && p.min_p >= 0.0f && p.min_p <= 1.0f;
	}

	nnc_status invalid_sampler(const char* function)
	{
		nnc_fail(NNC_ERR_INVALID_ARGUMENT, "%s: temperature must be >= 0, top_k >= 0, and top_p/min_p in [0, 1]",
		         function);
		return NNC_ERR_INVALID_ARGUMENT;
	}
}

// nnc_model_load(), once baseline.cpp has checked the CPU.
nnc_status nnc_model_load_avx(const char* path, const nnc_model_params* params, nnc_model** model)
{
	if (model) *model = nullptr;
	if (!path || !*path || !model) return invalid("nnc_model_load needs a path and somewhere to put the model");
	const nnc_model_params p = params ? *params : nnc_model_default_params();
	return guarded([&]
	{
		auto m = std::make_unique<nnc_model>();
		if (!model_load(path, *m, p.weights, p.progress, p.progress_user)) return nnc_last_status();
		*model = m.release();
		return NNC_OK;
	});
}

extern "C" {

// ---- models -------------------------------------------------------------

void nnc_model_free(nnc_model* model)
{
	delete model;
}

nnc_model_info nnc_model_get_info(const nnc_model* model)
{
	nnc_model_info info{};
	if (!model) return info;
	const auto& h = model->hparams;
	info.name = model->name.c_str();
	info.architecture = h.arch.c_str();
	info.weights = model->weights;
	info.n_vocab = h.n_vocab;
	info.n_ctx_train = h.n_ctx;
	info.n_embd = h.n_embd;
	info.n_layer = h.n_layer;
	info.n_head = h.n_head;
	info.n_head_kv = h.n_head_kv;
	info.n_params = model->n_params;
	info.file_bytes = model->gguf.mapped_size;
	info.resident_bytes = model->decoded_bytes;
	info.has_chat_template = model->vocab.chat != chat_format::none && h.n_embd_out == 0 && h.n_embd_target == 0;
	info.n_embd_out = h.n_embd_out;
	info.translator = !model->vocab.languages.empty();
	info.n_embd_target = h.n_embd_target;
	info.has_thinking = info.has_chat_template && (model->vocab.can_think || model->vocab.always_thinks);
	return info;
}

// ---- tokens -------------------------------------------------------------

nnc_token nnc_token_bos(const nnc_model* model) { return model ? model->vocab.bos_id : NNC_TOKEN_NULL; }
nnc_token nnc_token_eos(const nnc_model* model) { return model ? model->vocab.eos_id : NNC_TOKEN_NULL; }

bool nnc_token_is_eog(const nnc_model* model, const nnc_token token)
{
	return model && vocab_is_eog(model->vocab, token);
}

int32_t nnc_tokenize(const nnc_model* model, const char* text, const size_t length, const bool add_bos,
                     nnc_token* tokens, const int32_t capacity)
{
	if (!model || (!text && length > 0) || capacity < 0 || (!tokens && capacity > 0))
	{
		invalid("nnc_tokenize: invalid arguments");
		return -1;
	}
	int32_t n = -1;
	const nnc_status st = guarded([&]
	{
		const std::vector<int> ids = vocab_tokenize(model->vocab, std::string(text ? text : "", length), add_bos);
		n = static_cast<int32_t>(ids.size());
		const int32_t copy = n < capacity ? n : capacity;
		for (int32_t i = 0; i < copy; ++i) tokens[i] = ids[i];
		return NNC_OK;
	});
	return st == NNC_OK ? n : -1;
}

int32_t nnc_token_to_piece(const nnc_model* model, const nnc_token token, char* buffer, const int32_t capacity)
{
	if (!model || !model->vocab.valid(token) || capacity < 0 || (!buffer && capacity > 0))
	{
		invalid("nnc_token_to_piece: invalid arguments");
		return -1;
	}
	const std::string& piece = vocab_piece(model->vocab, token);
	const int32_t n = static_cast<int32_t>(piece.size());
	if (n > 0 && capacity > 0) std::memcpy(buffer, piece.data(), static_cast<size_t>(n < capacity ? n : capacity));
	return n;
}

const char* nnc_token_text(const nnc_model* model, const nnc_token token)
{
	if (!model || !model->vocab.valid(token)) return nullptr;
	return model->vocab.tokens[static_cast<size_t>(token)].c_str();
}

// ---- sessions -------------------------------------------------------------

nnc_status nnc_session_create(const nnc_model* model, const nnc_session_params* params, nnc_session** session)
{
	if (session) *session = nullptr;
	if (!model || !session) return invalid("nnc_session_create needs a model and somewhere to put the session");
	const nnc_session_params p = params ? *params : nnc_session_default_params();
	if (p.n_ctx < 0) return invalid("nnc_session_create: n_ctx must not be negative");
	if (p.n_draft < 0) return invalid("nnc_session_create: n_draft must not be negative");
	if (p.kv_type != NNC_KV_BF16 && p.kv_type != NNC_KV_F32)
		return invalid("nnc_session_create: kv_type must be NNC_KV_BF16 or NNC_KV_F32");
	return guarded([&]
	{
		auto s = std::make_unique<nnc_session>();
		if (!session_init(*s, *model, p)) return nnc_last_status();
		*session = s.release();
		return NNC_OK;
	});
}

void nnc_session_free(nnc_session* session)
{
	delete session;
}

void nnc_session_reset(nnc_session* session)
{
	if (session) session_reset(*session);
}

int32_t nnc_session_n_ctx(const nnc_session* session) { return session ? session->cache.n_ctx : 0; }

int32_t nnc_session_n_past(const nnc_session* session)
{
	return session ? static_cast<int32_t>(session->tokens.size()) : 0;
}

nnc_status nnc_session_eval(nnc_session* session, const nnc_token* tokens, const int32_t n_tokens)
{
	if (!session) return invalid("nnc_session_eval: no session");
	session->cancel.store(false);
	return guarded([&] { return session_eval(*session, tokens, n_tokens); });
}

const float* nnc_session_logits(nnc_session* session)
{
	if (!session) return nullptr;
	// Rebuilding a prediction lost to an earlier cancel is cancellable work
	// of this call, not a reason to fail it.
	session->cancel.store(false);
	const float* out = nullptr;
	guarded([&]
	{
		out = session_logits(*session);
		return NNC_OK;
	});
	return out;
}

nnc_token nnc_session_sample(nnc_session* session, const nnc_sampler_params* params)
{
	if (!session) return NNC_TOKEN_NULL;
	const nnc_sampler_params p = params ? *params : nnc_sampler_default_params();
	if (!valid_sampler(p))
	{
		invalid_sampler("nnc_session_sample");
		return NNC_TOKEN_NULL;
	}
	session->cancel.store(false);
	nnc_token tok = NNC_TOKEN_NULL;
	guarded([&]
	{
		tok = session_sample(*session, p);
		return NNC_OK;
	});
	return tok;
}

nnc_status nnc_session_generate(nnc_session* session, const nnc_token* prompt, const int32_t n_prompt,
                                const nnc_generate_params* params, nnc_generate_result* result)
{
	if (!session) return invalid("nnc_session_generate: no session");
	if (params && params->max_tokens < 0) return invalid("nnc_session_generate: max_tokens must not be negative");
	const nnc_generate_params p = generate_params(params);
	if (!valid_sampler(p.sampler)) return invalid_sampler("nnc_session_generate");
	session->cancel.store(false);
	nnc_generate_result r{};
	const nnc_status st = guarded([&] { return session_generate(*session, prompt, n_prompt, p, r); });
	if (result) *result = r;
	return st;
}

nnc_status nnc_session_chat(nnc_session* session, const char* message, const nnc_generate_params* params,
                            nnc_generate_result* result)
{
	if (!session || !message) return invalid("nnc_session_chat needs a session and a message");
	if (params && params->max_tokens < 0) return invalid("nnc_session_chat: max_tokens must not be negative");
	const nnc_generate_params p = generate_params(params);
	if (!valid_sampler(p.sampler)) return invalid_sampler("nnc_session_chat");
	session->cancel.store(false);
	nnc_generate_result r{};
	const nnc_status st = guarded([&] { return session_chat(*session, message, p, r); });
	if (result) *result = r;
	return st;
}

nnc_status nnc_session_translate(nnc_session* session, const char* text, const char* source, const char* target,
                                 const nnc_generate_params* params, nnc_generate_result* result)
{
	if (!session || !text || !source || !target)
		return invalid("nnc_session_translate needs a session, the text, and both languages");
	if (params && params->max_tokens < 0) return invalid("nnc_session_translate: max_tokens must not be negative");
	const nnc_generate_params p = generate_params(params);
	if (!valid_sampler(p.sampler)) return invalid_sampler("nnc_session_translate");
	session->cancel.store(false);
	nnc_generate_result r{};
	const nnc_status st = guarded([&] { return session_translate(*session, text, source, target, p, r); });
	if (result) *result = r;
	return st;
}

void nnc_session_cancel(nnc_session* session)
{
	if (session) session->cancel.store(true, std::memory_order_relaxed);
}

// ---- embeddings -------------------------------------------------------------

nnc_status nnc_embed(const nnc_model* model, const char* text, const size_t length, float* out,
                     const int32_t n_out)
{
	if (!model || (!text && length > 0) || !out || n_out <= 0)
		return invalid("nnc_embed needs a model, the text, and room for at least one value");
	return guarded([&]
	{
		if (!model_embed(*model, std::string(text ? text : "", length), out, n_out)) return nnc_last_status();
		return NNC_OK;
	});
}

} // extern "C"
