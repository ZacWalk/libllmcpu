/* nnc.h — the public API of libnnc.
 *
 * libnnc loads GGUF language models (Gemma 3, Gemma 4, Llama and Qwen
 * families) and text-embedding models (EmbeddingGemma) and runs them on
 * the CPU with JIT-compiled AVX2/FMA kernels. This header is the whole
 * public surface.
 * It is plain C, so it works from C, from C++ and from FFI bindings. Every
 * other header in src/libnnc is internal and may change without notice.
 *
 * Typical use:
 *
 *   nnc_model* model = NULL;
 *   if (nnc_model_load("gemma-3-1b-it-Q6_K.gguf", NULL, &model) != NNC_OK) {
 *       fprintf(stderr, "%s\n", nnc_last_error());
 *       return 1;
 *   }
 *   nnc_session* chat = NULL;
 *   nnc_session_create(model, NULL, &chat);
 *
 *   nnc_generate_params gen = nnc_generate_default_params();
 *   gen.on_text = print_text;              // streamed as it is generated
 *   nnc_session_chat(chat, "Why is the sky blue?", &gen, NULL);
 *
 *   nnc_session_free(chat);
 *   nnc_model_free(model);
 *
 * Threading: a model is immutable once loaded and can be shared by any
 * number of sessions on any threads. A session must be used by one thread
 * at a time; the one exception is nnc_session_cancel(), which may be called
 * from any thread or from a signal handler. All sessions share one
 * process-wide worker pool, so two sessions running at once is safe but not
 * faster than running them one after the other.
 *
 * Strings are UTF-8 throughout, including file paths on Windows.
 */

#ifndef NNC_H
#define NNC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- status and errors ------------------------------------------------- */

typedef enum nnc_status
{
	NNC_OK = 0,
	NNC_ERR_INVALID_ARGUMENT = 1, /* NULL pointer, out-of-range value, ... */
	NNC_ERR_UNSUPPORTED_CPU = 2, /* the CPU or OS lacks AVX2 + FMA */
	NNC_ERR_IO = 3, /* a file could not be opened, mapped or read */
	NNC_ERR_BAD_FORMAT = 4, /* not a GGUF file, or a corrupt one */
	NNC_ERR_UNSUPPORTED_MODEL = 5, /* an architecture or tensor type nnc cannot run */
	NNC_ERR_OUT_OF_MEMORY = 6,
	NNC_ERR_CONTEXT_FULL = 7, /* the session's context window has no room */
	NNC_ERR_CANCELLED = 8, /* stopped by a progress callback or nnc_session_cancel */
} nnc_status;

/* A short, static description of `status`, e.g. "context full". */
const char* nnc_status_string(nnc_status status);

/* Detail of the most recent failure on the calling thread, e.g. "cannot
 * open 'x.gguf': the system cannot find the file specified". Never NULL.
 * Valid until the next failing call on the same thread. */
const char* nnc_last_error(void);

/* ---- library ------------------------------------------------------------ */

/* The library version, e.g. "0.3.0". */
const char* nnc_version(void);

/* True when this CPU and OS can run nnc: x86-64 with AVX2 and FMA, with the
 * OS saving YMM state. Otherwise nnc_model_load() fails with
 * NNC_ERR_UNSUPPORTED_CPU. Every function that takes no model or session is
 * safe to call on any x86-64 CPU; on an unsupported one the thread and
 * profiler settings do nothing. */
bool nnc_cpu_supported(void);

typedef enum nnc_log_level
{
	NNC_LOG_DEBUG = 0,
	NNC_LOG_INFO = 1,
	NNC_LOG_WARN = 2,
	NNC_LOG_ERROR = 3,
} nnc_log_level;

/* Receives one complete message without a trailing newline. */
typedef void (*nnc_log_callback)(nnc_log_level level, const char* message, void* user);

/* Route library diagnostics to `callback`; NULL restores the default, which
 * writes warnings and errors to stderr. Failures are reported through
 * return values and nnc_last_error(), not through the log. Set this before
 * loading models: it is not synchronised with running work. */
void nnc_set_log_callback(nnc_log_callback callback, void* user);

/* Threads that share the large matrix-vector products: the calling thread
 * plus workers. 0 restores the default, which is $NNC_THREADS if set, else
 * min(8, logical CPUs). Capped at 64. Takes effect at the next kernel
 * dispatch, and waits for any dispatch in flight. */
void nnc_set_num_threads(int n_threads);
int nnc_num_threads(void);

/* ---- models -------------------------------------------------------------- */

typedef struct nnc_model nnc_model;

/* How weights are held in memory. They are decoded into this layout once,
 * at load, whatever format the file stores them in. */
typedef enum nnc_weight_format
{
	NNC_WEIGHTS_Q8_0 = 0, /* 8-bit, ~1.06 bytes/weight; near-lossless; the default */
	NNC_WEIGHTS_Q4 = 1, /* 4-bit, ~0.63 bytes/weight; ~1.6x faster, lower quality */
	NNC_WEIGHTS_BF16 = 2, /* 16-bit, used in place from the file where possible */
} nnc_weight_format;

/* Load progress in [0, 1]. Return false to cancel the load. */
typedef bool (*nnc_progress_callback)(float progress, void* user);

typedef struct nnc_model_params
{
	nnc_weight_format weights;
	nnc_progress_callback progress; /* optional */
	void* progress_user;
} nnc_model_params;

nnc_model_params nnc_model_default_params(void);

/* Load a GGUF model. `params` may be NULL for the defaults. On success
 * *model owns the loaded model; release it with nnc_model_free(). */
nnc_status nnc_model_load(const char* path, const nnc_model_params* params, nnc_model** model);
void nnc_model_free(nnc_model* model);

typedef struct nnc_model_info
{
	const char* name; /* general.name, or the file name; lives as long as the model */
	const char* architecture; /* "gemma3", "gemma4", "llama", "qwen2", "qwen3", "qwen35", "gemma-embedding", ... */
	nnc_weight_format weights;
	int32_t n_vocab;
	int32_t n_ctx_train; /* context length the model was trained for */
	int32_t n_embd;
	int32_t n_layer;
	int32_t n_head;
	int32_t n_head_kv;
	uint64_t n_params; /* weights in the file */
	uint64_t file_bytes; /* size of the GGUF file */
	uint64_t resident_bytes; /* weight bytes decoded into RAM (not counting the file mapping) */
	bool has_chat_template; /* nnc_session_chat() knows this model's turn format */
	int32_t n_embd_out; /* text-embedding models: values per nnc_embed() vector; 0 for generative models */
	bool translator; /* TranslateGemma: made for nnc_session_translate() */
	int32_t n_embd_target; /* MTP drafters: the width (n_embd) of the model they draft for; else 0 */
	bool has_thinking; /* it can reason before it answers: see nnc_session_params.thinking */
} nnc_model_info;

/* All fields zero when `model` is NULL. */
nnc_model_info nnc_model_get_info(const nnc_model* model);

/* ---- tokens -------------------------------------------------------------- */

typedef int32_t nnc_token;
#define NNC_TOKEN_NULL ((nnc_token)-1)

/* Special tokens; NNC_TOKEN_NULL when the model has none. */
nnc_token nnc_token_bos(const nnc_model* model);
nnc_token nnc_token_eos(const nnc_model* model);

/* True for tokens that end generation: end-of-sequence and end-of-turn. */
bool nnc_token_is_eog(const nnc_model* model, nnc_token token);

/* Tokenize `length` bytes of UTF-8 `text`. Spellings of control tokens in
 * the text ("<start_of_turn>") are kept as plain text. Writes at most
 * `capacity` tokens to `tokens` and returns how many the text needs, so a
 * result larger than `capacity` means the buffer was too small. Returns -1
 * for invalid arguments. */
int32_t nnc_tokenize(const nnc_model* model, const char* text, size_t length, bool add_bos,
                     nnc_token* tokens, int32_t capacity);

/* The bytes `token` renders to: byte tokens become the byte, control tokens
 * become nothing. A piece may be part of a UTF-8 character, because some
 * characters span several tokens. Copies at most `capacity` bytes without a
 * terminator and returns the piece's full length, or -1 for an invalid
 * token. */
int32_t nnc_token_to_piece(const nnc_model* model, nnc_token token, char* buffer, int32_t capacity);

/* The vocabulary entry for `token` as the file spells it ("▁Paris",
 * "<0x0A>", "<end_of_turn>"): NUL-terminated, valid as long as the model.
 * NULL for an invalid token. Meant for debugging. */
const char* nnc_token_text(const nnc_model* model, nnc_token token);

/* ---- sessions ------------------------------------------------------------- */

/* A session is one conversation: a context window of tokens, its key/value
 * cache, and the sampler's random state. */
typedef struct nnc_session nnc_session;

typedef enum nnc_kv_type
{
	NNC_KV_BF16 = 0,
	NNC_KV_F32 = 1
} nnc_kv_type;

typedef struct nnc_session_params
{
	int32_t n_ctx; /* context window in tokens; 0 = min(4096, n_ctx_train) */
	uint64_t seed; /* sampling seed; 0 = random */
	const char* system_prompt; /* optional, used by nnc_session_chat(); copied */

	/* Optional speculative decoding with a multi-token-prediction (MTP)
	 * drafter: a small model made for this one (Gemma 4's "assistant"
	 * GGUFs, e.g. mtp-gemma-4-E2B-it-Q8_0.gguf for Gemma 4 E2B). It guesses
	 * the next few tokens and the model checks them all in one pass, which
	 * makes generation faster without changing what is generated: with Q8_0
	 * or Q4 weights and the same seed, the text is exactly the same as
	 * without it. It must outlive the session; nnc_session_create() fails if
	 * it was not made for the model. */
	const nnc_model* draft_model;
	int32_t n_draft; /* most tokens drafted per step; 0 = the default (4; 2 for mixture-of-experts models) */

	/* Element type of the K/V cache. BF16 (the default) halves its memory
	 * and the bandwidth attention spends reading it; F32 keeps keys and
	 * values exactly as computed. */
	nnc_kv_type kv_type;

	/* Let the model reason before it answers, in every chat and translation
	 * turn, on models that can (nnc_model_info.has_thinking): Gemma 4, and
	 * Qwen models whose chat template has a thinking switch (Qwen3, 3.5, 3.6
	 * and 3.8). Off by default: it helps with hard problems such as code and
	 * maths, and makes every reply much longer. The reasoning goes to
	 * nnc_generate_params.on_thinking, never to on_text. It stays in the
	 * context, where the model sees it in later turns and where it counts
	 * towards n_ctx, so give a thinking session a larger context. Models
	 * without a thinking mode ignore this; models made to always reason
	 * reason anyway. */
	bool thinking;
} nnc_session_params;

nnc_session_params nnc_session_default_params(void);

/* `params` may be NULL for the defaults. The model must outlive the session. */
nnc_status nnc_session_create(const nnc_model* model, const nnc_session_params* params,
                              nnc_session** session);
void nnc_session_free(nnc_session* session);

/* Forget the conversation and start again from an empty context. */
void nnc_session_reset(nnc_session* session);

int32_t nnc_session_n_ctx(const nnc_session* session);

/* Tokens in the context so far. */
int32_t nnc_session_n_past(const nnc_session* session);

/* Append `n_tokens` tokens to the context and run the model over them
 * (in batches). Afterwards nnc_session_logits() and nnc_session_sample()
 * describe the token that would come next. Fails with NNC_ERR_CONTEXT_FULL,
 * changing nothing, when they do not fit. If cancelled, the context is put
 * back as it was and NNC_ERR_CANCELLED is returned. */
nnc_status nnc_session_eval(nnc_session* session, const nnc_token* tokens, int32_t n_tokens);

/* Logits (n_vocab floats) for the token after the context, or NULL when
 * nothing has been evaluated. Valid until the next call on the session. */
const float* nnc_session_logits(nnc_session* session);

typedef struct nnc_sampler_params
{
	float temperature; /* 0 = greedy: always the single most likely token */
	int32_t top_k; /* >= 0; keep the k most likely tokens; 0 = all */
	float top_p; /* [0, 1]; keep the fewest tokens reaching top_p; 1 = off */
	float min_p; /* [0, 1]; drop tokens below min_p * p(most likely); 0 = off */
} nnc_sampler_params;

/* temperature 0.7, top_k 40, top_p 0.95, min_p 0.05 */
nnc_sampler_params nnc_sampler_default_params(void);

/* Choose the token after the context. It is not added to the context.
 * NNC_TOKEN_NULL when nothing has been evaluated or params are invalid;
 * invalid params are described by nnc_last_error(). */
nnc_token nnc_session_sample(nnc_session* session, const nnc_sampler_params* params);

typedef enum nnc_stop_reason
{
	NNC_STOP_END_OF_TURN = 0, /* the model produced an end-of-generation token */
	NNC_STOP_MAX_TOKENS = 1, /* max_tokens were generated */
	NNC_STOP_CONTEXT_FULL = 2, /* the context window filled up */
	NNC_STOP_CALLBACK = 3, /* on_text returned false */
	NNC_STOP_CANCELLED = 4, /* nnc_session_cancel() was called */
} nnc_stop_reason;

/* Receives generated text as it is produced: valid UTF-8 in whole
 * characters, not NUL-terminated. A character split across tokens arrives
 * once it is complete; bytes that cannot form one arrive as U+FFFD. Return
 * false to stop generating; the callback is not called again in that call. */
typedef bool (*nnc_text_callback)(const char* text, size_t length, void* user);

typedef struct nnc_generate_params
{
	int32_t max_tokens; /* 0 = until the end of the turn or a full context; reasoning tokens count */
	nnc_sampler_params sampler;
	nnc_text_callback on_text; /* optional: the answer */
	void* user;
	/* Optional: the model's reasoning, as it is generated, without the
	 * markers around it - for a model that reasons (see
	 * nnc_session_params.thinking). Called with `user`, like on_text, and
	 * returning false stops generating in the same way. Without it the
	 * reasoning is generated but not delivered. */
	nnc_text_callback on_thinking;
} nnc_generate_params;

/* max_tokens 0 and nnc_sampler_default_params() */
nnc_generate_params nnc_generate_default_params(void);

typedef struct nnc_generate_result
{
	nnc_stop_reason stop;
	int32_t n_prompt; /* tokens run through the model before generating */
	int32_t n_generated; /* tokens generated, not counting the end-of-turn token */
	double prefill_ms; /* time spent on the prompt */
	double decode_ms; /* time spent generating */
	int32_t n_drafted; /* tokens an MTP drafter proposed (0 without one) */
	int32_t n_draft_accepted; /* ...and how many of them the model kept */
	int32_t n_thinking; /* of n_generated, the tokens of reasoning, its markers included */
} nnc_generate_result;

/* Append `prompt` to the context (it may be empty when the context already
 * ends where generation should start), then generate. Generated tokens join
 * the context, so a later call carries on from them; if this call stopped
 * part-way through a character (max_tokens, a cancel or on_text returning
 * false), a later call with an empty prompt delivers the rest of it.
 * `params` and `result` may be NULL. Invalid sampler settings return
 * NNC_ERR_INVALID_ARGUMENT. Cancelling while the prompt is being processed
 * puts the context back and returns NNC_ERR_CANCELLED; cancelling while
 * generating returns NNC_OK with stop reason NNC_STOP_CANCELLED. */
nnc_status nnc_session_generate(nnc_session* session, const nnc_token* prompt, int32_t n_prompt,
                                const nnc_generate_params* params, nnc_generate_result* result);

/* One chat turn: add `message` to the context as a user turn in the model's
 * chat template (closing the previous reply first, and including the system
 * prompt on the first turn), then generate the reply. A model without a
 * known template gets the text as it is. NNC_ERR_CONTEXT_FULL means the
 * turn does not fit; nnc_session_reset() makes room. In a thinking session
 * (nnc_session_params.thinking) the reply opens with the model's reasoning,
 * which goes to on_thinking.
 *
 * FunctionGemma takes the system prompt as its "developer" turn: there,
 * declare the functions it may call (see Google's FunctionGemma docs for
 * the <start_function_declaration> syntax). A reply that calls one ends
 * with <start_function_call>call:NAME{...}<end_function_call>, and
 * generation stops at the <start_function_response> after it. Run the
 * function, then continue the same turn with nnc_session_generate() on
 * the tokens of "response:NAME{...}<end_function_response>"; the model
 * then answers in words. Function tokens are matched whole when tokenized. */
nnc_status nnc_session_chat(nnc_session* session, const char* message,
                            const nnc_generate_params* params, nnc_generate_result* result);

/* One translation turn: ask the model to translate `text` from language
 * `source` into language `target`, then generate the translation. Give
 * languages as codes such as "en", "de" or "pt-BR". TranslateGemma knows
 * 581 codes, including regional variants, and gets exactly the prompt it
 * was trained on; any other chat model gets the same request as a user
 * message, and also takes a language's English name. The system prompt is
 * not used. Like a chat turn, the translation joins the context. */
nnc_status nnc_session_translate(nnc_session* session, const char* text, const char* source,
                                 const char* target, const nnc_generate_params* params,
                                 nnc_generate_result* result);

/* Ask the call running on `session` to stop at the next token boundary.
 * Safe from any thread and from a signal handler. Every call starts
 * uncancelled, so a cancel that arrives while no call is running is
 * ignored. */
void nnc_session_cancel(nnc_session* session);

/* ---- embeddings ------------------------------------------------------------ */

/* A text-embedding model (nnc_model_info.n_embd_out > 0) turns text into a
 * vector whose direction stands for its meaning: texts about the same thing
 * get vectors that point the same way. That is the basis of semantic
 * search, clustering and deduplication. Such a model cannot generate text,
 * so nnc_session_create() refuses it, and nnc_embed() refuses a generative
 * model.
 *
 * Write the embedding of `length` bytes of UTF-8 `text` to `out`, `n_out`
 * floats. The vector has unit length, so the dot product of two of them is
 * their cosine similarity: 1 for the same meaning, near 0 for unrelated
 * text. `n_out` may be smaller than n_embd_out to keep only the first
 * values; EmbeddingGemma is trained for 512, 256 and 128 as well as its
 * full 768, and the shorter vector is scaled to unit length again. Text
 * beyond the model's context (2048 tokens for EmbeddingGemma, 8192 for
 * EmbeddingGemma 2) is ignored.
 *
 * EmbeddingGemma expects a prefix naming the task, embedded with the text:
 *   "task: search result | query: "       a search query, and
 *   "title: none | text: "                the documents it is matched against
 *   "task: sentence similarity | query: " texts compared with each other
 *   "task: clustering | query: "          likewise "classification",
 *                                         "question answering", "fact checking"
 *                                         and "code retrieval"
 *
 * The model is only read, so any number of threads may embed with it at
 * once; each call allocates its own working memory. */
nnc_status nnc_embed(const nnc_model* model, const char* text, size_t length, float* out, int32_t n_out);

/* ---- profiling ------------------------------------------------------------- */

/* A process-wide phase profiler: off by default and free when off. When on,
 * every kernel phase accumulates wall time and weight traffic. Meant for one
 * session at a time. */
void nnc_perf_enable(bool enabled);
void nnc_perf_reset(void);

/* A table of the time and memory bandwidth per phase since the last reset;
 * `n_tokens` adds a per-token line when positive. The text stays valid until
 * the next call. */
const char* nnc_perf_report(const char* title, int32_t n_tokens);

#ifdef __cplusplus
}
#endif

#endif /* NNC_H */
