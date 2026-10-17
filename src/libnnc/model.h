// nnc — model loader and forward pass for the Gemma 3, Gemma 4, Llama and
// Qwen (2, 2.5, 3, 3.5, 3.6, 3.8) families, and the EmbeddingGemma
// text-embedding models built on Gemma.
//
// model_load() maps a GGUF file, reads the hyperparameters and vocabulary,
// and builds nnc_tensor views of every weight. F32/F16/BF16 weights are used
// in place from the mapping; anything else is decoded once, at load, into
// the requested in-memory format (Q8_0 by default). The forward pass calls
// the kernels in nn_ops directly: there is no graph.
//
// `struct nnc_model` is the type nnc.h hands out as an opaque handle.

#pragma once

#include "gguf.h"
#include "nn_ops.h"
#include "runtime.h"
#include "tokenizer.h"

#include <deque>
#include <memory>
#include <string>
#include <vector>

// The forward-pass family. EmbeddingGemma ("gemma-embedding") runs as
// gemma3 and EmbeddingGemma 2 ("gemma-embedding2") as gemma4, with
// bidirectional attention and a pooled output instead of an lm_head.
enum class model_arch
{
	gemma3,
	gemma4,
	llama,
	qwen2, // Qwen2, Qwen2.5: Llama layers with Q/K/V biases
	qwen3, // Qwen3: Llama layers with per-head Q/K norms
	qwen35, // Qwen3.5, 3.6, 3.8 ("qwen35"): Gated DeltaNet layers, gated attention every fourth layer
};

// How an embedding model turns per-token hidden states into one vector
// (GGUF <arch>.pooling_type).
enum class model_pooling
{
	none = 0, // not an embedding model
	mean = 1,
	cls = 2, // the first token
	last = 3,
};

struct model_layer
{
	// Pre-norms (RMSNorm gamma, F32 vectors).
	nnc_tensor* attn_norm = nullptr; // [n_embd]
	nnc_tensor* ffn_norm = nullptr; // [n_embd]
	nnc_tensor* post_attention_norm = nullptr; // [n_embd]
	nnc_tensor* post_ffw_norm = nullptr; // [n_embd]
	nnc_tensor* post_norm = nullptr; // [n_embd]
	nnc_tensor* attn_q_norm = nullptr; // [head_dim]
	nnc_tensor* attn_k_norm = nullptr; // [head_dim]

	// Attention projections.
	nnc_tensor* attn_q = nullptr; // [n_embd, n_head*head_dim]; [q | gate] per head when attn_gated
	nnc_tensor* attn_k = nullptr; // [n_embd, n_head_kv*head_dim]
	nnc_tensor* attn_v = nullptr; // [n_embd, n_head_kv*head_dim]; absent = V is K
	nnc_tensor* attn_output = nullptr; // [n_head*head_dim, n_embd]
	nnc_tensor* attn_q_b = nullptr; // [n_head*head_dim] projection biases (Qwen2)
	nnc_tensor* attn_k_b = nullptr; // [n_head_kv*head_dim]
	nnc_tensor* attn_v_b = nullptr; // [n_head_kv*head_dim]

	// Gated DeltaNet (Qwen3.5's linear-attention layers), in place of
	// attention. q, k and v come from one projection, through a short causal
	// convolution; a decaying delta rule writes v into a per-head matrix
	// along k, and q reads it back. Shapes are in model_hparams (ssm_*).
	nnc_tensor* ssm_qkv = nullptr; // attn_qkv [n_embd, 2*n_k*d_k + n_v*d_v]
	nnc_tensor* ssm_gate = nullptr; // attn_gate [n_embd, n_v*d_v]: the output gate z
	nnc_tensor* ssm_alpha = nullptr; // [n_embd, n_v] (F32): the decay's input
	nnc_tensor* ssm_beta = nullptr; // [n_embd, n_v] (F32): the write strength's input
	nnc_tensor* ssm_conv1d = nullptr; // [d_conv, 2*n_k*d_k + n_v*d_v] (F32)
	nnc_tensor* ssm_dt = nullptr; // [n_v] decay bias
	nnc_tensor* ssm_a = nullptr; // [n_v] -exp(A_log): the decay rate
	nnc_tensor* ssm_norm = nullptr; // [d_v] output norm gamma
	nnc_tensor* ssm_out = nullptr; // [n_v*d_v, n_embd]
	const float* ssm_conv_taps = nullptr; // ssm_conv1d transposed: [d_conv][conv channels]

	// MLP projections.
	nnc_tensor* ffn_gate = nullptr; // [n_embd, n_ff]
	nnc_tensor* ffn_up = nullptr; // [n_embd, n_ff]
	nnc_tensor* ffn_down = nullptr; // [n_ff, n_embd]

	// Per-layer-input embedding gate (Gemma 4 PLE).
	nnc_tensor* inp_gate = nullptr; // [n_embd, ple_dim]
	nnc_tensor* proj = nullptr; // [ple_dim, n_embd]
	nnc_tensor* layer_output_scale = nullptr; // [1]

	// Mixture of experts (Gemma 4 26B A4B): beside the dense MLP above, a
	// router picks n_expert_used of n_expert small gated MLPs per token.
	// Expert weights are stacked [cols, rows, n_expert] and stored as one
	// split-layout matrix per expert, back to back (see expert_matrix).
	nnc_tensor* ffn_gate_inp = nullptr; // [n_embd, n_expert] router (F32)
	nnc_tensor* ffn_gate_inp_scale = nullptr; // [n_embd] router input scale
	nnc_tensor* ffn_gate_up_exps = nullptr; // [n_embd, 2*n_ff_exp, n_expert]: gate rows, then up rows
	nnc_tensor* ffn_gate_exps = nullptr; // [n_embd, n_ff_exp, n_expert] when gate and up are separate
	nnc_tensor* ffn_up_exps = nullptr; // [n_embd, n_ff_exp, n_expert]
	nnc_tensor* ffn_down_exps = nullptr; // [n_ff_exp, n_embd, n_expert]
	nnc_tensor* ffn_down_exps_scale = nullptr; // [n_expert] per-expert output scale; optional
	nnc_tensor* pre_ffw_norm_2 = nullptr; // [n_embd] expert input norm
	nnc_tensor* post_ffw_norm_1 = nullptr; // [n_embd] dense MLP output norm
	nnc_tensor* post_ffw_norm_2 = nullptr; // [n_embd] expert output norm
	int n_ff_exp = 0; // 0: no experts in this layer

	// Per-layer shapes, resolved once at load.
	int n_ff = 0;
	int head_dim = 0;
	int n_head = 0;
	int n_head_kv = 0;
	int n_rot = 0;
	int kv_layer = 0; // the layer whose K/V cache this layer reads: itself unless shared; -1 for none
	int rs_index = -1; // a Gated DeltaNet layer's slot in the recurrent state; -1 for attention layers
	bool attn_gated = false; // Qwen3.5: attention output * sigmoid(gate), gate from attn_q
	bool sliding_window = false;
	// RoPE frequency divisors for this layer (n_rot/2 floats), or null.
	const float* rope_freqs = nullptr;
};

struct model_hparams
{
	std::string arch; // general.architecture: "gemma3", "gemma4", "llama", "qwen35", ...
	model_arch arch_id = model_arch::gemma3;
	int n_vocab = 0;
	int n_ctx = 0; // <arch>.context_length (training context)
	int n_embd = 0;
	int n_layer = 0; // layers run; a Qwen3.5 file's trailing MTP block is not one
	int n_head = 0;
	int n_head_kv = 0;
	int head_dim = 0;
	int head_dim_swa = 0;
	int rope_dim = 0;
	int rope_dim_swa = 0;
	float rope_freq_base = 0;
	float rope_freq_base_swa = 0;
	float rope_freq_scale = 1; // 1/rope.scaling.factor (linear scaling)
	nnc_rope_mode rope_mode = NNC_ROPE_NEOX;
	float rms_eps = 0;
	int sliding_window = 0;
	int shared_kv_layers = 0; // trailing layers that reuse an earlier layer's KV
	int n_layer_kv = 0; // layers [0, n_layer_kv) own a KV cache
	int ple_dim = 0; // embedding_length_per_layer_input (Gemma 4)
	int n_expert = 0; // experts per MoE layer (0: dense model)
	int n_expert_used = 0; // experts the router picks per token
	float final_logit_softcap = 0;
	float attention_scale = 1.0f;
	float embd_scale = 1.0f; // sqrt(n_embd) for Gemma, 1 for Llama and Qwen
	int file_type = 0; // raw GGUF general.file_type

	// Gated DeltaNet (qwen35): the causal convolution's width, the q/k heads
	// and their size, the value heads and their size, and how many layers
	// are Gated DeltaNet layers (the rest attend).
	int ssm_d_conv = 0;
	int ssm_n_k = 0;
	int ssm_d_k = 0;
	int ssm_n_v = 0;
	int ssm_d_v = 0;
	int n_layer_rs = 0;

	// Text-embedding models: every token sees the whole sequence (sliding
	// layers see |i - j| <= sliding_window / 2), and the output is a pooled,
	// projected vector of n_embd_out floats rather than logits.
	bool causal = true;
	model_pooling pooling = model_pooling::none;
	int n_embd_out = 0; // 0: a generative model

	// MTP drafters (gemma4-assistant): the hidden width of the model they
	// draft for; 0 for every other model.
	int n_embd_target = 0;
};

struct nnc_model
{
	nnc_model() = default;
	~nnc_model();
	nnc_model(const nnc_model&) = delete;
	nnc_model& operator=(const nnc_model&) = delete;

	std::string path;
	std::string name; // general.name, else the file name
	gguf_file gguf; // owns the mapping
	model_hparams hparams;
	model_vocab vocab;

	// Top-level weights.
	nnc_tensor* token_embd = nullptr; // [n_embd, n_vocab]
	nnc_tensor* per_layer_token_embd = nullptr; // [ple_dim*n_layer, n_vocab]
	nnc_tensor* per_layer_model_proj = nullptr; // [n_embd, ple_dim*n_layer]
	nnc_tensor* per_layer_proj_norm = nullptr; // [ple_dim] (F32)
	nnc_tensor* output_norm = nullptr; // [n_embd] (F32)
	nnc_tensor* output = nullptr; // [n_embd, n_vocab]; absent = tied to token_embd
	nnc_tensor* rope_freqs = nullptr; // [rope_dim/2] (F32)

	// Embedding models: linear maps applied in order to the pooled hidden
	// state (EmbeddingGemma: dense_2 then dense_3; EmbeddingGemma 2: its
	// output projection). Empty for generative models.
	std::vector<nnc_tensor*> embd_proj;

	// MTP drafters: [target embedding | target hidden] -> n_embd at the
	// input, and n_embd -> the next step's hidden at the output.
	nnc_tensor* nextn_pre = nullptr; // [2 * n_embd_target, n_embd]
	nnc_tensor* nextn_post = nullptr; // [n_embd, n_embd_target]

	std::vector<model_layer> layers;

	// Storage for the tensor descriptors above (a deque keeps them put).
	std::deque<nnc_tensor> tensors;

	// Loader-owned weight buffers: decoded packed tensors and quantized
	// copies. `tensor->data` points into one of these, or into the mapping.
	std::vector<std::unique_ptr<uint8_t[]>> buffers;

	nnc_weight_format weights = NNC_WEIGHTS_Q8_0;
	size_t decoded_bytes = 0; // total size of `buffers`
	uint64_t n_params = 0;
};

// Load a model. On failure records the reason with nnc_fail() and returns
// false; `out` is then left for its destructor to clean up.
bool model_load(const std::string& path, nnc_model& out, nnc_weight_format fmt,
                nnc_progress_callback progress, void* progress_user);

// Keys per attention block. Blocks are aligned to absolute positions, so a
// query's result does not depend on how its work is shared out, and a
// sliding layer's ring buffer is a whole number of blocks.
constexpr int MODEL_ATTN_BLOCK = 256;

// One layer's K/V storage, token-major: position pos is row pos % rows,
// kv_dim elements of BF16 (the default) or F32. A full buffer has n_ctx
// rows. A causal sliding-window layer keeps only a ring of
// round_up(window + n_batch, MODEL_ATTN_BLOCK) rows, which holds every key
// a query of a batch can see, plus a margin for going back (see
// model_kv_rewind).
struct model_kv_buffer
{
	std::vector<uint8_t> k, v;
	int rows = 0;
	int window = 0; // a ring's window; 0 for a full buffer
};

// Per-layer key/value cache plus all forward-pass scratch. K/V live in
// bufs[kv_buf[li]]. kv_dim can differ per layer (Gemma 4 global layers are
// wider than its sliding-window ones). Layers that reuse an earlier layer's
// cache own no storage, and a bidirectional (embedding) model keeps one
// buffer for all layers: each layer rewrites the whole sequence before it
// attends. An MTP drafter's cache owns none either: kv_ext points at its
// target's, which kv_buf indexes, and its queries see the keys the target
// has evaluated.
struct model_kv_cache
{
	int n_ctx = 0; // max positions
	int cur_pos = 0; // next slot to fill
	std::vector<int> kv_dim_per_layer;
	std::vector<int> kv_buf; // [n_layer] the buffer in bufs each layer uses
	std::vector<model_kv_buffer> bufs;
	bool kv_bf16 = true; // element type of every buffer
	int kv_written = 0; // the highest position written since position 0, plus one
	int max_rewind = 0; // how far below kv_written cur_pos may go back; INT_MAX without rings
	model_kv_cache* kv_ext = nullptr; // MTP drafter: the target's cache

	// Recurrent state of the Gated DeltaNet layers (Qwen3.5), in slot
	// model_layer::rs_index of each: the last d_conv - 1 inputs of the
	// layer's causal convolution, and its delta-rule matrices, [d_v][d_k]
	// per value head. Unlike K/V it sums up the whole history, so it always
	// describes exactly cur_pos tokens: model_forward() at position 0 clears
	// it, and going back means model_state_restore().
	std::vector<float> conv_state; // [n_layer_rs][d_conv - 1][conv_dim]
	std::vector<float> ssm_state; // [n_layer_rs][n_v][d_v][d_k]
	std::vector<float> conv_saved; // model_state_save()'s copies
	std::vector<float> ssm_saved;
	int saved_pos = -1; // cur_pos when they were saved; -1: nothing saved

	// Forward-pass scratch, sized once in model_kv_init to the maxima
	// across all layers and n_batch tokens. Nothing is allocated per token.
	std::vector<float> sx; // [n_batch, n_embd]  residual stream / final hidden
	std::vector<float> snormed; // [n_batch, n_embd]
	std::vector<float> sq; // [n_batch, max_q_dim]
	std::vector<float> sk_cur; // [n_batch, max_kv_dim]
	std::vector<float> sv_cur; // [n_batch, max_kv_dim]
	std::vector<float> sattn; // [n_batch, max_q_dim]
	std::vector<float> sgate; // [n_batch, max_ff]
	std::vector<float> sup; // [n_batch, max_ff]
	std::vector<float> sscores; // n_attn_slots x attn_slot: block scores, then two partials
	std::vector<float> sattn_part; // [n_head_max * n_blocks * NNC_ATTN_PART(head_dim)]  decode's per-block partials
	std::vector<float> sple_gate; // [n_batch, ple_dim]
	std::vector<float> sple_contrib; // [n_batch, n_embd]
	std::vector<float> sple_inputs; // [n_batch, ple_dim * n_layer]
	std::vector<float> sple_proj; // [n_batch, ple_dim * n_layer]; empty without per_layer_token_embd
	std::vector<float> sxsum; // [n_batch, max_cols/32]  Q4 per-block sums of x
	std::vector<float> sq_gate; // [n_batch, 2 * max_q_dim]  gated attention: [q | gate] per head, as projected
	std::vector<float> sattn_gate; // [n_batch, max_q_dim]  ...and its gate half

	// Gated DeltaNet scratch; empty for other models.
	std::vector<float> sdn_conv; // [d_conv - 1 + n_batch, conv_dim]  convolution input: history, then the batch
	std::vector<float> sdn_qkv; // [n_batch, conv_dim]  convolution output: q | k | v per token
	std::vector<float> sdn_z; // [n_batch, n_v * d_v]  output gate
	std::vector<float> sdn_gb; // [n_batch, 2 * n_v]  each head's decay, then its write strength
	std::vector<float> sdn_out; // [n_batch, n_v * d_v]

	// Mixture-of-experts scratch; empty for dense models.
	std::vector<float> smoe_in; // [n_batch, n_embd]  expert input (pre_ffw_norm_2)
	std::vector<float> smoe_out; // [n_batch, n_embd]  weighted sum of expert outputs
	std::vector<float> smoe_rows; // [n_batch, n_embd]  inputs gathered for one expert
	std::vector<float> smoe_down; // [n_batch, n_embd]  one expert's outputs
	std::vector<float> smoe_gate_up; // [n_batch, 2 * max_ff_exp]
	std::vector<float> smoe_router; // [n_embd]  router input of one token
	std::vector<float> smoe_probs; // [n_expert]
	std::vector<float> smoe_weight; // [n_batch, n_expert_used]
	std::vector<int> smoe_expert; // [n_batch, n_expert_used]  chosen experts
	std::vector<int> smoe_pairs; // [n_batch]  the (token, slot) pairs that chose one expert

	// MTP drafter scratch; empty for other models.
	std::vector<float> sdraft_in; // [2 * n_embd_target]  [target embedding | hidden]
	std::vector<float> sdraft_logits; // [n_vocab]

	int n_batch = 1; // most tokens one model_forward call may take
	int n_head_max = 1; // query heads in the widest layer
	int n_attn_slots = 1; // scores slots: how many chunks of attention work run at once
	size_t attn_slot = 0; // floats per slot
	int attn_part_max = 0; // the most n_head * NNC_ATTN_PART(head_dim) of any layer
	int n_attn_blocks = 0; // the most blocks one query's keys span
	size_t bytes = 0; // K/V storage, and the recurrent state
};

// Tokens per prefill batch. Large enough to amortise a weight pass over
// many tokens, small enough that the batch's activations stay in L2.
constexpr int MODEL_PREFILL_BATCH = 16;

// kv_f32 keeps K/V as F32 rather than BF16, at twice the memory.
bool model_kv_init(const nnc_model& m, model_kv_cache& cache, int n_ctx, int n_batch, bool kv_f32 = false);

// Move cur_pos back to `pos` (<= cur_pos), keeping the K/V below it. A ring
// buffer may have overwritten keys that position needs, when it is more
// than max_rewind below the highest position written; then this returns
// false and changes nothing, and the caller goes back to 0 instead (which
// always succeeds). Recurrent state is not touched: see model_state_restore.
bool model_kv_rewind(model_kv_cache& cache, int pos);

// A model with Gated DeltaNet layers keeps a recurrent state that cannot be
// cut back as K/V can (see model_kv_cache). model_state_save() copies it,
// with cur_pos; model_state_restore() puts both back, and returns false
// when nothing was saved. For other models neither does anything.
bool model_is_recurrent(const nnc_model& m);
void model_state_save(model_kv_cache& cache);
bool model_state_restore(model_kv_cache& cache);

// Run `n_tok` consecutive tokens (n_tok <= cache.n_batch) at positions
// [pos0, pos0 + n_tok), pos0 == cache.cur_pos, appending their K/V and
// advancing cache.cur_pos. Afterwards model_last_hidden() is the final
// hidden state of the last token, the input to the lm_head. A bidirectional
// model takes its whole sequence in one call, from position 0.
bool model_forward(const nnc_model& m, model_kv_cache& cache, const int* tokens, int n_tok, int pos0);
const float* model_last_hidden(const nnc_model& m, const model_kv_cache& cache, int n_tok);

// The keys that the query at `pos` attends to, [*first, *last], in a
// sequence whose keys fill [0, n_seq). A causal model sees its prefix up to
// `pos` (an MTP drafter's query, at the position after the keys, sees them
// all), and a sliding layer (window > 0) only the last `window` positions
// up to `pos`. A bidirectional model sees the whole sequence, and a sliding
// layer only |key - pos| <= window / 2.
void model_attn_span(bool causal, int window, int pos, int n_seq, int* first, int* last);

// The most tokens model_embed() looks at; longer input is cut.
constexpr int MODEL_EMBED_MAX_TOKENS = 8192;

// An embedding model's vector for `text`: the text between the model's
// BOS and EOS (cut to fit its context), run through the model with
// bidirectional attention, pooled, projected, cut to its first `n_out`
// values and scaled to unit length. Allocates its own context. On failure
// records the reason with nnc_fail() and returns false.
bool model_embed(const nnc_model& m, const std::string& text, float* out, int n_out);

// Pool `n_tok` rows of `n` floats (token-major) into `out`, as `pooling` says.
void model_pool(model_pooling pooling, const float* rows, int n_tok, int n, float* out);

// Scale the first `n` values of `x` to unit Euclidean length (left as they
// are when they are all zero).
void model_l2_normalize(float* x, int n);

// lm_head: logits (n_vocab floats, soft-capped) for a final hidden state.
void model_logits(const nnc_model& m, const float* hidden, float* logits);

// lm_head for the first `n` final hidden states the last model_forward()
// left in `cache`, in one pass over the weights: n rows of n_vocab logits,
// soft-capped if `soft_cap`. Each row is what model_logits() gives for that
// hidden state alone (Q8_0 and Q4 weights).
void model_logits_batch(const nnc_model& m, model_kv_cache& cache, int n, float* logits, bool soft_cap);

// MTP drafting with a Gemma 4 "assistant" drafter. model_draft_init sets up
// `dcache` to read `tcache`, the target's cache (both must stay put while
// in use); false, with the reason, when the drafter was not made for the
// target.
bool model_draft_init(const nnc_model& drafter, const nnc_model& target, model_kv_cache& tcache,
                      model_kv_cache& dcache);

// One draft step at position `pos`: `token` is the next token, which the
// target has not evaluated yet (its cache holds positions [0, pos)), and
// `h` the target's final hidden state before it, or the previous step's
// `h_out`. Returns the drafter's guess for the token after `token`, sets *p
// to the probability it gives that guess, and writes the next step's h to
// `h_out` (n_embd_target floats; may alias `h`). Every step of one round
// uses the same `pos`.
int model_draft_step(const nnc_model& drafter, const nnc_model& target, model_kv_cache& dcache, int token,
                     const float* h, int pos, float* h_out, float* p);

// argmax of the logits without soft-capping them (it is monotonic).
// `scratch` must hold n_vocab floats; the BF16 path streams instead.
int model_argmax(const nnc_model& m, const float* hidden, float* scratch);

// Dtype-dispatching building blocks, exposed for diagnostics.
void model_gemv(const nnc_tensor* W, const float* x, float* y);
void model_embed_row(const nnc_tensor* T, float* y, int token, size_t n, float scale);