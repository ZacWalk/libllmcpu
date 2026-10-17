// nnc — neural-net op surface.
// Declarations for the SIMD/JIT-routed kernels (gelu, softmax, norms, rope,
// dot, gemv, quantize/dequantize, elementwise). The gemvs share one
// process-wide worker pool, whose size nnc_set_num_threads() (nnc.h) sets.

#pragma once

#include <cstddef>
#include <cstdint>

// Run fn(first, last, user) over disjoint ranges covering [0, n), split
// evenly across the worker pool; returns when every range is done. For
// coarse, uniform work such as load-time weight decoding.
using nnc_range_fn = void (*)(uint32_t first, uint32_t last, void* user);
void nnc_parallel_for(uint32_t n, nnc_range_fn fn, void* user);

// y[i] = 0.5 * x[i] * (1 + tanh( sqrt(2/pi) * (x[i] + 0.044715 * x[i]^3) ))
// FP32 in / FP32 out. y and x may alias (out-of-place use is also fine).
void nnc_gelu_f32(float* y, const float* x, size_t n);

// In-place numerically-stable softmax over n contiguous floats:
//   m = max(p);  p[i] = exp(p[i] - m) / sum_j exp(p[j] - m).
// -INFINITY entries map to 0 (used for causal attention masks).
void nnc_softmax_f32_inplace(float* p, size_t n);

// Gemma 4's expert choice: the `k` largest of the probabilities p[0..n)
// (ties go to the lower index) into idx[0..k), and their probabilities,
// renormalised to sum to 1, into w[0..k). Requires 0 < k <= n.
void nnc_moe_top_k(const float* p, int n, int k, int* idx, float* w);

// Blocked attention. A query's keys are cut into blocks; each block gives a
// partial result, and the partials are merged in block order:
//
//   nnc_attn_block(..., part0); acc = part0;
//   nnc_attn_block(..., part1); nnc_attn_merge(acc, part1, ...); ...
//   nnc_attn_finish(out, ..., acc, ...);
//
// The arithmetic depends only on where the blocks fall, so a caller that
// fixes them (model.cpp aligns them to absolute positions) gets the same
// bits whether the blocks run one after another or on different threads.
//
// A partial holds, per head, NNC_ATTN_PART(head_dim) floats: the weighted
// sum of V rows o[head_dim], then the block's max score m, then the sum of
// its weights l = sum_t exp(s_t - m).
#define NNC_ATTN_PART(head_dim) ((head_dim) + 2)

// One block of `n_keys` keys for `n_heads` query heads that share one KV
// head (grouped-query attention: each K/V row is read once for all of
// them). q holds the heads at `q_stride` floats apart. K and V are rows of
// `kv_stride` elements, F32 or (kv_bf16) BF16, with this KV head at offset
// 0. `scores` is scratch for n_heads * n_keys floats; `part` receives
// n_heads partials.
void nnc_attn_block(const float* q, size_t q_stride, const void* K, const void* V, bool kv_bf16,
                    size_t kv_stride, size_t n_keys, size_t head_dim, size_t n_heads, float scale,
                    float* scores, float* part);

// acc <- the merge of acc (earlier keys) and part (the next block).
void nnc_attn_merge(float* acc, const float* part, size_t head_dim, size_t n_heads);

// out[h] = o / l for each head, `out_stride` floats apart.
void nnc_attn_finish(float* out, size_t out_stride, const float* acc, size_t head_dim, size_t n_heads);

// LayerNorm over n contiguous floats (mean 0, variance 1, then optional
// affine is applied by a separate op
//   m  = mean(x); v = mean((x-m)^2);  y[i] = (x[i] - m) / sqrt(v + eps).
// y and x may alias.
void nnc_layernorm_f32(float* y, const float* x, size_t n, float eps);

// RMSNorm over n contiguous floats. No mean subtraction:
//   r = sqrt(mean(x^2) + eps);  y[i] = x[i] / r.
// y and x may alias. Used by Gemma / Llama-style models. The (optional)
// per-channel learned scale is applied by the caller.
void nnc_rmsnorm_f32(float* y, const float* x, size_t n, float eps);

// Apply (RMSNorm + per-channel gamma multiply) to `n_groups` contiguous
// length-`dim` vectors. Equivalent to:
//   for (g) { rmsnorm(y+g*dim, x+g*dim, dim); for (i) y[g*dim+i] *= gamma[i]; }
// but the gamma vector stays hot in L1 across all groups.
void nnc_rmsnorm_gamma_multi_f32(float* y, const float* x,
                                 size_t n_groups, size_t dim,
                                 const float* gamma, float eps);

// AVX2 / FMA dot product of two FP32 vectors. 4-accumulator unroll for
// 8*4 = 32-element strides; tail handled scalarly. Intended for the
// per-head attention Q.K dot (head_dim=256 / 512), where the JITed
// dot_f32 kernel's call overhead would dominate.
float nnc_dot_f32_simd(const float* a, const float* b, size_t n);

// BF16-weights, FP32-activations gemv:
//   y[r] = sum_{k=0..cols-1} bf16_to_fp32(W[r*cols + k]) * x[k]   for r in [0, rows).
// W is BF16 (uint16_t holding the upper 16 bits of an IEEE-754 binary32),
// x and y are FP32. Routes to a JITted AVX2+FMA kernel cached by
// (rows, cols) when cols is a multiple of 8 (vpmovzxwd + vpslld 16 +
// vfmadd231ps); scalar fallback otherwise.
void nnc_gemv_bf16w_f32x(const void* W, const float* x, float* y,
                         uint32_t rows, uint32_t cols);

// Streaming "BF16-weight gemv + argmax" — computes y[r] = sum_k bf16->f32(W[r,k])*x[k]
// for all r in [0, rows) and returns argmax_r y[r] without ever materialising
// the full y[] vector. Used for the lm_head / final logits projection when the
// caller only needs the greedy-decode token id (no top-k / top-p sampling).
//
// Walks the row axis in groups of 4 (re-using the cached BF16 4-row JIT kernel),
// updating a running (best_val, best_idx) pair from each 4-element batch. This
// avoids a full vocab-sized write (~1 MB at vocab=262144) and the subsequent
// softcap + linear-scan argmax passes. Softcap is monotonic so it has no effect
// on argmax and is omitted entirely.
//
// rows must be > 0 and a multiple of 4; cols must be > 0 and a multiple of 8.
// Falls back to nnc_gemv_bf16w_f32x + scratch + scalar argmax when those
// preconditions don't hold.
int nnc_gemv_bf16w_argmax_f32x(const void* W, const float* x,
                               uint32_t rows, uint32_t cols);

// Q8_0 split-layout gemv: y[r] = sum_b scales[r,b] * sum_{k in block b} qs[r,k]*x[k]
// for r in [0, rows). qs is row-major int8 of size rows*cols; scales is
// row-major BF16 of size rows*(cols/32). cols must be a positive multiple
// of 32. Routes to a JIT 1-row kernel (cols baked) and parallelises the
// row axis through the same worker pool as the BF16 path when rows is
// large enough to amortise dispatch.
void nnc_gemv_q8_0_f32x(const int8_t* qs, const uint16_t* scales,
                        const float* x, float* y,
                        uint32_t rows, uint32_t cols);

// Q4 (nnc split 4-bit) gemv. Reconstruction is w = scale*q + bias with q
// an unsigned nibble in [0, 15], so
//   y[r] = sum_b [ scales[r,b] * sum_{k in b} q[r,k]*x[k] + biases[r,b] * S[b] ]
// where S[b] = sum_{k in b} x[k] depends only on x. `xsum` must hold those
// cols/32 block sums (see nnc_block_sums_f32); passing it in keeps the
// bias term out of the JIT kernel and off the per-row critical path.
//
// qs is row-major packed nibbles of size rows*cols/2; scales and biases
// are row-major BF16 of size rows*(cols/32). BF16 rather than FP32 halves
// the metadata traffic (0.75 -> 0.625 bytes/weight), and rather than FP16
// because a block scale can be arbitrarily small — BF16 keeps FP32's
// exponent range so it cannot flush to subnormal. cols must be a positive
// multiple of 32.
void nnc_gemv_q4_s_f32x(const uint8_t* qs, const uint16_t* scales,
                        const uint16_t* biases, const float* x,
                        const float* xsum, float* y,
                        uint32_t rows, uint32_t cols);

// out[b] = sum of x[b*32 .. b*32+31], for b in [0, n/32). n must be a
// positive multiple of 32.
void nnc_block_sums_f32(const float* x, float* out, size_t n);

// ---- batched gemv (prefill) -------------------------------------------
//
// Same maths as the single-vector gemvs above, but applied to `n_batch`
// activation vectors against one pass over the weights. The loop order is
// row-major outer / batch inner, so each weight row is pulled from DRAM
// once and then reused from L1 for every vector in the batch — which is
// the entire point, since decode and prefill are both bandwidth bound.
//
// `X` holds n_batch vectors of `cols` floats at `x_stride` floats apart;
// `Y` holds n_batch vectors of `rows` floats at `y_stride` floats apart.
// n_batch == 1 is exactly the unbatched call.
void nnc_gemv_bf16w_f32x_batch(const void* W, const float* X, size_t x_stride,
                               float* Y, size_t y_stride,
                               uint32_t rows, uint32_t cols, uint32_t n_batch);

void nnc_gemv_q8_0_f32x_batch(const int8_t* qs, const uint16_t* scales,
                              const float* X, size_t x_stride,
                              float* Y, size_t y_stride,
                              uint32_t rows, uint32_t cols, uint32_t n_batch);

// `XSUM` holds the per-32-block sums of each X vector (see
// nnc_block_sums_f32), n_batch runs of cols/32 floats at xsum_stride apart.
void nnc_gemv_q4_s_f32x_batch(const uint8_t* qs, const uint16_t* scales,
                              const uint16_t* biases,
                              const float* X, size_t x_stride,
                              const float* XSUM, size_t xsum_stride,
                              float* Y, size_t y_stride,
                              uint32_t rows, uint32_t cols, uint32_t n_batch);

// In-place quantize a row-major BF16 weight matrix [rows x cols] into the
// Q8_0 split layout: writes `qs[rows*cols]` (int8) followed by
// `scales[rows*(cols/32)]` (BF16). cols must be a positive multiple of 32.
// Uses absmax-per-block scaling: scale = max(|w|)/127, q = round(w/scale).
void nnc_quantize_bf16_to_q8_0(const uint16_t* W_bf16, int8_t* qs,
                               uint16_t* scales, size_t rows, size_t cols);

// Same, from FP32 input, over a flat run of `n` elements (n must be a
// positive multiple of 32). Because Q8_0 blocks are 32 wide and every
// quantized weight matrix has `cols % 32 == 0`, a flat walk produces
// exactly the row-major block order the split layout expects — so this
// can be called incrementally on chunks of a larger matrix.
void nnc_quantize_f32_to_q8_0(const float* src, int8_t* qs,
                              uint16_t* scales, size_t n);

// Quantize a flat run of `n` FP32 weights (n a positive multiple of 32)
// into the nnc split 4-bit layout: writes `qs[n/2]` packed nibbles plus
// one BF16 `scale` and one BF16 `bias` per 32-element block, such that
//   w[k] ~= scales[k/32] * q[k] + biases[k/32],   q in [0, 15].
// Asymmetric (min/max) rather than absmax, which is what makes 4 bits
// usable: scale = (max-min)/15, bias = min.
//
// Nibble packing within a block: byte i holds element i in its low
// nibble and element i+16 in its high nibble.
void nnc_quantize_f32_to_q4_s(const float* src, uint8_t* qs,
                              uint16_t* scales, uint16_t* biases, size_t n);

// Returns sum_i bf16_to_f32(a[i]) * b[i]. Used for the Q4 bias term.
float nnc_dot_bf16_f32_simd(const uint16_t* a, const float* b, size_t n);

// ---- load-time dequantizers -------------------------------------------
//
// Q8_0 uses a 32-element block: { fp16 d; int8 qs[32] } = 34 bytes.
// Q4_K/Q5_K/Q6_K use a 256-element super-block with fp16 super-scales
// and 6-bit (Q4_K/Q5_K) or 8-bit (Q6_K) per-sub-block scales. Block
// sizes (matching ggml `block_q*`):
//   Q8_0 =  34 bytes  (  32 elems)
//   Q4_K = 144 bytes  ( 256 elems, 4-bit qs)
//   Q5_K = 176 bytes  ( 256 elems, 4-bit qs + 1-bit qh)
//   Q6_K = 210 bytes  ( 256 elems, 4-bit ql + 2-bit qh + i8 scales)
//
// `n_elements` must be a positive multiple of the type's block size.
// `blocks` points at the first packed block; `dst` receives
// `n_elements` contiguous floats. Scalar implementations: these run
// once at load time, feeding the BF16 / Q8_0 gemv kernels.
void nnc_dequantize_q8_0_to_f32(const void* blocks, float* dst, size_t n_elements);
void nnc_dequantize_q4_0_to_f32(const void* blocks, float* dst, size_t n_elements);
void nnc_dequantize_q4_1_to_f32(const void* blocks, float* dst, size_t n_elements);
void nnc_dequantize_q5_0_to_f32(const void* blocks, float* dst, size_t n_elements);
void nnc_dequantize_q5_1_to_f32(const void* blocks, float* dst, size_t n_elements);
void nnc_dequantize_q4_k_to_f32(const void* blocks, float* dst, size_t n_elements);
void nnc_dequantize_q5_k_to_f32(const void* blocks, float* dst, size_t n_elements);
void nnc_dequantize_q6_k_to_f32(const void* blocks, float* dst, size_t n_elements);

// Elements per storage block for a GGUF / ggml quantized type code.
// Returns 0 for types this build cannot decode, and for the plain float
// types (which have no blocks).
uint32_t nnc_quant_block_elems(uint32_t ggml_type);

// Decode `n_elements` values to F32, dispatching on the GGUF / ggml type
// code: 0=F32, 1=F16, 30=BF16, 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1, 8=Q8_0,
// 12=Q4_K, 13=Q5_K, 14=Q6_K. Returns false (and leaves dst untouched) for
// other types.
bool nnc_dequantize_to_f32(uint32_t ggml_type, const void* blocks,
                           float* dst, size_t n_elements);

// Fused SwiGLU activation used by Gemma / Llama MLPs:
//   y[i] = silu(gate[i]) * up[i]   where silu(x) = x * sigmoid(x).
// y may alias gate or up.
void nnc_swiglu_f32(float* y, const float* gate, const float* up, size_t n);

// y[i] *= sigmoid(g[i]): Qwen3.5's gated attention output.
void nnc_mul_sigmoid_f32(float* y, const float* g, size_t n);

// ---- Gated DeltaNet (Qwen3.5's linear-attention layers) -------------------

// Scale each of `n_groups` contiguous length-`dim` vectors to length
// `scale`:  x[g] *= scale / sqrt(sum(x[g]^2) + eps).
void nnc_l2norm_multi_f32(float* x, size_t n_groups, size_t dim, float eps, float scale);

// Causal depthwise convolution over time, then SiLU. `x` holds k - 1 + n_tok
// rows of `ch` floats - the k - 1 inputs before the batch, then the batch's
// own - and `w` the taps transposed to k rows of `ch`, oldest first:
//   y[t][c] = silu( sum_j w[j][c] * x[t + j][c] )      for t in [0, n_tok).
void nnc_conv1d_silu_f32(float* y, const float* x, const float* w, size_t n_tok, size_t ch, size_t k);

// One token of the gated delta rule for one head. The state S is d_v rows
// of d_k floats: row j holds what value lane j has stored along each key
// lane. With k unit-length and q unit-length times 1/sqrt(d_k):
//   S = decay * S;   S += beta * (v - S k) k^T;   o = S q
// `decay` = exp(g) lies in (0, 1] and `beta` in (0, 1). Each row is read,
// updated and read again while it is in L1.
void nnc_delta_rule_f32(float* S, const float* q, const float* k, const float* v, float decay, float beta,
                        float* o, size_t d_k, size_t d_v);

// SIMD elementwise add: y[i] += b[i] for i in [0, n).
void nnc_add_inplace_f32(float* y, const float* b, size_t n);

// SIMD elementwise multiply: y[i] *= s[i] for i in [0, n). Used to apply
// a learned per-channel scale (e.g. Gemma's RMSNorm gamma) right after
// nnc_rmsnorm_f32.
void nnc_mul_inplace_f32(float* y, const float* s, size_t n);

// Fused gated-MLP activation:  y[i] = gelu(gate[i]) * up[i].
// Replaces a two-pass `nnc_gelu_f32(gate); nnc_mul_inplace_f32(gate, up)`
// pair, halving memory traffic on `gate` (one read + one write instead
// of two of each). Matches the Gemma 4 "FFN_GELU + PAR" inner step.
// `y` may alias `gate` or `up`.
void nnc_gelu_mul_f32(float* y, const float* gate, const float* up, size_t n);

// Rotary position embedding. For one token at position `pos`, rotates the
// first `n_rot` lanes of each head's `head_dim`-vector in pairs:
//
//   theta_i = pos * freq_base^(-2i / n_rot) / freq_factors[i]   for i in [0, n_rot/2)
//   (a, b)  = (cos(theta_i) * a - sin(theta_i) * b,  sin(theta_i) * a + cos(theta_i) * b)
//
// NNC_ROPE_NEOX pairs lane i with lane i + n_rot/2 (Gemma, Falcon, ...);
// NNC_ROPE_NORM pairs lanes 2i and 2i + 1 (Llama, whose Q/K weights the
// GGUF converter permutes to suit it). Lanes [n_rot, head_dim) pass through
// unchanged. `n_heads` heads are processed, `head_dim` floats apart.
// `freq_factors` (n_rot/2 floats, may be null) divides each pair's angle:
// Llama 3's long-context scaling and Gemma 4's proportional RoPE, whose
// 1e30 entries switch rotation off for a pair. Linear position scaling is
// the caller's job (pass pos * scale).
enum nnc_rope_mode
{
	NNC_ROPE_NEOX = 0,
	NNC_ROPE_NORM = 1,
};

void nnc_rope_f32(float* x, uint32_t n_heads, uint32_t head_dim,
                  uint32_t n_rot, float pos, float freq_base,
                  nnc_rope_mode mode = NNC_ROPE_NEOX, const float* freq_factors = nullptr);

// Logit soft-cap (Gemma final-layer & some attention-layer outputs):
//   y[i] = tanh(x[i] / cap) * cap.
// y and x may alias.
void nnc_softcap_f32(float* y, const float* x, size_t n, float cap);

// Look up one row of a BF16 embedding table and convert it to FP32:
//   y[i] = bf16_to_f32(table[token_id * n_embd + i]) * scale
// for i in [0, n_embd). `scale` is typically sqrt(n_embd) for Gemma.
void nnc_embed_row_bf16(float* y, const void* table, int token_id,
                        size_t n_embd, float scale);

// Same, from a Q8_0 split-layout table: qs is row-major int8 of size
// rows*n_embd, scales is row-major BF16 of size rows*(n_embd/32).
// n_embd must be a positive multiple of 32.
void nnc_embed_row_q8_0(float* y, const int8_t* qs, const uint16_t* scales,
                        int token_id, size_t n_embd, float scale);

// Same, from a Q4 split-layout table (BF16 scales/biases).
void nnc_embed_row_q4_s(float* y, const uint8_t* qs, const uint16_t* scales,
                        const uint16_t* biases, int token_id,
                        size_t n_embd, float scale);
