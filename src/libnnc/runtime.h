// nnc runtime: diagnostics (errors, logging, assert), CPU features, dtype
// metadata, tensor descriptors, timers and the phase profiler. Computation
// lives in nn_ops.cpp (SIMD) and jit_ops.cpp (JIT); model.cpp drives those
// kernels directly.

#pragma once

#include "nnc.h"

#include <cstdint>
#include <cstring>
#include <string>

#define NNC_MAX_DIMS 4

#ifndef NNC_ASSERT
// Custom assert: print "ASSERT failed: <expr> at <file>:<line>" to stderr,
// flush, and abort the process. No modal message box.
[[noreturn]] void nnc_assert_fail(const char* expr, const char* file, int line);
#define NNC_ASSERT(x) ((x) ? (void)0 : nnc_assert_fail(#x, __FILE__, __LINE__))
#endif

#if defined(__GNUC__)
#define NNC_PRINTF(fmt_index, first_arg) __attribute__((format(printf, fmt_index, first_arg)))
#else
#define NNC_PRINTF(fmt_index, first_arg)
#endif

// --- diagnostics --------------------------------------------------------
//
// Failures travel up as `false` / nullptr; the code that detects one records
// the status and a specific message here, and the public API hands both to
// the caller (the status as the return value, the message through
// nnc_last_error()). Nothing in the library prints a failure itself. These
// live in baseline.cpp, so they work before the CPU check has passed.

// Record a failure on this thread and return false, so a failing path is
//   if (!ok) return nnc_fail(NNC_ERR_IO, "cannot open '%s'", path);
bool nnc_fail(nnc_status status, const char* fmt, ...) NNC_PRINTF(2, 3);

// The status of the most recent nnc_fail() on this thread.
nnc_status nnc_last_status();
void nnc_clear_error();

// Send a message to the log callback (see nnc_set_log_callback).
void nnc_log(nnc_log_level level, const char* fmt, ...) NNC_PRINTF(2, 3);

// --- CPU features -------------------------------------------------------

struct cpu_features
{
	bool avx2 = false;
	bool fma = false;
	bool f16c = false;
};

// Detected once. Lives in baseline.cpp, which is compiled without AVX flags
// so the check itself runs on any x86-64 CPU.
const cpu_features& nnc_cpu_features();

// --- behind the CPU gate ------------------------------------------------
//
// The AVX-compiled halves of public functions whose front halves, in
// baseline.cpp, first check that the CPU can run them.
nnc_status nnc_model_load_avx(const char* path, const nnc_model_params* params, nnc_model** model);
void nnc_set_num_threads_avx(int n_threads);
int nnc_num_threads_avx();
void nnc_perf_enable_avx(bool enabled);
void nnc_perf_reset_avx();
const char* nnc_perf_report_avx(const char* title, int32_t n_tokens);

// --- dtypes -------------------------------------------------------------

using nnc_bf16_t = uint16_t;

enum nnc_type
{
	NNC_TYPE_F32 = 0,
	NNC_TYPE_F16 = 1,
	NNC_TYPE_I32 = 2,
	NNC_TYPE_BF16 = 3,
	// Q8_0 (split layout): row r of an [rows x cols] weight matrix is
	// stored as `int8 qs[rows*cols]` followed by `bf16 scales[rows*cols/32]`
	// in the same allocation. `tensor->data` points at `qs`; the kernel
	// computes the scales pointer from rows*cols. Block size is 32; the
	// quantizer fills cols-multiple-of-32 only.
	NNC_TYPE_Q8_0 = 4,
	// Q4 (split layout): `uint8 qs[rows*cols/2]` packed nibbles, then
	// `bf16 scales[rows*cols/32]`, then `bf16 biases[rows*cols/32]`, all
	// in one allocation. Reconstruction is w = scale*q + bias with q an
	// unsigned nibble in [0, 15]. Block size is 32; byte i of a block holds
	// element i in its low nibble and element i+16 in its high nibble.
	NNC_TYPE_Q4_S = 5,
	NNC_TYPE_COUNT,
};

// --- bf16 helpers (IEEE 754 truncation: u32(f32) >> 16) ---
static inline float nnc_bf16_to_f32(const nnc_bf16_t v)
{
	const uint32_t u = static_cast<uint32_t>(v) << 16;
	float f;
	std::memcpy(&f, &u, 4);
	return f;
}

static inline nnc_bf16_t nnc_f32_to_bf16(const float f)
{
	uint32_t u;
	std::memcpy(&u, &f, 4);
	// round-to-nearest-even
	const uint32_t rounding_bias = 0x7fff + ((u >> 16) & 1);
	return static_cast<nnc_bf16_t>((u + rounding_bias) >> 16);
}

// AVX2 batched bf16 -> f32 conversion (tail handled scalarly).
void nnc_bf16_to_f32_row(const nnc_bf16_t* src, float* dst, size_t n);

// A weight view. `data` points into the GGUF mmap or into a loader-owned
// decode buffer; the descriptor itself is owned by the model.
struct nnc_tensor
{
	nnc_type type;
	int n_dims;
	int ne[NNC_MAX_DIMS]; // shape (ne[0] = fastest-varying)
	size_t nb[NNC_MAX_DIMS]; // strides in bytes
	void* data;
};

// --- type metadata ---
// Bytes per storage block (34 for Q8_0's 32 elements, 2 for BF16, ...).
size_t nnc_type_size(nnc_type t);
size_t nnc_nbytes(const struct nnc_tensor* t);
int64_t nnc_nelements(const struct nnc_tensor* t);

// --- timing helpers ---
int64_t nnc_time_us(); // microseconds since the first call
int64_t nnc_time_ns(); // monotonic nanoseconds

// --- phase profiler -----------------------------------------------------
//
// Fixed-slot, allocation-free wall-clock accumulator. Off by default;
// nnc_perf_enable() (the app's `--perf`) turns it on. Scopes must not overlap
// within a slot; nesting different slots is fine but the outer slot's time
// then includes the inner one. Process-wide and unsynchronised, so it is
// meant for one session at a time. The public half (enable / reset / report)
// is declared in nnc.h.
enum nnc_perf_slot
{
	NNC_PERF_EMBED = 0, // token embedding row lookup
	NNC_PERF_PLE_PREP, // per-layer-input embedding table + projection
	NNC_PERF_NORM, // RMSNorm + gamma (all sites)
	NNC_PERF_ATTN_QKV, // Q/K/V projections
	NNC_PERF_ATTN_ROPE, // per-head Q/K norms + RoPE
	NNC_PERF_ATTN_CORE, // Q.K scores + fused softmax*V
	NNC_PERF_ATTN_OUT, // attention output projection
	NNC_PERF_DN_PROJ, // Gated DeltaNet input projections (q/k/v, output gate, decay, write strength)
	NNC_PERF_DN_CORE, // Gated DeltaNet convolution, delta-rule recurrence and gated norm
	NNC_PERF_DN_OUT, // Gated DeltaNet output projection
	NNC_PERF_FFN_GATE_UP, // gate + up projections
	NNC_PERF_FFN_ACT, // gelu(gate)*up / swiglu
	NNC_PERF_FFN_DOWN, // down projection
	NNC_PERF_MOE_ROUTE, // mixture-of-experts router + top-k
	NNC_PERF_MOE_EXPERTS, // the chosen experts' gated MLPs
	NNC_PERF_PLE_LAYER, // per-layer inp_gate + proj
	NNC_PERF_LM_HEAD, // final logits projection (+ argmax)
	NNC_PERF_DRAFT, // MTP draft steps, whole (the drafter's own phases are not split out)
	NNC_PERF_COUNT,
};

extern bool g_nnc_perf_on;

inline bool nnc_perf_enabled() { return g_nnc_perf_on; }
void nnc_perf_add(nnc_perf_slot slot, int64_t ns, uint64_t bytes);

// RAII scope. `bytes` is the weight/activation traffic attributed to the
// scope, used to derive an effective bandwidth in the report.
struct nnc_perf_scope
{
	explicit nnc_perf_scope(const nnc_perf_slot s, const uint64_t bytes = 0)
		: slot_(s), bytes_(bytes), t0_(g_nnc_perf_on ? nnc_time_ns() : 0)
	{
	}

	~nnc_perf_scope()
	{
		if (g_nnc_perf_on) nnc_perf_add(slot_, nnc_time_ns() - t0_, bytes_);
	}

	nnc_perf_scope(const nnc_perf_scope&) = delete;
	nnc_perf_scope& operator=(const nnc_perf_scope&) = delete;

private:
	nnc_perf_slot slot_;
	uint64_t bytes_;
	int64_t t0_;
};