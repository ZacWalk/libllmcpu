// nnc — the JIT kernel cache.
// Owns the (op, dtype, shape)-keyed cache that JITs each kernel once and
// hands out reusable typed function pointers. CPU detection lives in
// baseline.cpp.

#include "jit_kernel.h"

#include "jit_buffer.h"
#include "jit_ops.h"
#include "runtime.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

// ---- Kernel cache ------------------------------------------------------

struct jit_kernel_cache::impl
{
	struct entry
	{
		std::unique_ptr<jit_buffer> buf;
		void* fn = nullptr;
	};

	// Key packs (rows, cols) into a single 64-bit value.
	std::unordered_map<uint64_t, entry> gemv_;
	std::unordered_map<uint64_t, entry> gemv_bf16w_f32x_;
	std::unordered_map<uint32_t, entry> gemv_q8_0_1row_;
	std::unordered_map<uint32_t, entry> gemv_q4_s_1row_;
	std::unordered_map<uint32_t, entry> gemv_q8_0_3tok_;
	std::unordered_map<uint32_t, entry> gemv_q4_s_3tok_;
	// shared_mutex: hot-path lookups take a shared lock (concurrent
	// readers OK), and only first-time codegen takes the exclusive lock.
	// After first-token warmup every gemv hits the shared path. The maps
	// are append-only; once a slot is inserted its `fn` pointer never
	// changes, so reading it under a shared lock is safe.
	mutable std::shared_mutex mu_;
};

jit_kernel_cache::jit_kernel_cache() : p_(std::make_unique<impl>())
{
}

jit_kernel_cache::~jit_kernel_cache() = default;

static inline uint64_t pack(const uint32_t a, const uint32_t b)
{
	return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

// Generic shared/exclusive lookup helper. The shared (read) path is the
// steady-state hit; the exclusive (write) path runs once per shape on
// first touch. We re-check the map under the exclusive lock so two
// threads racing the same key only build the kernel once.
template <typename Map, typename Key, typename Build>
static void* lookup_or_build(std::shared_mutex& mu, Map& m, const Key& key, Build build)
{
	{
		std::shared_lock<std::shared_mutex> rk(mu);
		const auto it = m.find(key);
		if (it != m.end()) return it->second.fn;
	}
	auto buf = std::make_unique<jit_buffer>();
	build(*buf);
	void* fn = buf->commit();
	// Out of executable memory: there is no sensible way to carry on with
	// a gemv that has no kernel, so stop loudly rather than crash later.
	NNC_ASSERT(fn != nullptr && "JIT commit failed: the OS refused executable memory");
	std::unique_lock<std::shared_mutex> wk(mu);
	auto& slot = m[key];
	if (slot.fn != nullptr)
	{
		// Lost the race: another thread inserted first. Drop our buffer.
		return slot.fn;
	}
	slot.buf = std::move(buf);
	slot.fn = fn;
	return fn;
}

nnc_gemv_f32_fn jit_kernel_cache::get_gemv_f32(const uint32_t rows, const uint32_t cols)
{
	assert(rows > 0);
	assert(cols > 0 && (cols % 8) == 0);
	const uint64_t key = pack(rows, cols);
	return reinterpret_cast<nnc_gemv_f32_fn>(
		lookup_or_build(p_->mu_, p_->gemv_, key,
		                [&](jit_buffer& b) { nnc_build_gemv_f32(b, rows, cols); }));
}

size_t jit_kernel_cache::size() const
{
	std::shared_lock<std::shared_mutex> lk(p_->mu_);
	return p_->gemv_.size() + p_->gemv_bf16w_f32x_.size()
		+ p_->gemv_q8_0_1row_.size() + p_->gemv_q4_s_1row_.size()
		+ p_->gemv_q8_0_3tok_.size() + p_->gemv_q4_s_3tok_.size();
}

nnc_gemv_bf16w_f32x_fn jit_kernel_cache::get_gemv_bf16w_f32x(const uint32_t rows, const uint32_t cols)
{
	assert(rows > 0);
	assert(cols > 0 && (cols % 8) == 0);
	const uint64_t key = pack(rows, cols);
	return reinterpret_cast<nnc_gemv_bf16w_f32x_fn>(
		lookup_or_build(p_->mu_, p_->gemv_bf16w_f32x_, key,
		                [&](jit_buffer& b)
		                {
			                // Prefer the 4-rows-at-a-time builder when
			                // rows is a multiple of 4: it reuses each
			                // x-tile across 4 row accumulators, cutting
			                // x-side bandwidth by ~4x.
			                if ((rows % 4) == 0)
				                nnc_build_gemv_bf16w_f32x_4row(b, rows, cols);
			                else
				                nnc_build_gemv_bf16w_f32x(b, rows, cols);
		                }));
}

nnc_gemv_q8_0_1row_fn jit_kernel_cache::get_gemv_q8_0_1row(const uint32_t cols)
{
	assert(cols > 0 && (cols % 32) == 0);
	return reinterpret_cast<nnc_gemv_q8_0_1row_fn>(
		lookup_or_build(p_->mu_, p_->gemv_q8_0_1row_, cols,
		                [&](jit_buffer& b) { nnc_build_gemv_q8_0_f32x_1row(b, cols); }));
}

nnc_gemv_q4_s_1row_fn jit_kernel_cache::get_gemv_q4_s_1row(const uint32_t cols)
{
	assert(cols > 0 && (cols % 32) == 0);
	return reinterpret_cast<nnc_gemv_q4_s_1row_fn>(
		lookup_or_build(p_->mu_, p_->gemv_q4_s_1row_, cols,
		                [&](jit_buffer& b) { nnc_build_gemv_q4_s_f32x_1row(b, cols); }));
}

nnc_gemv_3tok_fn jit_kernel_cache::get_gemv_q8_0_3tok(const uint32_t cols)
{
	assert(cols > 0 && (cols % 32) == 0);
	return reinterpret_cast<nnc_gemv_3tok_fn>(
		lookup_or_build(p_->mu_, p_->gemv_q8_0_3tok_, cols,
		                [&](jit_buffer& b) { nnc_build_gemv_q8_0_f32x_3tok(b, cols); }));
}

nnc_gemv_3tok_fn jit_kernel_cache::get_gemv_q4_s_3tok(const uint32_t cols)
{
	assert(cols > 0 && (cols % 32) == 0);
	return reinterpret_cast<nnc_gemv_3tok_fn>(
		lookup_or_build(p_->mu_, p_->gemv_q4_s_3tok_, cols,
		                [&](jit_buffer& b) { nnc_build_gemv_q4_s_f32x_3tok(b, cols); }));
}
