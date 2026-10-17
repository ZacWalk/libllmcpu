// nnc — single test translation unit.
// All tests (app + jit) live here. Run with `nnc --test` (also -test, /test).
// Each test returns true on success. Exit code = number of failures.

#include "jit_buffer.h"
#include "jit_kernel.h"
#include "jit_ops.h"
#include "nn_ops.h"
#include "emitter_x64.h"
#include "emitter_avx2.h"
#include "gguf.h"
#include "platform.h"
#include "runtime.h"
#include "session.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace
{
	int g_pass = 0;
	int g_fail = 0;

	void report(const char* name, const bool ok, const char* msg = nullptr)
	{
		if (ok)
		{
			++g_pass;
			std::printf("[PASS] %s\n", name);
		}
		else
		{
			++g_fail;
			std::printf("[FAIL] %s%s%s\n", name, msg ? " : " : "", msg ? msg : "");
		}
	}

	bool close(const float a, const float b, const float rel_tol)
	{
		const float diff = std::fabs(a - b);
		const float scale = std::fmax(std::fabs(a), std::fabs(b));
		return diff <= rel_tol * std::fmax(scale, 1.0f);
	}

	// ---- CPU --------------------------------------------------------------

	bool test_cpu_has_avx2_fma()
	{
		const auto& f = nnc_cpu_features();
		return f.avx2 && f.fma;
	}

	bool test_nnc_api_defaults()
	{
		const nnc_model_params model = nnc_model_default_params();
		const nnc_session_params session = nnc_session_default_params();
		const nnc_sampler_params sampler = nnc_sampler_default_params();
		const nnc_generate_params generate = nnc_generate_default_params();
		return model.weights == NNC_WEIGHTS_Q8_0 && model.progress == nullptr
			&& session.n_ctx == 0 && session.seed == 0 && session.system_prompt == nullptr
			&& session.kv_type == NNC_KV_BF16 && !session.thinking && generate.on_thinking == nullptr
			&& sampler.temperature == 0.7f && sampler.top_k == 40
			&& sampler.top_p == 0.95f && sampler.min_p == 0.05f
			&& generate.max_tokens == 0 && generate.on_text == nullptr
			&& generate.sampler.temperature == sampler.temperature;
	}

	bool test_nnc_api_rejects_invalid_handles()
	{
		nnc_model* model = nullptr;
		nnc_session* session = nullptr;
		const nnc_status load = nnc_model_load(nullptr, nullptr, &model);
		const nnc_status create = nnc_session_create(nullptr, nullptr, &session);
		const int32_t token_count = nnc_tokenize(nullptr, nullptr, 0, false, nullptr, 0);
		return load == NNC_ERR_INVALID_ARGUMENT && model == nullptr
			&& create == NNC_ERR_INVALID_ARGUMENT && session == nullptr
			&& token_count == -1 && nnc_last_error()[0] != '\0';
	}

	// nnc_last_error() has a fixed size; a message cut to fit must not end
	// part-way through a UTF-8 character.
	bool test_nnc_last_error_cuts_between_characters()
	{
		std::string path = "x"; // an odd byte count, so the cut lands inside an é
		for (int i = 0; i < 1500; ++i) path += "\xC3\xA9";
		nnc_model* model = nullptr;
		if (nnc_model_load(path.c_str(), nullptr, &model) == NNC_OK) return false;
		const std::string message = nnc_last_error();
		std::string pending = message, checked;
		utf8_take(pending, checked, true);
		return message.size() > 1000 && message.size() < 2048 && checked == message;
	}

	// Streamed text is valid UTF-8 in whole characters: a character split
	// across tokens waits for the rest of it, and bytes that cannot form one
	// become U+FFFD, one per maximal invalid subpart.
	bool test_utf8_take()
	{
		const std::string fffd = "\xEF\xBF\xBD";
		std::string rest;
		auto take = [&rest](std::string pending, const bool final)
		{
			std::string out;
			utf8_take(pending, out, final);
			rest = pending;
			return out;
		};
		if (take("abc", false) != "abc" || !rest.empty()) return false;
		// 中 (E4 B8 AD) in two pieces, and never finished.
		if (take("a\xE4\xB8", false) != "a" || rest != "\xE4\xB8") return false;
		if (take(rest + "\xAD", false) != "\xE4\xB8\xAD" || !rest.empty()) return false;
		if (take("\xE4\xB8", true) != fffd || !rest.empty()) return false;
		// 😀 (F0 9F 98 80) after its first byte.
		if (!take("\xF0", false).empty() || rest != "\xF0") return false;
		if (take(rest + "\x9F\x98\x80", false) != "\xF0\x9F\x98\x80" || !rest.empty()) return false;
		// A broken character is replaced as soon as the break shows.
		if (take("\xE4\xB8" "a", false) != fffd + "a" || !rest.empty()) return false;
		// Stray continuation and lead bytes; overlong, surrogate, past U+10FFFF.
		if (take("\x80" "b\xFF", false) != fffd + "b" + fffd) return false;
		if (take("\xC0\xAF", false) != fffd + fffd) return false;
		if (take("\xED\xA0\x80", false) != fffd + fffd + fffd) return false;
		if (take("\xF4\x90\x80\x80", false) != fffd + fffd + fffd + fffd) return false;
		return true;
	}

	// ---- JIT plumbing -----------------------------------------------------

	bool test_jit_return_42()
	{
		jit_buffer buf;
		x64_emitter e(buf);
		e.mov_r32_imm32(gpr::rax, 42);
		e.ret();
		using fn_t = int (*)();
		const auto fn = reinterpret_cast<fn_t>(buf.commit());
		return fn() == 42;
	}

	bool test_jit_add_two_ints()
	{
		jit_buffer buf;
		x64_emitter e(buf);
		e.emit_win64_arg_shuffle(2); // (a, b)
		e.lea_r32_base_index(gpr::rax, gpr::rcx, gpr::rdx);
		e.ret();
		using fn_t = int (*)(int, int);
		const auto fn = reinterpret_cast<fn_t>(buf.commit());
		return fn(2, 3) == 5 && fn(-1, 1) == 0 && fn(1000, 2345) == 3345;
	}

	// ---- dot_f32 ----------------------------------------------------------

	float scalar_dot(const float* a, const float* b, const size_t n)
	{
		double s = 0.0;
		for (size_t i = 0; i < n; ++i) s += static_cast<double>(a[i]) * b[i];
		return static_cast<float>(s);
	}

	bool test_jit_dot_f32()
	{
		jit_buffer buf;
		nnc_build_dot_f32(buf);
		const auto fn = reinterpret_cast<nnc_dot_f32_fn>(buf.commit());

		std::mt19937 rng(0xC0FFEE);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

		constexpr size_t sizes[] = {8, 64, 768, 4096};
		for (const size_t n : sizes)
		{
			std::vector<float> a(n), b(n);
			for (size_t i = 0; i < n; ++i)
			{
				a[i] = dist(rng);
				b[i] = dist(rng);
			}
			const float got = fn(a.data(), b.data(), n);
			const float ref = scalar_dot(a.data(), b.data(), n);
			if (!close(got, ref, 1e-4f))
			{
				std::printf("    n=%zu  jit=%g  ref=%g\n", n, got, ref);
				return false;
			}
		}
		return true;
	}

	// ---- gemv_f32 ---------------------------------------------------------

	void scalar_gemv(const float* W, const float* x, float* y, const uint32_t rows, const uint32_t cols)
	{
		for (uint32_t r = 0; r < rows; ++r)
		{
			double s = 0.0;
			for (uint32_t k = 0; k < cols; ++k) s += static_cast<double>(W[r * cols + k]) * x[k];
			y[r] = static_cast<float>(s);
		}
	}

	bool gemv_one(jit_kernel_cache& cache, const uint32_t rows, const uint32_t cols, std::mt19937& rng)
	{
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		std::vector<float> W(static_cast<size_t>(rows) * cols);
		std::vector<float> x(cols);
		std::vector<float> y_jit(rows), y_ref(rows);

		for (auto& v : W) v = dist(rng);
		for (auto& v : x) v = dist(rng);

		const auto fn = cache.get_gemv_f32(rows, cols);
		fn(W.data(), x.data(), y_jit.data());
		scalar_gemv(W.data(), x.data(), y_ref.data(), rows, cols);

		for (uint32_t r = 0; r < rows; ++r)
		{
			if (!close(y_jit[r], y_ref[r], 1e-4f))
			{
				std::printf("    gemv rows=%u cols=%u r=%u jit=%g ref=%g\n",
				            rows, cols, r, y_jit[r], y_ref[r]);
				return false;
			}
		}
		return true;
	}

	bool test_jit_gemv_f32()
	{
		jit_kernel_cache cache;
		std::mt19937 rng(0xBEEF);

		struct shape
		{
			uint32_t rows, cols;
		};
		constexpr shape shapes[] = {
			{1, 8},
			{4, 64},
			{16, 128},
			{32, 768},
		};
		for (const auto s : shapes)
		{
			if (!gemv_one(cache, s.rows, s.cols, rng)) return false;
		}
		return true;
	}

	bool test_jit_gemv_cache_reuse()
	{
		jit_kernel_cache cache;
		const auto a = cache.get_gemv_f32(16, 64);
		const auto b = cache.get_gemv_f32(16, 64);
		const auto c = cache.get_gemv_f32(16, 128); // different cols
		const bool ok = (a == b) && (a != c) && (cache.size() == 2);
		if (!ok)
		{
			std::printf("    cache: a=%p b=%p c=%p size=%zu\n", a, b, c, cache.size());
		}
		return ok;
	}

	// ---- nn_ops: gelu (Stage A) -------------------------------------------

	float ref_gelu_scalar(const float v)
	{
		constexpr float SQRT_2_OVER_PI = 0.79788456080286535f;
		constexpr float A = 0.044715f;
		const float u = SQRT_2_OVER_PI * (v + A * v * v * v);
		return 0.5f * v * (1.0f + std::tanh(u));
	}

	bool test_nnc_gelu_f32()
	{
		std::mt19937 rng(0xDEAD);
		std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
		constexpr size_t n = 1024;
		std::vector<float> x(n), y(n);
		for (auto& v : x) v = dist(rng);

		nnc_gelu_f32(y.data(), x.data(), n);

		for (size_t i = 0; i < n; ++i)
		{
			const float ref = ref_gelu_scalar(x[i]);
			if (!close(y[i], ref, 5e-4f))
			{
				std::printf("    gelu i=%zu x=%g y=%g ref=%g\n", i, x[i], y[i], ref);
				return false;
			}
		}

		// Spot-check known values.
		constexpr float spot[] = {0.0f, 1.0f, -1.0f, 3.0f, -3.0f};
		float out[5];
		nnc_gelu_f32(out, spot, 5);
		if (!close(out[0], 0.0f, 1e-6f)) return false;
		if (!close(out[1], 0.841192f, 1e-4f)) return false; // GELU(1)  ~ 0.8412
		if (!close(out[2], -0.158808f, 1e-4f)) return false; // GELU(-1) ~ -0.1588
		return true;
	}

	// ---- nn_ops: softmax + layernorm (Stage C) -----------------------------

	bool test_nnc_softmax_f32()
	{
		// Known-good: softmax([1,2,3,4]) -> exp normalized.
		float p[4] = {1.0f, 2.0f, 3.0f, 4.0f};
		nnc_softmax_f32_inplace(p, 4);

		float e[4];
		double s = 0;
		for (int i = 0; i < 4; ++i)
		{
			e[i] = std::exp(static_cast<float>(i + 1) - 4.0f);
			s += e[i];
		}
		for (int i = 0; i < 4; ++i)
		{
			const float ref = static_cast<float>(e[i] / s);
			if (!close(p[i], ref, 1e-6f))
			{
				std::printf("    softmax i=%d got=%g ref=%g\n", i, p[i], ref);
				return false;
			}
		}

		// -INFINITY masks: those entries must become 0 and the rest sum to 1.
		constexpr float ninf = -std::numeric_limits<float>::infinity();
		float q[5] = {0.5f, ninf, 1.5f, ninf, 2.5f};
		nnc_softmax_f32_inplace(q, 5);
		if (q[1] != 0.0f || q[3] != 0.0f) return false;
		const float total = q[0] + q[1] + q[2] + q[3] + q[4];
		if (!close(total, 1.0f, 1e-6f)) return false;
		// Strictly increasing for the unmasked entries.
		if (!(q[0] < q[2] && q[2] < q[4])) return false;

		// Same, but long enough to exercise the AVX2 path (n >= 8).
		{
			float w[16];
			for (int i = 0; i < 16; ++i) w[i] = (i % 3 == 0) ? ninf : static_cast<float>(i);
			nnc_softmax_f32_inplace(w, 16);
			double wsum = 0;
			for (int i = 0; i < 16; ++i)
			{
				if (i % 3 == 0 && w[i] != 0.0f) return false;
				wsum += w[i];
			}
			if (std::fabs(wsum - 1.0) > 1e-6) return false;
		}

		// Random vector: result sums to 1, all >= 0.
		std::mt19937 rng(0xC0FFEE);
		std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
		constexpr size_t n = 1024;
		std::vector<float> v(n);
		for (auto& x : v) x = dist(rng);
		nnc_softmax_f32_inplace(v.data(), n);
		double sum = 0;
		for (const auto x : v)
		{
			if (x < 0.0f) return false;
			sum += x;
		}
		if (std::fabs(sum - 1.0) > 1e-5)
		{
			std::printf("    softmax sum=%g\n", sum);
			return false;
		}
		return true;
	}

	bool test_nnc_layernorm_f32()
	{
		std::mt19937 rng(0xBEEF);
		std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
		constexpr size_t n = 768;
		std::vector<float> x(n), y(n);
		for (auto& v : x) v = dist(rng);

		nnc_layernorm_f32(y.data(), x.data(), n, 1e-5f);

		double mean = 0, var = 0;
		for (const auto v : y) mean += v;
		mean /= n;
		for (const auto v : y) var += (v - mean) * (v - mean);
		var /= n;
		if (std::fabs(mean) > 1e-5)
		{
			std::printf("    norm mean=%g\n", mean);
			return false;
		}
		if (std::fabs(var - 1.0) > 1e-3)
		{
			std::printf("    norm var=%g\n", var);
			return false;
		}

		// Aliasing: y == x must work (in-place).
		std::vector<float> z = x;
		nnc_layernorm_f32(z.data(), z.data(), n, 1e-5f);
		for (size_t i = 0; i < n; ++i)
		{
			if (!close(z[i], y[i], 1e-5f))
			{
				std::printf("    norm alias i=%zu z=%g y=%g\n", i, z[i], y[i]);
				return false;
			}
		}
		return true;
	}

	bool test_nnc_rmsnorm_f32()
	{
		std::mt19937 rng(0xCAFE);
		std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
		// Cover Gemma E2B/E4B hidden sizes plus a small odd size for the tail.
		constexpr size_t sizes[] = {1536, 2560, 256, 13};
		for (const size_t n : sizes)
		{
			std::vector<float> x(n), y(n);
			for (auto& v : x) v = dist(rng);

			nnc_rmsnorm_f32(y.data(), x.data(), n, 1e-6f);

			// Scalar reference.
			double s2 = 0.0;
			for (const auto v : x) s2 += static_cast<double>(v) * v;
			const float scale = static_cast<float>(1.0 / std::sqrt(s2 / static_cast<double>(n) + 1e-6));
			for (size_t i = 0; i < n; ++i)
			{
				const float ref = x[i] * scale;
				if (!close(y[i], ref, 1e-5f))
				{
					std::printf("    rmsnorm n=%zu i=%zu y=%g ref=%g\n", n, i, y[i], ref);
					return false;
				}
			}

			// rms(y) should be ~1.
			double sy2 = 0.0;
			for (const auto v : y) sy2 += static_cast<double>(v) * v;
			const double rms_y = std::sqrt(sy2 / static_cast<double>(n));
			if (std::fabs(rms_y - 1.0) > 1e-3)
			{
				std::printf("    rmsnorm n=%zu rms(y)=%g\n", n, rms_y);
				return false;
			}

			// Aliasing.
			std::vector<float> z = x;
			nnc_rmsnorm_f32(z.data(), z.data(), n, 1e-6f);
			for (size_t i = 0; i < n; ++i)
				if (!close(z[i], y[i], 1e-5f)) return false;
		}
		return true;
	}

	// ---- BF16 -------------------------------------------------------------

	bool test_bf16_round_trip()
	{
		// Round-trip a handful of representative values; values with no
		// fractional bits beyond bit 16 must round-trip exactly.
		constexpr float exact[] = {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.5f, 1024.0f, -2048.0f};
		for (const float v : exact)
		{
			const float r = nnc_bf16_to_f32(nnc_f32_to_bf16(v));
			if (r != v) return false;
		}
		// Generic values: |relerr| <= 2^-7 (BF16 has 7 mantissa bits + 1 hidden).
		std::mt19937 rng(0xBF16);
		std::uniform_real_distribution<float> d(-3.0f, 3.0f);
		for (int i = 0; i < 1024; ++i)
		{
			const float v = d(rng);
			const float r = nnc_bf16_to_f32(nnc_f32_to_bf16(v));
			const float err = std::fabs(r - v);
			const float scale = std::fmax(std::fabs(v), 1.0f);
			if (err / scale > (1.0f / 128.0f)) return false;
		}
		return true;
	}

	bool test_bf16_to_f32_row()
	{
		// Compare AVX2 batched conversion to the scalar reference.
		std::mt19937 rng(0xBF16ABCD);
		std::uniform_real_distribution<float> d(-100.0f, 100.0f);
		constexpr size_t N = 137; // not a multiple of 8 -> exercise tail
		std::vector<nnc_bf16_t> src(N);
		for (size_t i = 0; i < N; ++i)
			src[i] = nnc_f32_to_bf16(d(rng));
		std::vector<float> got(N), ref(N);
		nnc_bf16_to_f32_row(src.data(), got.data(), N);
		for (size_t i = 0; i < N; ++i)
			ref[i] = nnc_bf16_to_f32(src[i]);
		for (size_t i = 0; i < N; ++i)
			if (got[i] != ref[i]) return false;
		return true;
	}

	// ---- nn_ops: gemv_bf16w_f32x ------------------------------------------

	bool test_nnc_gemv_bf16w_f32x()
	{
		std::mt19937 rng(0xB16E);
		std::uniform_real_distribution<float> d(-1.0f, 1.0f);
		struct sh
		{
			uint32_t rows, cols;
		};
		// E2B/E4B Gemma shapes plus odd small ones for the tail path.
		const sh shapes[] = {
			{1, 8}, {7, 5}, {16, 1536},
			{1536, 1536}, {2048, 1536}, {6144, 1536},
		};
		for (const auto s : shapes)
		{
			std::vector<nnc_bf16_t> W(static_cast<size_t>(s.rows) * s.cols);
			std::vector<float> Wf32(W.size());
			for (size_t i = 0; i < W.size(); ++i)
			{
				const float v = d(rng);
				W[i] = nnc_f32_to_bf16(v);
				Wf32[i] = nnc_bf16_to_f32(W[i]);
			}
			std::vector<float> x(s.cols), y(s.rows), ref(s.rows);
			for (auto& v : x) v = d(rng);

			nnc_gemv_bf16w_f32x(W.data(), x.data(), y.data(), s.rows, s.cols);

			for (uint32_t r = 0; r < s.rows; ++r)
			{
				double acc = 0.0;
				const float* row = Wf32.data() + static_cast<size_t>(r) * s.cols;
				for (uint32_t k = 0; k < s.cols; ++k)
					acc += static_cast<double>(row[k]) * x[k];
				ref[r] = static_cast<float>(acc);
			}
			// FMA reorders rounding vs scalar; allow modest relative error.
			for (uint32_t r = 0; r < s.rows; ++r)
			{
				if (!close(y[r], ref[r], 1e-3f))
				{
					std::printf("    gemv_bf16w r=%u y=%g ref=%g (rows=%u cols=%u)\n",
					            r, y[r], ref[r], s.rows, s.cols);
					return false;
				}
			}
		}
		return true;
	}

	// ---- nn_ops: gemv_bf16w_argmax_f32x -----------------------------------

	bool test_nnc_gemv_bf16w_argmax_f32x()
	{
		std::mt19937 rng(0xA12A);
		std::uniform_real_distribution<float> d(-1.0f, 1.0f);
		struct sh
		{
			uint32_t rows, cols;
		};
		// Cover the fast path (rows%4==0, cols%8==0) plus the fallback.
		const sh shapes[] = {
			{4, 32}, {16, 256}, {2048, 256}, {256, 1536}, {7, 5},
		};
		for (const auto s : shapes)
		{
			std::vector<nnc_bf16_t> W(static_cast<size_t>(s.rows) * s.cols);
			std::vector<float> Wf32(W.size());
			for (size_t i = 0; i < W.size(); ++i)
			{
				const float v = d(rng);
				W[i] = nnc_f32_to_bf16(v);
				Wf32[i] = nnc_bf16_to_f32(W[i]);
			}
			std::vector<float> x(s.cols);
			for (auto& v : x) v = d(rng);

			// Reference: full materialise then linear scan, BF16-accurate.
			std::vector<float> ref(s.rows);
			for (uint32_t r = 0; r < s.rows; ++r)
			{
				double acc = 0.0;
				const float* row = Wf32.data() + static_cast<size_t>(r) * s.cols;
				for (uint32_t k = 0; k < s.cols; ++k)
					acc += static_cast<double>(row[k]) * x[k];
				ref[r] = static_cast<float>(acc);
			}
			int ref_best = 0;
			for (uint32_t r = 1; r < s.rows; ++r)
				if (ref[r] > ref[ref_best]) ref_best = static_cast<int>(r);

			const int got = nnc_gemv_bf16w_argmax_f32x(W.data(), x.data(),
			                                           s.rows, s.cols);
			// FMA reorder may flip ties; if got != ref_best require their
			// values to be within tolerance.
			if (got != ref_best)
			{
				const float diff = std::fabs(ref[got] - ref[ref_best]);
				const float tol = 1e-3f * std::max(1.0f, std::fabs(ref[ref_best]));
				if (diff > tol)
				{
					std::printf("    argmax got=%d ref=%d ref_v=%g got_v=%g (rows=%u cols=%u)\n",
					            got, ref_best, ref[ref_best], ref[got],
					            s.rows, s.cols);
					return false;
				}
			}
		}
		return true;
	}

	// ---- nn_ops: q8_0 quantize roundtrip + gemv ---------------------------

	bool test_nnc_q8_0_quantize_roundtrip()
	{
		// Quantize a synthetic BF16 row and verify the recovered (scale*q)
		// values are within absmax/127 of the original — i.e. the worst
		// per-element error is bounded by the per-block step size.
		std::mt19937 rng(0xC0FFEE);
		std::uniform_real_distribution<float> d(-2.5f, 2.5f);
		constexpr size_t rows = 3, cols = 64;
		std::vector<nnc_bf16_t> W(rows * cols);
		std::vector<float> Wf(rows * cols);
		for (size_t i = 0; i < W.size(); ++i)
		{
			Wf[i] = d(rng);
			W[i] = nnc_f32_to_bf16(Wf[i]);
			Wf[i] = nnc_bf16_to_f32(W[i]); // re-quantize for fair compare
		}
		std::vector<int8_t> qs(rows * cols);
		std::vector<uint16_t> scales(rows * (cols / 32));
		nnc_quantize_bf16_to_q8_0(W.data(), qs.data(), scales.data(), rows, cols);

		for (size_t r = 0; r < rows; ++r)
		{
			for (size_t b = 0; b < cols / 32; ++b)
			{
				const float scale = nnc_bf16_to_f32(scales[r * (cols / 32) + b]);
				// Step size = scale; allow one step of error.
				for (size_t k = 0; k < 32; ++k)
				{
					const float got = scale * static_cast<float>(qs[r * cols + b * 32 + k]);
					const float ref = Wf[r * cols + b * 32 + k];
					if (std::fabs(got - ref) > scale + 1e-6f)
					{
						std::printf("    q8_0 roundtrip r=%zu b=%zu k=%zu got=%g ref=%g scale=%g\n",
						            r, b, k, got, ref, scale);
						return false;
					}
				}
			}
		}
		return true;
	}

	bool test_nnc_gemv_q8_0_f32x()
	{
		// Quantize a BF16 weight matrix, run JIT Q8_0 gemv, and compare
		// to a scalar reference computed from the SAME quantized weights
		// (so the only error source is FMA reorder vs scalar order).
		std::mt19937 rng(0xC8E2);
		std::uniform_real_distribution<float> d(-1.0f, 1.0f);
		struct sh
		{
			uint32_t rows, cols;
		};
		// cover small (single-thread) and parallel paths.
		const sh shapes[] = {
			{1, 32}, {8, 64}, {64, 128},
			{256, 256}, {512, 512}, {2048, 2048},
		};
		for (const auto s : shapes)
		{
			std::vector<nnc_bf16_t> W(static_cast<size_t>(s.rows) * s.cols);
			for (auto& w : W) w = nnc_f32_to_bf16(d(rng));

			std::vector<int8_t> qs(W.size());
			std::vector<uint16_t> scales(static_cast<size_t>(s.rows) * (s.cols / 32));
			nnc_quantize_bf16_to_q8_0(W.data(), qs.data(), scales.data(),
			                          s.rows, s.cols);

			std::vector<float> x(s.cols), y(s.rows), ref(s.rows);
			for (auto& v : x) v = d(rng);

			nnc_gemv_q8_0_f32x(qs.data(), scales.data(), x.data(), y.data(),
			                   s.rows, s.cols);

			// Reference: dequantize then dot, in double precision.
			const size_t scale_stride = s.cols / 32;
			for (uint32_t r = 0; r < s.rows; ++r)
			{
				double acc = 0.0;
				for (uint32_t b = 0; b < scale_stride; ++b)
				{
					const float scale = nnc_bf16_to_f32(scales[r * scale_stride + b]);
					for (uint32_t k = 0; k < 32; ++k)
					{
						const float w = scale * qs[r * s.cols + b * 32 + k];
						acc += static_cast<double>(w) * x[b * 32 + k];
					}
				}
				ref[r] = static_cast<float>(acc);
			}
			for (uint32_t r = 0; r < s.rows; ++r)
			{
				const float tol = 1e-3f * std::max(1.0f, std::fabs(ref[r]));
				if (std::fabs(y[r] - ref[r]) > tol)
				{
					std::printf("    q8_0 gemv r=%u y=%g ref=%g (rows=%u cols=%u)\n",
					            r, y[r], ref[r], s.rows, s.cols);
					return false;
				}
			}
		}
		return true;
	}

	// ---- nn_ops: Q4 (nnc split 4-bit) -------------------------------------

	bool test_nnc_q4_s_quantize_roundtrip()
	{
		// Case 1: scale and bias exactly representable in BF16 (powers of
		// two), and the values sit exactly on the 16 quantization levels.
		// Round-trip must then be bit-exact, which pins down the nibble
		// packing rather than just the numerics.
		{
			std::vector<float> src(64);
			for (int i = 0; i < 32; ++i) src[i] = -2.0f + static_cast<float>(i % 16) * 0.25f;
			for (int i = 0; i < 32; ++i) src[32 + i] = 4.0f + static_cast<float>(i % 16) * 0.5f;

			std::vector<uint8_t> qs(32);
			std::vector<uint16_t> scales(2), biases(2);
			nnc_quantize_f32_to_q4_s(src.data(), qs.data(), scales.data(), biases.data(), 64);

			const float expect_scale[2] = {0.25f, 0.5f};
			const float expect_bias[2] = {-2.0f, 4.0f};
			for (size_t b = 0; b < 2; ++b)
			{
				const float sc = nnc_bf16_to_f32(scales[b]);
				const float bi = nnc_bf16_to_f32(biases[b]);
				if (sc != expect_scale[b] || bi != expect_bias[b]) return false;
				for (size_t k = 0; k < 16; ++k)
				{
					const uint8_t byte = qs[b * 16 + k];
					// byte i holds element i low, element i+16 high.
					if (sc * (byte & 0x0F) + bi != src[b * 32 + k]) return false;
					if (sc * (byte >> 4) + bi != src[b * 32 + k + 16]) return false;
				}
			}
		}

		// Case 2: arbitrary values. BF16 carries 8 significand bits, so the
		// reconstruction error is bounded by the scale and bias rounding
		// amplified by the largest quant level:
		//   |err| <= 15*scale*2^-8 + |bias|*2^-8 = (range + |bias|) * 2^-8
		{
			std::mt19937 rng(0x4B15);
			std::uniform_real_distribution<float> d(-3.0f, 7.0f);
			std::vector<float> src(32);
			for (auto& v : src) v = d(rng);

			uint8_t qs[16];
			uint16_t sc_h, bi_h;
			nnc_quantize_f32_to_q4_s(src.data(), qs, &sc_h, &bi_h, 32);

			const float sc = nnc_bf16_to_f32(sc_h);
			const float bi = nnc_bf16_to_f32(bi_h);
			// Half a quant step, plus the BF16 rounding of scale and bias.
			const float step = 15.0f * sc;
			const float tol = 0.5f * sc + (step + std::fabs(bi)) * 0.00391f + 1e-6f;
			for (size_t k = 0; k < 16; ++k)
			{
				if (std::fabs(sc * (qs[k] & 0x0F) + bi - src[k]) > tol) return false;
				if (std::fabs(sc * (qs[k] >> 4) + bi - src[k + 16]) > tol) return false;
			}
		}

		// Degenerate block: all values equal -> scale 0, bias = the value.
		std::vector<float> flat(32, 3.5f);
		uint8_t fq[16];
		uint16_t fs, fb;
		nnc_quantize_f32_to_q4_s(flat.data(), fq, &fs, &fb, 32);
		if (nnc_bf16_to_f32(fs) != 0.0f) return false;
		if (nnc_bf16_to_f32(fb) != 3.5f) return false;
		for (const unsigned char v : fq) if (v != 0) return false;
		return true;
	}

	bool test_nnc_block_sums_f32()
	{
		std::vector<float> x(128);
		std::mt19937 rng(0xB10C);
		std::uniform_real_distribution<float> d(-4.0f, 4.0f);
		for (auto& v : x) v = d(rng);

		float out[4];
		nnc_block_sums_f32(x.data(), out, x.size());
		for (int b = 0; b < 4; ++b)
		{
			double ref = 0.0;
			for (int k = 0; k < 32; ++k) ref += x[b * 32 + k];
			if (!close(out[b], static_cast<float>(ref), 1e-5f)) return false;
		}
		return true;
	}

	bool test_nnc_gemv_q4_s_f32x()
	{
		// Compare the JIT 4-bit gemv against a scalar reference computed
		// from the SAME quantized weights, so the only error source is
		// summation order.
		std::mt19937 rng(0x40E5);
		std::uniform_real_distribution<float> d(-1.0f, 1.0f);
		struct sh
		{
			uint32_t rows, cols;
		};
		const sh shapes[] = {
			{1, 32}, {8, 64}, {64, 128},
			{256, 256}, {512, 512}, {1024, 2048},
		};
		for (const auto s : shapes)
		{
			const size_t n = static_cast<size_t>(s.rows) * s.cols;
			const size_t nblocks = s.cols / 32;
			std::vector<float> W(n);
			for (auto& w : W) w = d(rng);

			std::vector<uint8_t> qs(n / 2);
			std::vector<uint16_t> scales(static_cast<size_t>(s.rows) * nblocks);
			std::vector<uint16_t> biases(scales.size());
			// Row by row: the layout is row-major over blocks.
			for (uint32_t r = 0; r < s.rows; ++r)
				nnc_quantize_f32_to_q4_s(W.data() + r * s.cols,
				                         qs.data() + r * (s.cols / 2),
				                         scales.data() + r * nblocks,
				                         biases.data() + r * nblocks,
				                         s.cols);

			std::vector<float> x(s.cols), y(s.rows), xsum(nblocks);
			for (auto& v : x) v = d(rng);
			nnc_block_sums_f32(x.data(), xsum.data(), s.cols);

			nnc_gemv_q4_s_f32x(qs.data(), scales.data(), biases.data(),
			                   x.data(), xsum.data(), y.data(), s.rows, s.cols);

			for (uint32_t r = 0; r < s.rows; ++r)
			{
				double acc = 0.0;
				for (size_t b = 0; b < nblocks; ++b)
				{
					const float sc = nnc_bf16_to_f32(scales[r * nblocks + b]);
					const float bi = nnc_bf16_to_f32(biases[r * nblocks + b]);
					const uint8_t* blk = qs.data() + r * (s.cols / 2) + b * 16;
					for (size_t k = 0; k < 16; ++k)
					{
						acc += static_cast<double>(sc * (blk[k] & 0x0F) + bi) * x[b * 32 + k];
						acc += static_cast<double>(sc * (blk[k] >> 4) + bi) * x[b * 32 + k + 16];
					}
				}
				const float ref = static_cast<float>(acc);
				const float tol = 1e-3f * std::max(1.0f, std::fabs(ref));
				if (std::fabs(y[r] - ref) > tol)
				{
					std::printf("    q4 gemv r=%u y=%g ref=%g (rows=%u cols=%u)\n",
					            r, y[r], ref, s.rows, s.cols);
					return false;
				}
			}
		}
		return true;
	}

	// ---- nn_ops: load-time dequantizers ----------------------------------

	// The register-general encoders against hand-assembled bytes: extended
	// registers in every VEX field, SIB with and without an index, and
	// RSP as a base.
	bool test_emitter_high_registers()
	{
		jit_buffer buf;
		avx2_emitter v(buf);
		x64_emitter e(buf);
		v.g_vfmadd231ps(ymm::y8, ymm::y13, {gpr::r10, gpr::rax, 2, 32});
		v.g_vpslld(ymm::y15, ymm::y13, 28);
		v.g_vmovups_xmm_store({gpr::rsp, gpr::rsp, 0, 16}, ymm::y7);
		v.g_vpmovsxbd(ymm::y13, {gpr::rcx, gpr::r11, 0, 0});
		v.g_vextractf128(ymm::y9, ymm::y8, 1);
		e.mov_r64_load_disp8(gpr::r10, gpr::rdx, 8);
		const uint8_t want[] = {
			0xC4, 0x42, 0x15, 0xB8, 0x44, 0x82, 0x20, // vfmadd231ps ymm8, ymm13, [r10+rax*4+32]
			0xC4, 0xC1, 0x05, 0x72, 0xF5, 0x1C, // vpslld ymm15, ymm13, 28
			0xC4, 0xE1, 0x78, 0x11, 0x7C, 0x24, 0x10, // vmovups [rsp+16], xmm7
			0xC4, 0x22, 0x7D, 0x21, 0x2C, 0x19, // vpmovsxbd ymm13, [rcx+r11]
			0xC4, 0x43, 0x7D, 0x19, 0xC1, 0x01, // vextractf128 xmm9, ymm8, 1
			0x4C, 0x8B, 0x52, 0x08, // mov r10, [rdx+8]
		};
		return buf.size() == sizeof(want) && std::memcmp(buf.staging_data(), want, sizeof(want)) == 0;
	}

	// The batched gemvs (three tokens per kernel call, the rest one at a
	// time) give every token the bits the single-vector gemv gives it.
	bool test_gemv_batch_matches_rows()
	{
		std::mt19937 rng(0xBA7C);
		std::uniform_real_distribution<float> d(-1.0f, 1.0f);
		for (const uint32_t n_batch : {2u, 3u, 7u})
			for (const auto& [rows, cols] : {std::pair{64u, 256u}, std::pair{512u, 2048u}})
			{
				const size_t n = static_cast<size_t>(rows) * cols, nblocks = cols / 32;
				std::vector<float> W(n), X(static_cast<size_t>(n_batch) * cols), XS(n_batch * nblocks);
				for (float& w : W) w = d(rng);
				for (float& x : X) x = d(rng);
				for (uint32_t t = 0; t < n_batch; ++t)
					nnc_block_sums_f32(X.data() + t * cols, XS.data() + t * nblocks, cols);
				std::vector<nnc_bf16_t> Wb(n);
				for (size_t i = 0; i < n; ++i) Wb[i] = nnc_f32_to_bf16(W[i]);
				std::vector<int8_t> q8(n);
				std::vector<uint8_t> q4(n / 2);
				std::vector<uint16_t> s8(rows * nblocks), s4(rows * nblocks), b4(rows * nblocks);
				nnc_quantize_bf16_to_q8_0(Wb.data(), q8.data(), s8.data(), rows, cols);
				for (uint32_t r = 0; r < rows; ++r)
					nnc_quantize_f32_to_q4_s(W.data() + r * cols, q4.data() + r * (cols / 2), s4.data() + r * nblocks,
					                         b4.data() + r * nblocks, cols);
				std::vector<float> Y(static_cast<size_t>(n_batch) * rows), y(rows);
				nnc_gemv_q8_0_f32x_batch(q8.data(), s8.data(), X.data(), cols, Y.data(), rows, rows, cols, n_batch);
				for (uint32_t t = 0; t < n_batch; ++t)
				{
					nnc_gemv_q8_0_f32x(q8.data(), s8.data(), X.data() + t * cols, y.data(), rows, cols);
					if (std::memcmp(y.data(), Y.data() + t * rows, rows * sizeof(float)) != 0) return false;
				}
				nnc_gemv_q4_s_f32x_batch(q4.data(), s4.data(), b4.data(), X.data(), cols, XS.data(), nblocks, Y.data(),
				                         rows, rows, cols, n_batch);
				for (uint32_t t = 0; t < n_batch; ++t)
				{
					nnc_gemv_q4_s_f32x(q4.data(), s4.data(), b4.data(), X.data() + t * cols, XS.data() + t * nblocks,
					                   y.data(), rows, cols);
					if (std::memcmp(y.data(), Y.data() + t * rows, rows * sizeof(float)) != 0) return false;
				}
			}
		return true;
	}

	// IEEE 754 binary32 -> binary16, round-to-nearest-even. Used to build
	// the fp16 scales in the synthetic quantized blocks below.
	uint16_t fp32_to_fp16(const float f)
	{
		uint32_t bits;
		std::memcpy(&bits, &f, 4);
		const uint32_t s = (bits >> 31) & 0x1u;
		int32_t e = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
		const uint32_t m = bits & 0x7FFFFFu;

		if (e >= 31) return static_cast<uint16_t>((s << 15) | (0x1Fu << 10)); // overflow -> inf
		if (e <= 0) return static_cast<uint16_t>(s << 15); // underflow -> 0
		// round-to-nearest-even on the 13 dropped bits
		const uint32_t round = (m >> 12) & 1u;
		const uint32_t sticky = (m & 0xFFFu) ? 1u : 0u;
		uint32_t mh = (m >> 13) + (round & (sticky | ((m >> 13) & 1u)));
		if (mh & 0x400u)
		{
			mh = 0;
			++e;
			if (e >= 31) return static_cast<uint16_t>((s << 15) | (0x1Fu << 10));
		}
		return static_cast<uint16_t>((s << 15) | (static_cast<uint32_t>(e) << 10) | (mh & 0x3FFu));
	}

	// Helper: build a Q4_K block where every super-scale = 1, every
	// sub-block scale = 1 and min = 0, so dequant(qs) == per-nibble
	// integer values. Returns a 144-byte block image.
	void build_q4k_unit_block(uint8_t out[144], const uint8_t qs[128])
	{
		std::memset(out, 0, 144);
		const uint16_t one_h = fp32_to_fp16(1.0f);
		std::memcpy(out + 0, &one_h, 2); // d
		// dmin = 0 (already memset)
		uint8_t* sc = out + 4;
		// sub-blocks 0..3: scale in low 6 of sc[0..3]; min = 0 in sc[4..7].
		sc[0] = 1; sc[1] = 1; sc[2] = 1; sc[3] = 1;
		// sub-blocks 4..7: scale low 4 in sc[8..11], min in sc[8..11]>>4.
		// High 2 bits of scale come from sc[0..3]>>6 (=0). Want scale=1.
		sc[8] = 0x01; sc[9] = 0x01; sc[10] = 0x01; sc[11] = 0x01;
		std::memcpy(out + 16, qs, 128);
	}

	bool test_nnc_dequantize_q8_0()
	{
		// block_q8_0 = { fp16 d; int8 qs[32] }. Two blocks with distinct
		// scales; dequant must yield d * qs exactly.
		uint8_t blocks[68];
		const uint16_t d0 = fp32_to_fp16(0.25f);
		const uint16_t d1 = fp32_to_fp16(-2.0f);
		std::memcpy(blocks + 0, &d0, 2);
		std::memcpy(blocks + 34, &d1, 2);
		for (int k = 0; k < 32; ++k)
		{
			blocks[2 + k] = static_cast<uint8_t>(static_cast<int8_t>(k - 16));
			blocks[36 + k] = static_cast<uint8_t>(static_cast<int8_t>(127 - k * 4));
		}

		float out[64];
		nnc_dequantize_q8_0_to_f32(blocks, out, 64);
		for (int k = 0; k < 32; ++k)
		{
			if (!close(out[k], 0.25f * (k - 16), 1e-6f)) return false;
			if (!close(out[32 + k], -2.0f * (127 - k * 4), 1e-6f)) return false;
		}
		if (nnc_quant_block_elems(8) != 32) return false;
		if (nnc_quant_block_elems(14) != 256) return false;
		if (nnc_quant_block_elems(30) != 0) return false;
		return true;
	}

	// The classic 32-element formats, which mixed quantizations use for rows
	// that are not whole 256-element K-quant blocks: one block of each,
	// built per ggml's layout and decoded through the type dispatcher.
	bool test_nnc_dequantize_legacy()
	{
		const float d = 0.5f, m = -1.25f;
		const uint16_t dh = fp32_to_fp16(d), mh = fp32_to_fp16(m);
		struct format
		{
			uint32_t type;
			bool has_min, has_high;
		};
		for (const format f : {format{2, false, false}, format{3, true, false}, format{6, false, true},
		                       format{7, true, true}})
		{
			int q[32];
			for (int j = 0; j < 32; ++j) q[j] = f.has_high ? (j * 11 + 5) % 32 : (j * 7 + 3) % 16;
			uint8_t block[24] = {};
			size_t at = 0;
			std::memcpy(block + at, &dh, 2);
			at += 2;
			if (f.has_min)
			{
				std::memcpy(block + at, &mh, 2);
				at += 2;
			}
			if (f.has_high)
			{
				uint32_t qh = 0;
				for (int j = 0; j < 32; ++j) qh |= static_cast<uint32_t>((q[j] >> 4) & 1) << j;
				std::memcpy(block + at, &qh, 4);
				at += 4;
			}
			for (int j = 0; j < 16; ++j)
				block[at + j] = static_cast<uint8_t>((q[j] & 0x0F) | ((q[j + 16] & 0x0F) << 4));

			float out[32];
			if (nnc_quant_block_elems(f.type) != 32 || !nnc_dequantize_to_f32(f.type, block, out, 32)) return false;
			const int centre = f.has_min ? 0 : (f.has_high ? 16 : 8);
			for (int j = 0; j < 32; ++j)
				if (out[j] != static_cast<float>(q[j] - centre) * d + (f.has_min ? m : 0.0f)) return false;
		}
		return true;
	}

	bool test_nnc_dequantize_q4_k()
	{
		// qs[i] = 0x21 for all i: low nibble = 1, high nibble = 2.
		uint8_t qs[128];
		std::memset(qs, 0x21, sizeof(qs));
		uint8_t blk[144];
		build_q4k_unit_block(blk, qs);

		float out[256];
		nnc_dequantize_q4_k_to_f32(blk, out, 256);

		// Each of the 4 outer iterations writes 64 outputs:
		// 32 from low nibble (=1), 32 from high nibble (=2).
		for (int j = 0; j < 4; ++j)
		{
			for (int k = 0; k < 32; ++k)
				if (out[j * 64 + k] != 1.0f) return false;
			for (int k = 0; k < 32; ++k)
				if (out[j * 64 + 32 + k] != 2.0f) return false;
		}
		return true;
	}

	bool test_nnc_dequantize_q5_k()
	{
		// Same scale/min layout as Q4_K, plus qh: 1 bit per element
		// across all 4 outer iterations (u1,u2 walk bits 0/1, 2/3, 4/5, 6/7).
		uint8_t blk[176];
		std::memset(blk, 0, sizeof(blk));
		const uint16_t one_h = fp32_to_fp16(1.0f);
		std::memcpy(blk + 0, &one_h, 2); // d
		uint8_t* sc = blk + 4;
		sc[0] = sc[1] = sc[2] = sc[3] = 1;
		sc[8] = sc[9] = sc[10] = sc[11] = 0x01;
		uint8_t* qh = blk + 16;
		uint8_t* qs = blk + 16 + 32;

		// Case 1: qh all zero → no +16, output identical to Q4_K test.
		std::memset(qh, 0x00, 32);
		std::memset(qs, 0x21, 128);

		float out[256];
		nnc_dequantize_q5_k_to_f32(blk, out, 256);
		for (int j = 0; j < 4; ++j)
		{
			for (int k = 0; k < 32; ++k) if (out[j * 64 + k] != 1.0f) return false;
			for (int k = 0; k < 32; ++k) if (out[j * 64 + 32 + k] != 2.0f) return false;
		}

		// Case 2: qh all 0xFF → every element gets +16 added.
		std::memset(qh, 0xFF, 32);
		nnc_dequantize_q5_k_to_f32(blk, out, 256);
		for (int j = 0; j < 4; ++j)
		{
			for (int k = 0; k < 32; ++k) if (out[j * 64 + k] != 17.0f) return false;
			for (int k = 0; k < 32; ++k) if (out[j * 64 + 32 + k] != 18.0f) return false;
		}
		return true;
	}

	bool test_fp16_subnormal_dequant()
	{
		// Regression: fp16 subnormal decode previously had an off-by-one
		// that produced values 2x too small. K-quant `d` super-block scales
		// frequently land in subnormal range, biasing dequanted weights.
		//
		// Build a Q4_K block with d = smallest subnormal half (0x0001
		// = 2^-24), dmin = 0, sub-block scale[0] = 1, qs low nibble = 1.
		// Then y[0] should be 1 * d * 1 - 0 = 2^-24 ≈ 5.96e-8.
		uint8_t blk[144];
		std::memset(blk, 0, sizeof(blk));
		const uint16_t d_smallest_subnormal = 0x0001; // 2^-24
		std::memcpy(blk + 0, &d_smallest_subnormal, 2);
		// dmin = 0 (already)
		blk[4] = 1; // scales[0] = sc=1, mn=0
		blk[8] = 0; // scales[8..11] = 0 (unused for j<4)
		std::memset(blk + 16, 0x11, 128); // qs: low=1, high=1

		float out[256];
		nnc_dequantize_q4_k_to_f32(blk, out, 256);
		const float expected = std::ldexp(1.0f, -24); // 2^-24
		// Tight relative tolerance — decode should be exact.
		const float err = std::fabs(out[0] - expected);
		return err < 1e-12f;
	}

	bool test_nnc_dequantize_q6_k()
	{
		// d = 1, all 16 sub-block scales = 1, ql = 0x10 (low nibble 0,
		// high nibble 1), qh = 0 (top 2 bits of every quant = 0). Then
		// q1 = q2 = 0 - 32 = -32; q3 = q4 = 1 - 32 = -31.
		uint8_t blk[210];
		std::memset(blk, 0, sizeof(blk));
		std::memset(blk + 0, 0x10, 128); // ql
		// qh already zeroed
		std::memset(blk + 128 + 64, 1, 16); // scales (i8) = 1
		const uint16_t one_h = fp32_to_fp16(1.0f);
		std::memcpy(blk + 208, &one_h, 2); // d

		float out[256];
		nnc_dequantize_q6_k_to_f32(blk, out, 256);

		// For each of 2 outer iters writing 128 outputs:
		//   y[l +  0] = -32, y[l + 32] = -32, y[l + 64] = -31, y[l + 96] = -31
		for (int n = 0; n < 256; n += 128)
		{
			for (int l = 0; l < 32; ++l)
			{
				if (out[n + l + 0] != -32.0f) return false;
				if (out[n + l + 32] != -32.0f) return false;
				if (out[n + l + 64] != -31.0f) return false;
				if (out[n + l + 96] != -31.0f) return false;
			}
		}
		return true;
	}

	// ---- nn_ops: swiglu ---------------------------------------------------

	bool test_nnc_swiglu_f32()
	{
		std::mt19937 rng(0x5167);
		std::uniform_real_distribution<float> d(-3.0f, 3.0f);
		constexpr size_t N = 257;
		std::vector<float> g(N), u(N), y(N);
		for (size_t i = 0; i < N; ++i)
		{
			g[i] = d(rng);
			u[i] = d(rng);
		}

		nnc_swiglu_f32(y.data(), g.data(), u.data(), N);

		for (size_t i = 0; i < N; ++i)
		{
			const float sig = 1.0f / (1.0f + std::exp(-g[i]));
			const float ref = g[i] * sig * u[i];
			if (!close(y[i], ref, 1e-5f)) return false;
		}
		// Aliasing: y == gate must work.
		std::vector<float> z = g;
		nnc_swiglu_f32(z.data(), z.data(), u.data(), N);
		for (size_t i = 0; i < N; ++i)
			if (!close(z[i], y[i], 1e-5f)) return false;
		return true;
	}

	// ---- nn_ops: rope -----------------------------------------------------

	bool test_nnc_rope_f32()
	{
		// Property check: pos=0 must be the identity.
		{
			std::mt19937 rng(0x5051);
			std::uniform_real_distribution<float> d(-1.0f, 1.0f);
			constexpr uint32_t H = 8, D = 64, R = 64;
			std::vector<float> x(H * D), x0(H * D);
			for (auto& v : x) v = d(rng);
			x0 = x;
			nnc_rope_f32(x.data(), H, D, R, 0, 10000.0f);
			for (size_t i = 0; i < x.size(); ++i)
				if (!close(x[i], x0[i], 1e-6f)) return false;
		}

		// Numerical equivalence vs scalar reference.
		{
			std::mt19937 rng(0x5052);
			std::uniform_real_distribution<float> d(-1.0f, 1.0f);
			constexpr uint32_t H = 4, D = 64, R = 32; // R < D: trailing lanes pass through
			std::vector<float> x(H * D), ref(H * D);
			for (auto& v : x) v = d(rng);
			ref = x;

			constexpr int32_t pos = 17;
			constexpr float base = 1e6f;
			nnc_rope_f32(x.data(), H, D, R, pos, base);

			constexpr uint32_t half = R / 2;
			for (uint32_t h = 0; h < H; ++h)
			{
				float* rh = ref.data() + h * D;
				for (uint32_t i = 0; i < half; ++i)
				{
					const float inv = std::pow(base, -static_cast<float>(2 * i) / static_cast<float>(R));
					const float theta = pos * inv;
					const float c = std::cos(theta), s = std::sin(theta);
					const float a = rh[i], b = rh[i + half];
					rh[i] = c * a - s * b;
					rh[i + half] = s * a + c * b;
				}
			}
			for (size_t i = 0; i < x.size(); ++i)
				if (!close(x[i], ref[i], 1e-5f)) return false;
		}

		// Property: rotation preserves per-pair magnitude.
		{
			std::mt19937 rng(0x5053);
			std::uniform_real_distribution<float> d(-1.0f, 1.0f);
			constexpr uint32_t H = 2, D = 16, R = 16;
			std::vector<float> x(H * D), x0(H * D);
			for (auto& v : x) v = d(rng);
			x0 = x;
			nnc_rope_f32(x.data(), H, D, R, 42, 10000.0f);
			for (uint32_t h = 0; h < H; ++h)
			{
				const float* a = x0.data() + h * D;
				const float* b = x.data() + h * D;
				constexpr uint32_t half = R / 2;
				for (uint32_t i = 0; i < half; ++i)
				{
					const float m0 = a[i] * a[i] + a[i + half] * a[i + half];
					const float m1 = b[i] * b[i] + b[i + half] * b[i + half];
					if (std::fabs(m0 - m1) > 1e-4f * std::fmax(m0, 1e-6f)) return false;
				}
			}
		}
		return true;
	}

	// ---- nn_ops: softcap --------------------------------------------------

	bool test_nnc_softcap_f32()
	{
		constexpr float cap = 30.0f;
		constexpr float in[] = {0.0f, 1.0f, -1.0f, 5.0f, -5.0f, 30.0f, 100.0f, -100.0f, 1e6f};
		constexpr size_t N = sizeof(in) / sizeof(in[0]);
		float out[N];
		nnc_softcap_f32(out, in, N, cap);
		for (size_t i = 0; i < N; ++i)
		{
			const float ref = std::tanh(in[i] / cap) * cap;
			if (!close(out[i], ref, 1e-5f)) return false;
			if (std::fabs(out[i]) > cap + 1e-4f) return false;
		}
		// Soft-cap of 0 stays 0.
		if (out[0] != 0.0f) return false;
		// Saturates near +/- cap for large inputs.
		if (out[8] < cap - 1e-3f || out[8] > cap) return false;
		return true;
	}

	// ---- nn_ops: embed_row_bf16 ------------------------------------------

	bool test_nnc_embed_row_bf16()
	{
		constexpr size_t n_embd = 17;
		constexpr size_t n_vocab = 5;
		std::vector<nnc_bf16_t> table(n_vocab * n_embd);
		std::mt19937 rng(0xE3B0);
		std::uniform_real_distribution<float> d(-2.0f, 2.0f);
		for (auto& v : table) v = nnc_f32_to_bf16(d(rng));

		constexpr int tok = 3;
		constexpr float scale = 1.5f;
		std::vector<float> y(n_embd);
		nnc_embed_row_bf16(y.data(), table.data(), tok, n_embd, scale);
		for (size_t i = 0; i < n_embd; ++i)
		{
			const float ref = nnc_bf16_to_f32(table[tok * n_embd + i]) * scale;
			if (!close(y[i], ref, 1e-6f)) return false;
		}
		return true;
	}

	bool test_nnc_embed_row_q8_0()
	{
		// Quantize a synthetic table, then check the row lookup matches a
		// scalar dequant of the same row.
		constexpr size_t n_embd = 64; // multiple of 32
		constexpr size_t n_vocab = 7;
		std::vector<float> src(n_vocab * n_embd);
		std::mt19937 rng(0x0800);
		std::uniform_real_distribution<float> d(-2.0f, 2.0f);
		for (auto& v : src) v = d(rng);

		std::vector<int8_t> qs(n_vocab * n_embd);
		std::vector<uint16_t> scales(n_vocab * n_embd / 32);
		nnc_quantize_f32_to_q8_0(src.data(), qs.data(), scales.data(), src.size());

		constexpr int tok = 5;
		constexpr float scale = 0.75f;
		std::vector<float> y(n_embd);
		nnc_embed_row_q8_0(y.data(), qs.data(), scales.data(), tok, n_embd, scale);

		constexpr size_t nblocks = n_embd / 32;
		for (size_t i = 0; i < n_embd; ++i)
		{
			const size_t gi = tok * n_embd + i;
			const float ref = static_cast<float>(qs[gi])
				* nnc_bf16_to_f32(scales[tok * nblocks + i / 32]) * scale;
			if (!close(y[i], ref, 1e-5f)) return false;
			// And within Q8_0 error of the original value.
			if (std::fabs(y[i] - src[gi] * scale) > 0.05f) return false;
		}
		return true;
	}

	bool test_nnc_embed_row_q4_s()
	{
		constexpr size_t n_embd = 64;
		constexpr size_t n_vocab = 7;
		std::vector<float> src(n_vocab * n_embd);
		std::mt19937 rng(0x0400);
		std::uniform_real_distribution<float> d(-2.0f, 2.0f);
		for (auto& v : src) v = d(rng);

		constexpr size_t nblocks = n_embd / 32;
		std::vector<uint8_t> qs(n_vocab * n_embd / 2);
		std::vector<uint16_t> scales(n_vocab * nblocks), biases(n_vocab * nblocks);
		for (size_t r = 0; r < n_vocab; ++r)
			nnc_quantize_f32_to_q4_s(src.data() + r * n_embd,
			                         qs.data() + r * (n_embd / 2),
			                         scales.data() + r * nblocks,
			                         biases.data() + r * nblocks, n_embd);

		constexpr int tok = 5;
		constexpr float scale = 0.75f;
		std::vector<float> y(n_embd);
		nnc_embed_row_q4_s(y.data(), qs.data(), scales.data(), biases.data(),
		                   tok, n_embd, scale);

		for (size_t b = 0; b < nblocks; ++b)
		{
			const uint8_t* blk = qs.data() + tok * (n_embd / 2) + b * 16;
			const float sc = nnc_bf16_to_f32(scales[tok * nblocks + b]);
			const float bi = nnc_bf16_to_f32(biases[tok * nblocks + b]);
			for (size_t k = 0; k < 16; ++k)
			{
				if (!close(y[b * 32 + k], (sc * (blk[k] & 0x0F) + bi) * scale, 1e-5f))
					return false;
				if (!close(y[b * 32 + k + 16], (sc * (blk[k] >> 4) + bi) * scale, 1e-5f))
					return false;
			}
		}
		// 4-bit over a [-2,2] spread: max error is half a step ~ 0.13.
		for (size_t i = 0; i < n_embd; ++i)
			if (std::fabs(y[i] - src[tok * n_embd + i] * scale) > 0.2f) return false;
		return true;
	}

	// ---- GGUF -------------------------------------------------------------

	// Writes a GGUF with no tensors and the single KV `key` whose encoded
	// type and value bytes follow, under the UTF-8 file `name` in the temp
	// directory. Returns the full UTF-8 path, or "" on failure.
	std::string write_one_kv_gguf(const char* name, const char* key, const uint32_t type,
	                              const void* value, const size_t value_bytes)
	{
		std::error_code ec;
		const std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
		if (ec) return {};
		const std::filesystem::path path = dir / std::filesystem::path(
			std::u8string(reinterpret_cast<const char8_t*>(name), std::strlen(name)));
		std::ofstream f(path, std::ios::binary);
		const uint32_t magic = 0x46554747, version = 3;
		const uint64_t tensors = 0, kvs = 1, key_len = std::strlen(key);
		f.write(reinterpret_cast<const char*>(&magic), 4);
		f.write(reinterpret_cast<const char*>(&version), 4);
		f.write(reinterpret_cast<const char*>(&tensors), 8);
		f.write(reinterpret_cast<const char*>(&kvs), 8);
		f.write(reinterpret_cast<const char*>(&key_len), 8);
		f.write(key, static_cast<std::streamsize>(key_len));
		f.write(reinterpret_cast<const char*>(&type), 4);
		f.write(static_cast<const char*>(value), static_cast<std::streamsize>(value_bytes));
		if (!f) return {};
		const std::u8string u8 = path.u8string();
		return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
	}

	void remove_utf8(const std::string& path)
	{
		std::error_code ec;
		std::filesystem::remove(std::filesystem::path(
			std::u8string(reinterpret_cast<const char8_t*>(path.data()), path.size())), ec);
	}

	bool test_platform_replace_existing_file()
	{
		std::error_code ec;
		const auto dir = std::filesystem::temp_directory_path(ec);
		if (ec) return false;
		const std::string suffix = std::to_string(nnc_time_ns());
		const auto source = dir / ("nnc-test-replace-source-" + suffix);
		const auto destination = dir / ("nnc-test-replace-destination-" + suffix);
		bool wrote = false;
		{
			std::ofstream src(source, std::ios::binary);
			std::ofstream dst(destination, std::ios::binary);
			if (src && dst)
			{
				src << "verified replacement";
				dst << "old partial download";
				wrote = static_cast<bool>(src) && static_cast<bool>(dst);
			}
		}
		if (!wrote)
		{
			std::filesystem::remove(source, ec);
			std::filesystem::remove(destination, ec);
			return false;
		}

		std::string error;
		const bool replaced = platform_replace_file(path_u8(source), path_u8(destination), error);
		std::ifstream result(destination, std::ios::binary);
		const std::string contents((std::istreambuf_iterator<char>(result)), std::istreambuf_iterator<char>());
		const bool source_gone = !std::filesystem::exists(source, ec) && !ec;
		const bool ok = replaced && contents == "verified replacement" && source_gone;
		std::filesystem::remove(source, ec);
		std::filesystem::remove(destination, ec);
		return ok;
	}

	// Paths are UTF-8 throughout nnc (the model picker, the sys layer), so
	// gguf_load must open a file whose UTF-8 name is outside the Windows
	// ANSI code page.
	bool test_gguf_load_utf8_path()
	{
		const uint32_t alignment = 64;
		const std::string path = write_one_kv_gguf(
			"nnc-test-\xC3\xA9\xE2\x82\xAC\xE6\xBC\xA2.gguf", // é € 漢
			"general.alignment", 4 /* UINT32 */, &alignment, sizeof(alignment));
		if (path.empty()) return false;
		gguf_file g{};
		const bool ok = gguf_load(path, g) && g.version == 3 && g.kv.size() == 1
			&& g.alignment == 64 && g.tensors.empty();
		remove_utf8(path);
		return ok;
	}

	// A few bytes must not be able to claim gigabytes: an array whose length
	// the rest of the file cannot hold is refused before anything is sized.
	bool test_gguf_load_rejects_oversized_array()
	{
		unsigned char value[13];
		const uint32_t element_type = 0; // UINT8
		const uint64_t length = 1ull << 27;
		std::memcpy(value, &element_type, 4);
		std::memcpy(value + 4, &length, 8);
		value[12] = 7; // the only element actually present
		const std::string path = write_one_kv_gguf("nnc-test-array.gguf", "bogus",
		                                           9 /* ARRAY */, value, sizeof(value));
		if (path.empty()) return false;
		gguf_file g{};
		const bool ok = !gguf_load(path, g) && g.kv.size() == 1 && g.kv[0].value.arr.empty();
		remove_utf8(path);
		return ok;
	}

	void append_u32(std::vector<unsigned char>& bytes, const uint32_t v)
	{
		const auto* p = reinterpret_cast<const unsigned char*>(&v);
		bytes.insert(bytes.end(), p, p + 4);
	}

	void append_u64(std::vector<unsigned char>& bytes, const uint64_t v)
	{
		const auto* p = reinterpret_cast<const unsigned char*>(&v);
		bytes.insert(bytes.end(), p, p + 8);
	}

	// Each level of a nested array is a recursion frame for 12 bytes of file,
	// so a deep nest must be refused rather than overflow the stack, while
	// an array of arrays still loads.
	bool test_gguf_load_rejects_deep_array_nesting()
	{
		std::vector<unsigned char> deep;
		for (int level = 1; level < 100000; ++level)
		{
			append_u32(deep, 9); // ARRAY of ...
			append_u64(deep, 1);
		}
		append_u32(deep, 0); // ... UINT8
		append_u64(deep, 1);
		deep.push_back(7);
		const std::string deep_path = write_one_kv_gguf("nnc-test-nested.gguf", "deep",
		                                                9 /* ARRAY */, deep.data(), deep.size());
		std::vector<unsigned char> two;
		append_u32(two, 0);
		append_u64(two, 1);
		two.push_back(7);
		std::vector<unsigned char> outer;
		append_u32(outer, 9);
		append_u64(outer, 1);
		outer.insert(outer.end(), two.begin(), two.end());
		const std::string two_path = write_one_kv_gguf("nnc-test-nested2.gguf", "two",
		                                               9 /* ARRAY */, outer.data(), outer.size());
		if (deep_path.empty() || two_path.empty()) return false;
		gguf_file g{};
		const bool refused = !gguf_load(deep_path, g);
		gguf_file g2{};
		const bool accepted = gguf_load(two_path, g2) && g2.kv.size() == 1
			&& g2.kv[0].value.arr.size() == 1 && g2.kv[0].value.arr[0].arr.size() == 1
			&& g2.kv[0].value.arr[0].arr[0].u64 == 7;
		remove_utf8(deep_path);
		remove_utf8(two_path);
		return refused && accepted;
	}

	// Each metadata value is far larger in memory than its one byte of file,
	// so the parser budgets them: an array past the budget is refused before
	// any of it is read.
	bool test_gguf_load_rejects_metadata_bomb()
	{
		const uint64_t length = (256ull << 20) / sizeof(gguf_value) + 1;
		std::vector<unsigned char> value;
		append_u32(value, 0); // UINT8
		append_u64(value, length);
		value.resize(value.size() + static_cast<size_t>(length), 7);
		const std::string path = write_one_kv_gguf("nnc-test-bomb.gguf", "bomb", 9 /* ARRAY */,
		                                           value.data(), value.size());
		if (path.empty()) return false;
		gguf_file g{};
		const bool ok = !gguf_load(path, g) && nnc_last_status() == NNC_ERR_BAD_FORMAT
			&& g.kv.size() == 1 && g.kv[0].value.arr.empty();
		remove_utf8(path);
		return ok;
	}

	// Writes a GGUF holding one F32 tensor of shape `dims` and `data_bytes`
	// of zeroed data after the aligned descriptor table.
	std::string write_one_tensor_gguf(const char* name, const std::vector<uint64_t>& dims,
	                                  const size_t data_bytes)
	{
		std::vector<unsigned char> bytes;
		append_u32(bytes, 0x46554747);
		append_u32(bytes, 3);
		append_u64(bytes, 1); // tensors
		append_u64(bytes, 0); // KVs
		append_u64(bytes, 1);
		bytes.push_back('t');
		append_u32(bytes, static_cast<uint32_t>(dims.size()));
		for (const uint64_t d : dims) append_u64(bytes, d);
		append_u32(bytes, 0); // F32
		append_u64(bytes, 0); // offset
		bytes.resize((bytes.size() + 31) / 32 * 32 + data_bytes, 0);
		std::error_code ec;
		const std::filesystem::path path = std::filesystem::temp_directory_path(ec) / name;
		if (ec) return {};
		std::ofstream f(path, std::ios::binary);
		f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		if (!f) return {};
		const std::u8string u8 = path.u8string();
		return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
	}

	// nnc keeps dimensions as int and derives byte counts from the element
	// count, so a dimension past INT_MAX or a product that wraps is refused
	// at load: [2^62 + 1, 4] would otherwise count as 4 elements.
	bool test_gguf_load_rejects_impossible_shapes()
	{
		const std::vector<std::vector<uint64_t>> bad = {
			{1ull << 32, 1ull << 32},
			{(1ull << 62) + 1, 4},
			{1ull << 30, 1ull << 30, 1ull << 30, 1ull << 30},
			{1ull << 21, 1ull << 21, 1ull << 20}, // 2^62 F32 elements: 2^64 bytes wraps to 0
		};
		bool ok = true;
		for (const auto& dims : bad)
		{
			const std::string path = write_one_tensor_gguf("nnc-test-shape.gguf", dims, 64);
			gguf_file g{};
			ok = ok && !path.empty() && !gguf_load(path, g);
			remove_utf8(path);
		}
		const std::string good = write_one_tensor_gguf("nnc-test-shape-ok.gguf", {4, 2}, 32);
		gguf_file g{};
		ok = ok && !good.empty() && gguf_load(good, g) && g.tensors.size() == 1
			&& gguf_tensor_nelements(g.tensors[0]) == 8;
		remove_utf8(good);
		return ok;
	}

	// ---- mixture of experts, attention spans, embeddings -------------------

	bool test_moe_top_k()
	{
		const float p[6] = {0.10f, 0.30f, 0.05f, 0.30f, 0.20f, 0.05f};
		int idx[3];
		float w[3];
		nnc_moe_top_k(p, 6, 3, idx, w);
		// 0.30 twice: the lower index first; then 0.20.
		if (idx[0] != 1 || idx[1] != 3 || idx[2] != 4) return false;
		return close(w[0], 0.375f, 1e-6f) && close(w[1], 0.375f, 1e-6f) && close(w[2], 0.25f, 1e-6f);
	}

	bool test_attn_span()
	{
		struct expect
		{
			bool causal;
			int window, pos, n_seq, first, last;
		};
		const expect cases[] = {
			{true, 0, 5, 10, 0, 5}, // causal: the prefix
			{true, 3, 5, 10, 3, 5}, // sliding: the last 3 positions
			{true, 0, 5, 5, 0, 4}, // MTP drafter: every key the target evaluated
			{true, 3, 5, 5, 3, 4},
			{false, 0, 5, 10, 0, 9}, // bidirectional: everything
			{false, 4, 5, 10, 3, 7}, // ...within |key - pos| <= window / 2
			{false, 4, 0, 10, 0, 2},
			{false, 4, 9, 10, 7, 9},
			{false, 512, 300, 1000, 44, 556},
		};
		for (const expect& e : cases)
		{
			int first = -1, last = -1;
			model_attn_span(e.causal, e.window, e.pos, e.n_seq, &first, &last);
			if (first != e.first || last != e.last) return false;
		}
		return true;
	}

	bool test_embedding_pool_normalize()
	{
		const float rows[6] = {1, 2, 3, 4, 5, 6}; // three tokens of two values
		float out[2];
		model_pool(model_pooling::mean, rows, 3, 2, out);
		if (out[0] != 3.0f || out[1] != 4.0f) return false;
		model_pool(model_pooling::cls, rows, 3, 2, out);
		if (out[0] != 1.0f || out[1] != 2.0f) return false;
		model_pool(model_pooling::last, rows, 3, 2, out);
		if (out[0] != 5.0f || out[1] != 6.0f) return false;

		float v[3] = {3, 4, 12};
		model_l2_normalize(v, 2); // a Matryoshka prefix is normalised on its own
		if (!close(v[0], 0.6f, 1e-6f) || !close(v[1], 0.8f, 1e-6f) || v[2] != 12.0f) return false;
		float z[2] = {0, 0};
		model_l2_normalize(z, 2);
		return z[0] == 0.0f && z[1] == 0.0f;
	}

	// ---- tokenizer ---------------------------------------------------------

	void append_str(std::vector<unsigned char>& bytes, const std::string& s)
	{
		append_u64(bytes, s.size());
		bytes.insert(bytes.end(), s.begin(), s.end());
	}

	// A GGUF holding just a tokenizer: `tokens` (text, GGUF token type),
	// with descending scores, string KVs and, for byte-level BPE
	// vocabularies, `merges`. Returns its UTF-8 path.
	std::string write_vocab_gguf(const char* name, const std::vector<std::pair<std::string, int32_t>>& tokens,
	                             const std::vector<std::pair<std::string, std::string>>& strings,
	                             const std::vector<std::string>& merges = {})
	{
		std::vector<unsigned char> b;
		append_u32(b, 0x46554747);
		append_u32(b, 3);
		append_u64(b, 0); // tensors
		append_u64(b, 3 + strings.size() + (merges.empty() ? 0 : 1)); // KVs
		append_str(b, "tokenizer.ggml.tokens");
		append_u32(b, 9); // ARRAY
		append_u32(b, 8); // of STRING
		append_u64(b, tokens.size());
		for (const auto& t : tokens) append_str(b, t.first);
		append_str(b, "tokenizer.ggml.token_type");
		append_u32(b, 9);
		append_u32(b, 5); // of INT32
		append_u64(b, tokens.size());
		for (const auto& t : tokens) append_u32(b, static_cast<uint32_t>(t.second));
		append_str(b, "tokenizer.ggml.scores");
		append_u32(b, 9);
		append_u32(b, 6); // of FLOAT32
		append_u64(b, tokens.size());
		for (size_t i = 0; i < tokens.size(); ++i)
		{
			const float score = -static_cast<float>(i);
			uint32_t bits;
			std::memcpy(&bits, &score, 4);
			append_u32(b, bits);
		}
		for (const auto& [key, value] : strings)
		{
			append_str(b, key);
			append_u32(b, 8); // STRING
			append_str(b, value);
		}
		if (!merges.empty())
		{
			append_str(b, "tokenizer.ggml.merges");
			append_u32(b, 9);
			append_u32(b, 8); // of STRING
			append_u64(b, merges.size());
			for (const auto& m : merges) append_str(b, m);
		}
		std::error_code ec;
		const std::filesystem::path path = std::filesystem::temp_directory_path(ec) / name;
		if (ec) return {};
		std::ofstream f(path, std::ios::binary);
		f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
		if (!f) return {};
		const std::u8string u8 = path.u8string();
		return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
	}

	// User-defined tokens (FunctionGemma's tags, Gemma's newline and space
	// runs) are one token wherever they appear, longest first; control
	// tokens are not. A vocabulary with function tags is FunctionGemma's:
	// its system prompt gets a developer turn, and a function response ends
	// generation.
	bool test_vocab_user_defined_tokens()
	{
		const std::vector<std::pair<std::string, int32_t>> tokens = {
			{"<pad>", 3}, {"<eos>", 3}, {"<bos>", 3}, {"<unk>", 3}, {"a", 1}, {"b", 1}, {"\xE2\x96\x81", 1},
			{"<escape>", 4}, {"\n", 4}, {"\n\n", 4}, {"  ", 4}, {"<start_of_turn>", 3}, {"<end_of_turn>", 3},
			{"<start_function_declaration>", 4}, {"<start_function_response>", 4},
		};
		const std::string path = write_vocab_gguf("nnc-test-vocab.gguf", tokens, {{"tokenizer.ggml.model", "llama"}});
		if (path.empty()) return false;
		gguf_file g{};
		model_vocab v;
		const bool loaded = gguf_load(path, g) && vocab_load(g, v);
		remove_utf8(path);
		if (!loaded) return false;

		const std::vector<int> got = vocab_tokenize(v, "a<escape>b\n\n\na  b", false);
		const std::vector<int> want = {4, 7, 5, 9, 8, 4, 10, 5};
		return got == want && v.chat == chat_format::gemma && v.developer_turn && vocab_is_eog(v, 14)
			&& vocab_is_eog(v, 12) && !vocab_is_eog(v, 13);
	}

	// TranslateGemma's chat template carries its language list.
	bool test_vocab_translation_languages()
	{
		const std::vector<std::pair<std::string, int32_t>> tokens = {
			{"<pad>", 3}, {"<eos>", 3}, {"<bos>", 3}, {"<start_of_turn>", 3}, {"<end_of_turn>", 3}, {"\n", 4},
		};
		const std::string tpl = "{%- set languages = {\n    \"de\": \"German\",\n    \"pt-BR\": \"Portuguese\",\n}\n"
			"-%}{{ bos_token }}{{ content[\"source_lang_code\"] }}";
		const std::string path = write_vocab_gguf("nnc-test-vocab-langs.gguf", tokens,
		                                          {{"tokenizer.chat_template", tpl}});
		if (path.empty()) return false;
		gguf_file g{};
		model_vocab v;
		const bool loaded = gguf_load(path, g) && vocab_load(g, v);
		remove_utf8(path);
		return loaded && v.chat == chat_format::gemma && !v.developer_turn && v.languages.size() == 2
			&& v.languages["de"] == "German" && v.languages["pt-BR"] == "Portuguese";
	}

	// Qwen's pre-tokenizers take digits one at a time, llama-bpe's in runs of
	// up to three. All keep full-width and CJK punctuation and symbols such
	// as arrows out of the words beside them, and only Qwen3.5's keeps
	// combining marks (Devanagari's vowel signs) inside them.
	bool test_vocab_pretokenize_qwen()
	{
		const std::vector<std::pair<std::string, int32_t>> tokens = {{"a", 1}, {"b", 1}, {"ab", 1}};
		const char* const pres[] = {"llama-bpe", "qwen2", "qwen35"};
		model_vocab v[3];
		for (int i = 0; i < 3; ++i)
		{
			const std::string path = write_vocab_gguf("nnc-test-vocab-pre.gguf", tokens,
			                                          {{"tokenizer.ggml.model", "gpt2"}, {"tokenizer.ggml.pre", pres[i]}},
			                                          {"a b"});
			if (path.empty()) return false;
			gguf_file g{};
			const bool loaded = gguf_load(path, g) && vocab_load(g, v[i]);
			remove_utf8(path);
			if (!loaded) return false;
		}

		const std::string text = "ab 12345 \xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x9F\xE2\x86\x92x"; // "ab 12345 你好？→x"
		const std::vector<std::string> want_qwen = {
			"ab", " ", "1", "2", "3", "4", "5", " \xE4\xBD\xA0\xE5\xA5\xBD", "\xEF\xBC\x9F\xE2\x86\x92", "x"};
		const std::vector<std::string> want_llama = {
			"ab", " ", "123", "45", " \xE4\xBD\xA0\xE5\xA5\xBD", "\xEF\xBC\x9F\xE2\x86\x92", "x"};
		// नमस्ते: न म स ् त े, where ् and े are combining marks.
		const std::string hindi = "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87";
		const std::vector<std::string> want_split = {
			"\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8", "\xE0\xA5\x8D\xE0\xA4\xA4", "\xE0\xA5\x87"};
		return v[0].pre == vocab_pre::llama3 && v[1].pre == vocab_pre::qwen2 && v[2].pre == vocab_pre::qwen35
			&& vocab_pretokenize(v[0], text) == want_llama && vocab_pretokenize(v[1], text) == want_qwen
			&& vocab_pretokenize(v[2], text) == want_qwen && vocab_pretokenize(v[0], hindi) == want_split
			&& vocab_pretokenize(v[1], hindi) == want_split
			&& vocab_pretokenize(v[2], hindi) == std::vector<std::string>{hindi}
			&& vocab_tokenize(v[1], "ab", false) == std::vector<int>{2};
	}

	// Qwen3.5's generation prompt (thinking off unless asked for) and
	// Qwen3.8's (on unless switched off, with an instruction for its
	// reasoning effort), cut down to the parts nnc reads.
	const char* const k_qwen35_template =
		"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- if enable_thinking is defined and "
		"enable_thinking is true %}{{- '<think>\\n' }}{%- else %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
		"{%- endif %}";
	const char* const k_qwen38_template =
		"{%- if enable_thinking is undefined or enable_thinking is true %}"
		"{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}"
		"{%- if resolved_reasoning_effort == 'xhigh' %}"
		"{%- set reasoning_instructions = 'Think hard.\\nThen answer.' %}"
		"{%- elif resolved_reasoning_effort == 'low' %}{%- set reasoning_instructions = 'Be brief.' %}{%- endif %}"
		"{%- endif %}"
		"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- if enable_thinking is defined and "
		"enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- else %}{{- '<think>\\n' }}{%- endif %}"
		"{%- endif %}";

	// A vocabulary with <|im_start|> and <|im_end|> takes ChatML turns, and
	// the template says what nnc writes on its own: Qwen2.5's default system
	// prompt, the empty <think> block that turns thinking off (Qwen3, 3.5,
	// 3.8), or the open one of a model that always thinks - and, with
	// thinking on, how the reply opens (Qwen3.5 and 3.8 write "<think>\n",
	// Qwen3 nothing) and the instruction for Qwen3.8's default reasoning
	// effort.
	bool test_vocab_chatml()
	{
		const std::vector<std::pair<std::string, int32_t>> tokens = {
			{"<|endoftext|>", 3}, {"<|im_start|>", 3}, {"<|im_end|>", 3}, {"a", 1}, {"b", 1}, {"ab", 1},
			{"<think>", 4}, {"</think>", 4},
		};
		const std::string templates[] = {
			"{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}{%- else %}"
			"{{- '<|im_start|>system\\nYou are Qwen, a helpful assistant.<|im_end|>\\n' }}{%- endif %}"
			"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}",
			"{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
			"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- if enable_thinking is defined and "
			"enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}",
			"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}",
			k_qwen35_template,
			k_qwen38_template,
		};
		const char* const want_system[] = {"You are Qwen, a helpful assistant.", "", "", "", ""};
		const char* const want_open[] = {"", "<think>\n\n</think>\n\n", "<think>\n", "<think>\n\n</think>\n\n",
		                                 "<think>\n\n</think>\n\n"};
		const char* const want_open_think[] = {"", "", "", "<think>\n", "<think>\n"};
		const bool want_can_think[] = {false, true, false, true, true};
		const char* const want_instructions[] = {"", "", "", "", "Think hard.\nThen answer."};
		for (int i = 0; i < 5; ++i)
		{
			const std::string path = write_vocab_gguf("nnc-test-vocab-chatml.gguf", tokens,
			                                          {{"tokenizer.ggml.model", "gpt2"},
			                                           {"tokenizer.chat_template", templates[i]}}, {"a b"});
			if (path.empty()) return false;
			gguf_file g{};
			model_vocab v;
			const bool loaded = gguf_load(path, g) && vocab_load(g, v);
			remove_utf8(path);
			if (!loaded || v.chat != chat_format::chatml || v.tok_turn_start != 1 || v.tok_turn_end != 2
				|| v.default_system != want_system[i] || v.reply_open != want_open[i] || !vocab_is_eog(v, 2)
				|| !vocab_is_eog(v, 0) || v.tok_thought_open != 6 || v.tok_thought_close != 7
				|| v.reply_open_think != want_open_think[i] || v.can_think != want_can_think[i]
				|| v.always_thinks != (i == 2) || v.think_instructions != want_instructions[i])
				return false;
		}
		return true;
	}

	// ---- thinking ----------------------------------------------------------

	// A vocabulary in which every printable ASCII character is a token, so
	// that any prompt text encodes, and renders back, exactly; `extra` are
	// the model's special tokens. Byte-level (gpt2) for ChatML, SentencePiece
	// for Gemma.
	bool load_ascii_vocab(const std::vector<std::pair<std::string, int32_t>>& extra, const bool byte_level,
	                      const std::string& chat_template, model_vocab& v)
	{
		std::vector<std::pair<std::string, int32_t>> tokens = extra;
		for (int c = 0x21; c < 0x7F; ++c) tokens.emplace_back(std::string(1, static_cast<char>(c)), 1);
		std::vector<std::pair<std::string, std::string>> strings = {{"tokenizer.chat_template", chat_template}};
		std::vector<std::string> merges;
		if (byte_level)
		{
			tokens.emplace_back("\xC4\xA0", 1); // a space, byte-level
			tokens.emplace_back("\xC4\x8A", 1); // a newline
			tokens.emplace_back("ab", 1);
			strings.emplace_back("tokenizer.ggml.model", "gpt2");
			merges.emplace_back("a b");
		}
		else
		{
			tokens.emplace_back("\xE2\x96\x81", 1); // a space, SentencePiece
		}
		const std::string path = write_vocab_gguf("nnc-test-vocab-ascii.gguf", tokens, strings, merges);
		if (path.empty()) return false;
		gguf_file g{};
		const bool loaded = gguf_load(path, g) && vocab_load(g, v);
		remove_utf8(path);
		return loaded;
	}

	// Tokens as the template would write them: control tokens by name.
	std::string render_tokens(const model_vocab& v, const std::vector<int>& ids)
	{
		std::string s;
		for (const int id : ids)
			s += v.types[static_cast<size_t>(id)] == VOCAB_TOKEN_CONTROL ? v.tokens[static_cast<size_t>(id)]
			                                                             : vocab_piece(v, id);
		return s;
	}

	const std::vector<std::pair<std::string, int32_t>> k_chatml_specials = {
		{"<|endoftext|>", 3}, {"<|im_start|>", 3}, {"<|im_end|>", 3}, {"<think>", 4}, {"</think>", 4},
	};
	const std::vector<std::pair<std::string, int32_t>> k_gemma4_specials = {
		{"<pad>", 3}, {"<eos>", 3}, {"<bos>", 3}, {"<|turn>", 3}, {"<turn|>", 3}, {"\n", 4}, {"<|think|>", 3},
		{"<|channel>", 4}, {"<channel|>", 4},
	};

	// Chat turns with thinking on and off, as the templates write them:
	// Qwen3.8's reasoning instruction heads the system turn and the reply
	// opens with "<think>\n" (Qwen3's with nothing); Gemma 4's <|think|>
	// heads a system turn, written for it alone if need be, and replaces the
	// empty thought channel of the 26B and 31B templates.
	bool test_chat_turn_thinking()
	{
		model_vocab q38, q3, g4, g26;
		const std::string qwen3 =
			"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- if enable_thinking is defined and "
			"enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}";
		const std::string gemma4 = "{%- if enable_thinking -%}{{- '<|think|>\\n' -}}{%- endif -%}";
		if (!load_ascii_vocab(k_chatml_specials, true, k_qwen38_template, q38)
			|| !load_ascii_vocab(k_chatml_specials, true, qwen3, q3)
			|| !load_ascii_vocab(k_gemma4_specials, false, gemma4, g4)
			|| !load_ascii_vocab(k_gemma4_specials, false, gemma4 + "{{- '<|channel>thought\\n<channel|>' -}}", g26))
			return false;
		if (!q38.can_think || !q3.can_think || !g4.can_think || !g26.can_think || g4.empty_thought || !g26.empty_thought
			|| g4.chat != chat_format::gemma4 || !g4.thought_label || g4.tok_thought_open != g4.tok_channel_start)
			return false;

		auto turn = [](const model_vocab& v, const std::string& system, const bool think)
		{
			return render_tokens(v, session_turn_tokens(v, {}, system, "Hi", think));
		};
		const std::vector<int> first = session_turn_tokens(q38, {}, "Be helpful.", "Hi", true);
		std::vector<int> context = first;
		for (const int id : vocab_tokenize(q38, "Ok", false)) context.push_back(id);
		context.push_back(q38.tok_turn_end);
		return render_tokens(q38, first)
				== "<|im_start|>system\nThink hard.\nThen answer.\n\nBe helpful.<|im_end|>\n<|im_start|>user\nHi"
				"<|im_end|>\n<|im_start|>assistant\n<think>\n"
			&& turn(q38, "", true)
				== "<|im_start|>system\nThink hard.\nThen answer.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
				"<|im_start|>assistant\n<think>\n"
			&& turn(q38, "Be helpful.", false)
				== "<|im_start|>system\nBe helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
				"<think>\n\n</think>\n\n"
			&& render_tokens(q38, session_turn_tokens(q38, context, "Be helpful.", "More", true))
				== "\n<|im_start|>user\nMore<|im_end|>\n<|im_start|>assistant\n<think>\n"
			&& turn(q3, "", true) == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
			&& turn(q3, "", false) == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
			&& turn(g4, "Be helpful.", true)
				== "<|turn>system\n<|think|>\nBe helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
			&& turn(g4, "", true) == "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
			&& turn(g4, "", false) == "<|turn>user\nHi<turn|>\n<|turn>model\n"
			&& turn(g26, "", false) == "<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
			&& turn(g26, "", true) == "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n";
	}

	// A reply's reasoning and answer are told apart by the thought markers,
	// which show nowhere, as do a Gemma 4 channel's name line and the
	// newlines that open the reasoning and the answer. A call that carries
	// on picks up where the context left off, within the current turn.
	bool test_thought_split()
	{
		model_vocab q, g;
		const std::string gemma4 = "{%- if enable_thinking -%}{{- '<|think|>\\n' -}}{%- endif -%}";
		if (!load_ascii_vocab(k_chatml_specials, true, k_qwen38_template, q)
			|| !load_ascii_vocab(k_gemma4_specials, false, gemma4, g))
			return false;

		// Routes `reply` after `context`: the reasoning shown, the answer
		// shown, and how many tokens were reasoning.
		struct split_result
		{
			std::string thought, answer;
			int n_thought = 0;
		};
		auto run = [](const model_vocab& v, const std::vector<int>& context, const std::vector<int>& reply)
		{
			thought_split split;
			split.start(v, context);
			split_result r;
			std::string shown;
			for (const int t : reply)
			{
				const bool reasoning = split.route(t, vocab_piece(v, t), shown);
				(reasoning ? r.thought : r.answer) += shown;
				r.n_thought += reasoning ? 1 : 0;
			}
			return r;
		};
		auto ids = [](const model_vocab& v, std::initializer_list<std::pair<const char*, int>> parts)
		{
			std::vector<int> out;
			for (const auto& [text, id] : parts)
			{
				if (id >= 0) out.push_back(id);
				else
					for (const int t : vocab_tokenize(v, text, false)) out.push_back(t);
			}
			return out;
		};

		// Qwen: the prompt opened the reasoning; "Plan\n" is 5 tokens.
		const std::vector<int> qp = session_turn_tokens(q, {}, "", "Hi", true);
		const split_result a = run(q, qp, ids(q, {{"Plan\n", -1}, {"", q.tok_thought_close}, {"\n\nDone", -1}}));
		// Gemma 4: the model opens its thought channel and names it.
		const std::vector<int> gp = session_turn_tokens(g, {}, "", "Hi", true);
		const split_result b = run(g, gp, ids(g, {{"", g.tok_thought_open}, {"thought", -1}, {"", g.tok_newline},
		                                          {"Why", -1}, {"", g.tok_newline}, {"", g.tok_thought_close},
		                                          {"Sky", -1}}));
		// Carrying on in the middle of the channel's name.
		std::vector<int> gc = gp;
		for (const int t : ids(g, {{"", g.tok_thought_open}, {"tho", -1}})) gc.push_back(t);
		const split_result c = run(g, gc, ids(g, {{"ught", -1}, {"", g.tok_newline}, {"X", -1}}));
		// Thinking off: after the empty block, the answer's opening newline goes.
		const split_result d = run(q, session_turn_tokens(q, {}, "", "Hi", false), ids(q, {{"\nA", -1}}));
		// A reasoning block left open in an earlier turn does not reach this one.
		std::vector<int> qc = qp;
		for (const int t : ids(q, {{"x", -1}, {"", q.tok_turn_end}, {"\n", -1}, {"", q.tok_turn_start},
		                          {"assistant\n", -1}}))
			qc.push_back(t);
		const split_result e = run(q, qc, ids(q, {{"B", -1}}));
		return a.thought == "Plan\n" && a.answer == "Done" && a.n_thought == 6 && b.thought == "Why\n"
			&& b.answer == "Sky" && b.n_thought == 14 && c.thought == "X" && c.answer.empty() && d.answer == "A"
			&& d.thought.empty() && e.answer == "B" && e.thought.empty();
	}

	// ---- Gated DeltaNet kernels ------------------------------------------------

	bool test_nnc_mul_sigmoid_f32()
	{
		std::mt19937 rng(11);
		std::uniform_real_distribution<float> d(-12.0f, 12.0f);
		for (const size_t n : {size_t{1}, size_t{7}, size_t{8}, size_t{37}})
		{
			std::vector<float> y(n), g(n), want(n);
			for (size_t i = 0; i < n; ++i)
			{
				y[i] = d(rng);
				g[i] = d(rng);
				want[i] = y[i] / (1.0f + std::exp(-g[i]));
			}
			nnc_mul_sigmoid_f32(y.data(), g.data(), n);
			for (size_t i = 0; i < n; ++i)
				if (!close(y[i], want[i], 1e-5f)) return false;
		}
		return true;
	}

	bool test_nnc_l2norm_multi_f32()
	{
		std::mt19937 rng(12);
		std::normal_distribution<float> d(0.0f, 2.0f);
		const size_t groups = 3, dim = 37;
		const float eps = 1e-6f, scale = 0.5f;
		std::vector<float> x(groups * dim), want(x.size());
		for (float& v : x) v = d(rng);
		for (size_t g = 0; g < groups; ++g)
		{
			double ss = 0.0;
			for (size_t i = 0; i < dim; ++i) ss += static_cast<double>(x[g * dim + i]) * x[g * dim + i];
			for (size_t i = 0; i < dim; ++i)
				want[g * dim + i] = static_cast<float>(x[g * dim + i] * scale / std::sqrt(ss + eps));
		}
		nnc_l2norm_multi_f32(x.data(), groups, dim, eps, scale);
		for (size_t i = 0; i < x.size(); ++i)
			if (!close(x[i], want[i], 1e-5f)) return false;
		return true;
	}

	// y[t][c] = silu(sum_j w[j][c] * x[t + j][c]) over the history rows and
	// the batch, with a channel count that leaves a scalar tail.
	bool test_nnc_conv1d_silu_f32()
	{
		std::mt19937 rng(13);
		std::normal_distribution<float> d(0.0f, 1.0f);
		const size_t ch = 21, k = 4, n_tok = 3;
		std::vector<float> x((k - 1 + n_tok) * ch), w(k * ch), y(n_tok * ch);
		for (float& v : x) v = d(rng);
		for (float& v : w) v = d(rng);
		nnc_conv1d_silu_f32(y.data(), x.data(), w.data(), n_tok, ch, k);
		for (size_t t = 0; t < n_tok; ++t)
			for (size_t c = 0; c < ch; ++c)
			{
				double acc = 0.0;
				for (size_t j = 0; j < k; ++j) acc += static_cast<double>(w[j * ch + c]) * x[(t + j) * ch + c];
				const float want = static_cast<float>(acc / (1.0 + std::exp(-acc)));
				if (!close(y[t * ch + c], want, 1e-5f)) return false;
			}
		return true;
	}

	// Several steps of the delta rule against its textbook form:
	//   S = decay S;  S += beta (v - S k) k^T;  o = S q.
	bool test_nnc_delta_rule_f32()
	{
		std::mt19937 rng(14);
		std::normal_distribution<float> d(0.0f, 1.0f);
		std::uniform_real_distribution<float> u(0.0f, 1.0f);
		const size_t d_k = 43, d_v = 5; // 43 = 2 * 16 + 8 + 3: every loop of the kernel
		std::vector<float> S(d_v * d_k, 0.0f), R(S.size(), 0.0f), q(d_k), k(d_k), v(d_v), o(d_v);
		for (int step = 0; step < 6; ++step)
		{
			for (float& x : q) x = d(rng);
			for (float& x : v) x = d(rng);
			double kk = 0.0;
			for (float& x : k)
			{
				x = d(rng);
				kk += static_cast<double>(x) * x;
			}
			for (float& x : k) x = static_cast<float>(x / std::sqrt(kk));
			const float decay = 0.5f + 0.5f * u(rng), beta = u(rng);
			nnc_delta_rule_f32(S.data(), q.data(), k.data(), v.data(), decay, beta, o.data(), d_k, d_v);
			for (size_t j = 0; j < d_v; ++j)
			{
				double sk = 0.0;
				for (size_t i = 0; i < d_k; ++i)
				{
					R[j * d_k + i] *= decay;
					sk += static_cast<double>(R[j * d_k + i]) * k[i];
				}
				const double delta = beta * (v[j] - sk);
				double out = 0.0;
				for (size_t i = 0; i < d_k; ++i)
				{
					R[j * d_k + i] = static_cast<float>(R[j * d_k + i] + delta * k[i]);
					out += static_cast<double>(R[j * d_k + i]) * q[i];
				}
				if (!close(o[j], static_cast<float>(out), 1e-4f)) return false;
			}
			for (size_t i = 0; i < S.size(); ++i)
				if (!close(S[i], R[i], 1e-4f)) return false;
		}
		return true;
	}

	// ---- Qwen3.5: a tiny synthetic hybrid model ----------------------------

	// A GGUF of metadata and F32 tensors, for synthetic models.
	struct gguf_builder
	{
		struct tensor
		{
			std::string name;
			std::vector<uint64_t> ne;
			std::vector<float> data;
		};

		std::vector<unsigned char> kvs;
		uint64_t n_kv = 0;
		std::vector<tensor> tensors;

		void key(const std::string& k, const uint32_t type)
		{
			append_str(kvs, k);
			append_u32(kvs, type);
			++n_kv;
		}

		void u32(const std::string& k, const uint32_t v)
		{
			key(k, 4);
			append_u32(kvs, v);
		}

		void f32(const std::string& k, const float v)
		{
			uint32_t bits;
			std::memcpy(&bits, &v, 4);
			key(k, 6);
			append_u32(kvs, bits);
		}

		void str(const std::string& k, const std::string& v)
		{
			key(k, 8);
			append_str(kvs, v);
		}

		void strs(const std::string& k, const std::vector<std::string>& v)
		{
			key(k, 9);
			append_u32(kvs, 8);
			append_u64(kvs, v.size());
			for (const auto& s : v) append_str(kvs, s);
		}

		void i32s(const std::string& k, const std::vector<int32_t>& v)
		{
			key(k, 9);
			append_u32(kvs, 5);
			append_u64(kvs, v.size());
			for (const int32_t x : v) append_u32(kvs, static_cast<uint32_t>(x));
		}

		void bools(const std::string& k, const std::vector<bool>& v)
		{
			key(k, 9);
			append_u32(kvs, 7);
			append_u64(kvs, v.size());
			for (const bool x : v) kvs.push_back(x ? 1 : 0);
		}

		void add(const std::string& name, const std::vector<uint64_t>& ne, std::vector<float> data)
		{
			tensors.push_back({name, ne, std::move(data)});
		}

		// The file `name` in the temp directory; returns its UTF-8 path, or "".
		std::string write(const char* name) const
		{
			std::vector<unsigned char> b;
			append_u32(b, 0x46554747);
			append_u32(b, 3);
			append_u64(b, tensors.size());
			append_u64(b, n_kv);
			b.insert(b.end(), kvs.begin(), kvs.end());
			uint64_t offset = 0;
			for (const auto& t : tensors)
			{
				append_str(b, t.name);
				append_u32(b, static_cast<uint32_t>(t.ne.size()));
				for (const uint64_t d : t.ne) append_u64(b, d);
				append_u32(b, 0); // F32
				append_u64(b, offset);
				offset += (t.data.size() * sizeof(float) + 31) / 32 * 32;
			}
			for (const auto& t : tensors)
			{
				b.resize((b.size() + 31) / 32 * 32, 0);
				const auto* p = reinterpret_cast<const unsigned char*>(t.data.data());
				b.insert(b.end(), p, p + t.data.size() * sizeof(float));
			}
			std::error_code ec;
			const std::filesystem::path path = std::filesystem::temp_directory_path(ec) / name;
			if (ec) return {};
			std::ofstream f(path, std::ios::binary);
			f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
			if (!f) return {};
			const std::u8string u8 = path.u8string();
			return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
		}
	};

	constexpr int TINY_EMBD = 64;
	constexpr int TINY_VOCAB = 30;

	// Qwen3.5 in miniature: three Gated DeltaNet layers, then one gated
	// attention layer, and a trailing multi-token-prediction block that the
	// loader must skip. Random F32 weights; a 30-token byte-level vocabulary.
	// `extra` names a tensor to add that nnc has no use for.
	std::string write_tiny_qwen35(const char* name, const char* extra = nullptr)
	{
		constexpr int n_embd = TINY_EMBD, n_ff = 96, n_layer = 4;
		constexpr int head_dim = 32, n_head = 2, n_kv = 1;
		constexpr int d_k = 32, n_k = 2, n_v = 4, d_conv = 4;
		constexpr int v_dim = n_v * d_k, conv_dim = 2 * n_k * d_k + v_dim;
		std::mt19937 rng(35);
		std::normal_distribution<float> nd(0.0f, 1.0f);
		auto rnd = [&](const size_t n, const float scale)
		{
			std::vector<float> v(n);
			for (float& x : v) x = nd(rng) * scale;
			return v;
		};
		auto gamma = [&](const size_t n)
		{
			std::vector<float> v(n);
			for (float& x : v) x = 1.0f + 0.1f * nd(rng);
			return v;
		};

		gguf_builder g;
		g.str("general.architecture", "qwen35");
		g.u32("qwen35.block_count", n_layer + 1);
		g.u32("qwen35.nextn_predict_layers", 1);
		g.u32("qwen35.context_length", 256);
		g.u32("qwen35.embedding_length", n_embd);
		g.u32("qwen35.feed_forward_length", n_ff);
		g.u32("qwen35.attention.head_count", n_head);
		g.u32("qwen35.attention.head_count_kv", n_kv);
		g.u32("qwen35.attention.key_length", head_dim);
		g.f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
		g.f32("qwen35.rope.freq_base", 10000.0f);
		g.u32("qwen35.rope.dimension_count", 16);
		g.u32("qwen35.ssm.conv_kernel", d_conv);
		g.u32("qwen35.ssm.state_size", d_k);
		g.u32("qwen35.ssm.group_count", n_k);
		g.u32("qwen35.ssm.time_step_rank", n_v);
		g.u32("qwen35.ssm.inner_size", v_dim);
		g.bools("qwen35.attention.recurrent_layers", {true, true, true, false, false});
		std::vector<std::string> toks;
		std::vector<int32_t> types;
		for (int i = 0; i < 26; ++i)
		{
			toks.emplace_back(1, static_cast<char>('a' + i));
			types.push_back(1);
		}
		for (const char* t : {"ab", "cd"})
		{
			toks.emplace_back(t);
			types.push_back(1);
		}
		for (const char* t : {"<|im_start|>", "<|im_end|>"})
		{
			toks.emplace_back(t);
			types.push_back(3);
		}
		g.str("tokenizer.ggml.model", "gpt2");
		g.str("tokenizer.ggml.pre", "qwen35");
		g.strs("tokenizer.ggml.tokens", toks);
		g.i32s("tokenizer.ggml.token_type", types);
		g.strs("tokenizer.ggml.merges", {"a b", "c d"});

		g.add("token_embd.weight", {n_embd, TINY_VOCAB}, rnd(n_embd * TINY_VOCAB, 0.5f));
		g.add("output_norm.weight", {n_embd}, gamma(n_embd));
		for (int li = 0; li < n_layer; ++li)
		{
			const std::string p = "blk." + std::to_string(li) + ".";
			g.add(p + "attn_norm.weight", {n_embd}, gamma(n_embd));
			g.add(p + "post_attention_norm.weight", {n_embd}, gamma(n_embd));
			g.add(p + "ffn_gate.weight", {n_embd, n_ff}, rnd(n_embd * n_ff, 0.15f));
			g.add(p + "ffn_up.weight", {n_embd, n_ff}, rnd(n_embd * n_ff, 0.15f));
			g.add(p + "ffn_down.weight", {n_ff, n_embd}, rnd(n_ff * n_embd, 0.1f));
			if (li < 3)
			{
				std::vector<float> a(n_v);
				for (float& x : a) x = -std::exp(0.5f * nd(rng));
				g.add(p + "attn_qkv.weight", {n_embd, conv_dim}, rnd(n_embd * conv_dim, 0.15f));
				g.add(p + "attn_gate.weight", {n_embd, v_dim}, rnd(n_embd * v_dim, 0.15f));
				g.add(p + "ssm_alpha.weight", {n_embd, n_v}, rnd(n_embd * n_v, 0.1f));
				g.add(p + "ssm_beta.weight", {n_embd, n_v}, rnd(n_embd * n_v, 0.1f));
				g.add(p + "ssm_conv1d.weight", {d_conv, conv_dim}, rnd(d_conv * conv_dim, 0.4f));
				g.add(p + "ssm_dt.bias", {n_v}, rnd(n_v, 0.5f));
				g.add(p + "ssm_a", {n_v}, a);
				g.add(p + "ssm_norm.weight", {d_k}, gamma(d_k));
				g.add(p + "ssm_out.weight", {v_dim, n_embd}, rnd(v_dim * n_embd, 0.1f));
			}
			else
			{
				g.add(p + "attn_q.weight", {n_embd, 2 * n_head * head_dim}, rnd(n_embd * 2 * n_head * head_dim, 0.15f));
				g.add(p + "attn_k.weight", {n_embd, n_kv * head_dim}, rnd(n_embd * n_kv * head_dim, 0.15f));
				g.add(p + "attn_v.weight", {n_embd, n_kv * head_dim}, rnd(n_embd * n_kv * head_dim, 0.15f));
				g.add(p + "attn_q_norm.weight", {head_dim}, gamma(head_dim));
				g.add(p + "attn_k_norm.weight", {head_dim}, gamma(head_dim));
				g.add(p + "attn_output.weight", {n_head * head_dim, n_embd}, rnd(n_head * head_dim * n_embd, 0.1f));
			}
		}
		g.add("blk.4.nextn.eh_proj.weight", {2 * n_embd, n_embd}, rnd(2 * n_embd * n_embd, 0.1f));
		if (extra) g.add(extra, {1}, {1.0f});
		return g.write(name);
	}

	const int k_tiny_tokens[11] = {3, 7, 1, 29, 5, 12, 0, 18, 26, 9, 4};

	// ---- Blocked attention, BF16 K/V and sliding-window rings ---------------

	// Scalar softmax attention of `group` heads over n_keys K/V rows.
	void ref_attention(const float* q, const float* K, const float* V, const size_t kv_stride, const size_t n_keys,
	                   const size_t hd, const size_t group, const float scale, float* out)
	{
		for (size_t h = 0; h < group; ++h)
		{
			std::vector<double> s(n_keys);
			double m = -1e300, sum = 0.0;
			for (size_t t = 0; t < n_keys; ++t)
			{
				double d = 0.0;
				for (size_t i = 0; i < hd; ++i) d += static_cast<double>(q[h * hd + i]) * K[t * kv_stride + i];
				s[t] = d * scale;
				m = std::max(m, s[t]);
			}
			for (size_t t = 0; t < n_keys; ++t) sum += (s[t] = std::exp(s[t] - m));
			for (size_t i = 0; i < hd; ++i)
			{
				double o = 0.0;
				for (size_t t = 0; t < n_keys; ++t) o += s[t] * V[t * kv_stride + i];
				out[h * hd + i] = static_cast<float>(o / sum);
			}
		}
	}

	// Blocks of keys merged in order give softmax attention over them all,
	// with F32 and with BF16 K/V, including a head size off the 8-wide path.
	bool test_nnc_attn_blocks()
	{
		std::mt19937 rng(7);
		std::normal_distribution<float> nd(0.0f, 1.0f);
		for (const bool bf16 : {false, true})
			for (const size_t hd : {size_t{64}, size_t{36}})
			{
				const size_t group = 3, n_kv = 2, kv_stride = n_kv * hd, n_keys = 600;
				const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
				std::vector<float> q(group * hd), K(n_keys * kv_stride), V(n_keys * kv_stride);
				for (float& x : q) x = nd(rng);
				for (float& x : K) x = 2.0f * nd(rng);
				for (float& x : V) x = nd(rng);
				std::vector<nnc_bf16_t> Kb(K.size()), Vb(V.size());
				for (size_t i = 0; i < K.size(); ++i)
				{
					Kb[i] = nnc_f32_to_bf16(K[i]);
					Vb[i] = nnc_f32_to_bf16(V[i]);
					if (bf16)
					{
						K[i] = nnc_bf16_to_f32(Kb[i]);
						V[i] = nnc_bf16_to_f32(Vb[i]);
					}
				}
				// KV head 1, in blocks of 100, 256 and 244 keys.
				const size_t esz = bf16 ? 2 : 4, off = hd;
				const auto* kp = bf16 ? static_cast<const void*>(Kb.data()) : K.data();
				const auto* vp = bf16 ? static_cast<const void*>(Vb.data()) : V.data();
				const size_t pw = group * NNC_ATTN_PART(hd);
				std::vector<float> scores(group * 256), acc(pw), part(pw), out(group * hd), ref(group * hd);
				size_t k0 = 0;
				for (const size_t n : {size_t{100}, size_t{256}, size_t{244}})
				{
					const auto* kb = static_cast<const unsigned char*>(kp) + (k0 * kv_stride + off) * esz;
					const auto* vb = static_cast<const unsigned char*>(vp) + (k0 * kv_stride + off) * esz;
					nnc_attn_block(q.data(), hd, kb, vb, bf16, kv_stride, n, hd, group, scale, scores.data(),
					               k0 == 0 ? acc.data() : part.data());
					if (k0 > 0) nnc_attn_merge(acc.data(), part.data(), hd, group);
					k0 += n;
				}
				nnc_attn_finish(out.data(), hd, acc.data(), hd, group);
				ref_attention(q.data(), K.data() + off, V.data() + off, kv_stride, n_keys, hd, group, scale,
				              ref.data());
				for (size_t i = 0; i < out.size(); ++i)
					if (std::fabs(out[i] - ref[i]) > 1e-4f * (1.0f + std::fabs(ref[i]))) return false;
			}
		return true;
	}

	// A dense Qwen3 in miniature with two attention layers, the first
	// sliding over `window` keys, the second global.
	std::string write_tiny_qwen3(const char* name, const int window)
	{
		constexpr int n_embd = TINY_EMBD, n_ff = 96, n_layer = 2;
		constexpr int head_dim = 32, n_head = 4, n_kv = 2;
		std::mt19937 rng(3);
		std::normal_distribution<float> nd(0.0f, 1.0f);
		auto rnd = [&](const size_t n, const float scale)
		{
			std::vector<float> v(n);
			for (float& x : v) x = nd(rng) * scale;
			return v;
		};
		auto gamma = [&](const size_t n)
		{
			std::vector<float> v(n);
			for (float& x : v) x = 1.0f + 0.1f * nd(rng);
			return v;
		};
		gguf_builder g;
		g.str("general.architecture", "qwen3");
		g.u32("qwen3.block_count", n_layer);
		g.u32("qwen3.context_length", 4096);
		g.u32("qwen3.embedding_length", n_embd);
		g.u32("qwen3.feed_forward_length", n_ff);
		g.u32("qwen3.attention.head_count", n_head);
		g.u32("qwen3.attention.head_count_kv", n_kv);
		g.u32("qwen3.attention.key_length", head_dim);
		g.f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
		g.f32("qwen3.rope.freq_base", 10000.0f);
		g.u32("qwen3.attention.sliding_window", static_cast<uint32_t>(window));
		g.bools("qwen3.attention.sliding_window_pattern", {true, false});
		std::vector<std::string> toks;
		std::vector<int32_t> types;
		for (int i = 0; i < 26; ++i)
		{
			toks.emplace_back(1, static_cast<char>('a' + i));
			types.push_back(1);
		}
		for (const char* t : {"ab", "cd"})
		{
			toks.emplace_back(t);
			types.push_back(1);
		}
		for (const char* t : {"<|im_start|>", "<|im_end|>"})
		{
			toks.emplace_back(t);
			types.push_back(3);
		}
		g.str("tokenizer.ggml.model", "gpt2");
		g.str("tokenizer.ggml.pre", "qwen2");
		g.strs("tokenizer.ggml.tokens", toks);
		g.i32s("tokenizer.ggml.token_type", types);
		g.strs("tokenizer.ggml.merges", {"a b", "c d"});

		g.add("token_embd.weight", {n_embd, TINY_VOCAB}, rnd(n_embd * TINY_VOCAB, 0.5f));
		g.add("output_norm.weight", {n_embd}, gamma(n_embd));
		for (int li = 0; li < n_layer; ++li)
		{
			const std::string p = "blk." + std::to_string(li) + ".";
			g.add(p + "attn_norm.weight", {n_embd}, gamma(n_embd));
			g.add(p + "ffn_norm.weight", {n_embd}, gamma(n_embd));
			g.add(p + "ffn_gate.weight", {n_embd, n_ff}, rnd(n_embd * n_ff, 0.15f));
			g.add(p + "ffn_up.weight", {n_embd, n_ff}, rnd(n_embd * n_ff, 0.15f));
			g.add(p + "ffn_down.weight", {n_ff, n_embd}, rnd(n_ff * n_embd, 0.1f));
			g.add(p + "attn_q.weight", {n_embd, n_head * head_dim}, rnd(n_embd * n_head * head_dim, 0.3f));
			g.add(p + "attn_k.weight", {n_embd, n_kv * head_dim}, rnd(n_embd * n_kv * head_dim, 0.3f));
			g.add(p + "attn_v.weight", {n_embd, n_kv * head_dim}, rnd(n_embd * n_kv * head_dim, 0.3f));
			g.add(p + "attn_q_norm.weight", {head_dim}, gamma(head_dim));
			g.add(p + "attn_k_norm.weight", {head_dim}, gamma(head_dim));
			g.add(p + "attn_output.weight", {n_head * head_dim, n_embd}, rnd(n_head * head_dim * n_embd, 0.1f));
		}
		return g.write(name);
	}

	int tiny_token(const int i) { return k_tiny_tokens[(i * 7 + i / 11) % 11]; }

	// Feed tokens [from, to) in batches of up to `batch`; true if each call
	// leaves the same final hidden states in both caches.
	bool forward_both(const nnc_model& m, model_kv_cache& a, model_kv_cache& b, const int from, const int to,
	                  const int batch)
	{
		int ids[16];
		for (int p = from; p < to;)
		{
			const int n = std::min(batch, to - p);
			for (int i = 0; i < n; ++i) ids[i] = tiny_token(p + i);
			if (!model_forward(m, a, ids, n, p) || !model_forward(m, b, ids, n, p)) return false;
			if (std::memcmp(a.sx.data(), b.sx.data(), static_cast<size_t>(n) * TINY_EMBD * sizeof(float)) != 0)
				return false;
			p += n;
		}
		return true;
	}

	// A sliding layer's ring buffer gives exactly what a full buffer does,
	// through many wraps and a step back, and refuses to go back further
	// than it holds.
	bool test_sliding_ring_matches_full()
	{
		const std::string path = write_tiny_qwen3("nnc-test-ring.gguf", 40);
		if (path.empty()) return false;
		bool ok = false;
		{
			nnc_model m;
			model_kv_cache ring, full;
			ok = model_load(path, m, NNC_WEIGHTS_Q8_0, nullptr, nullptr) && m.layers[0].sliding_window
				&& !m.layers[1].sliding_window && model_kv_init(m, ring, 1200, 16) && model_kv_init(m, full, 1200, 16);
			ok = ok && ring.bufs[0].rows == 256 && ring.bufs[0].window == 40 && ring.bufs[1].rows == 1200
				&& ring.max_rewind == 216;
			if (ok)
			{
				// The same cache with the sliding layer kept in full.
				model_kv_buffer& b = full.bufs[0];
				b.rows = 1200;
				b.window = 0;
				b.k.assign(static_cast<size_t>(b.rows) * 64 * 2, 0);
				b.v.assign(b.k.size(), 0);
				full.bytes += 2 * (b.k.size() - ring.bufs[0].k.size());
				full.max_rewind = std::numeric_limits<int>::max();
			}
			ok = ok && ring.bytes < full.bytes && forward_both(m, ring, full, 0, 700, 16)
				&& forward_both(m, ring, full, 700, 720, 1);
			ok = ok && model_kv_rewind(ring, 705) && model_kv_rewind(full, 705) && forward_both(m, ring, full, 705, 760, 16);
			ok = ok && !model_kv_rewind(ring, 760 - 217) && ring.cur_pos == 760 && model_kv_rewind(ring, 760 - 216)
				&& model_kv_rewind(ring, 0) && model_kv_rewind(full, 0) && forward_both(m, ring, full, 0, 300, 16);
		}
		remove_utf8(path);
		return ok;
	}

	// Decode attention over a long span is shared out by (KV head, block),
	// yet each token gets the bits a batch gives it - with F32 or BF16 K/V.
	// A session asked for any other K/V type is refused.
	bool test_attn_decode_matches_batch()
	{
		const std::string path = write_tiny_qwen3("nnc-test-decode.gguf", 0);
		if (path.empty()) return false;
		bool ok = false;
		{
			nnc_model m;
			ok = model_load(path, m, NNC_WEIGHTS_Q8_0, nullptr, nullptr);
			nnc_session_params sp = nnc_session_default_params();
			sp.kv_type = static_cast<nnc_kv_type>(2);
			nnc_session* s = nullptr;
			ok = ok && nnc_session_create(&m, &sp, &s) == NNC_ERR_INVALID_ARGUMENT && s == nullptr;
			for (const bool f32 : {false, true})
			{
				model_kv_cache a, b;
				ok = ok && model_kv_init(m, a, 1200, 16, f32) && model_kv_init(m, b, 1200, 16, f32)
					&& forward_both(m, a, b, 0, 1100, 16);
				int ids[3] = {tiny_token(1100), tiny_token(1101), tiny_token(1102)};
				ok = ok && model_forward(m, a, ids, 3, 1100);
				const std::vector<float> batch(a.sx.begin(), a.sx.begin() + 3 * TINY_EMBD);
				for (int t = 0; t < 3 && ok; ++t)
					ok = model_forward(m, b, ids + t, 1, 1100 + t)
						&& std::memcmp(b.sx.data(), batch.data() + t * TINY_EMBD, TINY_EMBD * sizeof(float)) == 0;
			}
		}
		remove_utf8(path);
		return ok;
	}

	// A batch through the Gated DeltaNet layers (the convolution window
	// spanning the saved history and the batch, the recurrence carried from
	// token to token) gives each token the state that one token at a time
	// does.
	bool test_qwen35_batch_matches_steps()
	{
		const std::string path = write_tiny_qwen35("nnc-test-qwen35.gguf");
		if (path.empty()) return false;
		bool ok = false;
		{
			nnc_model m;
			model_kv_cache a, b;
			ok = model_load(path, m, NNC_WEIGHTS_Q8_0, nullptr, nullptr) && m.hparams.n_layer == 4
				&& m.hparams.n_layer_rs == 3 && m.layers[3].attn_gated && m.layers[3].rs_index < 0
				&& model_kv_init(m, a, 64, 16) && model_kv_init(m, b, 64, 16)
				&& model_forward(m, a, k_tiny_tokens, 11, 0);
			for (int t = 0; t < 11 && ok; ++t)
			{
				ok = model_forward(m, b, k_tiny_tokens + t, 1, t);
				for (int i = 0; i < TINY_EMBD && ok; ++i)
					ok = close(b.sx[static_cast<size_t>(i)], a.sx[static_cast<size_t>(t * TINY_EMBD + i)], 1e-4f);
			}
		}
		remove_utf8(path);
		return ok;
	}

	// The recurrent state cannot be cut back: a saved copy puts it, and the
	// position, back exactly, and a sequence that starts again at position 0
	// starts from a clear state.
	bool test_qwen35_state_save_restore()
	{
		const std::string path = write_tiny_qwen35("nnc-test-qwen35-state.gguf");
		if (path.empty()) return false;
		bool ok = false;
		{
			nnc_model m;
			model_kv_cache c;
			ok = model_load(path, m, NNC_WEIGHTS_Q8_0, nullptr, nullptr) && model_kv_init(m, c, 64, 16)
				&& model_forward(m, c, k_tiny_tokens, 6, 0);
			model_state_save(c);
			const size_t rows = 5 * TINY_EMBD;
			ok = ok && model_forward(m, c, k_tiny_tokens + 6, 5, 6);
			const std::vector<float> first(c.sx.begin(), c.sx.begin() + static_cast<ptrdiff_t>(rows));
			ok = ok && model_state_restore(c) && c.cur_pos == 6 && model_forward(m, c, k_tiny_tokens + 6, 5, 6)
				&& std::memcmp(first.data(), c.sx.data(), rows * sizeof(float)) == 0;
			c.cur_pos = 0;
			ok = ok && model_forward(m, c, k_tiny_tokens, 11, 0);
			for (int i = 0; i < TINY_EMBD && ok; ++i)
				ok = close(c.sx[static_cast<size_t>(10 * TINY_EMBD + i)], first[static_cast<size_t>(4 * TINY_EMBD + i)],
				           1e-5f);
		}
		remove_utf8(path);
		return ok;
	}

	// A session over a recurrent model: a lost prediction is rebuilt from the
	// state saved when a multi-batch call began, a cancelled call changes
	// nothing, and a reset starts afresh - each giving exactly the logits of
	// a straight run.
	bool test_qwen35_session_rewind()
	{
		const std::string path = write_tiny_qwen35("nnc-test-qwen35-session.gguf");
		if (path.empty()) return false;
		bool ok = false;
		{
			nnc_model m;
			nnc_session s;
			nnc_session_params sp = nnc_session_default_params();
			sp.n_ctx = 64;
			const size_t n = TINY_VOCAB;
			auto logits = [&]
			{
				const float* l = session_logits(s);
				return l ? std::vector<float>(l, l + n) : std::vector<float>();
			};
			// Two runs of 20 tokens: each takes two prefill batches.
			std::vector<int> a, b;
			for (int i = 0; i < 20; ++i)
			{
				a.push_back(k_tiny_tokens[i % 11]);
				b.push_back(k_tiny_tokens[(i * 7 + 3) % 11]);
			}
			ok = model_load(path, m, NNC_WEIGHTS_Q8_0, nullptr, nullptr) && session_init(s, m, sp)
				&& session_eval(s, a.data(), 20) == NNC_OK;
			const std::vector<float> la = ok ? logits() : std::vector<float>();
			ok = ok && session_eval(s, b.data(), 20) == NNC_OK && s.cache.saved_pos == 20;
			const std::vector<float> lb = ok ? logits() : std::vector<float>();

			s.have_hidden = false; // as a cancel can leave it
			ok = ok && logits() == lb && s.cache.cur_pos == 40;
			s.cancel.store(true);
			ok = ok && session_eval(s, a.data(), 20) == NNC_ERR_CANCELLED;
			s.cancel.store(false);
			ok = ok && s.tokens.size() == 40 && logits() == lb;
			session_reset(s);
			ok = ok && session_eval(s, a.data(), 20) == NNC_OK && logits() == la;
		}
		remove_utf8(path);
		return ok;
	}

	// Files nnc would run wrongly are refused: a Qwen model holding a tensor
	// nnc has no use for, and the Qwen architectures it does not implement.
	bool test_qwen_refusals()
	{
		const std::string path = write_tiny_qwen35("nnc-test-qwen35-extra.gguf", "blk.0.attn_qkv.scale");
		if (path.empty()) return false;
		bool ok;
		{
			nnc_model m;
			ok = !model_load(path, m, NNC_WEIGHTS_Q8_0, nullptr, nullptr)
				&& nnc_last_status() == NNC_ERR_UNSUPPORTED_MODEL
				&& std::strstr(nnc_last_error(), "blk.0.attn_qkv.scale") != nullptr;
		}
		remove_utf8(path);
		for (const char* arch : {"qwen4exp", "qwen35moe"})
		{
			std::vector<unsigned char> value;
			append_str(value, arch);
			const std::string p = write_one_kv_gguf("nnc-test-qwen-arch.gguf", "general.architecture", 8 /* STRING */,
			                                        value.data(), value.size());
			{
				nnc_model m;
				ok = ok && !p.empty() && !model_load(p, m, NNC_WEIGHTS_Q8_0, nullptr, nullptr)
					&& nnc_last_status() == NNC_ERR_UNSUPPORTED_MODEL;
			}
			remove_utf8(p);
		}
		return ok;
	}
}

int run_tests()
{
	std::printf("nnc: running tests\n");

	report("cpu_has_avx2_fma", test_cpu_has_avx2_fma());
	report("nnc_api_defaults", test_nnc_api_defaults());
	report("nnc_api_rejects_invalid_handles", test_nnc_api_rejects_invalid_handles());
	report("nnc_last_error_cuts_between_characters", test_nnc_last_error_cuts_between_characters());
	report("utf8_take", test_utf8_take());
	report("platform_replace_existing_file", test_platform_replace_existing_file());
	report("jit_return_42", test_jit_return_42());
	report("jit_add_two_ints", test_jit_add_two_ints());
	report("jit_dot_f32", test_jit_dot_f32());
	report("jit_gemv_f32", test_jit_gemv_f32());
	report("jit_gemv_cache_reuse", test_jit_gemv_cache_reuse());
	report("nnc_gelu_f32", test_nnc_gelu_f32());
	report("nnc_softmax_f32", test_nnc_softmax_f32());
	report("nnc_layernorm_f32", test_nnc_layernorm_f32());
	report("nnc_rmsnorm_f32", test_nnc_rmsnorm_f32());
	report("bf16_round_trip", test_bf16_round_trip());
	report("bf16_to_f32_row", test_bf16_to_f32_row());
	report("nnc_gemv_bf16w_f32x", test_nnc_gemv_bf16w_f32x());
	report("nnc_gemv_bf16w_argmax_f32x", test_nnc_gemv_bf16w_argmax_f32x());
	report("nnc_q8_0_quantize_roundtrip", test_nnc_q8_0_quantize_roundtrip());
	report("nnc_gemv_q8_0_f32x", test_nnc_gemv_q8_0_f32x());
	report("nnc_q4_s_quantize_roundtrip", test_nnc_q4_s_quantize_roundtrip());
	report("nnc_block_sums_f32", test_nnc_block_sums_f32());
	report("nnc_gemv_q4_s_f32x", test_nnc_gemv_q4_s_f32x());
	report("nnc_dequantize_q8_0", test_nnc_dequantize_q8_0());
	report("nnc_dequantize_legacy", test_nnc_dequantize_legacy());
	report("nnc_dequantize_q4_k", test_nnc_dequantize_q4_k());
	report("nnc_dequantize_q5_k", test_nnc_dequantize_q5_k());
	report("fp16_subnormal_dequant", test_fp16_subnormal_dequant());
	report("nnc_dequantize_q6_k", test_nnc_dequantize_q6_k());
	report("nnc_swiglu_f32", test_nnc_swiglu_f32());
	report("nnc_mul_sigmoid_f32", test_nnc_mul_sigmoid_f32());
	report("nnc_l2norm_multi_f32", test_nnc_l2norm_multi_f32());
	report("nnc_conv1d_silu_f32", test_nnc_conv1d_silu_f32());
	report("nnc_delta_rule_f32", test_nnc_delta_rule_f32());
	report("nnc_rope_f32", test_nnc_rope_f32());
	report("nnc_softcap_f32", test_nnc_softcap_f32());
	report("nnc_embed_row_bf16", test_nnc_embed_row_bf16());
	report("nnc_embed_row_q8_0", test_nnc_embed_row_q8_0());
	report("nnc_embed_row_q4_s", test_nnc_embed_row_q4_s());
	report("gguf_load_utf8_path", test_gguf_load_utf8_path());
	report("gguf_load_rejects_oversized_array", test_gguf_load_rejects_oversized_array());
	report("gguf_load_rejects_deep_array_nesting", test_gguf_load_rejects_deep_array_nesting());
	report("gguf_load_rejects_metadata_bomb", test_gguf_load_rejects_metadata_bomb());
	report("gguf_load_rejects_impossible_shapes", test_gguf_load_rejects_impossible_shapes());
	report("moe_top_k", test_moe_top_k());
	report("attn_span", test_attn_span());
	report("embedding_pool_normalize", test_embedding_pool_normalize());
	report("vocab_user_defined_tokens", test_vocab_user_defined_tokens());
	report("vocab_translation_languages", test_vocab_translation_languages());
	report("vocab_pretokenize_qwen", test_vocab_pretokenize_qwen());
	report("vocab_chatml", test_vocab_chatml());
	report("chat_turn_thinking", test_chat_turn_thinking());
	report("thought_split", test_thought_split());
	report("qwen35_batch_matches_steps", test_qwen35_batch_matches_steps());
	report("qwen35_state_save_restore", test_qwen35_state_save_restore());
	report("qwen35_session_rewind", test_qwen35_session_rewind());
	report("qwen_refusals", test_qwen_refusals());
	report("nnc_attn_blocks", test_nnc_attn_blocks());
	report("emitter_high_registers", test_emitter_high_registers());
	report("gemv_batch_matches_rows", test_gemv_batch_matches_rows());
	report("sliding_ring_matches_full", test_sliding_ring_matches_full());
	report("attn_decode_matches_batch", test_attn_decode_matches_batch());

	std::printf("nnc: %d passed, %d failed\n", g_pass, g_fail);
	return g_fail;
}
