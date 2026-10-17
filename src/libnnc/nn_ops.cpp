// nnc — neural-net op implementations.
// Hand-written AVX2/FMA SIMD kernels (gelu, softmax, layernorm, rmsnorm,
// rope, elementwise) plus thin wrappers that route the BF16 / Q8_0 gemvs
// to JITted kernels from jit_kernel.cpp and parallelise them across a
// small static worker pool.

#include "nn_ops.h"

#include "jit_kernel.h"
#include "runtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Horizontal reduce: sum of 8 floats in a ymm register.
static inline float hsum_ymm(const __m256 v)
{
	const __m128 hi = _mm256_extractf128_ps(v, 1);
	const __m128 lo = _mm256_castps256_ps128(v);
	__m128 s = _mm_add_ps(lo, hi);
	s = _mm_hadd_ps(s, s);
	s = _mm_hadd_ps(s, s);
	return _mm_cvtss_f32(s);
}

// Horizontal reduce: max of 8 floats in a ymm register.
static inline float hmax_ymm(const __m256 v)
{
	const __m128 hi = _mm256_extractf128_ps(v, 1);
	const __m128 lo = _mm256_castps256_ps128(v);
	__m128 m = _mm_max_ps(lo, hi);
	m = _mm_max_ps(m, _mm_movehl_ps(m, m));
	m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x55));
	return _mm_cvtss_f32(m);
}

// AVX2 exp(x) approximation. Uses range reduction x = n*ln(2) + r with r
// in [-ln(2)/2, ln(2)/2], then a 5-degree minimax polynomial for exp(r),
// followed by ldexp via integer add into the FP32 exponent. Max relative
// error ~3e-7 across [-87.3, 88.7]; outside that range the result
// saturates to 0 / +inf which is exactly what softmax wants for the
// negative-large / positive-large extremes (the negative side flushes
// underflows to 0; the positive side cannot occur because softmax has
// already subtracted the max).
static inline __m256 nnc_exp_ps(__m256 x)
{
	const __m256 v_lo = _mm256_set1_ps(-87.336544f); // ~ln(FLT_MIN)
	const __m256 v_hi = _mm256_set1_ps(88.722839f); // ~ln(FLT_MAX)
	x = _mm256_min_ps(_mm256_max_ps(x, v_lo), v_hi);

	const __m256 LOG2EF = _mm256_set1_ps(1.44269504088896341f);
	const __m256 C1 = _mm256_set1_ps(0.693359375f); // ln(2) hi
	const __m256 C2 = _mm256_set1_ps(-2.12194440e-4f); // ln(2) lo
	const __m256 ONE = _mm256_set1_ps(1.0f);

	// fx = round(x * log2(e))
	__m256 fx = _mm256_mul_ps(x, LOG2EF);
	fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	// r = x - fx*ln(2)  (computed in two halves for accuracy)
	__m256 r = _mm256_fnmadd_ps(fx, C1, x);
	r = _mm256_fnmadd_ps(fx, C2, r);

	// 5-degree polynomial for exp(r) on [-ln2/2, ln2/2].
	const __m256 P0 = _mm256_set1_ps(1.9875691500e-4f);
	const __m256 P1 = _mm256_set1_ps(1.3981999507e-3f);
	const __m256 P2 = _mm256_set1_ps(8.3334519073e-3f);
	const __m256 P3 = _mm256_set1_ps(4.1665795894e-2f);
	const __m256 P4 = _mm256_set1_ps(1.6666665459e-1f);
	const __m256 P5 = _mm256_set1_ps(5.0000001201e-1f);

	__m256 y = _mm256_fmadd_ps(P0, r, P1);
	y = _mm256_fmadd_ps(y, r, P2);
	y = _mm256_fmadd_ps(y, r, P3);
	y = _mm256_fmadd_ps(y, r, P4);
	y = _mm256_fmadd_ps(y, r, P5);
	const __m256 r2 = _mm256_mul_ps(r, r);
	y = _mm256_fmadd_ps(y, r2, r);
	y = _mm256_add_ps(y, ONE);

	// ldexp: result *= 2^fx
	const __m256i emm = _mm256_slli_epi32(
		_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(0x7f)), 23);
	return _mm256_mul_ps(y, _mm256_castsi256_ps(emm));
}

void nnc_gelu_f32(float* y, const float* x, const size_t n)
{
	// GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
	//
	// SIMD tanh: Padé[7/6] over [-5, 5], saturates to sign(u) for |u| >= 5.
	//   tanh(x) ~= x*(135135 + 17325*x^2 + 378*x^4 + x^6)
	//             / (135135 + 62370*x^2 + 3150*x^4 + 28*x^6)
	// Accuracy: < 5e-4 absolute in GELU output (tested).
	const __m256 v_a = _mm256_set1_ps(0.044715f);
	const __m256 v_c = _mm256_set1_ps(0.79788456080286535f); // sqrt(2/pi)
	const __m256 v_half = _mm256_set1_ps(0.5f);
	const __m256 v_one = _mm256_set1_ps(1.0f);
	const __m256 v_135135 = _mm256_set1_ps(135135.0f);
	const __m256 v_17325 = _mm256_set1_ps(17325.0f);
	const __m256 v_378 = _mm256_set1_ps(378.0f);
	const __m256 v_62370 = _mm256_set1_ps(62370.0f);
	const __m256 v_3150 = _mm256_set1_ps(3150.0f);
	const __m256 v_28 = _mm256_set1_ps(28.0f);
	const __m256 v_5 = _mm256_set1_ps(5.0f);
	const __m256 v_sign_mask = _mm256_set1_ps(-0.0f);

	size_t i = 0;
	const size_t np = n & ~size_t{7};
	for (; i < np; i += 8)
	{
		const __m256 v = _mm256_loadu_ps(x + i);
		const __m256 v2 = _mm256_mul_ps(v, v);
		const __m256 v3 = _mm256_mul_ps(v2, v);
		const __m256 u = _mm256_mul_ps(v_c, _mm256_fmadd_ps(v_a, v3, v));

		const __m256 u2 = _mm256_mul_ps(u, u);
		const __m256 u4 = _mm256_mul_ps(u2, u2);
		const __m256 u6 = _mm256_mul_ps(u4, u2);

		// num = u*(135135 + 17325*u^2 + 378*u^4 + u^6)
		__m256 num = _mm256_add_ps(u6,
		                           _mm256_fmadd_ps(v_378, u4,
		                                           _mm256_fmadd_ps(v_17325, u2, v_135135)));
		num = _mm256_mul_ps(u, num);
		// den = 135135 + 62370*u^2 + 3150*u^4 + 28*u^6
		const __m256 den = _mm256_fmadd_ps(v_28, u6,
		                                   _mm256_fmadd_ps(v_3150, u4,
		                                                   _mm256_fmadd_ps(v_62370, u2, v_135135)));
		const __m256 t_pade = _mm256_div_ps(num, den);

		const __m256 u_abs = _mm256_andnot_ps(v_sign_mask, u);
		const __m256 sat = _mm256_cmp_ps(u_abs, v_5, _CMP_GE_OQ);
		const __m256 sign_u = _mm256_or_ps(_mm256_and_ps(u, v_sign_mask), v_one);

		const __m256 t = _mm256_blendv_ps(t_pade, sign_u, sat);
		const __m256 r = _mm256_mul_ps(_mm256_mul_ps(v_half, v),
		                               _mm256_add_ps(v_one, t));
		_mm256_storeu_ps(y + i, r);
	}

	constexpr float SQRT_2_OVER_PI = 0.79788456080286535f;
	constexpr float GELU_COEF_A = 0.044715f;
	for (; i < n; ++i)
	{
		const float v = x[i];
		const float u = SQRT_2_OVER_PI * (v + GELU_COEF_A * v * v * v);
		y[i] = 0.5f * v * (1.0f + std::tanh(u));
	}
}

// ---- FP16 -> FP32 scalar decode ----------------------------------------

// IEEE 754 binary16 -> binary32, software only (no F16C dependency).
// Used by the K-quant dequantizers for their fp16 super-block scales.
static inline float fp16_to_fp32_scalar(const uint16_t h)
{
	const uint32_t s = (h >> 15) & 0x1u;
	const uint32_t e = (h >> 10) & 0x1Fu;
	const uint32_t m = h & 0x3FFu;

	uint32_t bits;
	if (e == 0)
	{
		if (m == 0)
		{
			bits = s << 31;
		}
		else
		{
			// subnormal: normalize. mm initially equals m; shift left until
			// bit 10 (0x400) is set — that bit becomes the implicit leading 1.
			// Number of shifts performed = 10 - (position of MSB in m).
			// The final exponent of the half-precision value is
			//   -14 - shifts (since min normal half = 2^-14, and each shift
			//   halves the magnitude).
			// Biased single-precision exponent = (-14 - shifts) + 127.
			uint32_t mm = m;
			uint32_t shifts = 0;
			while ((mm & 0x400u) == 0)
			{
				mm <<= 1;
				++shifts;
			}
			mm &= 0x3FFu;
			bits = (s << 31)
				| (static_cast<uint32_t>(127 - 14 - shifts) << 23)
				| (mm << 13);
		}
	}
	else if (e == 31)
	{
		bits = (s << 31) | (0xFFu << 23) | (m << 13);
	}
	else
	{
		bits = (s << 31) | ((e + (127 - 15)) << 23) | (m << 13);
	}

	float f;
	std::memcpy(&f, &bits, 4);
	return f;
}

namespace
{
	jit_kernel_cache& global_cache()
	{
		static jit_kernel_cache c;
		return c;
	}
}

// ---- mixture-of-experts routing ----------------------------------------

void nnc_moe_top_k(const float* p, const int n, const int k, int* idx, float* w)
{
	// k is small (8 of 128 for Gemma 4), so k passes of a linear scan beat
	// sorting the whole distribution.
	float sum = 0.0f;
	for (int j = 0; j < k; ++j)
	{
		int best = -1;
		for (int e = 0; e < n; ++e)
		{
			bool taken = false;
			for (int q = 0; q < j && !taken; ++q) taken = idx[q] == e;
			if (!taken && (best < 0 || p[e] > p[best])) best = e;
		}
		idx[j] = best;
		w[j] = p[best];
		sum += p[best];
	}
	// As llama.cpp does: never divide by less than the smallest normal FP16.
	const float denom = std::max(sum, 6.103515625e-5f);
	for (int j = 0; j < k; ++j) w[j] /= denom;
}

// ---- softmax (in place) ------------------------------------------------

void nnc_softmax_f32_inplace(float* p, const size_t n)
{
	if (n == 0) return;

	constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

	// Pass 1: find max via AVX2.
	float m = NEG_INF;
	{
		size_t i = 0;
		if (n >= 8)
		{
			__m256 vmax = _mm256_loadu_ps(p);
			i = 8;
			for (; i + 8 <= n; i += 8)
			{
				vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(p + i));
			}
			m = hmax_ymm(vmax);
		}
		for (; i < n; ++i)
		{
			if (p[i] > m) m = p[i];
		}
	}

	// Pass 2: exp(p[i] - m), accumulate sum. AVX2 polynomial nnc_exp_ps
	// keeps softmax fully vectorised. nnc_exp_ps clamps its argument to
	// ~ln(FLT_MIN), so -INFINITY lanes would come out as ~1e-38 rather
	// than 0; mask them explicitly to honour the causal-mask contract.
	double sum = 0.0;
	{
		const __m256 vm = _mm256_set1_ps(m);
		const __m256 vninf = _mm256_set1_ps(NEG_INF);
		__m256 vsum = _mm256_setzero_ps();
		size_t i = 0;
		for (; i + 8 <= n; i += 8)
		{
			const __m256 raw = _mm256_loadu_ps(p + i);
			__m256 v = nnc_exp_ps(_mm256_sub_ps(raw, vm));
			v = _mm256_andnot_ps(_mm256_cmp_ps(raw, vninf, _CMP_EQ_OQ), v);
			_mm256_storeu_ps(p + i, v);
			vsum = _mm256_add_ps(vsum, v);
		}
		sum = static_cast<double>(hsum_ymm(vsum));
		for (; i < n; ++i)
		{
			if (p[i] == NEG_INF)
			{
				p[i] = 0.0f;
			}
			else
			{
				const float v = std::exp(p[i] - m);
				p[i] = v;
				sum += v;
			}
		}
	}

	// Pass 3: scale by 1/sum via AVX2. If `sum` is zero (e.g. every input
	// was -INFINITY because the entire row is masked), the outputs are
	// already zero from pass 2 — guard against producing NaN.
	const float inv = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
	const __m256 vinv = _mm256_set1_ps(inv);
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		_mm256_storeu_ps(p + i, _mm256_mul_ps(_mm256_loadu_ps(p + i), vinv));
	}
	for (; i < n; ++i) p[i] *= inv;
}

// ---- blocked attention --------------------------------------------------

namespace
{
	inline __m256 load8_kv(const float* p) { return _mm256_loadu_ps(p); }

	inline __m256 load8_kv(const uint16_t* p)
	{
		const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
		return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
	}

	inline float kv_scalar(const float* p) { return *p; }

	inline float kv_scalar(const uint16_t* p)
	{
		const uint32_t u = static_cast<uint32_t>(*p) << 16;
		float f;
		std::memcpy(&f, &u, sizeof f);
		return f;
	}

	// Scores of key row k against NH query heads at once, so the row is
	// loaded (and, for BF16, widened) once for all of them.
	template <size_t NH, typename T>
	void attn_scores(const float* q, const size_t q_stride, const T* k, const size_t hd, const float scale,
	                 float* scores, const size_t n_keys, const size_t t)
	{
		__m256 acc[NH];
		for (size_t g = 0; g < NH; ++g) acc[g] = _mm256_setzero_ps();
		const size_t hd8 = hd & ~size_t{7};
		for (size_t i = 0; i < hd8; i += 8)
		{
			const __m256 kv = load8_kv(k + i);
			for (size_t g = 0; g < NH; ++g)
				acc[g] = _mm256_fmadd_ps(_mm256_loadu_ps(q + g * q_stride + i), kv, acc[g]);
		}
		for (size_t g = 0; g < NH; ++g)
		{
			float s = hsum_ymm(acc[g]);
			for (size_t i = hd8; i < hd; ++i) s += q[g * q_stride + i] * kv_scalar(k + i);
			scores[g * n_keys + t] = s * scale;
		}
	}

	template <typename T>
	void attn_block_t(const float* q, const size_t q_stride, const T* K, const T* V, const size_t kv_stride,
	                  const size_t n_keys, const size_t hd, const size_t n_heads, const float scale,
	                  float* scores, float* part)
	{
		const size_t pw = NNC_ATTN_PART(hd);
		const size_t hd8 = hd & ~size_t{7};

		for (size_t t = 0; t < n_keys; ++t)
		{
			const T* k = K + t * kv_stride;
			size_t h = 0;
			for (; h + 4 <= n_heads; h += 4)
				attn_scores<4>(q + h * q_stride, q_stride, k, hd, scale, scores + h * n_keys, n_keys, t);
			const float* qh = q + h * q_stride;
			float* sh = scores + h * n_keys;
			switch (n_heads - h)
			{
			case 3: attn_scores<3>(qh, q_stride, k, hd, scale, sh, n_keys, t); break;
			case 2: attn_scores<2>(qh, q_stride, k, hd, scale, sh, n_keys, t); break;
			case 1: attn_scores<1>(qh, q_stride, k, hd, scale, sh, n_keys, t); break;
			default: break;
			}
		}

		// Each head's weights exp(s - m), with m the block's own max.
		for (size_t h = 0; h < n_heads; ++h)
		{
			float* s = scores + h * n_keys;
			float m = s[0];
			for (size_t t = 1; t < n_keys; ++t) m = std::max(m, s[t]);
			const __m256 vm = _mm256_set1_ps(m);
			__m256 vsum = _mm256_setzero_ps();
			size_t t = 0;
			for (; t + 8 <= n_keys; t += 8)
			{
				const __m256 w = nnc_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(s + t), vm));
				_mm256_storeu_ps(s + t, w);
				vsum = _mm256_add_ps(vsum, w);
			}
			float l = hsum_ymm(vsum);
			for (; t < n_keys; ++t)
			{
				s[t] = std::exp(s[t] - m);
				l += s[t];
			}
			part[h * pw + hd] = m;
			part[h * pw + hd + 1] = l;
		}

		// Weighted sum of V rows, t outermost so each row is read once for
		// all heads; the first row writes, the rest accumulate.
		for (size_t t = 0; t < n_keys; ++t)
		{
			const T* vr = V + t * kv_stride;
			for (size_t h = 0; h < n_heads; ++h)
			{
				const float w = scores[h * n_keys + t];
				const __m256 vw = _mm256_set1_ps(w);
				float* o = part + h * pw;
				size_t i = 0;
				if (t == 0)
				{
					for (; i < hd8; i += 8) _mm256_storeu_ps(o + i, _mm256_mul_ps(vw, load8_kv(vr + i)));
					for (; i < hd; ++i) o[i] = w * kv_scalar(vr + i);
				}
				else
				{
					for (; i < hd8; i += 8)
						_mm256_storeu_ps(o + i, _mm256_fmadd_ps(vw, load8_kv(vr + i), _mm256_loadu_ps(o + i)));
					for (; i < hd; ++i) o[i] += w * kv_scalar(vr + i);
				}
			}
		}
	}
}

void nnc_attn_block(const float* q, const size_t q_stride, const void* K, const void* V, const bool kv_bf16,
                    const size_t kv_stride, const size_t n_keys, const size_t head_dim, const size_t n_heads,
                    const float scale, float* scores, float* part)
{
	if (n_keys == 0 || n_heads == 0) return;
	if (kv_bf16)
		attn_block_t(q, q_stride, static_cast<const uint16_t*>(K), static_cast<const uint16_t*>(V), kv_stride,
		             n_keys, head_dim, n_heads, scale, scores, part);
	else
		attn_block_t(q, q_stride, static_cast<const float*>(K), static_cast<const float*>(V), kv_stride, n_keys,
		             head_dim, n_heads, scale, scores, part);
}

void nnc_attn_merge(float* acc, const float* part, const size_t head_dim, const size_t n_heads)
{
	const size_t pw = NNC_ATTN_PART(head_dim);
	const size_t hd8 = head_dim & ~size_t{7};
	for (size_t h = 0; h < n_heads; ++h)
	{
		float* a = acc + h * pw;
		const float* b = part + h * pw;
		const float m = std::max(a[head_dim], b[head_dim]);
		const float ca = std::exp(a[head_dim] - m);
		const float cb = std::exp(b[head_dim] - m);
		const __m256 vca = _mm256_set1_ps(ca);
		const __m256 vcb = _mm256_set1_ps(cb);
		size_t i = 0;
		for (; i < hd8; i += 8)
			_mm256_storeu_ps(a + i, _mm256_fmadd_ps(_mm256_loadu_ps(b + i), vcb,
			                                        _mm256_mul_ps(_mm256_loadu_ps(a + i), vca)));
		for (; i < head_dim; ++i) a[i] = a[i] * ca + b[i] * cb;
		a[head_dim] = m;
		a[head_dim + 1] = a[head_dim + 1] * ca + b[head_dim + 1] * cb;
	}
}

void nnc_attn_finish(float* out, const size_t out_stride, const float* acc, const size_t head_dim,
                     const size_t n_heads)
{
	const size_t pw = NNC_ATTN_PART(head_dim);
	for (size_t h = 0; h < n_heads; ++h)
	{
		const float* a = acc + h * pw;
		const float l = a[head_dim + 1];
		const float inv = l > 0.0f ? 1.0f / l : 0.0f;
		const __m256 vinv = _mm256_set1_ps(inv);
		float* o = out + h * out_stride;
		size_t i = 0;
		for (; i + 8 <= head_dim; i += 8) _mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), vinv));
		for (; i < head_dim; ++i) o[i] = a[i] * inv;
	}
}

// ---- layernorm (no affine; matches NNC_OP_NORM) ------------------

void nnc_layernorm_f32(float* y, const float* x, const size_t n, const float eps)
{
	if (n == 0) return;

	// Pass 1: sum.
	double mean = 0.0;
	{
		size_t i = 0;
		if (n >= 8)
		{
			__m256 vs = _mm256_setzero_ps();
			for (; i + 8 <= n; i += 8)
			{
				vs = _mm256_add_ps(vs, _mm256_loadu_ps(x + i));
			}
			mean = static_cast<double>(hsum_ymm(vs));
		}
		for (; i < n; ++i) mean += x[i];
		mean /= static_cast<double>(n);
	}

	// Pass 2: y[i] = x[i] - mean ; accumulate sum-of-squares.
	double sum2 = 0.0;
	{
		const __m256 vmean = _mm256_set1_ps(static_cast<float>(mean));
		__m256 vs2 = _mm256_setzero_ps();
		size_t i = 0;
		for (; i + 8 <= n; i += 8)
		{
			const __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
			_mm256_storeu_ps(y + i, v);
			vs2 = _mm256_fmadd_ps(v, v, vs2);
		}
		sum2 = static_cast<double>(hsum_ymm(vs2));
		for (; i < n; ++i)
		{
			const float v = static_cast<float>(x[i] - mean);
			y[i] = v;
			sum2 += static_cast<double>(v) * v;
		}
	}

	// Pass 3: y[i] *= scale.
	const float scale = static_cast<float>(1.0 / std::sqrt(sum2 / static_cast<double>(n) + eps));
	const __m256 vscale = _mm256_set1_ps(scale);
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		_mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), vscale));
	}
	for (; i < n; ++i) y[i] *= scale;
}

// ---- rmsnorm (no affine; y[i] = x[i] / sqrt(mean(x^2) + eps)) ----

void nnc_rmsnorm_f32(float* y, const float* x, const size_t n, const float eps)
{
	if (n == 0) return;

	// Pass 1: sum-of-squares.
	double sum2 = 0.0;
	{
		size_t i = 0;
		if (n >= 8)
		{
			__m256 vs2 = _mm256_setzero_ps();
			for (; i + 8 <= n; i += 8)
			{
				const __m256 v = _mm256_loadu_ps(x + i);
				vs2 = _mm256_fmadd_ps(v, v, vs2);
			}
			sum2 = static_cast<double>(hsum_ymm(vs2));
		}
		for (; i < n; ++i) sum2 += static_cast<double>(x[i]) * x[i];
	}

	// Pass 2: y[i] = x[i] * scale.
	const float scale = static_cast<float>(1.0 / std::sqrt(sum2 / static_cast<double>(n) + eps));
	const __m256 vscale = _mm256_set1_ps(scale);
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		_mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vscale));
	}
	for (; i < n; ++i) y[i] = x[i] * scale;
}

void nnc_rmsnorm_gamma_multi_f32(float* y, const float* x,
                                 const size_t n_groups, const size_t dim,
                                 const float* gamma, const float eps)
{
	if (n_groups == 0 || dim == 0) return;
	const float inv_dim = 1.0f / static_cast<float>(dim);
	for (size_t g = 0; g < n_groups; ++g)
	{
		const float* xg = x + g * dim;
		float* yg = y + g * dim;

		// Pass 1: sum-of-squares.
		double sum2 = 0.0;
		{
			size_t i = 0;
			if (dim >= 8)
			{
				__m256 vs2 = _mm256_setzero_ps();
				for (; i + 8 <= dim; i += 8)
				{
					const __m256 v = _mm256_loadu_ps(xg + i);
					vs2 = _mm256_fmadd_ps(v, v, vs2);
				}
				sum2 = static_cast<double>(hsum_ymm(vs2));
			}
			for (; i < dim; ++i) sum2 += static_cast<double>(xg[i]) * xg[i];
		}
		const float scale = static_cast<float>(
			1.0 / std::sqrt(sum2 * inv_dim + eps));

		// Pass 2 fused with gamma multiply: y[i] = x[i] * scale * gamma[i].
		const __m256 vscale = _mm256_set1_ps(scale);
		size_t i = 0;
		for (; i + 8 <= dim; i += 8)
		{
			__m256 v = _mm256_loadu_ps(xg + i);
			v = _mm256_mul_ps(v, vscale);
			v = _mm256_mul_ps(v, _mm256_loadu_ps(gamma + i));
			_mm256_storeu_ps(yg + i, v);
		}
		for (; i < dim; ++i) yg[i] = xg[i] * scale * gamma[i];
	}
}

float nnc_dot_f32_simd(const float* a, const float* b, const size_t n)
{
	size_t i = 0;
	__m256 s0 = _mm256_setzero_ps();
	__m256 s1 = _mm256_setzero_ps();
	__m256 s2 = _mm256_setzero_ps();
	__m256 s3 = _mm256_setzero_ps();
	for (; i + 32 <= n; i += 32)
	{
		s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 0),
		                     _mm256_loadu_ps(b + i + 0), s0);
		s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
		                     _mm256_loadu_ps(b + i + 8), s1);
		s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),
		                     _mm256_loadu_ps(b + i + 16), s2);
		s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),
		                     _mm256_loadu_ps(b + i + 24), s3);
	}
	for (; i + 8 <= n; i += 8)
	{
		s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
		                     _mm256_loadu_ps(b + i), s0);
	}
	s0 = _mm256_add_ps(s0, s1);
	s2 = _mm256_add_ps(s2, s3);
	s0 = _mm256_add_ps(s0, s2);
	float acc = hsum_ymm(s0);
	for (; i < n; ++i) acc += a[i] * b[i];
	return acc;
}

// ---- elementwise bias add ----------------------------------------------

void nnc_add_inplace_f32(float* y, const float* b, const size_t n)
{
	size_t i = 0;
	const size_t np = n & ~size_t{7};
	for (; i < np; i += 8)
	{
		_mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(y + i), _mm256_loadu_ps(b + i)));
	}
	for (; i < n; ++i) y[i] += b[i];
}

// ---- elementwise multiply ----------------------------------------------

void nnc_mul_inplace_f32(float* y, const float* s, const size_t n)
{
	size_t i = 0;
	const size_t np = n & ~size_t{7};
	for (; i < np; i += 8)
	{
		_mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), _mm256_loadu_ps(s + i)));
	}
	for (; i < n; ++i) y[i] *= s[i];
}

// ---- fused GELU * up (gated-MLP inner step) ----------------------------
//
// y[i] = gelu(gate[i]) * up[i]. Reuses the AVX2 Pade[7/6] tanh path from
// nnc_gelu_f32 — we just multiply the GELU result by up[i] before the
// store, eliminating one full read+write pass over the n_ff activation
// tile (~32KB / layer for Gemma E2B at n_ff=8192).
void nnc_gelu_mul_f32(float* y, const float* gate, const float* up, const size_t n)
{
	const __m256 v_a = _mm256_set1_ps(0.044715f);
	const __m256 v_c = _mm256_set1_ps(0.79788456080286535f);
	const __m256 v_half = _mm256_set1_ps(0.5f);
	const __m256 v_one = _mm256_set1_ps(1.0f);
	const __m256 v_135135 = _mm256_set1_ps(135135.0f);
	const __m256 v_17325 = _mm256_set1_ps(17325.0f);
	const __m256 v_378 = _mm256_set1_ps(378.0f);
	const __m256 v_62370 = _mm256_set1_ps(62370.0f);
	const __m256 v_3150 = _mm256_set1_ps(3150.0f);
	const __m256 v_28 = _mm256_set1_ps(28.0f);
	const __m256 v_5 = _mm256_set1_ps(5.0f);
	const __m256 v_sign_mask = _mm256_set1_ps(-0.0f);

	size_t i = 0;
	const size_t np = n & ~size_t{7};
	for (; i < np; i += 8)
	{
		const __m256 v = _mm256_loadu_ps(gate + i);
		const __m256 u_hi = _mm256_loadu_ps(up + i);
		const __m256 v2 = _mm256_mul_ps(v, v);
		const __m256 v3 = _mm256_mul_ps(v2, v);
		const __m256 u = _mm256_mul_ps(v_c, _mm256_fmadd_ps(v_a, v3, v));
		const __m256 u2 = _mm256_mul_ps(u, u);
		const __m256 u4 = _mm256_mul_ps(u2, u2);
		const __m256 u6 = _mm256_mul_ps(u4, u2);

		__m256 num = _mm256_add_ps(u6,
		                           _mm256_fmadd_ps(v_378, u4,
		                                           _mm256_fmadd_ps(v_17325, u2, v_135135)));
		num = _mm256_mul_ps(u, num);
		const __m256 den = _mm256_fmadd_ps(v_28, u6,
		                                   _mm256_fmadd_ps(v_3150, u4,
		                                                   _mm256_fmadd_ps(v_62370, u2, v_135135)));
		const __m256 t_pade = _mm256_div_ps(num, den);

		const __m256 u_abs = _mm256_andnot_ps(v_sign_mask, u);
		const __m256 sat = _mm256_cmp_ps(u_abs, v_5, _CMP_GE_OQ);
		const __m256 sign_u = _mm256_or_ps(_mm256_and_ps(u, v_sign_mask), v_one);
		const __m256 t = _mm256_blendv_ps(t_pade, sign_u, sat);
		__m256 r = _mm256_mul_ps(_mm256_mul_ps(v_half, v),
		                         _mm256_add_ps(v_one, t));
		r = _mm256_mul_ps(r, u_hi);
		_mm256_storeu_ps(y + i, r);
	}

	constexpr float SQRT_2_OVER_PI = 0.79788456080286535f;
	constexpr float GELU_COEF_A = 0.044715f;
	for (; i < n; ++i)
	{
		const float v = gate[i];
		const float uu = SQRT_2_OVER_PI * (v + GELU_COEF_A * v * v * v);
		y[i] = 0.5f * v * (1.0f + std::tanh(uu)) * up[i];
	}
}

// ---- BF16-weight, FP32-activation gemv ---------------------------------
//
// AVX2 path processes 8 cols at a time. BF16 -> F32 is a free shift:
//   f32 = (u32)(bf16) << 16
// Implemented as vpmovzxwd (zero-extend 8 u16 to 8 u32) + vpslld(16) +
// vfmadd231ps. Falls through to a scalar reference for the column tail
// or when cols < 8.
//
// Multi-threaded fast path: when the gemv is "big enough" (rows >= 256 and
// cols >= 256) and rows is a multiple of 4, we split the row axis across
// a static worker pool. Each worker calls a single shared (rows=4)-baked
// JIT kernel on its slice — this keeps the kernel cache to one entry per
// `cols` while still benefitting from the 4-row inner-loop. The pool is
// created lazily on first use (workers spin briefly then yield while
// idle, avoiding condition-variable wakeup latency on the hot path).

namespace
{
	// Lightweight bring-your-own-task pool. Idle workers spin on `ticket_`
	// for a couple of milliseconds — so back-to-back gemvs within a token
	// see sub-microsecond dispatch — and then park on a condition variable,
	// so a process sitting at a prompt costs no CPU. dispatch() is
	// serialised by a mutex: concurrent callers (two sessions on two
	// threads) take turns rather than corrupting each other's task.
	class nnc_gemv_pool
	{
	public:
		// fn(tid, n_threads, user): tid in [0, n_threads); the caller is tid 0.
		using task_fn_t = void (*)(int tid, int n_threads, const void* user);

		// Bounds the fixed-size per-thread arrays used by dispatch() and by
		// the argmax reduction below.
		static constexpr int MAX_THREADS = 64;

		static nnc_gemv_pool& global()
		{
			static nnc_gemv_pool p;
			return p;
		}

		// Total participating threads = caller + workers. A hint for sizing
		// work: dispatch() passes the authoritative count to the task.
		int n_threads() const { return n_total_.load(std::memory_order_relaxed); }

		// Restart with `total` threads (caller included). Waits for any
		// dispatch in flight.
		void configure(int total)
		{
			if (total < 1) total = 1;
			if (total > MAX_THREADS) total = MAX_THREADS;
			std::lock_guard<std::mutex> lk(dispatch_mu_);
			if (total == n_total_.load(std::memory_order_relaxed)) return;
			stop_workers();
			start_workers(total - 1);
		}

		// Run `fn(tid, n, user)` for tid in [0, n). The caller runs tid 0;
		// the workers run tids 1..n-1. Blocks until all participants finish.
		void dispatch(const task_fn_t fn, void* user)
		{
			std::lock_guard<std::mutex> lk(dispatch_mu_);
			const int nw = static_cast<int>(workers_.size());
			if (nw == 0)
			{
				fn(0, 1, user);
				return;
			}
			task_fn_ = fn;
			task_user_ = user;
			task_n_ = nw + 1;
			// Snapshot per-worker baselines, advance the ticket, wake any
			// parked workers, run the caller's share, then spin-wait.
			unsigned baselines[MAX_THREADS];
			for (int i = 0; i < nw; ++i)
				baselines[i] = workers_[i]->done.load(std::memory_order_acquire);
			ticket_.fetch_add(1, std::memory_order_seq_cst);
			// Pairs with the sleeper's increment-then-recheck: either we see
			// it parked and notify under the lock, or it sees the new ticket.
			if (sleepers_.load(std::memory_order_seq_cst) > 0)
			{
				std::lock_guard<std::mutex> sl(sleep_mu_);
				sleep_cv_.notify_all();
			}
			fn(0, nw + 1, user);
			for (int i = 0; i < nw; ++i)
			{
				while (workers_[i]->done.load(std::memory_order_acquire) == baselines[i])
					_mm_pause();
			}
		}

		~nnc_gemv_pool()
		{
			std::lock_guard<std::mutex> lk(dispatch_mu_);
			stop_workers();
		}

	private:
		struct slot
		{
			std::thread thr;
			std::atomic<unsigned> done{0};
		};

		// How long an idle worker spins before parking. Long enough to
		// bridge the gaps between gemvs in a token (norms, attention,
		// sampling); short enough that a waiting prompt is idle.
		static constexpr auto SPIN_BEFORE_PARK = std::chrono::milliseconds(2);

		nnc_gemv_pool() { start_workers(default_threads() - 1); }

		static int default_threads()
		{
			// Env override for perf experiments: NNC_THREADS = total
			// threads (workers + main). Otherwise default cap at 8.
			int total = 0;
			if (const char* e = std::getenv("NNC_THREADS")) total = std::atoi(e);
			if (total <= 0)
			{
				const unsigned hw = std::thread::hardware_concurrency();
				total = static_cast<int>(hw > 0 ? hw : 1u);
				if (total > 8) total = 8;
			}
			if (total > MAX_THREADS) total = MAX_THREADS;
			return total;
		}

	public:
		static int resolve_threads(const int requested)
		{
			if (requested <= 0) return default_threads();
			return requested > MAX_THREADS ? MAX_THREADS : requested;
		}

	private:
		void start_workers(const int n)
		{
			stop_.store(false, std::memory_order_release);
			const unsigned start = ticket_.load(std::memory_order_acquire);
			workers_.reserve(static_cast<size_t>(n));
			for (int i = 0; i < n; ++i)
			{
				workers_.push_back(std::make_unique<slot>());
				workers_.back()->thr = std::thread([this, i, start] { worker_loop(i, start); });
			}
			n_total_.store(n + 1, std::memory_order_relaxed);
		}

		void stop_workers()
		{
			if (workers_.empty()) return;
			stop_.store(true, std::memory_order_seq_cst);
			ticket_.fetch_add(1, std::memory_order_seq_cst);
			{
				std::lock_guard<std::mutex> sl(sleep_mu_);
				sleep_cv_.notify_all();
			}
			for (const auto& w : workers_) w->thr.join();
			workers_.clear();
			n_total_.store(1, std::memory_order_relaxed);
		}

		void worker_loop(const int i, unsigned last)
		{
			auto& w = *workers_[i];
			for (;;)
			{
				unsigned t;
				auto spin_start = std::chrono::steady_clock::now();
				for (int spins = 1;; ++spins)
				{
					t = ticket_.load(std::memory_order_acquire);
					if (t != last) break;
					if ((spins & 1023) != 0)
					{
						_mm_pause();
						continue;
					}
					if (std::chrono::steady_clock::now() - spin_start < SPIN_BEFORE_PARK)
						continue;
					std::unique_lock<std::mutex> sl(sleep_mu_);
					sleepers_.fetch_add(1, std::memory_order_seq_cst);
					sleep_cv_.wait(sl, [&] { return ticket_.load(std::memory_order_seq_cst) != last; });
					sleepers_.fetch_sub(1, std::memory_order_relaxed);
					spin_start = std::chrono::steady_clock::now();
				}
				last = t;
				if (stop_.load(std::memory_order_acquire)) return;
				task_fn_(i + 1, task_n_, task_user_);
				w.done.fetch_add(1, std::memory_order_release);
			}
		}

		std::mutex dispatch_mu_;
		std::vector<std::unique_ptr<slot>> workers_;
		std::atomic<int> n_total_{1};
		std::atomic<unsigned> ticket_{0};
		std::atomic<bool> stop_{false};
		std::atomic<int> sleepers_{0};
		std::mutex sleep_mu_;
		std::condition_variable sleep_cv_;
		task_fn_t task_fn_ = nullptr;
		void* task_user_ = nullptr;
		int task_n_ = 1;
	};

	// [first, last) of `total` items for thread `tid` of `n`: an even split,
	// with the remainder spread over the lowest tids.
	void split_range(const uint32_t total, const int tid, const int n, uint32_t& first, uint32_t& last)
	{
		const uint32_t per = total / static_cast<uint32_t>(n);
		const uint32_t rem = total % static_cast<uint32_t>(n);
		const uint32_t t = static_cast<uint32_t>(tid);
		first = t * per + std::min(t, rem);
		last = first + per + (t < rem ? 1u : 0u);
	}

	struct bf16_gemv_ctx
	{
		const uint8_t* W;
		const float* x;
		float* y;
		uint32_t cols;
		uint32_t total_quads; // rows / 4
		nnc_gemv_bf16w_f32x_fn fn4;
	};

	void bf16_gemv_worker(const int tid, const int nth, const void* u)
	{
		const auto* c = static_cast<const bf16_gemv_ctx*>(u);
		uint32_t q0, q1;
		split_range(c->total_quads, tid, nth, q0, q1);
		if (q0 == q1) return;
		const size_t row_stride = static_cast<size_t>(c->cols) * 2;
		const size_t group_stride = row_stride * 4;
		const uint8_t* W_tid = c->W + q0 * group_stride;
		float* y_tid = c->y + q0 * 4;
		for (uint32_t q = q0; q < q1; ++q)
		{
			c->fn4(W_tid, c->x, y_tid);
			W_tid += group_stride;
			y_tid += 4;
		}
	}
}

void nnc_gemv_bf16w_f32x(const void* W, const float* x, float* y,
                         const uint32_t rows, const uint32_t cols)
{
	if (rows == 0 || cols == 0) return;

	if ((cols & 7u) == 0)
	{
		// Parallel path: rows must be a multiple of 4 (so we can use the
		// shared 4-row JIT kernel) and the gemv must be big enough to
		// amortise dispatch overhead. Threshold ~256 rows captures every
		// significant Gemma weight (Q/K/V/O/gate/up/down + lm_head) but
		// keeps tiny kernels (e.g. PLE on small ple_dim) on the fast
		// single-threaded path.
		auto& pool = nnc_gemv_pool::global();
		if (pool.n_threads() > 1 && (rows & 3u) == 0 && rows >= 256 && cols >= 256)
		{
			const auto fn4 = global_cache().get_gemv_bf16w_f32x(4, cols);
			bf16_gemv_ctx ctx{
				static_cast<const uint8_t*>(W), x, y, cols,
				rows / 4, fn4
			};
			pool.dispatch(&bf16_gemv_worker, &ctx);
			return;
		}

		const auto fn = global_cache().get_gemv_bf16w_f32x(rows, cols);
		fn(W, x, y);
		return;
	}

	// Scalar fallback (cols not a multiple of 8).
	const auto Wb = static_cast<const uint16_t*>(W);
	for (uint32_t r = 0; r < rows; ++r)
	{
		const uint16_t* row = Wb + static_cast<size_t>(r) * cols;
		double s = 0.0;
		for (uint32_t k = 0; k < cols; ++k)
		{
			const uint32_t u = static_cast<uint32_t>(row[k]) << 16;
			float wf;
			std::memcpy(&wf, &u, 4);
			s += static_cast<double>(wf) * x[k];
		}
		y[r] = static_cast<float>(s);
	}
}

// ---- Streaming BF16-weight gemv + argmax -------------------------------
//
// For lm_head / final logits projection in greedy decode: compute the
// gemv 4 rows at a time directly into a tiny scratch and update a
// running argmax. Never materialises the full vocab-sized logits buffer.
// Saves ~1 MB of writes per token (vocab=262144) plus the softcap pass
// (softcap is monotonic and doesn't change argmax).

int nnc_gemv_bf16w_argmax_f32x(const void* W, const float* x,
                               const uint32_t rows, const uint32_t cols)
{
	if (rows == 0 || cols == 0) return -1;

	if ((rows & 3u) == 0 && (cols & 7u) == 0)
	{
		const auto fn4 = global_cache().get_gemv_bf16w_f32x(4, cols);
		const auto Wb = static_cast<const uint8_t*>(W);
		const size_t row_stride = static_cast<size_t>(cols) * 2; // BF16 bytes/row

		// Parallel path: split the row axis. Each worker computes its
		// local (best_val, best_idx); main reduces across workers. The
		// argmax is the dominant cost at lm_head (rows = vocab = 262144),
		// so this is the highest-payoff parallel site in the model.
		auto& pool = nnc_gemv_pool::global();
		if (pool.n_threads() > 1 && rows >= 256 && cols >= 256)
		{
			struct local
			{
				float bv;
				int bi;
			};
			// Every slot starts empty, so the reduction is right whatever
			// thread count the dispatch actually runs with.
			alignas(64) local locals[nnc_gemv_pool::MAX_THREADS]{};
			for (auto& l : locals)
			{
				l.bv = -std::numeric_limits<float>::infinity();
				l.bi = -1;
			}

			struct argmax_ctx
			{
				const uint8_t* W;
				const float* x;
				size_t row_stride;
				uint32_t total_quads;
				nnc_gemv_bf16w_f32x_fn fn4;
				local* locals;
			} c{
				Wb, x, row_stride, rows / 4, fn4, locals
			};

			pool.dispatch([](const int tid, const int nth, const void* u)
			{
				const auto* cc = static_cast<const argmax_ctx*>(u);
				uint32_t q0, q1;
				split_range(cc->total_quads, tid, nth, q0, q1);
				if (q0 == q1) return;
				alignas(16) float scratch[4];
				float bv = -std::numeric_limits<float>::infinity();
				int bi = -1;
				const size_t group_stride = cc->row_stride * 4;
				const uint8_t* Wp = cc->W + q0 * group_stride;
				for (uint32_t q = q0; q < q1; ++q)
				{
					cc->fn4(Wp, cc->x, scratch);
					const int base = static_cast<int>(q) * 4;
					if (scratch[0] > bv)
					{
						bv = scratch[0];
						bi = base;
					}
					if (scratch[1] > bv)
					{
						bv = scratch[1];
						bi = base + 1;
					}
					if (scratch[2] > bv)
					{
						bv = scratch[2];
						bi = base + 2;
					}
					if (scratch[3] > bv)
					{
						bv = scratch[3];
						bi = base + 3;
					}
					Wp += group_stride;
				}
				cc->locals[tid].bv = bv;
				cc->locals[tid].bi = bi;
			}, &c);

			int best = -1;
			float bv = -std::numeric_limits<float>::infinity();
			for (const auto& l : locals)
			{
				if (l.bi >= 0 && (best < 0 || l.bv > bv))
				{
					bv = l.bv;
					best = l.bi;
				}
			}
			return best >= 0 ? best : 0; // all-NaN logits: any token is as good
		}

		// Single-threaded fast path.
		alignas(16) float scratch[4];
		int best = 0;
		float bv = -std::numeric_limits<float>::infinity();
		for (uint32_t r = 0; r < rows; r += 4)
		{
			fn4(Wb + static_cast<size_t>(r) * row_stride, x, scratch);
			if (scratch[0] > bv)
			{
				bv = scratch[0];
				best = static_cast<int>(r);
			}
			if (scratch[1] > bv)
			{
				bv = scratch[1];
				best = static_cast<int>(r) + 1;
			}
			if (scratch[2] > bv)
			{
				bv = scratch[2];
				best = static_cast<int>(r) + 2;
			}
			if (scratch[3] > bv)
			{
				bv = scratch[3];
				best = static_cast<int>(r) + 3;
			}
		}
		return best;
	}

	// Generic fallback: full materialise then linear scan.
	std::vector<float> tmp(rows);
	nnc_gemv_bf16w_f32x(W, x, tmp.data(), rows, cols);
	int best = 0;
	float bv = tmp[0];
	for (uint32_t r = 1; r < rows; ++r)
	{
		if (tmp[r] > bv)
		{
			bv = tmp[r];
			best = static_cast<int>(r);
		}
	}
	return best;
}

// ---- Q8_0 quantize + gemv ---------------------------------------------
//
// Block size 32. For each row's block, scale = max(|w|) / 127 and
// q[k] = round(w[k] / scale) clamped to [-127, 127]. Standard Q8_0
// (matching what GGML / llama.cpp ships) but written into the SPLIT
// layout (qs first, scales second) so the JIT kernel can use cheap
// scaled-q + FMA without an interleaved per-block fp16 unpack.

void nnc_quantize_bf16_to_q8_0(const uint16_t* W_bf16, int8_t* qs, uint16_t* scales,
                               const size_t rows, const size_t cols)
{
	NNC_ASSERT(W_bf16 && qs && scales);
	NNC_ASSERT(cols > 0 && (cols % 32) == 0);

	float vals[32];
	const size_t n = rows * cols;
	for (size_t b = 0; b < n; b += 32)
	{
		for (size_t k = 0; k < 32; ++k) vals[k] = nnc_bf16_to_f32(W_bf16[b + k]);
		nnc_quantize_f32_to_q8_0(vals, qs + b, scales + b / 32, 32);
	}
}

void nnc_quantize_f32_to_q8_0(const float* src, int8_t* qs, uint16_t* scales,
                              const size_t n)
{
	NNC_ASSERT(src && qs && scales);
	NNC_ASSERT(n > 0 && (n % 32) == 0);

	for (size_t b = 0; b < n; b += 32)
	{
		float amax = 0.0f;
		for (size_t k = 0; k < 32; ++k)
		{
			const float a = std::fabs(src[b + k]);
			if (a > amax) amax = a;
		}
		// Round-trip the scale through BF16 first so the encoder quantizes
		// against exactly the value the kernel will multiply by.
		const float scale = nnc_bf16_to_f32(nnc_f32_to_bf16(amax / 127.0f));
		scales[b / 32] = nnc_f32_to_bf16(scale);
		const float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
		for (size_t k = 0; k < 32; ++k)
		{
			int q = static_cast<int>(std::lrintf(src[b + k] * inv));
			if (q > 127) q = 127;
			if (q < -127) q = -127;
			qs[b + k] = static_cast<int8_t>(q);
		}
	}
}

void nnc_quantize_f32_to_q4_s(const float* src, uint8_t* qs,
                              uint16_t* scales, uint16_t* biases, const size_t n)
{
	NNC_ASSERT(src && qs && scales && biases);
	NNC_ASSERT(n > 0 && (n % 32) == 0);

	for (size_t b = 0; b < n; b += 32)
	{
		float lo = src[b], hi = src[b];
		for (size_t k = 1; k < 32; ++k)
		{
			const float v = src[b + k];
			if (v < lo) lo = v;
			if (v > hi) hi = v;
		}
		// Round-trip scale and bias through BF16 before quantizing the
		// values, so the encoder sees exactly what the kernel will use.
		const float scale = nnc_bf16_to_f32(nnc_f32_to_bf16((hi - lo) / 15.0f));
		const float bias = nnc_bf16_to_f32(nnc_f32_to_bf16(lo));
		scales[b / 32] = nnc_f32_to_bf16(scale);
		biases[b / 32] = nnc_f32_to_bf16(bias);
		const float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

		// byte i packs element i (low nibble) and element i+16 (high).
		uint8_t* blk = qs + b / 2;
		for (size_t k = 0; k < 16; ++k)
		{
			int q0 = static_cast<int>(std::lrintf((src[b + k] - bias) * inv));
			int q1 = static_cast<int>(std::lrintf((src[b + k + 16] - bias) * inv));
			if (q0 < 0) q0 = 0;
			if (q0 > 15) q0 = 15;
			if (q1 < 0) q1 = 0;
			if (q1 > 15) q1 = 15;
			blk[k] = static_cast<uint8_t>(q0 | (q1 << 4));
		}
	}
}

float nnc_dot_bf16_f32_simd(const uint16_t* a, const float* b, const size_t n)
{
	__m256 acc = _mm256_setzero_ps();
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
		const __m256i wide = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16);
		acc = _mm256_fmadd_ps(_mm256_castsi256_ps(wide), _mm256_loadu_ps(b + i), acc);
	}
	float s = hsum_ymm(acc);
	for (; i < n; ++i) s += nnc_bf16_to_f32(a[i]) * b[i];
	return s;
}

void nnc_block_sums_f32(const float* x, float* out, const size_t n)
{
	NNC_ASSERT(n > 0 && (n % 32) == 0);
	for (size_t b = 0; b < n; b += 32)
	{
		__m256 acc = _mm256_loadu_ps(x + b);
		acc = _mm256_add_ps(acc, _mm256_loadu_ps(x + b + 8));
		acc = _mm256_add_ps(acc, _mm256_loadu_ps(x + b + 16));
		acc = _mm256_add_ps(acc, _mm256_loadu_ps(x + b + 24));
		out[b / 32] = hsum_ymm(acc);
	}
}

namespace
{
	struct q8_gemv_ctx
	{
		const int8_t* qs;
		const uint16_t* scales;
		const float* x;
		float* y;
		uint32_t cols;
		uint32_t rows;
		nnc_gemv_q8_0_1row_fn fn1;
	};

	void q8_gemv_worker(const int tid, const int nth, const void* u)
	{
		const auto* c = static_cast<const q8_gemv_ctx*>(u);
		uint32_t r0, r1;
		split_range(c->rows, tid, nth, r0, r1);
		if (r0 == r1) return;
		const size_t qs_stride = c->cols;
		const size_t scales_stride = c->cols / 32;
		const int8_t* qs_tid = c->qs + r0 * qs_stride;
		const uint16_t* scales_tid = c->scales + r0 * scales_stride;
		float* y_tid = c->y + r0;
		for (uint32_t r = r0; r < r1; ++r)
		{
			c->fn1(qs_tid, c->x, y_tid, scales_tid);
			qs_tid += qs_stride;
			scales_tid += scales_stride;
			y_tid += 1;
		}
	}
}

void nnc_gemv_q8_0_f32x(const int8_t* qs, const uint16_t* scales,
                        const float* x, float* y,
                        const uint32_t rows, const uint32_t cols)
{
	if (rows == 0 || cols == 0) return;
	NNC_ASSERT((cols % 32) == 0);

	const auto fn1 = global_cache().get_gemv_q8_0_1row(cols);

	auto& pool = nnc_gemv_pool::global();
	if (pool.n_threads() > 1 && rows >= 256 && cols >= 256)
	{
		q8_gemv_ctx ctx{qs, scales, x, y, cols, rows, fn1};
		pool.dispatch(&q8_gemv_worker, &ctx);
		return;
	}

	const size_t qs_stride = cols;
	const size_t scales_stride = cols / 32;
	for (uint32_t r = 0; r < rows; ++r)
	{
		fn1(qs + r * qs_stride, x, y + r, scales + r * scales_stride);
	}
}

// ---- Q4 (nnc split 4-bit) gemv ----------------------------------------
//
// The JIT kernel handles sum_b scale_b * sum q*x. The bias term factors
// into biases[r] . xsum, a plain f32 dot over cols/32 elements that the
// worker does inline right after each row's kernel call.

namespace
{
	struct q4_gemv_ctx
	{
		const uint8_t* qs;
		const uint16_t* scales;
		const uint16_t* biases;
		const float* x;
		const float* xsum;
		float* y;
		uint32_t cols;
		uint32_t rows;
		nnc_gemv_q4_s_1row_fn fn1;
	};

	void q4_gemv_worker(const int tid, const int nth, const void* u)
	{
		const auto* c = static_cast<const q4_gemv_ctx*>(u);
		uint32_t r0, r1;
		split_range(c->rows, tid, nth, r0, r1);
		if (r0 == r1) return;
		const size_t nblocks = c->cols / 32;
		const size_t qs_stride = c->cols / 2;
		const uint8_t* qs_tid = c->qs + r0 * qs_stride;
		const uint16_t* sc_tid = c->scales + r0 * nblocks;
		const uint16_t* bi_tid = c->biases + r0 * nblocks;
		float* y_tid = c->y + r0;
		for (uint32_t r = r0; r < r1; ++r)
		{
			c->fn1(qs_tid, c->x, y_tid, sc_tid);
			*y_tid += nnc_dot_bf16_f32_simd(bi_tid, c->xsum, nblocks);
			qs_tid += qs_stride;
			sc_tid += nblocks;
			bi_tid += nblocks;
			y_tid += 1;
		}
	}
}

void nnc_gemv_q4_s_f32x(const uint8_t* qs, const uint16_t* scales,
                        const uint16_t* biases, const float* x,
                        const float* xsum, float* y,
                        const uint32_t rows, const uint32_t cols)
{
	if (rows == 0 || cols == 0) return;
	NNC_ASSERT((cols % 32) == 0);

	const auto fn1 = global_cache().get_gemv_q4_s_1row(cols);

	auto& pool = nnc_gemv_pool::global();
	if (pool.n_threads() > 1 && rows >= 256 && cols >= 256)
	{
		q4_gemv_ctx ctx{qs, scales, biases, x, xsum, y, cols, rows, fn1};
		pool.dispatch(&q4_gemv_worker, &ctx);
		return;
	}

	const size_t nblocks = cols / 32;
	const size_t qs_stride = cols / 2;
	for (uint32_t r = 0; r < rows; ++r)
	{
		fn1(qs + r * qs_stride, x, y + r, scales + r * nblocks);
		y[r] += nnc_dot_bf16_f32_simd(biases + r * nblocks, xsum, nblocks);
	}
}

// ---- batched gemv (prefill) -------------------------------------------
//
// One pass over the weights serves the whole batch: the outer loop walks
// rows (so a row is fetched from DRAM once) and the inner loop walks the
// batch (hitting that row in L1). Threading splits the row axis exactly
// as the single-vector paths do.

namespace
{
	struct batch_ctx
	{
		const void* w; // BF16 W / Q8_0 qs / Q4 qs
		const void* scales; // f32 (Q8_0) or bf16 (Q4)
		const uint16_t* biases; // Q4 only
		const float* X;
		const float* XSUM; // Q4 only
		float* Y;
		size_t x_stride;
		size_t xsum_stride;
		size_t y_stride;
		uint32_t cols;
		uint32_t rows;
		uint32_t n_batch;
		const void* fn;
		const void* fn3; // three-token kernel (Q8_0 / Q4)
	};

	void bf16_batch_worker(const int tid, const int nth, const void* u)
	{
		const auto* c = static_cast<const batch_ctx*>(u);
		uint32_t q0, q1;
		split_range(c->rows, tid, nth, q0, q1);
		if (q0 == q1) return;
		const auto fn4 = reinterpret_cast<nnc_gemv_bf16w_f32x_fn>(const_cast<void*>(c->fn));
		const size_t group_stride = static_cast<size_t>(c->cols) * 2 * 4;
		const auto* W = static_cast<const uint8_t*>(c->w) + q0 * group_stride;
		for (uint32_t q = q0; q < q1; ++q)
		{
			for (uint32_t n = 0; n < c->n_batch; ++n)
				fn4(W, c->X + n * c->x_stride, c->Y + n * c->y_stride + q * 4);
			W += group_stride;
		}
	}

	void q8_batch_worker(const int tid, const int nth, const void* u)
	{
		const auto* c = static_cast<const batch_ctx*>(u);
		uint32_t r0, r1;
		split_range(c->rows, tid, nth, r0, r1);
		if (r0 == r1) return;
		const auto fn1 = reinterpret_cast<nnc_gemv_q8_0_1row_fn>(const_cast<void*>(c->fn));
		const auto fn3 = reinterpret_cast<nnc_gemv_3tok_fn>(const_cast<void*>(c->fn3));
		const size_t nblocks = c->cols / 32;
		const auto* qs = static_cast<const int8_t*>(c->w) + static_cast<size_t>(r0) * c->cols;
		const uint16_t* sc = static_cast<const uint16_t*>(c->scales) + r0 * nblocks;
		for (uint32_t r = r0; r < r1; ++r)
		{
			uint32_t n = 0;
			for (; n + 3 <= c->n_batch; n += 3)
			{
				const float* xs[3] = {c->X + n * c->x_stride, c->X + (n + 1) * c->x_stride,
				                      c->X + (n + 2) * c->x_stride};
				float* ys[3] = {c->Y + n * c->y_stride + r, c->Y + (n + 1) * c->y_stride + r,
				                c->Y + (n + 2) * c->y_stride + r};
				fn3(qs, xs, ys, sc);
			}
			for (; n < c->n_batch; ++n)
				fn1(qs, c->X + n * c->x_stride, c->Y + n * c->y_stride + r, sc);
			qs += c->cols;
			sc += nblocks;
		}
	}

	void q4_batch_worker(const int tid, const int nth, const void* u)
	{
		const auto* c = static_cast<const batch_ctx*>(u);
		uint32_t r0, r1;
		split_range(c->rows, tid, nth, r0, r1);
		if (r0 == r1) return;
		const auto fn1 = reinterpret_cast<nnc_gemv_q4_s_1row_fn>(const_cast<void*>(c->fn));
		const auto fn3 = reinterpret_cast<nnc_gemv_3tok_fn>(const_cast<void*>(c->fn3));
		const size_t nblocks = c->cols / 32;
		const size_t qs_stride = c->cols / 2;
		const auto* qs = static_cast<const uint8_t*>(c->w) + r0 * qs_stride;
		const uint16_t* sc = static_cast<const uint16_t*>(c->scales) + r0 * nblocks;
		const uint16_t* bi = c->biases + r0 * nblocks;
		for (uint32_t r = r0; r < r1; ++r)
		{
			uint32_t n = 0;
			for (; n + 3 <= c->n_batch; n += 3)
			{
				const float* xs[3] = {c->X + n * c->x_stride, c->X + (n + 1) * c->x_stride,
				                      c->X + (n + 2) * c->x_stride};
				float* ys[3] = {c->Y + n * c->y_stride + r, c->Y + (n + 1) * c->y_stride + r,
				                c->Y + (n + 2) * c->y_stride + r};
				fn3(qs, xs, ys, sc);
				for (uint32_t t = 0; t < 3; ++t)
					*ys[t] += nnc_dot_bf16_f32_simd(bi, c->XSUM + (n + t) * c->xsum_stride, nblocks);
			}
			for (; n < c->n_batch; ++n)
			{
				float* yp = c->Y + n * c->y_stride + r;
				fn1(qs, c->X + n * c->x_stride, yp, sc);
				*yp += nnc_dot_bf16_f32_simd(bi, c->XSUM + n * c->xsum_stride, nblocks);
			}
			qs += qs_stride;
			sc += nblocks;
			bi += nblocks;
		}
	}

	// Shared dispatch: run `fn` over the row axis, threaded when the shape
	// is big enough to pay for it.
	void run_batch(batch_ctx& ctx, const nnc_gemv_pool::task_fn_t fn,
	               const uint32_t work_rows)
	{
		auto& pool = nnc_gemv_pool::global();
		ctx.rows = work_rows;
		if (pool.n_threads() > 1 && work_rows >= 64 && ctx.cols >= 256)
		{
			pool.dispatch(fn, &ctx);
			return;
		}
		fn(0, 1, &ctx);
	}
}

void nnc_set_num_threads_avx(const int n_threads)
{
	nnc_gemv_pool::global().configure(nnc_gemv_pool::resolve_threads(n_threads));
}

int nnc_num_threads_avx()
{
	return nnc_gemv_pool::global().n_threads();
}

void nnc_parallel_for(const uint32_t n, const nnc_range_fn fn, void* user)
{
	if (n == 0) return;
	auto& pool = nnc_gemv_pool::global();
	if (n == 1 || pool.n_threads() == 1)
	{
		fn(0, n, user);
		return;
	}
	struct ctx_t
	{
		uint32_t n;
		nnc_range_fn fn;
		void* user;
	} ctx{n, fn, user};
	pool.dispatch([](const int tid, const int nth, const void* u)
	{
		const auto* c = static_cast<const ctx_t*>(u);
		uint32_t first, last;
		split_range(c->n, tid, nth, first, last);
		if (first < last) c->fn(first, last, c->user);
	}, &ctx);
}

void nnc_gemv_bf16w_f32x_batch(const void* W, const float* X, const size_t x_stride,
                               float* Y, const size_t y_stride,
                               const uint32_t rows, const uint32_t cols,
                               const uint32_t n_batch)
{
	if (rows == 0 || cols == 0 || n_batch == 0) return;
	if (n_batch == 1)
	{
		nnc_gemv_bf16w_f32x(W, X, Y, rows, cols);
		return;
	}
	// The 4-row kernel is the batched path's unit of work; anything it
	// can't express falls back to repeating the single-vector gemv.
	if ((rows & 3u) != 0 || (cols & 7u) != 0)
	{
		for (uint32_t n = 0; n < n_batch; ++n)
			nnc_gemv_bf16w_f32x(W, X + n * x_stride, Y + n * y_stride, rows, cols);
		return;
	}

	batch_ctx ctx{};
	ctx.w = W;
	ctx.X = X;
	ctx.Y = Y;
	ctx.x_stride = x_stride;
	ctx.y_stride = y_stride;
	ctx.cols = cols;
	ctx.n_batch = n_batch;
	ctx.fn = reinterpret_cast<const void*>(global_cache().get_gemv_bf16w_f32x(4, cols));
	run_batch(ctx, &bf16_batch_worker, rows / 4);
}

void nnc_gemv_q8_0_f32x_batch(const int8_t* qs, const uint16_t* scales,
                              const float* X, const size_t x_stride,
                              float* Y, const size_t y_stride,
                              const uint32_t rows, const uint32_t cols,
                              const uint32_t n_batch)
{
	if (rows == 0 || cols == 0 || n_batch == 0) return;
	if (n_batch == 1)
	{
		nnc_gemv_q8_0_f32x(qs, scales, X, Y, rows, cols);
		return;
	}
	NNC_ASSERT((cols % 32) == 0);

	batch_ctx ctx{};
	ctx.w = qs;
	ctx.scales = scales;
	ctx.X = X;
	ctx.Y = Y;
	ctx.x_stride = x_stride;
	ctx.y_stride = y_stride;
	ctx.cols = cols;
	ctx.n_batch = n_batch;
	ctx.fn = reinterpret_cast<const void*>(global_cache().get_gemv_q8_0_1row(cols));
	ctx.fn3 = reinterpret_cast<const void*>(global_cache().get_gemv_q8_0_3tok(cols));
	run_batch(ctx, &q8_batch_worker, rows);
}

void nnc_gemv_q4_s_f32x_batch(const uint8_t* qs, const uint16_t* scales,
                              const uint16_t* biases,
                              const float* X, const size_t x_stride,
                              const float* XSUM, const size_t xsum_stride,
                              float* Y, const size_t y_stride,
                              const uint32_t rows, const uint32_t cols,
                              const uint32_t n_batch)
{
	if (rows == 0 || cols == 0 || n_batch == 0) return;
	if (n_batch == 1)
	{
		nnc_gemv_q4_s_f32x(qs, scales, biases, X, XSUM, Y, rows, cols);
		return;
	}
	NNC_ASSERT((cols % 32) == 0);

	batch_ctx ctx{};
	ctx.w = qs;
	ctx.scales = scales;
	ctx.biases = biases;
	ctx.X = X;
	ctx.XSUM = XSUM;
	ctx.Y = Y;
	ctx.x_stride = x_stride;
	ctx.xsum_stride = xsum_stride;
	ctx.y_stride = y_stride;
	ctx.cols = cols;
	ctx.n_batch = n_batch;
	ctx.fn = reinterpret_cast<const void*>(global_cache().get_gemv_q4_s_1row(cols));
	ctx.fn3 = reinterpret_cast<const void*>(global_cache().get_gemv_q4_s_3tok(cols));
	run_batch(ctx, &q4_batch_worker, rows);
}

// ---- SwiGLU ------------------------------------------------------------
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// y[i] = silu(gate[i]) * up[i]
//
// Reuses the AVX2 exp approximation from the softmax path; the reciprocal
// is a true divide (not rcp_ps) to keep ~1 ulp accuracy.

void nnc_swiglu_f32(float* y, const float* gate, const float* up, const size_t n)
{
	const __m256 one = _mm256_set1_ps(1.0f);
	size_t i = 0;
	const size_t np = n & ~size_t{7};
	for (; i < np; i += 8)
	{
		const __m256 g = _mm256_loadu_ps(gate + i);
		const __m256 e = nnc_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), g));
		const __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
		const __m256 r = _mm256_mul_ps(_mm256_mul_ps(g, sig), _mm256_loadu_ps(up + i));
		_mm256_storeu_ps(y + i, r);
	}
	for (; i < n; ++i)
	{
		const float g = gate[i];
		y[i] = g / (1.0f + std::exp(-g)) * up[i];
	}
}

void nnc_mul_sigmoid_f32(float* y, const float* g, const size_t n)
{
	const __m256 one = _mm256_set1_ps(1.0f);
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		const __m256 e = nnc_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_loadu_ps(g + i)));
		const __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
		_mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), sig));
	}
	for (; i < n; ++i) y[i] /= 1.0f + std::exp(-g[i]);
}

// ---- Gated DeltaNet ------------------------------------------------------

void nnc_l2norm_multi_f32(float* x, const size_t n_groups, const size_t dim, const float eps, const float scale)
{
	for (size_t g = 0; g < n_groups; ++g)
	{
		float* xg = x + g * dim;
		const float ss = nnc_dot_f32_simd(xg, xg, dim);
		const float s = scale / std::sqrt(ss + eps);
		const __m256 vs = _mm256_set1_ps(s);
		size_t i = 0;
		for (; i + 8 <= dim; i += 8) _mm256_storeu_ps(xg + i, _mm256_mul_ps(_mm256_loadu_ps(xg + i), vs));
		for (; i < dim; ++i) xg[i] *= s;
	}
}

void nnc_conv1d_silu_f32(float* y, const float* x, const float* w, const size_t n_tok, const size_t ch,
                         const size_t k)
{
	const __m256 one = _mm256_set1_ps(1.0f);
	for (size_t t = 0; t < n_tok; ++t)
	{
		const float* xt = x + t * ch;
		float* yt = y + t * ch;
		size_t c = 0;
		for (; c + 8 <= ch; c += 8)
		{
			__m256 acc = _mm256_setzero_ps();
			for (size_t j = 0; j < k; ++j)
				acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + j * ch + c), _mm256_loadu_ps(xt + j * ch + c), acc);
			const __m256 e = nnc_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), acc));
			_mm256_storeu_ps(yt + c, _mm256_div_ps(acc, _mm256_add_ps(one, e)));
		}
		for (; c < ch; ++c)
		{
			float acc = 0.0f;
			for (size_t j = 0; j < k; ++j) acc += w[j * ch + c] * xt[j * ch + c];
			yt[c] = acc / (1.0f + std::exp(-acc));
		}
	}
}

void nnc_delta_rule_f32(float* S, const float* q, const float* k, const float* v, const float decay,
                        const float beta, float* o, const size_t d_k, const size_t d_v)
{
	const __m256 vdecay = _mm256_set1_ps(decay);
	for (size_t j = 0; j < d_v; ++j)
	{
		float* row = S + j * d_k;
		// The decayed row's projection on k, then the step towards v[j].
		const float delta = beta * (v[j] - decay * nnc_dot_f32_simd(row, k, d_k));
		const __m256 vdelta = _mm256_set1_ps(delta);
		__m256 acc0 = _mm256_setzero_ps();
		__m256 acc1 = _mm256_setzero_ps();
		size_t i = 0;
		for (; i + 16 <= d_k; i += 16)
		{
			const __m256 s0 = _mm256_fmadd_ps(vdelta, _mm256_loadu_ps(k + i),
			                                  _mm256_mul_ps(_mm256_loadu_ps(row + i), vdecay));
			const __m256 s1 = _mm256_fmadd_ps(vdelta, _mm256_loadu_ps(k + i + 8),
			                                  _mm256_mul_ps(_mm256_loadu_ps(row + i + 8), vdecay));
			_mm256_storeu_ps(row + i, s0);
			_mm256_storeu_ps(row + i + 8, s1);
			acc0 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(q + i), acc0);
			acc1 = _mm256_fmadd_ps(s1, _mm256_loadu_ps(q + i + 8), acc1);
		}
		for (; i + 8 <= d_k; i += 8)
		{
			const __m256 s0 = _mm256_fmadd_ps(vdelta, _mm256_loadu_ps(k + i),
			                                  _mm256_mul_ps(_mm256_loadu_ps(row + i), vdecay));
			_mm256_storeu_ps(row + i, s0);
			acc0 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(q + i), acc0);
		}
		float out = hsum_ymm(_mm256_add_ps(acc0, acc1));
		for (; i < d_k; ++i)
		{
			const float s = row[i] * decay + delta * k[i];
			row[i] = s;
			out += s * q[i];
		}
		o[j] = out;
	}
}

// ---- RoPE ----------------------------------------------------------------

void nnc_rope_f32(float* x, const uint32_t n_heads, const uint32_t head_dim,
                  const uint32_t n_rot, const float pos, const float freq_base,
                  const nnc_rope_mode mode, const float* freq_factors)
{
	if (n_rot == 0) return;
	NNC_ASSERT((n_rot & 1u) == 0 && "nnc_rope_f32: n_rot must be even");
	const uint32_t half = n_rot / 2;
	// theta_i depends only on the lane index, not the head, so the
	// cos/sin pair is shared by all n_heads. freq_base^(-2/n_rot) is the
	// per-step ratio of the geometric inv schedule: compute once, then
	// accumulate (avoids one std::pow per pair).
	constexpr uint32_t MAX_HALF = 256; // head_dim <= 512 for every model we load
	NNC_ASSERT(half <= MAX_HALF && "nnc_rope_f32: n_rot/2 exceeds table size");
	float cs[MAX_HALF], sn[MAX_HALF];
	{
		const float ratio = std::pow(freq_base, -2.0f / static_cast<float>(n_rot));
		float inv = 1.0f;
		for (uint32_t i = 0; i < half; ++i)
		{
			float theta = pos * inv;
			if (freq_factors) theta /= freq_factors[i];
			cs[i] = std::cos(theta);
			sn[i] = std::sin(theta);
			inv *= ratio;
		}
	}
	// NeoX pairs lane i with lane i + n_rot/2; "normal" pairs lanes 2i and
	// 2i + 1. Which one a model needs is fixed by how its Q/K weights were
	// laid out at conversion time.
	const uint32_t stride = (mode == NNC_ROPE_NEOX) ? 1 : 2;
	const uint32_t partner = (mode == NNC_ROPE_NEOX) ? half : 1;
	for (uint32_t h = 0; h < n_heads; ++h)
	{
		float* xh = x + static_cast<size_t>(h) * head_dim;
		for (uint32_t i = 0; i < half; ++i)
		{
			float* p = xh + i * stride;
			const float a = p[0];
			const float b = p[partner];
			p[0] = cs[i] * a - sn[i] * b;
			p[partner] = sn[i] * a + cs[i] * b;
		}
	}
}

// ---- soft-cap (tanh(x/c)*c) --------------------------------------------

void nnc_softcap_f32(float* y, const float* x, const size_t n, const float cap)
{
	const float inv = 1.0f / cap;
	for (size_t i = 0; i < n; ++i)
		y[i] = std::tanh(x[i] * inv) * cap;
}

// ---- BF16 embedding-row lookup with optional scale ---------------------

void nnc_embed_row_bf16(float* y, const void* table, const int token_id,
                        const size_t n_embd, const float scale)
{
	const auto* row = static_cast<const uint16_t*>(table)
		+ static_cast<size_t>(token_id) * n_embd;
	// Inflate BF16 -> F32 then multiply by `scale`. Two-pass keeps the
	// SIMD code in nnc_bf16_to_f32_row simple.
	nnc_bf16_to_f32_row(row, y, n_embd);
	if (scale != 1.0f)
	{
		const __m256 vs = _mm256_set1_ps(scale);
		size_t i = 0;
		for (; i + 8 <= n_embd; i += 8)
			_mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), vs));
		for (; i < n_embd; ++i) y[i] *= scale;
	}
}

void nnc_embed_row_q8_0(float* y, const int8_t* qs, const uint16_t* scales,
                        const int token_id, const size_t n_embd, const float scale)
{
	NNC_ASSERT((n_embd % 32) == 0);
	const size_t nblocks = n_embd / 32;
	const int8_t* row = qs + static_cast<size_t>(token_id) * n_embd;
	const uint16_t* rs = scales + static_cast<size_t>(token_id) * nblocks;

	for (size_t b = 0; b < nblocks; ++b)
	{
		const __m256 vs = _mm256_set1_ps(nnc_bf16_to_f32(rs[b]) * scale);
		const int8_t* p = row + b * 32;
		for (size_t j = 0; j < 32; j += 8)
		{
			const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + j));
			const __m256 f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
			_mm256_storeu_ps(y + b * 32 + j, _mm256_mul_ps(f, vs));
		}
	}
}

void nnc_embed_row_q4_s(float* y, const uint8_t* qs, const uint16_t* scales,
                        const uint16_t* biases, const int token_id,
                        const size_t n_embd, const float scale)
{
	NNC_ASSERT((n_embd % 32) == 0);
	const size_t nblocks = n_embd / 32;
	const uint8_t* row = qs + static_cast<size_t>(token_id) * (n_embd / 2);
	const uint16_t* rs = scales + static_cast<size_t>(token_id) * nblocks;
	const uint16_t* rb = biases + static_cast<size_t>(token_id) * nblocks;

	for (size_t b = 0; b < nblocks; ++b)
	{
		const float s = nnc_bf16_to_f32(rs[b]) * scale;
		const float bias = nnc_bf16_to_f32(rb[b]) * scale;
		const uint8_t* blk = row + b * 16;
		float* out = y + b * 32;
		for (size_t k = 0; k < 16; ++k)
		{
			out[k] = s * static_cast<float>(blk[k] & 0x0F) + bias;
			out[k + 16] = s * static_cast<float>(blk[k] >> 4) + bias;
		}
	}
}

//
// Layouts mirror ggml's `block_q*_K` structs (little-endian, x86-only).
// Block size is 256 elements = 8 sub-blocks of 32 (Q4_K/Q5_K) or 16
// sub-blocks of 16 (Q6_K). Decoders are scalar; intent is to dequant
// once at load time into BF16 / F32 storage that the existing gemv
// kernels can consume.

namespace
{
	struct block_q4_K
	{
		uint16_t d; // fp16, super-block scale-of-scales
		uint16_t dmin; // fp16, super-block scale-of-mins
		uint8_t scales[12]; // 8 sub-blocks * (6-bit scale + 6-bit min), packed
		uint8_t qs[128]; // 256 4-bit quants (low/high nibble)
	};
	static_assert(sizeof(block_q4_K) == 144, "Q4_K block layout mismatch");

	struct block_q5_K
	{
		uint16_t d;
		uint16_t dmin;
		uint8_t scales[12];
		uint8_t qh[32]; // 256 1-bit "high" quants (1 bit per element)
		uint8_t qs[128]; // 256 4-bit "low" quants
	};
	static_assert(sizeof(block_q5_K) == 176, "Q5_K block layout mismatch");

	struct block_q6_K
	{
		uint8_t ql[128]; // 256 4-bit "low" quants
		uint8_t qh[64]; // 256 2-bit "high" quants
		int8_t scales[16]; // 16 sub-block scales (i8)
		uint16_t d; // fp16 super-block scale
	};
	static_assert(sizeof(block_q6_K) == 210, "Q6_K block layout mismatch");

	// 6-bit scale/min unpacker shared by Q4_K and Q5_K.
	// `j` selects sub-block 0..7; the 12 scale bytes encode one 6-bit
	// scale + 6-bit min per sub-block via the canonical ggml packing.
	inline void get_scale_min_k4(const int j, const uint8_t* q,
	                             uint8_t& d, uint8_t& m)
	{
		if (j < 4)
		{
			d = q[j] & 63u;
			m = q[j + 4] & 63u;
		}
		else
		{
			d = (q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4);
			m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
		}
	}
} // namespace

void nnc_dequantize_q8_0_to_f32(const void* blocks, float* dst,
                                const size_t n_elements)
{
	NNC_ASSERT(blocks && dst);
	NNC_ASSERT(n_elements > 0 && (n_elements % 32) == 0);

	// block_q8_0 = { fp16 d; int8 qs[32] }, 34 bytes, no padding.
	const auto* p = static_cast<const uint8_t*>(blocks);
	const size_t nb = n_elements / 32;
	for (size_t i = 0; i < nb; ++i, p += 34)
	{
		uint16_t dh;
		std::memcpy(&dh, p, 2);
		const float d = fp16_to_fp32_scalar(dh);
		const auto* qs = reinterpret_cast<const int8_t*>(p + 2);
		for (int k = 0; k < 32; ++k) dst[i * 32 + k] = d * qs[k];
	}
}

namespace
{
	// The four classic 32-element formats differ only in an optional fp16
	// min and an optional fifth bit per value. Each block is
	//   fp16 d; [fp16 m]; [uint32 qh]; uint8 qs[16]
	// and value j (or j+16) is (low nibble of qs[j] (or high), plus bit j
	// (or j+16) of qh as bit 4) * d, then + m, or centred on 8 or 16 when
	// there is no m.
	template <bool HAS_MIN, bool HAS_HIGH>
	void dequantize_legacy(const void* blocks, float* dst, const size_t n_elements)
	{
		NNC_ASSERT(blocks && dst);
		NNC_ASSERT(n_elements > 0 && (n_elements % 32) == 0);
		constexpr size_t bytes = 2 + (HAS_MIN ? 2 : 0) + (HAS_HIGH ? 4 : 0) + 16;
		constexpr int centre = HAS_MIN ? 0 : (HAS_HIGH ? 16 : 8);
		const auto* p = static_cast<const uint8_t*>(blocks);
		for (size_t i = 0; i < n_elements / 32; ++i, p += bytes)
		{
			uint16_t h;
			std::memcpy(&h, p, 2);
			const float d = fp16_to_fp32_scalar(h);
			float m = 0.0f;
			if (HAS_MIN)
			{
				std::memcpy(&h, p + 2, 2);
				m = fp16_to_fp32_scalar(h);
			}
			uint32_t qh = 0;
			if (HAS_HIGH) std::memcpy(&qh, p + 2 + (HAS_MIN ? 2 : 0), 4);
			const uint8_t* qs = p + bytes - 16;
			float* y = dst + i * 32;
			for (int j = 0; j < 16; ++j)
			{
				const int hi0 = HAS_HIGH ? static_cast<int>(((qh >> j) << 4) & 0x10) : 0;
				const int hi1 = HAS_HIGH ? static_cast<int>((qh >> (j + 12)) & 0x10) : 0;
				y[j] = static_cast<float>(((qs[j] & 0x0F) | hi0) - centre) * d + m;
				y[j + 16] = static_cast<float>(((qs[j] >> 4) | hi1) - centre) * d + m;
			}
		}
	}
}

void nnc_dequantize_q4_0_to_f32(const void* blocks, float* dst, const size_t n_elements)
{
	dequantize_legacy<false, false>(blocks, dst, n_elements);
}

void nnc_dequantize_q4_1_to_f32(const void* blocks, float* dst, const size_t n_elements)
{
	dequantize_legacy<true, false>(blocks, dst, n_elements);
}

void nnc_dequantize_q5_0_to_f32(const void* blocks, float* dst, const size_t n_elements)
{
	dequantize_legacy<false, true>(blocks, dst, n_elements);
}

void nnc_dequantize_q5_1_to_f32(const void* blocks, float* dst, const size_t n_elements)
{
	dequantize_legacy<true, true>(blocks, dst, n_elements);
}

void nnc_dequantize_q4_k_to_f32(const void* blocks, float* dst,
                                const size_t n_elements)
{
	NNC_ASSERT(blocks && dst);
	NNC_ASSERT(n_elements > 0 && (n_elements % 256) == 0);

	const auto* b = static_cast<const block_q4_K*>(blocks);
	const size_t nb = n_elements / 256;
	float* y = dst;

	for (size_t i = 0; i < nb; ++i)
	{
		const float d = fp16_to_fp32_scalar(b[i].d);
		const float dmin = fp16_to_fp32_scalar(b[i].dmin);
		const uint8_t* q = b[i].qs;

		// 4 outer iterations of 64 outputs each (256 = 4*64).
		// Each iteration consumes 32 qs bytes and uses sub-blocks
		// `is` (low nibble) and `is+1` (high nibble), with `is` += 2.
		int is = 0;
		for (int j = 0; j < 256; j += 64)
		{
			uint8_t sc, mn;
			get_scale_min_k4(is + 0, b[i].scales, sc, mn);
			const float d1 = d * sc;
			const float m1 = dmin * mn;
			get_scale_min_k4(is + 1, b[i].scales, sc, mn);
			const float d2 = d * sc;
			const float m2 = dmin * mn;

			for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0x0F) - m1;
			for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
			q += 32;
			is += 2;
		}
	}
}

void nnc_dequantize_q5_k_to_f32(const void* blocks, float* dst,
                                const size_t n_elements)
{
	NNC_ASSERT(blocks && dst);
	NNC_ASSERT(n_elements > 0 && (n_elements % 256) == 0);

	const auto* b = static_cast<const block_q5_K*>(blocks);
	const size_t nb = n_elements / 256;
	float* y = dst;

	for (size_t i = 0; i < nb; ++i)
	{
		const float d = fp16_to_fp32_scalar(b[i].d);
		const float dmin = fp16_to_fp32_scalar(b[i].dmin);
		const uint8_t* ql = b[i].qs;
		const uint8_t* qh = b[i].qh;

		int is = 0;
		uint8_t u1 = 1, u2 = 2;
		for (int j = 0; j < 256; j += 64)
		{
			uint8_t sc, mn;
			get_scale_min_k4(is + 0, b[i].scales, sc, mn);
			const float d1 = d * sc;
			const float m1 = dmin * mn;
			get_scale_min_k4(is + 1, b[i].scales, sc, mn);
			const float d2 = d * sc;
			const float m2 = dmin * mn;

			for (int l = 0; l < 32; ++l)
				*y++ = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
			for (int l = 0; l < 32; ++l)
				*y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
			ql += 32;
			is += 2;
			u1 <<= 2;
			u2 <<= 2;
		}
	}
}

void nnc_dequantize_q6_k_to_f32(const void* blocks, float* dst,
                                const size_t n_elements)
{
	NNC_ASSERT(blocks && dst);
	NNC_ASSERT(n_elements > 0 && (n_elements % 256) == 0);

	const auto* b = static_cast<const block_q6_K*>(blocks);
	const size_t nb = n_elements / 256;
	float* y = dst;

	for (size_t i = 0; i < nb; ++i)
	{
		const float d = fp16_to_fp32_scalar(b[i].d);
		const uint8_t* ql = b[i].ql;
		const uint8_t* qh = b[i].qh;
		const int8_t* sc = b[i].scales;

		// 2 outer iterations of 128 outputs each.
		for (int n = 0; n < 256; n += 128)
		{
			for (int l = 0; l < 32; ++l)
			{
				const int is = l / 16;
				const int q1 = static_cast<int8_t>(
					(ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
				const int q2 = static_cast<int8_t>(
					(ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
				const int q3 = static_cast<int8_t>(
					(ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
				const int q4 = static_cast<int8_t>(
					(ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
				y[l + 0] = d * sc[is + 0] * q1;
				y[l + 32] = d * sc[is + 2] * q2;
				y[l + 64] = d * sc[is + 4] * q3;
				y[l + 96] = d * sc[is + 6] * q4;
			}
			y += 128;
			ql += 64;
			qh += 32;
			sc += 8;
		}
	}
}

bool nnc_dequantize_to_f32(const uint32_t ggml_type, const void* blocks,
                           float* dst, const size_t n_elements)
{
	switch (ggml_type)
	{
		case 0: std::memcpy(dst, blocks, n_elements * sizeof(float));
			return true;
		case 1:
			{
				const auto* h = static_cast<const uint16_t*>(blocks);
				for (size_t i = 0; i < n_elements; ++i) dst[i] = fp16_to_fp32_scalar(h[i]);
				return true;
			}
		case 30: nnc_bf16_to_f32_row(static_cast<const nnc_bf16_t*>(blocks), dst, n_elements);
			return true;
		case 8: nnc_dequantize_q8_0_to_f32(blocks, dst, n_elements);
			return true;
		case 2: nnc_dequantize_q4_0_to_f32(blocks, dst, n_elements);
			return true;
		case 3: nnc_dequantize_q4_1_to_f32(blocks, dst, n_elements);
			return true;
		case 6: nnc_dequantize_q5_0_to_f32(blocks, dst, n_elements);
			return true;
		case 7: nnc_dequantize_q5_1_to_f32(blocks, dst, n_elements);
			return true;
		case 12: nnc_dequantize_q4_k_to_f32(blocks, dst, n_elements);
			return true;
		case 13: nnc_dequantize_q5_k_to_f32(blocks, dst, n_elements);
			return true;
		case 14: nnc_dequantize_q6_k_to_f32(blocks, dst, n_elements);
			return true;
		default: return false;
	}
}

uint32_t nnc_quant_block_elems(const uint32_t ggml_type)
{
	switch (ggml_type)
	{
		case 2: // Q4_0
		case 3: // Q4_1
		case 6: // Q5_0
		case 7: // Q5_1
		case 8: return 32; // Q8_0
		case 12: // Q4_K
		case 13: // Q5_K
		case 14: return 256; // Q6_K
		default: return 0;
	}
}
