// nnc — model loader and forward pass. See model.h.

#include "model.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

// ----------------------------------------------------------------------------
// Metadata helpers.
// ----------------------------------------------------------------------------

namespace
{
	bool value_int(const gguf_value* v, int& out)
	{
		switch (v->type)
		{
		case GGUF_TYPE_UINT8:
		case GGUF_TYPE_UINT16:
		case GGUF_TYPE_UINT32:
		case GGUF_TYPE_UINT64:
			if (v->u64 > static_cast<uint64_t>(INT32_MAX)) return false;
			out = static_cast<int>(v->u64);
			return true;
		case GGUF_TYPE_INT8:
		case GGUF_TYPE_INT16:
		case GGUF_TYPE_INT32:
		case GGUF_TYPE_INT64:
			if (v->i64 < INT32_MIN || v->i64 > INT32_MAX) return false;
			out = static_cast<int>(v->i64);
			return true;
		default: return false;
		}
	}

	// An integer hparam: a scalar, or the first element of a per-layer array.
	bool kv_int(const gguf_file& f, const std::string& key, int& out)
	{
		const gguf_value* v = gguf_find_kv(f, key);
		if (!v) return false;
		if (v->type == GGUF_TYPE_ARRAY)
		{
			if (v->arr.empty()) return false;
			v = &v->arr[0];
		}
		return value_int(v, out);
	}

	// A per-layer integer hparam: element `li` of an array, or the scalar.
	bool kv_int_at(const gguf_file& f, const std::string& key, const int li, int& out)
	{
		const gguf_value* v = gguf_find_kv(f, key);
		if (!v) return false;
		if (v->type == GGUF_TYPE_ARRAY)
		{
			if (li < 0 || static_cast<size_t>(li) >= v->arr.size()) return false;
			v = &v->arr[static_cast<size_t>(li)];
		}
		return value_int(v, out);
	}

	bool kv_float(const gguf_file& f, const std::string& key, float& out)
	{
		const gguf_value* v = gguf_find_kv(f, key);
		if (!v) return false;
		if (v->type == GGUF_TYPE_FLOAT32 || v->type == GGUF_TYPE_FLOAT64)
		{
			out = static_cast<float>(v->f64);
			return true;
		}
		return false;
	}

	bool kv_str(const gguf_file& f, const std::string& key, std::string& out)
	{
		const gguf_value* v = gguf_find_kv(f, key);
		if (!v || v->type != GGUF_TYPE_STRING) return false;
		out = v->str;
		return true;
	}

	bool kv_bool(const gguf_file& f, const std::string& key, bool& out)
	{
		const gguf_value* v = gguf_find_kv(f, key);
		if (!v || v->type != GGUF_TYPE_BOOL) return false;
		out = v->u64 != 0;
		return true;
	}

	// A per-layer flag: element `li` of an array of bools (or integers).
	bool kv_flag_at(const gguf_file& f, const std::string& key, const int li, bool& out)
	{
		const gguf_value* v = gguf_find_kv(f, key);
		if (!v || v->type != GGUF_TYPE_ARRAY || li < 0 || static_cast<size_t>(li) >= v->arr.size()) return false;
		const gguf_value& e = v->arr[static_cast<size_t>(li)];
		int i = 0;
		if (e.type == GGUF_TYPE_BOOL) i = e.u64 != 0;
		else if (!value_int(&e, i)) return false;
		out = i != 0;
		return true;
	}

	const char* fmt_name(const nnc_weight_format fmt)
	{
		switch (fmt)
		{
		case NNC_WEIGHTS_Q8_0: return "Q8_0";
		case NNC_WEIGHTS_Q4: return "Q4";
		default: return "BF16";
		}
	}

	// Byte size of `n` weights in `fmt`, secondary arrays included.
	size_t fmt_bytes(const nnc_weight_format fmt, const uint64_t n)
	{
		switch (fmt)
		{
		case NNC_WEIGHTS_Q8_0: return static_cast<size_t>(n) + static_cast<size_t>(n / 32) * 2;
		case NNC_WEIGHTS_Q4: return static_cast<size_t>(n / 2) + static_cast<size_t>(n / 32) * 4;
		default: return static_cast<size_t>(n) * sizeof(nnc_bf16_t);
		}
	}

	nnc_type fmt_type(const nnc_weight_format fmt)
	{
		switch (fmt)
		{
		case NNC_WEIGHTS_Q8_0: return NNC_TYPE_Q8_0;
		case NNC_WEIGHTS_Q4: return NNC_TYPE_Q4_S;
		default: return NNC_TYPE_BF16;
		}
	}

	// Encode `take` F32 weights starting at flat offset `off` into `buf`,
	// which holds all `n` weights of a tensor in `fmt`'s split layout.
	void encode_chunk(const nnc_weight_format fmt, uint8_t* buf, const uint64_t n,
	                  const float* src, const size_t off, const size_t take)
	{
		switch (fmt)
		{
		case NNC_WEIGHTS_Q8_0:
			nnc_quantize_f32_to_q8_0(src,
			                         reinterpret_cast<int8_t*>(buf) + off,
			                         reinterpret_cast<uint16_t*>(buf + n) + off / 32,
			                         take);
			break;
		case NNC_WEIGHTS_Q4:
			{
				auto* scales = reinterpret_cast<uint16_t*>(buf + n / 2);
				nnc_quantize_f32_to_q4_s(src, buf + off / 2,
				                         scales + off / 32,
				                         scales + n / 32 + off / 32,
				                         take);
				break;
			}
		default:
			{
				auto* bf = reinterpret_cast<nnc_bf16_t*>(buf);
				for (size_t i = 0; i < take; ++i) bf[off + i] = nnc_f32_to_bf16(src[i]);
				break;
			}
		}
	}

	// Elements per storage block of a type nnc_dequantize_to_f32 can read,
	// or 0.
	uint32_t decode_block_elems(const uint32_t ggml_type)
	{
		if (ggml_type == 0 || ggml_type == 1 || ggml_type == 30) return 1; // F32, F16, BF16
		return nnc_quant_block_elems(ggml_type);
	}

	// ------------------------------------------------------------------------
	// Load progress, measured in bytes of the file processed.
	// ------------------------------------------------------------------------

	struct load_progress
	{
		nnc_progress_callback fn = nullptr;
		void* user = nullptr;
		uint64_t total = 0;
		uint64_t done = 0;
		float reported = -1.0f;
		bool cancelled = false;

		// Returns false once the callback has asked to cancel.
		bool advance(const uint64_t bytes)
		{
			done += bytes;
			if (!fn || cancelled) return !cancelled;
			const double p = total ? static_cast<double>(done) / static_cast<double>(total) : 1.0;
			const float pf = static_cast<float>(p > 1.0 ? 1.0 : p);
			if (pf - reported < 0.005f) return true;
			reported = pf;
			if (!fn(pf, user)) cancelled = true;
			return !cancelled;
		}
	};

	// ------------------------------------------------------------------------
	// Tensor construction.
	//
	// Every tensor is loaded for a role. A `matrix` (anything a gemv or an
	// embedding lookup reads) always ends up BF16, Q8_0 or Q4 — whatever the
	// file stores — because those are the layouts the kernels take; so does
	// each matrix of an `experts` stack. A `vector` (norm gammas, scales,
	// RoPE factors, and the small MoE router, whose ranking quantization
	// could upset) always ends up F32. Anything already in its target layout
	// is used in place from the mapping; everything else is decoded once,
	// here, in parallel.
	// ------------------------------------------------------------------------

	enum class role
	{
		matrix,
		vector,
		experts, // [cols, rows, n_expert]: one split-layout matrix per expert, back to back
	};

	// 16384 elements: a multiple of every block size in play (32 and 256),
	// so no source block straddles a chunk, and small enough that the F32
	// staging buffer of each thread stays in L2.
	constexpr size_t CHUNK = 256 * 64;

	struct decode_job
	{
		uint32_t ggml_type;
		const uint8_t* src;
		size_t block_bytes;
		uint32_t block_elems;
		uint64_t n;
		bool to_f32; // vector role: write F32 straight into dst
		nnc_weight_format fmt; // matrix role: encode into this
		uint8_t* dst;
		std::atomic<bool> failed{false};
	};

	void decode_range(const uint32_t first, const uint32_t last, void* user)
	{
		auto* j = static_cast<decode_job*>(user);
		thread_local std::vector<float> tmp;
		if (tmp.size() < CHUNK) tmp.resize(CHUNK);
		for (uint32_t c = first; c < last; ++c)
		{
			const uint64_t off = static_cast<uint64_t>(c) * CHUNK;
			const size_t take = static_cast<size_t>(std::min<uint64_t>(CHUNK, j->n - off));
			const uint8_t* s = j->src + (off / j->block_elems) * j->block_bytes;
			float* out = j->to_f32 ? reinterpret_cast<float*>(j->dst) + off : tmp.data();
			if (!nnc_dequantize_to_f32(j->ggml_type, s, out, take))
			{
				j->failed = true;
				return;
			}
			if (!j->to_f32) encode_chunk(j->fmt, j->dst, j->n, tmp.data(), static_cast<size_t>(off), take);
		}
	}

	void set_shape(nnc_tensor& t, const gguf_tensor_info& gt)
	{
		t.n_dims = static_cast<int>(gt.n_dims);
		for (int d = 0; d < NNC_MAX_DIMS; ++d)
			t.ne[d] = (d < static_cast<int>(gt.n_dims)) ? static_cast<int>(gt.ne[d]) : 1;
		size_t elem = 1;
		switch (t.type)
		{
		case NNC_TYPE_F32: elem = 4;
			break;
		case NNC_TYPE_BF16: elem = 2;
			break;
		default: elem = 1; // the int8 / nibble stream of the split layouts
			break;
		}
		t.nb[0] = elem;
		for (int d = 1; d < NNC_MAX_DIMS; ++d) t.nb[d] = t.nb[d - 1] * static_cast<size_t>(t.ne[d - 1]);
	}

	// Load tensor `name` for `r`. Absent tensors leave `out` null and return
	// true; the caller decides which are required.
	bool load_tensor(nnc_model& m, load_progress& prog, const std::string& name, const role r,
	                 nnc_tensor*& out)
	{
		out = nullptr;
		const size_t idx = gguf_find_tensor(m.gguf, name);
		if (idx == SIZE_MAX) return true;
		const gguf_tensor_info& gt = m.gguf.tensors[idx];
		const uint64_t n = gguf_tensor_nelements(gt);
		const uint64_t src_bytes = gguf_tensor_nbytes(gt);
		const uint32_t be = decode_block_elems(gt.ggml_type);
		if (be == 0 || src_bytes == 0)
			return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
			                "tensor '%s' is %s, which nnc cannot decode "
			                "(it reads F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K and Q6_K)",
			                name.c_str(), gguf_ggml_type_name(gt.ggml_type));
		if (n == 0 || (n % be) != 0)
			return nnc_fail(NNC_ERR_BAD_FORMAT, "tensor '%s' has %" PRIu64 " elements, not whole %u-element blocks",
			                name.c_str(), n, be);
		const void* src = gguf_tensor_data(m.gguf, idx);
		if (!src)
			return nnc_fail(NNC_ERR_BAD_FORMAT, "tensor '%s' lies outside the file (is the download incomplete?)",
			                name.c_str());

		// An expert stack is decoded as n_mats independent matrices, each in
		// its own complete split layout, so one expert is a plain matrix at
		// an offset (expert_matrix) that every gemv path takes unchanged.
		const bool stacked = (r == role::experts);
		if (stacked && (gt.n_dims != 3 || gt.ne[2] == 0))
			return nnc_fail(NNC_ERR_BAD_FORMAT, "tensor '%s' is not a stack of expert matrices", name.c_str());
		const uint64_t n_mats = stacked ? gt.ne[2] : 1;
		const uint64_t n_per = n / n_mats;
		if (n_per % be != 0)
			return nnc_fail(NNC_ERR_BAD_FORMAT, "tensor '%s' does not divide into whole-block matrices", name.c_str());

		nnc_weight_format fmt = m.weights;
		if (r != role::vector && fmt != NNC_WEIGHTS_BF16 && (gt.ne[0] % 32) != 0)
			fmt = NNC_WEIGHTS_BF16; // the split layouts need whole 32-element blocks per row

		nnc_tensor& t = m.tensors.emplace_back();
		t.type = (r == role::vector) ? NNC_TYPE_F32 : fmt_type(fmt);
		set_shape(t, gt);

		const bool in_place = (r == role::vector)
			                      ? gt.ggml_type == 0
			                      : (fmt == NNC_WEIGHTS_BF16 && gt.ggml_type == 30);
		if (in_place)
		{
			t.data = const_cast<void*>(src);
		}
		else
		{
			const size_t per_mat = (r == role::vector) ? static_cast<size_t>(n_per) * sizeof(float)
				                       : fmt_bytes(fmt, n_per);
			const size_t total = per_mat * static_cast<size_t>(n_mats);
			auto buf = std::unique_ptr<uint8_t[]>(new(std::nothrow) uint8_t[total]);
			if (!buf)
				return nnc_fail(NNC_ERR_OUT_OF_MEMORY, "out of memory decoding tensor '%s' (%.1f MB)",
				                name.c_str(), static_cast<double>(total) / (1024.0 * 1024.0));
			const size_t src_per_mat = static_cast<size_t>(src_bytes / n_mats);
			for (uint64_t e = 0; e < n_mats; ++e)
			{
				decode_job job;
				job.ggml_type = gt.ggml_type;
				job.src = static_cast<const uint8_t*>(src) + e * src_per_mat;
				job.block_elems = be;
				job.block_bytes = static_cast<size_t>(src_bytes / (n / be));
				job.n = n_per;
				job.to_f32 = (r == role::vector);
				job.fmt = fmt;
				job.dst = buf.get() + e * per_mat;
				nnc_parallel_for(static_cast<uint32_t>((n_per + CHUNK - 1) / CHUNK), decode_range, &job);
				if (job.failed)
					return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "cannot decode tensor '%s'", name.c_str());
			}
			t.data = buf.get();
			m.decoded_bytes += total;
			m.buffers.push_back(std::move(buf));
		}
		out = &t;
		if (!prog.advance(src_bytes)) return nnc_fail(NNC_ERR_CANCELLED, "loading was cancelled");
		return true;
	}

	// ------------------------------------------------------------------------
	// Shape checks. The kernels trust these shapes, so a file that breaks
	// them is refused at load rather than read out of bounds later.
	// ------------------------------------------------------------------------

	bool check_vector(const nnc_tensor* t, const char* what, const int li, const int len)
	{
		if (!t) return true;
		if (t->n_dims != 1 || t->ne[0] != len)
		{
			if (li >= 0)
				return nnc_fail(NNC_ERR_BAD_FORMAT, "blk.%d.%s has %d values, expected %d", li, what, t->ne[0], len);
			return nnc_fail(NNC_ERR_BAD_FORMAT, "%s has %d values, expected %d", what, t->ne[0], len);
		}
		return true;
	}

	bool check_matrix(const nnc_tensor* t, const char* what, const int li, const int cols, const int rows)
	{
		if (!t) return true;
		if (t->n_dims != 2 || t->ne[0] != cols || (rows >= 0 && t->ne[1] != rows))
		{
			if (li >= 0)
				return nnc_fail(NNC_ERR_BAD_FORMAT, "blk.%d.%s is [%d, %d], expected [%d, %d]",
				                li, what, t->ne[0], t->ne[1], cols, rows);
			return nnc_fail(NNC_ERR_BAD_FORMAT, "%s is [%d, %d], expected [%d, %d]",
			                what, t->ne[0], t->ne[1], cols, rows);
		}
		return true;
	}

	bool require(const nnc_tensor* t, const int li, const char* what)
	{
		if (t) return true;
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "the model has no blk.%d.%s.weight", li, what);
	}

	bool check_experts(const nnc_tensor* t, const char* what, const int li, const int cols, const int rows,
	                   const int n_expert)
	{
		if (!t) return true;
		if (t->n_dims != 3 || t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != n_expert)
			return nnc_fail(NNC_ERR_BAD_FORMAT, "blk.%d.%s is [%d, %d, %d], expected [%d, %d, %d]", li, what,
			                t->ne[0], t->ne[1], t->ne[2], cols, rows, n_expert);
		return true;
	}

	// ------------------------------------------------------------------------
	// Weight access helpers. Tensors store ne[0] = cols (the fast dim, the
	// input length of the dot) and ne[1] = rows (the output length).
	// Quantized types use the split layout, so the secondary arrays are found
	// by offset from `data`.
	// ------------------------------------------------------------------------

	const uint16_t* q8_scales(const nnc_tensor* T)
	{
		return reinterpret_cast<const uint16_t*>(
			static_cast<const int8_t*>(T->data)
			+ static_cast<size_t>(T->ne[1]) * static_cast<size_t>(T->ne[0]));
	}

	const uint16_t* q4_scales(const nnc_tensor* T)
	{
		return reinterpret_cast<const uint16_t*>(
			static_cast<const uint8_t*>(T->data)
			+ static_cast<size_t>(T->ne[1]) * static_cast<size_t>(T->ne[0]) / 2);
	}

	const uint16_t* q4_biases(const nnc_tensor* T)
	{
		return q4_scales(T)
			+ static_cast<size_t>(T->ne[1]) * (static_cast<size_t>(T->ne[0]) / 32);
	}

	// Bytes of weight storage a gemv over `T` has to stream. Used only
	// for the perf report's effective-bandwidth column.
	uint64_t weight_bytes(const nnc_tensor* T)
	{
		const uint64_t n = static_cast<uint64_t>(T->ne[0]) * static_cast<uint64_t>(T->ne[1]);
		switch (T->type)
		{
		case NNC_TYPE_Q8_0: return n + n / 32 * 2;
		case NNC_TYPE_Q4_S: return n / 2 + n / 32 * 4;
		default: return n * 2;
		}
	}

	// Largest `cols` the stack-resident block-sum scratch supports.
	constexpr int MAX_XSUM_BLOCKS = 1024; // cols <= 32768

	// Most experts per layer; bounds the stack-resident bookkeeping.
	constexpr int MAX_EXPERTS = 512;

	// Expert `e` of a stacked [cols, rows, n_expert] tensor as the 2D
	// matrix the gemv paths take: the loader stores each expert as a
	// complete split-layout matrix, weight_bytes() long, back to back.
	nnc_tensor expert_matrix(const nnc_tensor* T, const int e)
	{
		nnc_tensor v = *T;
		v.n_dims = 2;
		v.ne[2] = 1;
		v.data = static_cast<uint8_t*>(T->data) + static_cast<size_t>(e) * static_cast<size_t>(weight_bytes(T));
		return v;
	}
}

void model_gemv(const nnc_tensor* T, const float* x, float* y)
{
	const uint32_t cols = static_cast<uint32_t>(T->ne[0]);
	const uint32_t rows = static_cast<uint32_t>(T->ne[1]);
	if (T->type == NNC_TYPE_Q8_0)
	{
		nnc_gemv_q8_0_f32x(static_cast<const int8_t*>(T->data), q8_scales(T),
		                   x, y, rows, cols);
		return;
	}
	if (T->type == NNC_TYPE_Q4_S)
	{
		// The per-block sums of x are shared by every row, so they are
		// computed once here rather than inside the row kernel.
		NNC_ASSERT(cols / 32 <= MAX_XSUM_BLOCKS);
		alignas(32) float xsum[MAX_XSUM_BLOCKS];
		nnc_block_sums_f32(x, xsum, cols);
		nnc_gemv_q4_s_f32x(static_cast<const uint8_t*>(T->data), q4_scales(T),
		                   q4_biases(T), x, xsum, y, rows, cols);
		return;
	}
	nnc_gemv_bf16w_f32x(T->data, x, y, rows, cols);
}

void model_embed_row(const nnc_tensor* T, float* y, const int token, const size_t n, const float scale)
{
	if (T->type == NNC_TYPE_Q8_0)
	{
		nnc_embed_row_q8_0(y, static_cast<const int8_t*>(T->data),
		                   q8_scales(T), token, n, scale);
		return;
	}
	if (T->type == NNC_TYPE_Q4_S)
	{
		nnc_embed_row_q4_s(y, static_cast<const uint8_t*>(T->data),
		                   q4_scales(T), q4_biases(T), token, n, scale);
		return;
	}
	nnc_embed_row_bf16(y, T->data, token, n, scale);
}

// ----------------------------------------------------------------------------
// Loading.
// ----------------------------------------------------------------------------

nnc_model::~nnc_model()
{
	gguf_unmap(gguf);
}

namespace
{
	bool read_hparams(nnc_model& m)
	{
		const gguf_file& f = m.gguf;
		auto& h = m.hparams;
		const std::string& arch = h.arch;
		auto akey = [&](const char* suffix) { return arch + "." + suffix; };

		bool ok = kv_int(f, akey("context_length"), h.n_ctx);
		ok = kv_int(f, akey("embedding_length"), h.n_embd) && ok;
		ok = kv_int(f, akey("block_count"), h.n_layer) && ok;
		ok = kv_int(f, akey("attention.head_count"), h.n_head) && ok;
		if (!ok || h.n_embd <= 0 || h.n_layer <= 0 || h.n_head <= 0)
			return nnc_fail(NNC_ERR_BAD_FORMAT,
			                "the file lacks the basic %s dimensions (context length, embedding "
			                "length, block count, head count)", arch.c_str());
		if (h.n_ctx <= 0) h.n_ctx = 4096;

		// A Qwen3.5 file may carry its multi-token-prediction block after the
		// trunk, counted in block_count. nnc does not run it.
		int n_nextn = 0;
		if (h.arch_id == model_arch::qwen35 && kv_int(f, akey("nextn_predict_layers"), n_nextn) && n_nextn > 0
			&& n_nextn < h.n_layer)
			h.n_layer -= n_nextn;
		if (!kv_int(f, akey("attention.head_count_kv"), h.n_head_kv) || h.n_head_kv <= 0)
			h.n_head_kv = h.n_head;

		// llama doesn't publish key_length; derive from rope.dimension_count or n_embd/n_head.
		if (!kv_int(f, akey("attention.key_length"), h.head_dim) || h.head_dim <= 0)
		{
			int rd = 0;
			if (kv_int(f, akey("rope.dimension_count"), rd) && rd > 0) h.head_dim = rd;
			else h.head_dim = h.n_embd / h.n_head;
		}
		if (!kv_float(f, akey("rope.freq_base"), h.rope_freq_base) || h.rope_freq_base <= 0)
			h.rope_freq_base = 10000.0f;
		if (!kv_float(f, akey("attention.layer_norm_rms_epsilon"), h.rms_eps) || h.rms_eps <= 0)
			h.rms_eps = 1e-6f;

		kv_int(f, akey("attention.key_length_swa"), h.head_dim_swa);
		if (h.head_dim_swa <= 0) h.head_dim_swa = h.head_dim;
		kv_int(f, akey("rope.dimension_count"), h.rope_dim);
		if (h.rope_dim <= 0) h.rope_dim = h.head_dim; // gemma3 omits this
		kv_int(f, akey("rope.dimension_count_swa"), h.rope_dim_swa);
		if (h.rope_dim_swa <= 0) h.rope_dim_swa = h.rope_dim;
		kv_float(f, akey("rope.freq_base_swa"), h.rope_freq_base_swa);
		if (h.rope_freq_base_swa <= 0)
		{
			// gemma3 uses a hardcoded local-rope theta of 10000 for sliding
			// layers (HF: rope_local_base_freq); it is not stored in the GGUF.
			h.rope_freq_base_swa = (h.arch_id == model_arch::gemma3) ? 10000.0f : h.rope_freq_base;
		}
		kv_int(f, akey("attention.sliding_window"), h.sliding_window);
		kv_int(f, akey("attention.shared_kv_layers"), h.shared_kv_layers);
		kv_int(f, akey("embedding_length_per_layer_input"), h.ple_dim);
		kv_int(f, akey("expert_count"), h.n_expert);
		kv_int(f, akey("expert_used_count"), h.n_expert_used);
		if (h.n_expert < 0) h.n_expert = 0;
		kv_float(f, akey("final_logit_softcapping"), h.final_logit_softcap);
		kv_int(f, "general.file_type", h.file_type);

		if (!h.causal)
		{
			bool causal = false;
			if (kv_bool(f, akey("attention.causal"), causal) && causal)
				return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "this %s file asks for causal attention", arch.c_str());
			// llama.cpp's pooling enum: 0 none, 1 mean, 2 cls, 3 last, 4 rank.
			int pooling = static_cast<int>(model_pooling::mean);
			kv_int(f, akey("pooling_type"), pooling);
			if (pooling < 1 || pooling > 3)
				return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
				                "the model pools its output with method %d; nnc does mean, first-token and "
				                "last-token pooling", pooling);
			h.pooling = static_cast<model_pooling>(pooling);
		}

		if (h.shared_kv_layers < 0 || h.shared_kv_layers > h.n_layer - 2) h.shared_kv_layers = 0;
		h.n_layer_kv = h.n_layer - h.shared_kv_layers;
		if (h.n_embd_target > 0) h.n_layer_kv = 0; // a drafter reads its target's cache
		// A bidirectional model's layers share one K/V buffer, each rewriting
		// it before it attends (model_kv_init), so no layer can read another's.
		if (!h.causal && h.shared_kv_layers > 0)
			return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
			                "this %s model shares K/V between layers, which nnc does not support for "
			                "bidirectional models", arch.c_str());

		if (!kv_float(f, akey("attention.scale"), h.attention_scale) || h.attention_scale <= 0)
		{
			// Gemma 4 folds the scale into its q_norm (1.0). Gemma 3 27B scales
			// by 1/sqrt(n_embd / n_head) rather than 1/sqrt(head_dim), per
			// gemma_pytorch's config; everything else uses 1/sqrt(head_dim).
			if (h.arch_id == model_arch::gemma4)
				h.attention_scale = 1.0f;
			else if (h.arch_id == model_arch::gemma3 && h.n_layer == 62)
				h.attention_scale = 1.0f / std::sqrt(static_cast<float>(h.n_embd / h.n_head));
			else
				h.attention_scale = 1.0f / std::sqrt(static_cast<float>(h.head_dim));
		}

		// RoPE linear scaling: theta_i is scaled by `1 / factor` (i.e.
		// effective_pos = pos / factor). Both gemma3-4B and recent llama
		// finetunes use this.
		{
			std::string scaling_type;
			kv_str(f, akey("rope.scaling.type"), scaling_type);
			float factor = 0.0f;
			kv_float(f, akey("rope.scaling.factor"), factor);
			if (scaling_type == "linear" && factor > 0) h.rope_freq_scale = 1.0f / factor;
		}

		// Llama's GGUF converter permutes Q/K for adjacent-pair RoPE; Gemma
		// and Qwen keep the half-split (NeoX) layout. Qwen3.5's interleaved
		// multimodal RoPE is NeoX RoPE over its first rope_dim lanes when, as
		// for text, every position component is the token's position.
		h.rope_mode = (h.arch_id == model_arch::llama) ? NNC_ROPE_NORM : NNC_ROPE_NEOX;
		const bool gemma = h.arch_id == model_arch::gemma3 || h.arch_id == model_arch::gemma4;
		h.embd_scale = gemma ? std::sqrt(static_cast<float>(h.n_embd)) : 1.0f;

		if (h.arch_id == model_arch::qwen35)
		{
			int d_inner = 0;
			ok = kv_int(f, akey("ssm.conv_kernel"), h.ssm_d_conv);
			ok = kv_int(f, akey("ssm.state_size"), h.ssm_d_k) && ok;
			ok = kv_int(f, akey("ssm.group_count"), h.ssm_n_k) && ok;
			ok = kv_int(f, akey("ssm.time_step_rank"), h.ssm_n_v) && ok;
			ok = kv_int(f, akey("ssm.inner_size"), d_inner) && ok;
			if (!ok || h.ssm_d_conv < 1 || h.ssm_d_conv > 16 || h.ssm_d_k <= 0 || h.ssm_d_k > 1024 || h.ssm_n_k <= 0
				|| h.ssm_n_v <= 0 || h.ssm_n_v % h.ssm_n_k != 0 || d_inner <= 0 || d_inner % h.ssm_n_v != 0)
				return nnc_fail(NNC_ERR_BAD_FORMAT, "the file lacks or garbles the Gated DeltaNet dimensions (%s.ssm.*)",
				                arch.c_str());
			h.ssm_d_v = d_inner / h.ssm_n_v;
		}
		return true;
	}

	// A Gated DeltaNet layer's tensors, checked against the dimensions in
	// hparams. Its convolution taps are transposed here, once, into the
	// time-major order nnc_conv1d_silu_f32 reads.
	bool resolve_delta_net(nnc_model& m, model_layer& L, const int li)
	{
		const auto& h = m.hparams;
		const char* const names[] = {"attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_conv1d",
		                             "ssm_dt", "ssm_a", "ssm_norm", "ssm_out"};
		const nnc_tensor* const parts[] = {L.ssm_qkv, L.ssm_gate, L.ssm_alpha, L.ssm_beta, L.ssm_conv1d,
		                                   L.ssm_dt, L.ssm_a, L.ssm_norm, L.ssm_out};
		for (size_t i = 0; i < std::size(parts); ++i)
			if (!parts[i])
				return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "the model has no blk.%d.%s tensor for its Gated DeltaNet "
				                "layer", li, names[i]);
		const int n_embd = h.n_embd;
		const int v_dim = h.ssm_n_v * h.ssm_d_v;
		const int conv_dim = 2 * h.ssm_n_k * h.ssm_d_k + v_dim;
		if (!check_matrix(L.ssm_qkv, "attn_qkv", li, n_embd, conv_dim)
			|| !check_matrix(L.ssm_gate, "attn_gate", li, n_embd, v_dim)
			|| !check_matrix(L.ssm_alpha, "ssm_alpha", li, n_embd, h.ssm_n_v)
			|| !check_matrix(L.ssm_beta, "ssm_beta", li, n_embd, h.ssm_n_v)
			|| !check_matrix(L.ssm_conv1d, "ssm_conv1d", li, h.ssm_d_conv, conv_dim)
			|| !check_vector(L.ssm_dt, "ssm_dt", li, h.ssm_n_v) || !check_vector(L.ssm_a, "ssm_a", li, h.ssm_n_v)
			|| !check_vector(L.ssm_norm, "ssm_norm", li, h.ssm_d_v)
			|| !check_matrix(L.ssm_out, "ssm_out", li, v_dim, n_embd))
			return false;

		const size_t taps = static_cast<size_t>(h.ssm_d_conv) * static_cast<size_t>(conv_dim);
		auto buf = std::unique_ptr<uint8_t[]>(new(std::nothrow) uint8_t[taps * sizeof(float)]);
		if (!buf) return nnc_fail(NNC_ERR_OUT_OF_MEMORY, "out of memory loading blk.%d.ssm_conv1d", li);
		auto* t = reinterpret_cast<float*>(buf.get());
		const auto* w = static_cast<const float*>(L.ssm_conv1d->data);
		for (size_t c = 0; c < static_cast<size_t>(conv_dim); ++c)
			for (size_t j = 0; j < static_cast<size_t>(h.ssm_d_conv); ++j)
				t[j * static_cast<size_t>(conv_dim) + c] = w[c * static_cast<size_t>(h.ssm_d_conv) + j];
		L.ssm_conv_taps = t;
		m.decoded_bytes += taps * sizeof(float);
		m.buffers.push_back(std::move(buf));
		L.kv_layer = -1;
		return true;
	}

	// An attention layer's heads, and the K/V cache it reads.
	bool resolve_attention(nnc_model& m, model_layer& L, const int li)
	{
		const auto& h = m.hparams;
		const int n_embd = h.n_embd;
		if (!require(L.attn_q, li, "attn_q") || !require(L.attn_output, li, "attn_output")) return false;
		const bool owns_kv = li < h.n_layer_kv;
		if (owns_kv && !require(L.attn_k, li, "attn_k")) return false;

		L.head_dim = L.attn_q_norm ? L.attn_q_norm->ne[0] : (L.sliding_window ? h.head_dim_swa : h.head_dim);
		if (L.head_dim <= 0 || (L.head_dim & 1) != 0 || L.head_dim > 512)
			return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "blk.%d has head size %d; nnc supports even sizes up to 512",
			                li, L.head_dim);
		if (!check_matrix(L.attn_q, "attn_q", li, n_embd, -1)) return false;
		// Qwen3.5's projection gives each head its query, then a gate for
		// the head's output.
		L.attn_gated = h.arch_id == model_arch::qwen35;
		const int per_head = L.attn_gated ? 2 * L.head_dim : L.head_dim;
		const int q_rows = L.attn_q->ne[1];
		if (q_rows % per_head != 0)
			return nnc_fail(NNC_ERR_BAD_FORMAT, "blk.%d.attn_q has %d outputs, not whole heads of %d",
			                li, q_rows, per_head);
		L.n_head = q_rows / per_head;
		const int q_dim = L.n_head * L.head_dim;

		int kv_dim = 0;
		if (owns_kv)
		{
			if (!check_matrix(L.attn_k, "attn_k", li, n_embd, -1)) return false;
			kv_dim = L.attn_k->ne[1];
			if (!check_matrix(L.attn_v, "attn_v", li, n_embd, kv_dim)) return false;
			L.kv_layer = li;
		}
		else if (h.n_embd_target > 0)
		{
			// An MTP drafter reads its target's cache; model_draft_init
			// checks that these shapes match it.
			int n_kv = h.n_head_kv;
			kv_int_at(m.gguf, h.arch + ".attention.head_count_kv", li, n_kv);
			L.kv_layer = -1;
			kv_dim = n_kv * L.head_dim;
		}
		else
		{
			// Shared KV (Gemma 4): the trailing layers read the cache of the
			// last cache-owning layer of the same kind, global or sliding.
			L.kv_layer = h.n_layer_kv - (L.sliding_window ? 2 : 1);
			if (L.kv_layer < 0 || m.layers[L.kv_layer].sliding_window != L.sliding_window)
				return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
				                "blk.%d shares a KV cache in a layer pattern nnc does not recognise", li);
			const model_layer& S = m.layers[L.kv_layer];
			if (S.head_dim != L.head_dim)
				return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "blk.%d shares a KV cache of a different head size", li);
			kv_dim = S.n_head_kv * S.head_dim;
		}
		if (kv_dim <= 0 || kv_dim % L.head_dim != 0)
			return nnc_fail(NNC_ERR_BAD_FORMAT, "blk.%d has %d K/V outputs, not whole heads of %d",
			                li, kv_dim, L.head_dim);
		L.n_head_kv = kv_dim / L.head_dim;
		if (L.n_head % L.n_head_kv != 0)
			return nnc_fail(NNC_ERR_BAD_FORMAT, "blk.%d has %d query heads for %d KV heads",
			                li, L.n_head, L.n_head_kv);

		if (!check_matrix(L.attn_output, "attn_output", li, q_dim, n_embd)
			|| !check_vector(L.attn_q_norm, "attn_q_norm", li, L.head_dim)
			|| !check_vector(L.attn_k_norm, "attn_k_norm", li, L.head_dim)
			|| !check_vector(L.attn_q_b, "attn_q.bias", li, q_rows)
			|| !check_vector(L.attn_k_b, "attn_k.bias", li, kv_dim)
			|| !check_vector(L.attn_v_b, "attn_v.bias", li, kv_dim))
			return false;

		int n_rot = L.sliding_window ? h.rope_dim_swa : h.rope_dim;
		if (n_rot > L.head_dim) n_rot = L.head_dim;
		L.n_rot = n_rot & ~1;

		// RoPE frequency factors: Llama 3's long-context scaling on every
		// layer, Gemma 4's proportional RoPE on its global layers only.
		const bool wants_freqs = h.arch_id == model_arch::llama
			|| (h.arch_id == model_arch::gemma4 && !L.sliding_window);
		if (wants_freqs && m.rope_freqs && m.rope_freqs->ne[0] == L.n_rot / 2)
			L.rope_freqs = static_cast<const float*>(m.rope_freqs->data);
		return true;
	}

	// Per-layer shapes, from the tensors themselves (some GGUFs disagree with
	// their own key_length), checked against everything the forward pass
	// will assume.
	bool resolve_layers(nnc_model& m)
	{
		auto& h = m.hparams;
		const int n_embd = h.n_embd;
		for (int li = 0; li < h.n_layer; ++li)
		{
			model_layer& L = m.layers[li];
			if (!require(L.attn_norm, li, "attn_norm") || !require(L.ffn_norm, li, "ffn_norm")
				|| !require(L.ffn_gate, li, "ffn_gate") || !require(L.ffn_up, li, "ffn_up")
				|| !require(L.ffn_down, li, "ffn_down"))
				return false;
			if (!(L.rs_index >= 0 ? resolve_delta_net(m, L, li) : resolve_attention(m, L, li))) return false;

			if (!check_matrix(L.ffn_gate, "ffn_gate", li, n_embd, -1)) return false;
			L.n_ff = L.ffn_gate->ne[1];
			if (!check_matrix(L.ffn_up, "ffn_up", li, n_embd, L.n_ff)) return false;
			if (!check_matrix(L.ffn_down, "ffn_down", li, L.n_ff, n_embd)) return false;
			if (!check_vector(L.attn_norm, "attn_norm", li, n_embd) || !check_vector(L.ffn_norm, "ffn_norm", li, n_embd)
				|| !check_vector(L.post_attention_norm, "post_attention_norm", li, n_embd)
				|| !check_vector(L.post_ffw_norm, "post_ffw_norm", li, n_embd)
				|| !check_vector(L.post_norm, "post_norm", li, n_embd))
				return false;
			if (L.layer_output_scale && L.layer_output_scale->ne[0] < 1)
				return nnc_fail(NNC_ERR_BAD_FORMAT, "blk.%d.layer_output_scale is empty", li);
			if (L.inp_gate || L.proj)
			{
				if (!L.inp_gate || !L.proj || h.ple_dim <= 0)
					return nnc_fail(NNC_ERR_BAD_FORMAT, "blk.%d has half a per-layer-embedding block", li);
				if (!check_matrix(L.inp_gate, "inp_gate", li, n_embd, h.ple_dim)
					|| !check_matrix(L.proj, "proj", li, h.ple_dim, n_embd))
					return false;
			}

			if (L.ffn_gate_inp || L.ffn_gate_up_exps || L.ffn_gate_exps || L.ffn_up_exps || L.ffn_down_exps)
			{
				if (!L.ffn_gate_inp || !L.ffn_gate_inp_scale || !L.ffn_down_exps
					|| !(L.ffn_gate_up_exps || (L.ffn_gate_exps && L.ffn_up_exps))
					|| !L.pre_ffw_norm_2 || !L.post_ffw_norm_1 || !L.post_ffw_norm_2)
					return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
					                "blk.%d has a mixture-of-experts layout nnc does not recognise", li);
				const int n_expert = h.n_expert;
				if (n_expert <= 0 || n_expert > MAX_EXPERTS || h.n_expert_used <= 0 || h.n_expert_used > n_expert)
					return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
					                "the model routes each token to %d of %d experts; nnc supports up to %d",
					                h.n_expert_used, n_expert, MAX_EXPERTS);
				const int n_ff_exp = L.ffn_down_exps->ne[0];
				if (n_ff_exp <= 0) return nnc_fail(NNC_ERR_BAD_FORMAT, "blk.%d has empty experts", li);
				if (!check_matrix(L.ffn_gate_inp, "ffn_gate_inp", li, n_embd, n_expert)
					|| !check_vector(L.ffn_gate_inp_scale, "ffn_gate_inp.scale", li, n_embd)
					|| !check_experts(L.ffn_gate_up_exps, "ffn_gate_up_exps", li, n_embd, 2 * n_ff_exp, n_expert)
					|| !check_experts(L.ffn_gate_exps, "ffn_gate_exps", li, n_embd, n_ff_exp, n_expert)
					|| !check_experts(L.ffn_up_exps, "ffn_up_exps", li, n_embd, n_ff_exp, n_expert)
					|| !check_experts(L.ffn_down_exps, "ffn_down_exps", li, n_ff_exp, n_embd, n_expert)
					|| !check_vector(L.ffn_down_exps_scale, "ffn_down_exps.scale", li, n_expert)
					|| !check_vector(L.pre_ffw_norm_2, "pre_ffw_norm_2", li, n_embd)
					|| !check_vector(L.post_ffw_norm_1, "post_ffw_norm_1", li, n_embd)
					|| !check_vector(L.post_ffw_norm_2, "post_ffw_norm_2", li, n_embd))
					return false;
				L.n_ff_exp = n_ff_exp;
			}
		}
		return true;
	}
}

bool model_load(const std::string& path, nnc_model& m, const nnc_weight_format fmt,
                const nnc_progress_callback progress, void* progress_user)
{
	if (fmt != NNC_WEIGHTS_Q8_0 && fmt != NNC_WEIGHTS_Q4 && fmt != NNC_WEIGHTS_BF16)
		return nnc_fail(NNC_ERR_INVALID_ARGUMENT, "unknown weight format %d", static_cast<int>(fmt));

	const int64_t t0 = nnc_time_us();
	m.path = path;
	m.weights = fmt;
	if (!gguf_mmap(path, m.gguf)) return false;
	const gguf_file& f = m.gguf;

	// --- architecture -------------------------------------------------
	auto& h = m.hparams;
	if (!kv_str(f, "general.architecture", h.arch))
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "'%s' does not say what model it holds (no general.architecture)",
		                path.c_str());
	if (h.arch == "gemma3") h.arch_id = model_arch::gemma3;
	else if (h.arch == "gemma4") h.arch_id = model_arch::gemma4;
	else if (h.arch == "llama") h.arch_id = model_arch::llama;
	else if (h.arch == "qwen2") h.arch_id = model_arch::qwen2;
	else if (h.arch == "qwen3") h.arch_id = model_arch::qwen3;
	else if (h.arch == "qwen35") h.arch_id = model_arch::qwen35;
	else if (h.arch == "gemma-embedding" || h.arch == "gemma-embedding2")
	{
		// EmbeddingGemma: Gemma 3 (or, for version 2, Gemma 4) layers that
		// attend both ways, pooled into one vector.
		h.arch_id = (h.arch == "gemma-embedding") ? model_arch::gemma3 : model_arch::gemma4;
		h.causal = false;
		h.pooling = model_pooling::mean;
	}
	else if (h.arch == "gemma4-assistant")
	{
		// Gemma 4's MTP drafter: query-only Gemma 4 layers over the target
		// model's K/V cache (model_draft_step).
		h.arch_id = model_arch::gemma4;
		if (!kv_int(f, "gemma4-assistant.embedding_length_out", h.n_embd_target) || h.n_embd_target <= 0)
			return nnc_fail(NNC_ERR_BAD_FORMAT, "the drafter does not say how wide its target model is");
	}
	else if (h.arch == "clip")
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
		                "'%s' is a vision projector (mmproj), not a language model", path.c_str());
	else if (h.arch == "qwen4exp")
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
		                "'%s' is Qwen3.8-Flash-Next (qwen4exp), whose sparse attention, hyper-connections, n-gram "
		                "embeddings and 512 experts nnc does not implement; nnc runs Qwen's dense models (qwen2, "
		                "qwen3 and qwen35, which includes Qwen3.8-27B)", path.c_str());
	else if (h.arch == "qwen2moe" || h.arch == "qwen3moe" || h.arch == "qwen35moe" || h.arch == "qwen3next")
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
		                "'%s' is a mixture-of-experts Qwen model (%s); nnc runs Qwen's dense models (qwen2, qwen3 "
		                "and qwen35) and Gemma 4's experts only", path.c_str(), h.arch.c_str());
	else
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
		                "'%s' is a '%s' model; nnc runs gemma3, gemma4, llama, qwen2, qwen3, qwen35, "
		                "gemma-embedding and gemma-embedding2", path.c_str(), h.arch.c_str());
	const bool qwen = h.arch_id == model_arch::qwen2 || h.arch_id == model_arch::qwen3
		|| h.arch_id == model_arch::qwen35;

	// Weights rotated by a Hadamard transform need the rotation undone at
	// run time, which nnc does not do.
	if (gguf_find_kv(f, "prism.hadamard.version"))
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
		                "'%s' holds Hadamard-rotated weights, which nnc cannot run", path.c_str());
	// Qwen3-Embedding is a qwen3 model that pools its output into a vector.
	if (qwen && gguf_find_kv(f, h.arch + ".pooling_type"))
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
		                "'%s' is a Qwen text-embedding model, which nnc does not run", path.c_str());

	// Gemma 4's mixture of experts (a dense MLP plus routed experts) is the
	// only one nnc runs; other architectures route differently.
	int n_expert = 0;
	if (h.arch_id != model_arch::gemma4
		&& ((kv_int(f, h.arch + ".expert_count", n_expert) && n_expert > 0)
			|| gguf_find_tensor(f, "blk.0.ffn_gate_inp.weight") != SIZE_MAX))
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
		                "'%s' is a mixture-of-experts %s model; nnc runs Gemma 4's experts only", path.c_str(),
		                h.arch.c_str());

	if (!kv_str(f, "general.name", m.name) || m.name.empty())
	{
		const size_t slash = path.find_last_of("/\\");
		m.name = (slash == std::string::npos) ? path : path.substr(slash + 1);
	}

	if (!read_hparams(m)) return false;
	// llama.cpp's mtp-*.gguf files for Qwen3.5 hold only the trailing
	// multi-token-prediction block, with the embeddings it shares.
	if (h.arch_id == model_arch::qwen35 && gguf_find_tensor(f, "blk.0.attn_norm.weight") == SIZE_MAX
		&& gguf_find_tensor(f, "blk." + std::to_string(h.n_layer) + ".nextn.eh_proj.weight") != SIZE_MAX)
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL,
		                "'%s' holds only a Qwen multi-token-prediction drafter, which nnc does not run",
		                path.c_str());
	if (!vocab_load(f, m.vocab)) return false;
	h.n_vocab = m.vocab.n_vocab();

	// --- tensors ------------------------------------------------------
	load_progress prog;
	prog.fn = progress;
	prog.user = progress_user;
	for (const auto& t : f.tensors)
	{
		prog.total += gguf_tensor_nbytes(t);
		m.n_params += gguf_tensor_nelements(t);
	}
	gguf_prefetch(m.gguf);
	if (!prog.advance(0)) return nnc_fail(NNC_ERR_CANCELLED, "loading was cancelled");

	// Every tensor a load consumes is marked, so that a Qwen file holding
	// tensors nnc would silently ignore can be refused.
	std::vector<uint8_t> used(f.tensors.size(), 0);
	auto load = [&](const std::string& name, const role r, nnc_tensor*& out)
	{
		if (!load_tensor(m, prog, name, r, out)) return false;
		if (out) used[gguf_find_tensor(f, name)] = 1;
		return true;
	};
	// A generative model's output.weight is its lm_head. An embedding model
	// has no lm_head: EmbeddingGemma 2's output.weight projects the pooled
	// state, and EmbeddingGemma's projections are dense_2 and dense_3.
	const bool embedding = !h.causal;
	if (!load("token_embd.weight", role::matrix, m.token_embd)
		|| !load("output_norm.weight", role::vector, m.output_norm)
		|| (!embedding && !load("output.weight", role::matrix, m.output))
		|| !load("rope_freqs.weight", role::vector, m.rope_freqs)
		|| !load("per_layer_token_embd.weight", role::matrix, m.per_layer_token_embd)
		|| !load("per_layer_model_proj.weight", role::matrix, m.per_layer_model_proj)
		|| !load("per_layer_proj_norm.weight", role::vector, m.per_layer_proj_norm))
		return false;
	if (!m.token_embd)
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "the model has no token_embd.weight");
	if (!check_matrix(m.token_embd, "token_embd.weight", -1, h.n_embd, h.n_vocab)
		|| !check_matrix(m.output, "output.weight", -1, h.n_embd, h.n_vocab)
		|| !check_vector(m.output_norm, "output_norm.weight", -1, h.n_embd))
		return false;

	if (h.n_embd_target > 0)
	{
		if (!load("nextn.pre_projection.weight", role::matrix, m.nextn_pre)
			|| !load("nextn.post_projection.weight", role::matrix, m.nextn_post))
			return false;
		if (!m.nextn_pre || !m.nextn_post || !m.output_norm)
			return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "the drafter lacks its input or output projection");
		if (!check_matrix(m.nextn_pre, "nextn.pre_projection.weight", -1, 2 * h.n_embd_target, h.n_embd)
			|| !check_matrix(m.nextn_post, "nextn.post_projection.weight", -1, h.n_embd, h.n_embd_target))
			return false;
	}

	if (embedding)
	{
		const bool v2 = h.arch_id == model_arch::gemma4;
		const char* const v1_names[] = {"dense_2.weight", "dense_3.weight"};
		const char* const v2_names[] = {"output.weight"};
		const char* const* names = v2 ? v2_names : v1_names;
		const size_t n_names = v2 ? std::size(v2_names) : std::size(v1_names);
		int width = h.n_embd;
		for (size_t i = 0; i < n_names; ++i)
		{
			nnc_tensor* t = nullptr;
			if (!load(names[i], role::matrix, t)) return false;
			if (!t) continue; // EmbeddingGemma converted without its dense layers
			if (!check_matrix(t, names[i], -1, width, -1)) return false;
			width = static_cast<int>(t->ne[1]);
			m.embd_proj.push_back(t);
		}
		if (v2 && m.embd_proj.empty())
			return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "the model has no output.weight projection");
		int declared = 0;
		if (kv_int(f, h.arch + ".embedding_length_out", declared) && declared != width)
			return nnc_fail(NNC_ERR_BAD_FORMAT, "the model says its embeddings have %d values, but its "
			                "projections make %d", declared, width);
		h.n_embd_out = width;
	}

	// The per-token table is optional: EmbeddingGemma 2 derives its
	// per-layer inputs from the projection alone.
	const bool have_ple = m.per_layer_token_embd || m.per_layer_model_proj || m.per_layer_proj_norm;
	if (have_ple)
	{
		const int ple_total = h.ple_dim * h.n_layer;
		if (h.ple_dim <= 0 || !m.per_layer_model_proj || !m.per_layer_proj_norm)
			return nnc_fail(NNC_ERR_BAD_FORMAT, "the per-layer embedding tables are incomplete");
		if (!check_matrix(m.per_layer_token_embd, "per_layer_token_embd.weight", -1, ple_total, h.n_vocab)
			|| !check_matrix(m.per_layer_model_proj, "per_layer_model_proj.weight", -1, h.n_embd, ple_total)
			|| !check_vector(m.per_layer_proj_norm, "per_layer_proj_norm.weight", -1, h.ple_dim))
			return false;
	}
	else
	{
		h.ple_dim = 0;
	}

	m.layers.resize(h.n_layer);
	// Qwen3.5: which layers are Gated DeltaNet layers. Newer files list
	// them; older ones give the period of the attention layers instead.
	if (h.arch_id == model_arch::qwen35)
	{
		int interval = 4;
		kv_int(f, h.arch + ".full_attention_interval", interval);
		for (int li = 0; li < h.n_layer; ++li)
		{
			bool recurrent = interval > 0 && (li + 1) % interval != 0;
			kv_flag_at(f, h.arch + ".attention.recurrent_layers", li, recurrent);
			if (recurrent) m.layers[li].rs_index = h.n_layer_rs++;
		}
	}
	const gguf_value* swp = gguf_find_kv(f, h.arch + ".attention.sliding_window_pattern");
	int swa_period = 6;
	if (swp && swp->type != GGUF_TYPE_ARRAY) kv_int(f, h.arch + ".attention.sliding_window_pattern", swa_period);
	for (int li = 0; li < h.n_layer; ++li)
	{
		model_layer& L = m.layers[li];
		if (swp && swp->type == GGUF_TYPE_ARRAY && swp->arr.size() == static_cast<size_t>(h.n_layer))
			L.sliding_window = (swp->arr[li].u64 != 0);
		else if ((h.arch_id == model_arch::gemma3 || embedding) && h.sliding_window > 0)
			// Every sixth layer is global; the rest slide. Matches Hugging
			// Face's `is_sliding = bool((i + 1) % sliding_window_pattern)`.
			L.sliding_window = swa_period <= 0 || ((li + 1) % swa_period) != 0;
		if (h.sliding_window <= 0) L.sliding_window = false;

		const std::string p = "blk." + std::to_string(li) + ".";
		if (!load(p + "attn_norm.weight", role::vector, L.attn_norm)
			|| !load(p + "ffn_norm.weight", role::vector, L.ffn_norm)
			|| !load(p + "post_attention_norm.weight", role::vector, L.post_attention_norm)
			|| !load(p + "post_ffw_norm.weight", role::vector, L.post_ffw_norm)
			|| !load(p + "post_norm.weight", role::vector, L.post_norm)
			|| !load(p + "attn_q_norm.weight", role::vector, L.attn_q_norm)
			|| !load(p + "attn_k_norm.weight", role::vector, L.attn_k_norm)
			|| !load(p + "layer_output_scale.weight", role::vector, L.layer_output_scale)
			|| !load(p + "attn_q.weight", role::matrix, L.attn_q)
			|| !load(p + "attn_output.weight", role::matrix, L.attn_output)
			|| !load(p + "ffn_gate.weight", role::matrix, L.ffn_gate)
			|| !load(p + "ffn_up.weight", role::matrix, L.ffn_up)
			|| !load(p + "ffn_down.weight", role::matrix, L.ffn_down)
			|| !load(p + "inp_gate.weight", role::matrix, L.inp_gate)
			|| !load(p + "proj.weight", role::matrix, L.proj)
			|| !load(p + "ffn_gate_inp.weight", role::vector, L.ffn_gate_inp)
			|| !load(p + "ffn_gate_inp.scale", role::vector, L.ffn_gate_inp_scale)
			|| !load(p + "ffn_gate_up_exps.weight", role::experts, L.ffn_gate_up_exps)
			|| !load(p + "ffn_gate_exps.weight", role::experts, L.ffn_gate_exps)
			|| !load(p + "ffn_up_exps.weight", role::experts, L.ffn_up_exps)
			|| !load(p + "ffn_down_exps.weight", role::experts, L.ffn_down_exps)
			|| !load(p + "ffn_down_exps.scale", role::vector, L.ffn_down_exps_scale)
			|| !load(p + "pre_ffw_norm_2.weight", role::vector, L.pre_ffw_norm_2)
			|| !load(p + "post_ffw_norm_1.weight", role::vector, L.post_ffw_norm_1)
			|| !load(p + "post_ffw_norm_2.weight", role::vector, L.post_ffw_norm_2))
			return false;
		// Layers that reuse an earlier layer's KV never read their own K/V
		// projections, so those are not loaded at all.
		if (li < h.n_layer_kv
			&& (!load(p + "attn_k.weight", role::matrix, L.attn_k) || !load(p + "attn_v.weight", role::matrix, L.attn_v)))
			return false;
		if (L.rs_index >= 0)
		{
			// The decay and write-strength rows are few, and steer the state
			// for the rest of the sequence, so they stay F32 like a router.
			if (!load(p + "attn_qkv.weight", role::matrix, L.ssm_qkv)
				|| !load(p + "attn_gate.weight", role::matrix, L.ssm_gate)
				|| !load(p + "ssm_alpha.weight", role::vector, L.ssm_alpha)
				|| !load(p + "ssm_beta.weight", role::vector, L.ssm_beta)
				|| !load(p + "ssm_conv1d.weight", role::vector, L.ssm_conv1d)
				|| !load(p + "ssm_dt.bias", role::vector, L.ssm_dt)
				|| !load(p + "ssm_a", role::vector, L.ssm_a)
				|| !load(p + "ssm_norm.weight", role::vector, L.ssm_norm)
				|| !load(p + "ssm_out.weight", role::matrix, L.ssm_out))
				return false;
		}
		else if (!load(p + "attn_q.bias", role::vector, L.attn_q_b) || !load(p + "attn_k.bias", role::vector, L.attn_k_b)
			|| !load(p + "attn_v.bias", role::vector, L.attn_v_b))
		{
			return false;
		}
		// Qwen3.5 calls its pre-MLP norm post_attention_norm; Gemma's norm of
		// that name scales the attention output instead.
		if (h.arch_id == model_arch::qwen35)
		{
			L.ffn_norm = L.post_attention_norm;
			L.post_attention_norm = nullptr;
		}
	}
	if (!resolve_layers(m)) return false;
	if (qwen)
	{
		for (size_t i = 0; i < f.tensors.size(); ++i)
		{
			const std::string& name = f.tensors[i].name;
			// Blocks past the last layer are Qwen3.5's multi-token predictor.
			if (used[i] || (name.compare(0, 4, "blk.") == 0 && std::atoi(name.c_str() + 4) >= h.n_layer)) continue;
			return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "'%s' holds a tensor nnc does not know how to use (%s)",
			                path.c_str(), name.c_str());
		}
	}
	if (progress && !prog.cancelled && prog.reported < 1.0f) progress(1.0f, progress_user);

	nnc_log(NNC_LOG_INFO, "loaded %s: %s, %d layers, %d-token vocabulary, %.1f MB of %s weights decoded in %.1f s",
	        m.name.c_str(), h.arch.c_str(), h.n_layer, h.n_vocab,
	        static_cast<double>(m.decoded_bytes) / (1024.0 * 1024.0), fmt_name(fmt),
	        static_cast<double>(nnc_time_us() - t0) / 1e6);
	return true;
}

// ----------------------------------------------------------------------------
// Forward pass.
// ----------------------------------------------------------------------------

namespace
{
	struct layer_scratch
	{
		float* normed; // [n_batch, n_embd]
		float* q; // [n_batch, max_q_dim]
		float* k_cur; // [n_batch, max_kv_dim]  pre-cache K
		float* v_cur; // [n_batch, max_kv_dim]  pre-cache V
		float* attn_out; // [n_batch, max_q_dim]
		float* gate; // [n_batch, max_ff]
		float* up; // [n_batch, max_ff]
		float* scores; // [max_n_head, n_ctx]
		float* ple_gate; // [n_batch, ple_dim]
		float* ple_contrib; // [n_batch, n_embd]
		float* xsum; // [n_batch, max_cols/32]  Q4 block sums
		const float* ple_for_layer; // [n_batch, ple_dim] slice for this layer
		int ple_stride; // element stride between tokens in ple_for_layer
	};

	// Batched weight gemv: `n_tok` activation vectors against one pass
	// over the weights. n_tok == 1 routes to the single-vector path.
	void gemv_batch(const nnc_tensor* T, const float* X, const size_t x_stride,
	                float* Y, const size_t y_stride, const int n_tok,
	                float* xsum_scratch)
	{
		const uint32_t cols = static_cast<uint32_t>(T->ne[0]);
		const uint32_t rows = static_cast<uint32_t>(T->ne[1]);
		const uint32_t nb = static_cast<uint32_t>(n_tok);

		if (T->type == NNC_TYPE_Q8_0)
		{
			nnc_gemv_q8_0_f32x_batch(static_cast<const int8_t*>(T->data), q8_scales(T),
			                         X, x_stride, Y, y_stride, rows, cols, nb);
			return;
		}
		if (T->type == NNC_TYPE_Q4_S)
		{
			const size_t nblocks = cols / 32;
			for (int t = 0; t < n_tok; ++t)
				nnc_block_sums_f32(X + t * x_stride, xsum_scratch + t * nblocks, cols);
			nnc_gemv_q4_s_f32x_batch(static_cast<const uint8_t*>(T->data), q4_scales(T),
			                         q4_biases(T), X, x_stride,
			                         xsum_scratch, nblocks, Y, y_stride, rows, cols, nb);
			return;
		}
		nnc_gemv_bf16w_f32x_batch(T->data, X, x_stride, Y, y_stride, rows, cols, nb);
	}

	const float* f32(const nnc_tensor* t) { return static_cast<const float*>(t->data); }

	layer_scratch scratch_for(model_kv_cache& cache, const int ple_stride)
	{
		layer_scratch s{};
		s.normed = cache.snormed.data();
		s.q = cache.sq.data();
		s.k_cur = cache.sk_cur.data();
		s.v_cur = cache.sv_cur.data();
		s.attn_out = cache.sattn.data();
		s.gate = cache.sgate.data();
		s.up = cache.sup.data();
		s.scores = cache.sscores.data();
		s.ple_gate = cache.sple_gate.data();
		s.ple_contrib = cache.sple_contrib.data();
		s.xsum = cache.sxsum.data();
		s.ple_for_layer = nullptr;
		s.ple_stride = ple_stride;
		return s;
	}

	// One layer's attention over a batch of query tokens.
	struct attn_job
	{
		const float* q; // [n_tok, n_head * head_dim]
		float* out; // [n_tok, n_head * head_dim]
		const uint8_t* K; // the layer's cache: row pos % rows, kv_dim elements
		const uint8_t* V;
		size_t esize; // bytes per element: 2 (BF16) or 4
		int rows;
		float* scores; // n_chunks slots of `slot` floats
		size_t slot;
		float* parts; // decode: a partial per (KV head, block)
		int n_tok, pos0, n_seq, window;
		int n_head, n_kv, head_dim;
		int n_chunks;
		int blk0, n_blk; // decode: the blocks of the one query's keys
		float scale;
		bool causal;
	};

	// Partial attention of query token t's heads h0 .. h0 + group over the
	// part of its span [first, last] that falls in block b.
	void attend_block(const attn_job& j, const int t, const int kv_h, const int first, const int last, const int b,
	                  float* scores, float* part)
	{
		const int k0 = std::max(first, b * MODEL_ATTN_BLOCK);
		const int k1 = std::min(last, b * MODEL_ATTN_BLOCK + MODEL_ATTN_BLOCK - 1);
		const size_t head_dim = static_cast<size_t>(j.head_dim);
		const size_t kv_dim = static_cast<size_t>(j.n_kv) * head_dim;
		const size_t group = static_cast<size_t>(j.n_head / j.n_kv);
		const size_t off = (static_cast<size_t>(k0 % j.rows) * kv_dim + kv_h * head_dim) * j.esize;
		const float* qh = j.q + static_cast<size_t>(t) * j.n_head * head_dim + kv_h * group * head_dim;
		nnc_attn_block(qh, head_dim, j.K + off, j.V + off, j.esize == 2, kv_dim, static_cast<size_t>(k1 + 1 - k0),
		               head_dim, group, j.scale, scores, part);
	}

	// Query heads that share a KV head are processed together: with GQA
	// (Gemma 3 1B is 4 query heads to 1 KV head) that turns n_head passes
	// over the K/V cache into one. The blocks are merged in order.
	void attend_token(const attn_job& j, const int t, float* slot)
	{
		int first, last;
		model_attn_span(j.causal, j.window, j.pos0 + t, j.n_seq, &first, &last);
		const size_t head_dim = static_cast<size_t>(j.head_dim);
		const int group = j.n_head / j.n_kv;
		const size_t part = static_cast<size_t>(group) * NNC_ATTN_PART(head_dim);
		float* scores = slot;
		float* acc = slot + static_cast<size_t>(group) * MODEL_ATTN_BLOCK;
		float* tmp = acc + part;
		float* at = j.out + static_cast<size_t>(t) * j.n_head * head_dim;
		for (int kv_h = 0; kv_h < j.n_kv; ++kv_h)
		{
			const int b0 = first / MODEL_ATTN_BLOCK, b1 = last / MODEL_ATTN_BLOCK;
			attend_block(j, t, kv_h, first, last, b0, scores, acc);
			for (int b = b0 + 1; b <= b1; ++b)
			{
				attend_block(j, t, kv_h, first, last, b, scores, tmp);
				nnc_attn_merge(acc, tmp, head_dim, static_cast<size_t>(group));
			}
			nnc_attn_finish(at + static_cast<size_t>(kv_h) * group * head_dim, head_dim, acc, head_dim,
			                static_cast<size_t>(group));
		}
	}

	// Chunk c takes tokens c, c + n_chunks, ... - interleaved, so that a
	// causal batch's longer spans are shared out - and scores slot c.
	void attend_chunks(const uint32_t c0, const uint32_t c1, void* user)
	{
		const auto& j = *static_cast<const attn_job*>(user);
		for (uint32_t c = c0; c < c1; ++c)
			for (int t = static_cast<int>(c); t < j.n_tok; t += j.n_chunks)
				attend_token(j, t, j.scores + c * j.slot);
	}

	// Decode: one query, its (KV head, block) pairs shared out. Chunk c
	// takes pairs c, c + n_chunks, ..., each into its own partial; the
	// caller merges them in block order, as attend_token would.
	void attend_blocks(const uint32_t c0, const uint32_t c1, void* user)
	{
		const auto& j = *static_cast<const attn_job*>(user);
		int first, last;
		model_attn_span(j.causal, j.window, j.pos0, j.n_seq, &first, &last);
		const size_t part = static_cast<size_t>(j.n_head / j.n_kv) * NNC_ATTN_PART(static_cast<size_t>(j.head_dim));
		const int n_items = j.n_kv * j.n_blk;
		for (uint32_t c = c0; c < c1; ++c)
			for (int i = static_cast<int>(c); i < n_items; i += j.n_chunks)
				attend_block(j, 0, i / j.n_blk, first, last, j.blk0 + i % j.n_blk, j.scores + c * j.slot,
				             j.parts + static_cast<size_t>(i) * part);
	}

	// Gemma 4's routed experts for `n_tok` tokens whose residual stream
	// after attention is X ([n_tok, n_embd]). Leaves
	// post_ffw_norm_2(sum over the chosen experts of weight * expert(x)) in
	// cache.smoe_out; the dense MLP beside it is the shared expert.
	void moe_forward(const nnc_model& m, model_kv_cache& cache, const model_layer& L, const float* X,
	                 const int n_tok, float* xsum)
	{
		const auto& h = m.hparams;
		const int n_embd = h.n_embd;
		const int n_expert = h.n_expert;
		const int k = h.n_expert_used;
		const int n_ff = L.n_ff_exp;
		int* chosen = cache.smoe_expert.data();
		float* weight = cache.smoe_weight.data();

		// Router: softmax over rms_norm(x) / sqrt(n_embd) * scale against
		// the F32 router weights, then the k most likely experts.
		uint64_t used[MAX_EXPERTS / 64] = {};
		{
			nnc_perf_scope ps(NNC_PERF_MOE_ROUTE);
			const float* router = f32(L.ffn_gate_inp);
			const float* scale = f32(L.ffn_gate_inp_scale);
			const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(n_embd));
			float* r = cache.smoe_router.data();
			float* p = cache.smoe_probs.data();
			for (int t = 0; t < n_tok; ++t)
			{
				nnc_rmsnorm_f32(r, X + static_cast<size_t>(t) * n_embd, static_cast<size_t>(n_embd), h.rms_eps);
				for (int i = 0; i < n_embd; ++i) r[i] *= inv_sqrt * scale[i];
				for (int e = 0; e < n_expert; ++e)
					p[e] = nnc_dot_f32_simd(router + static_cast<size_t>(e) * n_embd, r, static_cast<size_t>(n_embd));
				nnc_softmax_f32_inplace(p, static_cast<size_t>(n_expert));
				nnc_moe_top_k(p, n_expert, k, chosen + t * k, weight + t * k);
				for (int j = 0; j < k; ++j) used[chosen[t * k + j] / 64] |= 1ull << (chosen[t * k + j] % 64);
			}
		}

		const nnc_tensor* first = L.ffn_gate_up_exps ? L.ffn_gate_up_exps : L.ffn_gate_exps;
		uint64_t expert_bytes = weight_bytes(first) + weight_bytes(L.ffn_down_exps);
		if (!L.ffn_gate_up_exps) expert_bytes += weight_bytes(L.ffn_up_exps);
		uint64_t n_used = 0;
		for (const uint64_t word : used) n_used += static_cast<uint64_t>(std::popcount(word));
		nnc_perf_scope ps(NNC_PERF_MOE_EXPERTS, n_used * expert_bytes);

		float* in = cache.smoe_in.data();
		float* out = cache.smoe_out.data();
		const size_t n_embd_all = static_cast<size_t>(n_tok) * n_embd;
		nnc_rmsnorm_gamma_multi_f32(in, X, n_tok, n_embd, f32(L.pre_ffw_norm_2), h.rms_eps);
		std::fill(out, out + n_embd_all, 0.0f);

		// Each chosen expert runs once, over every token that picked it, so
		// its weights are streamed once per batch rather than once per token.
		int* pairs = cache.smoe_pairs.data();
		float* rows = cache.smoe_rows.data();
		float* gate_up = cache.smoe_gate_up.data();
		float* down = cache.smoe_down.data();
		const size_t gu_stride = 2 * static_cast<size_t>(n_ff);
		for (int e = 0; e < n_expert; ++e)
		{
			if (!(used[e / 64] >> (e % 64) & 1)) continue;
			int cnt = 0;
			for (int i = 0; i < n_tok * k; ++i)
				if (chosen[i] == e) pairs[cnt++] = i;
			for (int c = 0; c < cnt; ++c)
				std::memcpy(rows + static_cast<size_t>(c) * n_embd, in + static_cast<size_t>(pairs[c] / k) * n_embd,
				            sizeof(float) * static_cast<size_t>(n_embd));

			// Gate rows come first, then up rows: [gate | up] per token.
			if (L.ffn_gate_up_exps)
			{
				const nnc_tensor W = expert_matrix(L.ffn_gate_up_exps, e);
				gemv_batch(&W, rows, n_embd, gate_up, gu_stride, cnt, xsum);
			}
			else
			{
				const nnc_tensor G = expert_matrix(L.ffn_gate_exps, e);
				const nnc_tensor U = expert_matrix(L.ffn_up_exps, e);
				gemv_batch(&G, rows, n_embd, gate_up, gu_stride, cnt, xsum);
				gemv_batch(&U, rows, n_embd, gate_up + n_ff, gu_stride, cnt, xsum);
			}
			for (int c = 0; c < cnt; ++c)
			{
				float* g = gate_up + static_cast<size_t>(c) * gu_stride;
				nnc_gelu_mul_f32(g, g, g + n_ff, static_cast<size_t>(n_ff));
			}
			const nnc_tensor D = expert_matrix(L.ffn_down_exps, e);
			gemv_batch(&D, gate_up, gu_stride, down, n_embd, cnt, xsum);

			const float s = L.ffn_down_exps_scale ? f32(L.ffn_down_exps_scale)[e] : 1.0f;
			for (int c = 0; c < cnt; ++c)
			{
				const float w = weight[pairs[c]] * s;
				float* o = out + static_cast<size_t>(pairs[c] / k) * n_embd;
				const float* d = down + static_cast<size_t>(c) * n_embd;
				for (int i = 0; i < n_embd; ++i) o[i] += w * d[i];
			}
		}
		nnc_rmsnorm_gamma_multi_f32(out, out, n_tok, n_embd, f32(L.post_ffw_norm_2), h.rms_eps);
	}

	// A batch through a Gated DeltaNet layer's delta rule. A value head's
	// state depends only on its own inputs, so heads are shared out across
	// the pool, each running through the batch's tokens in order.
	struct delta_net_job
	{
		float* state; // the layer's [n_v][d_v][d_k] matrices
		const float* qkv; // [n_tok, conv_dim]: unit q (scaled) | unit k | v
		const float* gb; // [n_tok, 2 * n_v]: each head's decay, then its write strength
		float* out; // [n_tok, n_v * d_v]
		int n_tok, n_k, n_v, d_k, d_v;
	};

	void delta_net_heads(const uint32_t h0, const uint32_t h1, void* user)
	{
		const auto& j = *static_cast<const delta_net_job*>(user);
		const size_t d_k = static_cast<size_t>(j.d_k), d_v = static_cast<size_t>(j.d_v);
		const size_t n_v = static_cast<size_t>(j.n_v);
		const size_t qk_dim = static_cast<size_t>(j.n_k) * d_k;
		const size_t conv_dim = 2 * qk_dim + n_v * d_v;
		for (size_t hv = h0; hv < h1; ++hv)
		{
			// The converter orders value heads so that head hv reads key head
			// hv mod n_k (ggml's broadcast), not hv / (n_v / n_k).
			const size_t hk = hv % static_cast<size_t>(j.n_k);
			float* S = j.state + hv * d_v * d_k;
			for (int t = 0; t < j.n_tok; ++t)
			{
				const float* row = j.qkv + static_cast<size_t>(t) * conv_dim;
				const float* gb = j.gb + static_cast<size_t>(t) * 2 * n_v;
				nnc_delta_rule_f32(S, row + hk * d_k, row + qk_dim + hk * d_k, row + 2 * qk_dim + hv * d_v, gb[hv],
				                   gb[n_v + hv], j.out + static_cast<size_t>(t) * n_v * d_v + hv * d_v, d_k, d_v);
			}
		}
	}

	// A Gated DeltaNet layer (Qwen3.5) in place of attention: reads the
	// normed hidden states in s.normed and leaves the layer's output
	// projection there, carrying the layer's recurrent state through the
	// batch. Per token, after the projections:
	//   q, k, v = silu(causal_conv(x W_qkv))       q and k unit length per head
	//   decay   = exp(softplus(x W_alpha + dt) * A),  beta = sigmoid(x W_beta)
	//   S = decay S + beta (v - decay S k) k^T,    o = S q / sqrt(d_k)
	//   out     = W_out (rmsnorm(o) * gamma * silu(x W_gate))
	void delta_net_forward(const nnc_model& m, model_kv_cache& cache, const model_layer& L, const int n_tok,
	                       const layer_scratch& s)
	{
		const auto& h = m.hparams;
		const size_t n_embd = static_cast<size_t>(h.n_embd);
		const int n_v = h.ssm_n_v;
		const size_t qk_dim = static_cast<size_t>(h.ssm_n_k) * h.ssm_d_k;
		const size_t v_dim = static_cast<size_t>(n_v) * h.ssm_d_v;
		const size_t conv_dim = 2 * qk_dim + v_dim;
		const size_t hist = static_cast<size_t>(h.ssm_d_conv) - 1;
		const size_t state_floats = v_dim * h.ssm_d_k;
		const float* x = s.normed;

		// The convolution reads [its last d_conv - 1 inputs | the batch], so
		// the projection lands straight after the saved history.
		float* conv_in = cache.sdn_conv.data();
		float* history = cache.conv_state.data() + static_cast<size_t>(L.rs_index) * hist * conv_dim;
		float* z = cache.sdn_z.data();
		float* gb = cache.sdn_gb.data();
		std::memcpy(conv_in, history, hist * conv_dim * sizeof(float));
		{
			const uint64_t wb = weight_bytes(L.ssm_qkv) + weight_bytes(L.ssm_gate)
				+ 2ull * static_cast<uint64_t>(n_v) * n_embd * sizeof(float);
			nnc_perf_scope ps(NNC_PERF_DN_PROJ, wb);
			gemv_batch(L.ssm_qkv, x, n_embd, conv_in + hist * conv_dim, conv_dim, n_tok, s.xsum);
			gemv_batch(L.ssm_gate, x, n_embd, z, v_dim, n_tok, s.xsum);
			const float* wa = f32(L.ssm_alpha);
			const float* wb_ = f32(L.ssm_beta);
			for (int t = 0; t < n_tok; ++t)
			{
				const float* xt = x + static_cast<size_t>(t) * n_embd;
				float* g = gb + static_cast<size_t>(t) * 2 * n_v;
				for (int hv = 0; hv < n_v; ++hv)
				{
					g[hv] = nnc_dot_f32_simd(wa + static_cast<size_t>(hv) * n_embd, xt, n_embd);
					g[n_v + hv] = nnc_dot_f32_simd(wb_ + static_cast<size_t>(hv) * n_embd, xt, n_embd);
				}
			}
		}

		float* qkv = cache.sdn_qkv.data();
		float* o = cache.sdn_out.data();
		{
			nnc_perf_scope ps(NNC_PERF_DN_CORE, 2ull * n_tok * state_floats * sizeof(float));
			nnc_conv1d_silu_f32(qkv, conv_in, L.ssm_conv_taps, static_cast<size_t>(n_tok), conv_dim, hist + 1);
			std::memcpy(history, conv_in + static_cast<size_t>(n_tok) * conv_dim, hist * conv_dim * sizeof(float));

			const float q_scale = 1.0f / std::sqrt(static_cast<float>(h.ssm_d_k));
			const float* dt = f32(L.ssm_dt);
			const float* A = f32(L.ssm_a);
			for (int t = 0; t < n_tok; ++t)
			{
				float* row = qkv + static_cast<size_t>(t) * conv_dim;
				nnc_l2norm_multi_f32(row, static_cast<size_t>(h.ssm_n_k), static_cast<size_t>(h.ssm_d_k), h.rms_eps,
				                     q_scale);
				nnc_l2norm_multi_f32(row + qk_dim, static_cast<size_t>(h.ssm_n_k), static_cast<size_t>(h.ssm_d_k),
				                     h.rms_eps, 1.0f);
				float* g = gb + static_cast<size_t>(t) * 2 * n_v;
				for (int hv = 0; hv < n_v; ++hv)
				{
					const float a = g[hv] + dt[hv];
					const float softplus = a > 20.0f ? a : std::log1p(std::exp(a));
					g[hv] = std::exp(softplus * A[hv]);
					g[n_v + hv] = 1.0f / (1.0f + std::exp(-g[n_v + hv]));
				}
			}

			delta_net_job job{};
			job.state = cache.ssm_state.data() + static_cast<size_t>(L.rs_index) * state_floats;
			job.qkv = qkv;
			job.gb = gb;
			job.out = o;
			job.n_tok = n_tok;
			job.n_k = h.ssm_n_k;
			job.n_v = n_v;
			job.d_k = h.ssm_d_k;
			job.d_v = h.ssm_d_v;
			// A head's state is d_v * d_k floats (64 KB at 128 x 128); below a
			// few hundred KB of work in all, dispatch would cost more than it saves.
			if (n_v > 1 && static_cast<size_t>(n_tok) * state_floats >= (64u << 10))
				nnc_parallel_for(static_cast<uint32_t>(n_v), delta_net_heads, &job);
			else
				delta_net_heads(0, static_cast<uint32_t>(n_v), &job);

			nnc_rmsnorm_gamma_multi_f32(o, o, static_cast<size_t>(n_tok) * n_v, static_cast<size_t>(h.ssm_d_v),
			                            f32(L.ssm_norm), h.rms_eps);
			nnc_swiglu_f32(o, z, o, static_cast<size_t>(n_tok) * v_dim);
		}
		{
			nnc_perf_scope ps(NNC_PERF_DN_OUT, weight_bytes(L.ssm_out));
			gemv_batch(L.ssm_out, o, v_dim, s.normed, n_embd, n_tok, s.xsum);
		}
	}

	// Attention for a batch of `n_tok` tokens at positions pos0 ...: reads
	// the normed hidden states in s.normed, appends the batch's K/V to the
	// layer's cache, attends, and leaves the output projection in s.normed.
	void attention_forward(const nnc_model& m, model_kv_cache& cache, const int li, const int pos0, const int n_tok,
	                       const layer_scratch& s)
	{
		const auto& h = m.hparams;
		const model_layer& L = m.layers[li];
		const int n_embd = h.n_embd;
		const int head_dim = L.head_dim;
		const int n_head = L.n_head;
		const int n_kv = L.n_head_kv;
		const int q_dim = n_head * head_dim;
		const int kv_dim = n_kv * head_dim;
		const float freq = L.sliding_window ? h.rope_freq_base_swa : h.rope_freq_base;
		const bool reuse_kv = L.kv_layer != li;
		const float* normed = s.normed;

		float* q = s.q;
		float* k = s.k_cur;
		float* v = s.v_cur;
		float* gate = cache.sattn_gate.data();
		{
			uint64_t wb = weight_bytes(L.attn_q);
			if (!reuse_kv) wb += weight_bytes(L.attn_k) + (L.attn_v ? weight_bytes(L.attn_v) : 0);
			nnc_perf_scope ps(NNC_PERF_ATTN_QKV, wb);
			if (L.attn_gated)
			{
				// Each head's rows are its query, then its output gate.
				float* qg = cache.sq_gate.data();
				gemv_batch(L.attn_q, normed, n_embd, qg, 2 * static_cast<size_t>(q_dim), n_tok, s.xsum);
				const size_t hd = static_cast<size_t>(head_dim);
				for (size_t r = 0; r < static_cast<size_t>(n_tok) * n_head; ++r)
				{
					std::memcpy(q + r * hd, qg + 2 * r * hd, hd * sizeof(float));
					std::memcpy(gate + r * hd, qg + (2 * r + 1) * hd, hd * sizeof(float));
				}
			}
			else
			{
				gemv_batch(L.attn_q, normed, n_embd, q, q_dim, n_tok, s.xsum);
			}
			if (!reuse_kv)
			{
				gemv_batch(L.attn_k, normed, n_embd, k, kv_dim, n_tok, s.xsum);
				if (L.attn_v)
					gemv_batch(L.attn_v, normed, n_embd, v, kv_dim, n_tok, s.xsum);
				else // Gemma 4 global layers without v_proj: V is the raw K projection
					std::memcpy(v, k, sizeof(float) * static_cast<size_t>(kv_dim) * n_tok);
			}
			// Projection biases (Qwen2).
			for (int t = 0; t < n_tok && (L.attn_q_b || L.attn_k_b || L.attn_v_b); ++t)
			{
				if (L.attn_q_b) nnc_add_inplace_f32(q + static_cast<size_t>(t) * q_dim, f32(L.attn_q_b), q_dim);
				if (reuse_kv) continue;
				if (L.attn_k_b) nnc_add_inplace_f32(k + static_cast<size_t>(t) * kv_dim, f32(L.attn_k_b), kv_dim);
				if (L.attn_v_b) nnc_add_inplace_f32(v + static_cast<size_t>(t) * kv_dim, f32(L.attn_v_b), kv_dim);
			}
		}

		// Per-head Q / K norms (Gemma; Llama has none). Every (token, head)
		// pair is an independent length-head_dim group, so the whole batch
		// is one call.
		{
			nnc_perf_scope ps(NNC_PERF_NORM);
			if (L.attn_q_norm)
				nnc_rmsnorm_gamma_multi_f32(q, q, static_cast<size_t>(n_tok) * n_head,
				                            head_dim, f32(L.attn_q_norm), h.rms_eps);

			if (!reuse_kv)
			{
				if (L.attn_k_norm)
					nnc_rmsnorm_gamma_multi_f32(k, k, static_cast<size_t>(n_tok) * n_kv,
					                            head_dim, f32(L.attn_k_norm), h.rms_eps);

				// Gemma 4 also RMS-normalises V per head, with no learned gamma.
				if (h.arch_id == model_arch::gemma4)
				{
					for (int g = 0; g < n_tok * n_kv; ++g)
					{
						float* vh = v + static_cast<size_t>(g) * head_dim;
						nnc_rmsnorm_f32(vh, vh, static_cast<size_t>(head_dim), h.rms_eps);
					}
				}
			}
		}

		// RoPE on Q always; K only when freshly computed (cached K is already
		// rotated). Gemma 3 applies its linear position scaling to global
		// layers only (HF scales rope_theta, not rope_local_base_freq).
		{
			nnc_perf_scope ps(NNC_PERF_ATTN_ROPE);
			const float layer_rope_scale = (h.arch_id == model_arch::gemma3 && L.sliding_window)
				                               ? 1.0f
				                               : h.rope_freq_scale;
			for (int t = 0; t < n_tok; ++t)
			{
				const float rpos = static_cast<float>(pos0 + t) * layer_rope_scale;
				nnc_rope_f32(q + static_cast<size_t>(t) * q_dim, n_head, head_dim, L.n_rot, rpos, freq,
				             h.rope_mode, L.rope_freqs);
				if (!reuse_kv)
					nnc_rope_f32(k + static_cast<size_t>(t) * kv_dim, n_kv, head_dim, L.n_rot, rpos, freq,
					             h.rope_mode, L.rope_freqs);
			}
		}

		// Append K/V to the layer's cache (only when freshly computed). An
		// MTP drafter's layers read its target's cache instead.
		model_kv_cache& kv = cache.kv_ext ? *cache.kv_ext : cache;
		model_kv_buffer& buf = kv.bufs[cache.kv_buf[li]];
		const size_t esize = kv.kv_bf16 ? 2 : 4;
		NNC_ASSERT(buf.window == 0 || (L.sliding_window && buf.window == h.sliding_window));
		if (!reuse_kv)
		{
			for (int t = 0; t < n_tok; ++t)
			{
				const size_t row = static_cast<size_t>((pos0 + t) % buf.rows) * kv_dim;
				const float* kt = k + static_cast<size_t>(t) * kv_dim;
				const float* vt = v + static_cast<size_t>(t) * kv_dim;
				if (kv.kv_bf16)
				{
					auto* Kd = reinterpret_cast<nnc_bf16_t*>(buf.k.data()) + row;
					auto* Vd = reinterpret_cast<nnc_bf16_t*>(buf.v.data()) + row;
					for (int i = 0; i < kv_dim; ++i)
					{
						Kd[i] = nnc_f32_to_bf16(kt[i]);
						Vd[i] = nnc_f32_to_bf16(vt[i]);
					}
				}
				else
				{
					std::memcpy(reinterpret_cast<float*>(buf.k.data()) + row, kt, kv_dim * sizeof(float));
					std::memcpy(reinterpret_cast<float*>(buf.v.data()) + row, vt, kv_dim * sizeof(float));
				}
			}
		}

		float* attn = s.attn_out;
		{
			attn_job job{};
			job.q = q;
			job.out = attn;
			job.K = buf.k.data();
			job.V = buf.v.data();
			job.esize = esize;
			job.rows = buf.rows;
			job.scores = s.scores;
			job.slot = cache.attn_slot;
			job.parts = cache.sattn_part.data();
			job.n_tok = n_tok;
			job.pos0 = pos0;
			job.n_seq = cache.kv_ext ? kv.cur_pos : pos0 + n_tok;
			job.window = L.sliding_window ? h.sliding_window : 0;
			job.n_head = n_head;
			job.n_kv = n_kv;
			job.head_dim = head_dim;
			job.scale = h.attention_scale;
			job.causal = h.causal;

			uint64_t attn_bytes = 0;
			for (int t = 0; t < n_tok; ++t)
			{
				int first, last;
				model_attn_span(job.causal, job.window, pos0 + t, job.n_seq, &first, &last);
				attn_bytes += 2ull * static_cast<uint64_t>(last + 1 - first) * kv_dim * esize;
			}
			nnc_perf_scope ps(NNC_PERF_ATTN_CORE, attn_bytes);

			// Short spans run here. Decode with a long span shares its
			// (KV head, block) pairs across the pool; a batch with long
			// spans (a long prompt, or an embedding model's whole sequence)
			// shares out its tokens. Each chunk has its own scores slot.
			// Either way the blocks and their merge order are the same, so
			// the result does not depend on the path taken.
			constexpr uint64_t parallel_bytes = 256u << 10;
			if (n_tok == 1 && attn_bytes >= parallel_bytes && cache.n_attn_slots > 1)
			{
				int first, last;
				model_attn_span(job.causal, job.window, pos0, job.n_seq, &first, &last);
				job.blk0 = first / MODEL_ATTN_BLOCK;
				job.n_blk = last / MODEL_ATTN_BLOCK + 1 - job.blk0;
				const int n_items = n_kv * job.n_blk;
				job.n_chunks = std::min(n_items, cache.n_attn_slots);
				nnc_parallel_for(static_cast<uint32_t>(job.n_chunks), attend_blocks, &job);
				const size_t hd = static_cast<size_t>(head_dim);
				const size_t group = static_cast<size_t>(n_head / n_kv);
				const size_t part = group * NNC_ATTN_PART(hd);
				for (int kv_h = 0; kv_h < n_kv; ++kv_h)
				{
					float* acc = job.parts + static_cast<size_t>(kv_h) * job.n_blk * part;
					for (int b = 1; b < job.n_blk; ++b) nnc_attn_merge(acc, acc + b * part, hd, group);
					nnc_attn_finish(attn + kv_h * group * hd, hd, acc, hd, group);
				}
			}
			else
			{
				job.n_chunks = std::min(n_tok, cache.n_attn_slots);
				if (job.n_chunks > 1 && attn_bytes >= parallel_bytes)
					nnc_parallel_for(static_cast<uint32_t>(job.n_chunks), attend_chunks, &job);
				else
					for (int t = 0; t < n_tok; ++t) attend_token(job, t, job.scores);
			}

			// Qwen3.5 scales each head's output by its gate.
			if (L.attn_gated) nnc_mul_sigmoid_f32(attn, gate, static_cast<size_t>(n_tok) * q_dim);
		}

		// Output projection: [q_dim] -> [n_embd], into s.normed (dead after
		// the QKV gemvs).
		{
			nnc_perf_scope ps(NNC_PERF_ATTN_OUT, weight_bytes(L.attn_output));
			gemv_batch(L.attn_output, attn, q_dim, s.normed, n_embd, n_tok, s.xsum);
		}
	}

	// Runs `n_tok` consecutive tokens (positions pos0 .. pos0+n_tok-1)
	// through one transformer block. X is [n_tok, n_embd], token-major and
	// contiguous; it is updated in place. n_tok == 1 is the decode path.
	void layer_forward(const nnc_model& m, model_kv_cache& cache, const int li,
	                   const int pos0, const int n_tok, float* X,
	                   const layer_scratch& s)
	{
		const auto& h = m.hparams;
		const model_layer& L = m.layers[li];
		const int n_embd = h.n_embd;

		// --- attention block, or a Gated DeltaNet in its place ---
		float* normed = s.normed;
		{
			nnc_perf_scope ps(NNC_PERF_NORM);
			nnc_rmsnorm_gamma_multi_f32(normed, X, n_tok, n_embd, f32(L.attn_norm), h.rms_eps);
		}
		if (L.rs_index >= 0) delta_net_forward(m, cache, L, n_tok, s);
		else attention_forward(m, cache, li, pos0, n_tok, s);
		float* attn_out = s.normed;

		const size_t n_embd_all = static_cast<size_t>(n_tok) * n_embd;

		// Post-attention RMSNorm (Gemma) + residual.
		{
			nnc_perf_scope ps(NNC_PERF_NORM);
			if (L.post_attention_norm)
				nnc_rmsnorm_gamma_multi_f32(attn_out, attn_out, n_tok, n_embd,
				                            f32(L.post_attention_norm), h.rms_eps);
			nnc_add_inplace_f32(X, attn_out, n_embd_all);
		}

		// --- MLP block (gated) ---
		// Gemma uses gelu(gate)*up; Llama and Qwen use silu(gate)*up (SwiGLU).
		{
			nnc_perf_scope ps(NNC_PERF_NORM);
			nnc_rmsnorm_gamma_multi_f32(normed, X, n_tok, n_embd, f32(L.ffn_norm), h.rms_eps);
		}

		const int n_ff = L.n_ff;
		float* gate = s.gate;
		float* up = s.up;
		{
			nnc_perf_scope ps(NNC_PERF_FFN_GATE_UP,
			                  weight_bytes(L.ffn_gate) + weight_bytes(L.ffn_up));
			gemv_batch(L.ffn_gate, normed, n_embd, gate, n_ff, n_tok, s.xsum);
			gemv_batch(L.ffn_up, normed, n_embd, up, n_ff, n_tok, s.xsum);
		}
		{
			nnc_perf_scope ps(NNC_PERF_FFN_ACT);
			const size_t n_ff_all = static_cast<size_t>(n_tok) * n_ff;
			if (h.arch_id == model_arch::gemma3 || h.arch_id == model_arch::gemma4)
				nnc_gelu_mul_f32(gate, gate, up, n_ff_all);
			else
				nnc_swiglu_f32(gate, gate, up, n_ff_all);
		}

		float* ffn_out = s.normed;
		{
			nnc_perf_scope ps(NNC_PERF_FFN_DOWN, weight_bytes(L.ffn_down));
			gemv_batch(L.ffn_down, gate, n_ff, ffn_out, n_embd, n_tok, s.xsum);
		}

		// Gemma 4 MoE: the dense MLP above is the shared expert. Both
		// halves are normalised on their own, then summed.
		if (L.n_ff_exp > 0)
		{
			{
				nnc_perf_scope ps(NNC_PERF_NORM);
				nnc_rmsnorm_gamma_multi_f32(ffn_out, ffn_out, n_tok, n_embd, f32(L.post_ffw_norm_1), h.rms_eps);
			}
			moe_forward(m, cache, L, X, n_tok, s.xsum);
			nnc_add_inplace_f32(ffn_out, cache.smoe_out.data(), n_embd_all);
		}

		// Post-FFW RMSNorm (Gemma) + residual.
		{
			nnc_perf_scope ps(NNC_PERF_NORM);
			if (L.post_ffw_norm)
				nnc_rmsnorm_gamma_multi_f32(ffn_out, ffn_out, n_tok, n_embd,
				                            f32(L.post_ffw_norm), h.rms_eps);
			nnc_add_inplace_f32(X, ffn_out, n_embd_all);
		}

		// --- Per-Layer Embedding (PLE) block (Gemma 4) ---
		// g = gelu(inp_gate(x)) * inp_per_layer[li]
		// x += post_norm(proj(g))
		if (L.inp_gate && L.proj && s.ple_for_layer)
		{
			nnc_perf_scope ps(NNC_PERF_PLE_LAYER,
			                  weight_bytes(L.inp_gate) + weight_bytes(L.proj));
			const int ple_dim = L.inp_gate->ne[1]; // [n_embd, ple_dim]
			float* g = s.ple_gate;
			gemv_batch(L.inp_gate, X, n_embd, g, ple_dim, n_tok, s.xsum);
			nnc_gelu_f32(g, g, static_cast<size_t>(n_tok) * ple_dim);
			for (int t = 0; t < n_tok; ++t)
				nnc_mul_inplace_f32(g + static_cast<size_t>(t) * ple_dim,
				                    s.ple_for_layer + static_cast<size_t>(t) * s.ple_stride,
				                    static_cast<size_t>(ple_dim));

			float* contrib = s.ple_contrib;
			gemv_batch(L.proj, g, ple_dim, contrib, n_embd, n_tok, s.xsum);
			if (L.post_norm)
				nnc_rmsnorm_gamma_multi_f32(contrib, contrib, n_tok, n_embd, f32(L.post_norm), h.rms_eps);
			nnc_add_inplace_f32(X, contrib, n_embd_all);
		}

		// --- per-layer output scale (Gemma 4 layer_scalar) ---
		if (L.layer_output_scale)
		{
			const float s_val = f32(L.layer_output_scale)[0];
			for (size_t i = 0; i < n_embd_all; ++i) X[i] *= s_val;
		}
	}
}

// ----------------------------------------------------------------------------
// KV cache.
// ----------------------------------------------------------------------------

bool model_kv_init(const nnc_model& m, model_kv_cache& cache, const int n_ctx, const int n_batch, const bool kv_f32)
{
	const auto& h = m.hparams;
	if (n_ctx <= 0) return nnc_fail(NNC_ERR_INVALID_ARGUMENT, "the context size must be positive");
	const int nb = (n_batch > 0) ? n_batch : 1;
	const int n_layer = h.n_layer;
	try
	{
		cache = model_kv_cache{};
		cache.n_ctx = n_ctx;
		cache.n_batch = nb;
		cache.kv_bf16 = !kv_f32;
		cache.max_rewind = std::numeric_limits<int>::max();
		cache.kv_dim_per_layer.assign(n_layer, 0);
		cache.kv_buf.assign(n_layer, 0);
		int max_q_dim = 0, max_kv_dim = 0, max_ff = 0, max_n_head = 1, max_ff_exp = 0, max_part = 0;
		bool gated = false;
		for (int li = 0; li < n_layer; ++li)
		{
			const model_layer& L = m.layers[li];
			const int kv_dim = L.n_head_kv * L.head_dim;
			cache.kv_dim_per_layer[li] = kv_dim;
			max_q_dim = std::max(max_q_dim, L.n_head * L.head_dim);
			max_kv_dim = std::max(max_kv_dim, kv_dim);
			max_ff = std::max(max_ff, L.n_ff);
			max_n_head = std::max(max_n_head, L.n_head);
			max_ff_exp = std::max(max_ff_exp, L.n_ff_exp);
			max_part = std::max(max_part, L.n_head * static_cast<int>(NNC_ATTN_PART(L.head_dim)));
			gated = gated || L.attn_gated;
		}

		// One buffer per cache-owning layer; layers that reuse another
		// layer's cache, and every layer of a bidirectional model, share.
		const size_t esize = cache.kv_bf16 ? 2 : 4;
		const size_t n_ctx_z = static_cast<size_t>(n_ctx);
		if (!h.causal)
		{
			cache.bufs.resize(1);
			cache.bufs[0].rows = n_ctx;
			cache.bufs[0].k.assign(n_ctx_z * max_kv_dim * esize, 0);
			cache.bufs[0].v.assign(n_ctx_z * max_kv_dim * esize, 0);
			cache.bytes = 2 * n_ctx_z * max_kv_dim * esize;
		}
		else
		{
			// A sliding layer's buffer becomes a ring when that saves
			// memory and every layer reading it slides the same way.
			const int ring = (h.sliding_window + nb + MODEL_ATTN_BLOCK - 1) / MODEL_ATTN_BLOCK * MODEL_ATTN_BLOCK;
			std::vector<bool> can_ring(n_layer, h.sliding_window > 0 && ring < n_ctx);
			for (int li = 0; li < n_layer; ++li)
				if (!m.layers[li].sliding_window && m.layers[li].kv_layer >= 0) can_ring[m.layers[li].kv_layer] = false;
			cache.bufs.resize(n_layer);
			for (int li = 0; li < n_layer; ++li)
			{
				const model_layer& L = m.layers[li];
				cache.kv_buf[li] = L.kv_layer;
				if (L.kv_layer != li) continue;
				model_kv_buffer& b = cache.bufs[li];
				b.rows = can_ring[li] ? ring : n_ctx;
				b.window = can_ring[li] ? h.sliding_window : 0;
				if (b.window) cache.max_rewind = std::min(cache.max_rewind, b.rows - b.window);
				const size_t n = static_cast<size_t>(b.rows) * cache.kv_dim_per_layer[li] * esize;
				b.k.assign(n, 0);
				b.v.assign(n, 0);
				cache.bytes += 2 * n;
			}
		}

		const int ple_dim = h.ple_dim > 0 ? h.ple_dim : 0;
		const size_t ple_total = static_cast<size_t>(ple_dim) * n_layer;
		const size_t B = static_cast<size_t>(nb);
		cache.sx.resize(B * h.n_embd);
		cache.snormed.resize(B * h.n_embd);
		cache.sq.resize(B * max_q_dim);
		cache.sk_cur.resize(B * max_kv_dim);
		cache.sv_cur.resize(B * max_kv_dim);
		cache.sattn.resize(B * max_q_dim);
		cache.sgate.resize(B * max_ff);
		cache.sup.resize(B * max_ff);
		// Scores are held for every query head of a KV head at once so the
		// K/V cache is walked once per KV head rather than once per query
		// head; with them, a slot holds two partials. Attention runs in
		// parallel, one slot per chunk of its work, up to one per thread.
		cache.n_head_max = max_n_head;
		cache.attn_part_max = max_part;
		cache.attn_slot = static_cast<size_t>(max_n_head) * MODEL_ATTN_BLOCK + 2 * static_cast<size_t>(max_part);
		cache.n_attn_slots = std::max(1, std::min(nnc_num_threads_avx(), 64));
		cache.sscores.resize(cache.attn_slot * cache.n_attn_slots);
		cache.n_attn_blocks = (n_ctx + MODEL_ATTN_BLOCK - 1) / MODEL_ATTN_BLOCK + 1;
		cache.sattn_part.resize(static_cast<size_t>(max_part) * cache.n_attn_blocks);
		cache.sple_gate.resize(B * std::max(ple_dim, 1));
		cache.sple_contrib.resize(B * h.n_embd);
		cache.sple_inputs.resize(B * std::max<size_t>(ple_total, 1));
		// The projection half of the per-layer inputs needs its own buffer
		// only when there is a token half to combine it with.
		cache.sple_proj.resize(m.per_layer_token_embd ? B * std::max<size_t>(ple_total, 1) : 0);
		// Q4 block sums: one run of (cols/32) floats per batch slot, sized to
		// the widest gemv input we will ever see (an expert's down projection
		// reads n_ff_exp values, a Gated DeltaNet's output projection n_v * d_v).
		const int v_dim = h.ssm_n_v * h.ssm_d_v;
		const int max_cols = std::max({h.n_embd, max_q_dim, max_ff, ple_dim, max_ff_exp, v_dim, 32});
		cache.sxsum.resize(B * (static_cast<size_t>(max_cols) / 32 + 1));
		if (gated)
		{
			cache.sq_gate.resize(B * 2 * static_cast<size_t>(max_q_dim));
			cache.sattn_gate.resize(B * static_cast<size_t>(max_q_dim));
		}
		if (h.n_layer_rs > 0)
		{
			const size_t conv_dim = 2 * static_cast<size_t>(h.ssm_n_k) * h.ssm_d_k + static_cast<size_t>(v_dim);
			const size_t hist = static_cast<size_t>(h.ssm_d_conv) - 1;
			const size_t n_conv = static_cast<size_t>(h.n_layer_rs) * hist * conv_dim;
			const size_t n_ssm = static_cast<size_t>(h.n_layer_rs) * v_dim * h.ssm_d_k;
			cache.conv_state.assign(n_conv, 0.0f);
			cache.ssm_state.assign(n_ssm, 0.0f);
			cache.conv_saved.assign(n_conv, 0.0f);
			cache.ssm_saved.assign(n_ssm, 0.0f);
			cache.bytes += 2 * (n_conv + n_ssm) * sizeof(float);
			cache.sdn_conv.resize((hist + B) * conv_dim);
			cache.sdn_qkv.resize(B * conv_dim);
			cache.sdn_z.resize(B * v_dim);
			cache.sdn_gb.resize(B * 2 * static_cast<size_t>(h.ssm_n_v));
			cache.sdn_out.resize(B * v_dim);
		}
		if (max_ff_exp > 0)
		{
			const size_t k = static_cast<size_t>(h.n_expert_used);
			cache.smoe_in.resize(B * h.n_embd);
			cache.smoe_out.resize(B * h.n_embd);
			cache.smoe_rows.resize(B * h.n_embd);
			cache.smoe_down.resize(B * h.n_embd);
			cache.smoe_gate_up.resize(B * 2 * static_cast<size_t>(max_ff_exp));
			cache.smoe_router.resize(static_cast<size_t>(h.n_embd));
			cache.smoe_probs.resize(static_cast<size_t>(h.n_expert));
			cache.smoe_weight.resize(B * k);
			cache.smoe_expert.resize(B * k);
			cache.smoe_pairs.resize(B);
		}
		if (h.n_embd_target > 0)
		{
			cache.sdraft_in.resize(2 * static_cast<size_t>(h.n_embd_target));
			cache.sdraft_logits.resize(static_cast<size_t>(h.n_vocab));
		}
	}
	catch (const std::bad_alloc&)
	{
		cache = model_kv_cache{};
		return nnc_fail(NNC_ERR_OUT_OF_MEMORY, "out of memory allocating a %d-token context", n_ctx);
	}
	return true;
}

bool model_forward(const nnc_model& m, model_kv_cache& cache, const int* tokens, const int n_tok, const int pos0)
{
	const auto& h = m.hparams;
	if (n_tok <= 0 || n_tok > cache.n_batch)
		return nnc_fail(NNC_ERR_INVALID_ARGUMENT, "%d tokens in one forward pass (at most %d)", n_tok, cache.n_batch);
	if (pos0 != cache.cur_pos)
		return nnc_fail(NNC_ERR_INVALID_ARGUMENT, "position %d does not follow the cache (%d)", pos0, cache.cur_pos);
	if (!h.causal && pos0 != 0)
		return nnc_fail(NNC_ERR_INVALID_ARGUMENT, "a bidirectional model takes its whole sequence in one pass");
	if (pos0 + n_tok > cache.n_ctx)
		return nnc_fail(NNC_ERR_CONTEXT_FULL, "%d more tokens do not fit the %d-token context", n_tok, cache.n_ctx);
	for (int t = 0; t < n_tok; ++t)
		if (tokens[t] < 0 || tokens[t] >= h.n_vocab)
			return nnc_fail(NNC_ERR_INVALID_ARGUMENT, "token %d is outside the vocabulary [0, %d)", tokens[t],
			                h.n_vocab);
	// A sequence that starts again starts its recurrent state afresh.
	if (pos0 == 0)
	{
		std::fill(cache.conv_state.begin(), cache.conv_state.end(), 0.0f);
		std::fill(cache.ssm_state.begin(), cache.ssm_state.end(), 0.0f);
	}

	const int ple_dim = h.ple_dim;
	const int ple_total = ple_dim * h.n_layer;
	float* xp = cache.sx.data();

	layer_scratch s = scratch_for(cache, ple_total);

	{
		nnc_perf_scope ps(NNC_PERF_EMBED);
		for (int t = 0; t < n_tok; ++t)
			model_embed_row(m.token_embd, xp + static_cast<size_t>(t) * h.n_embd, tokens[t],
			                static_cast<size_t>(h.n_embd), h.embd_scale);
	}

	const bool have_ple = ple_dim > 0;
	if (have_ple)
	{
		// Per-layer inputs: rms_norm(per_layer_model_proj(x) / sqrt(n_embd)),
		// averaged with the token's own row of per_layer_token_embd (scaled
		// by sqrt(ple_dim)) when the model has that table.
		nnc_perf_scope ps(NNC_PERF_PLE_PREP, weight_bytes(m.per_layer_model_proj));
		float* ple_inputs = cache.sple_inputs.data();
		const bool have_table = m.per_layer_token_embd != nullptr;
		float* bproj = have_table ? cache.sple_proj.data() : ple_inputs;

		if (have_table)
		{
			const float tok_scale = std::sqrt(static_cast<float>(ple_dim));
			for (int t = 0; t < n_tok; ++t)
				model_embed_row(m.per_layer_token_embd,
				                ple_inputs + static_cast<size_t>(t) * ple_total,
				                tokens[t], static_cast<size_t>(ple_total), tok_scale);
		}

		gemv_batch(m.per_layer_model_proj, xp, h.n_embd, bproj, ple_total, n_tok, s.xsum);

		const float inv_sqrt_n_embd = 1.0f / std::sqrt(static_cast<float>(h.n_embd));
		const size_t ple_all = static_cast<size_t>(n_tok) * ple_total;
		for (size_t i = 0; i < ple_all; ++i) bproj[i] *= inv_sqrt_n_embd;

		// Each layer's slice is an independent RMSNorm group, and the
		// gamma is shared, so the whole batch is one call.
		nnc_rmsnorm_gamma_multi_f32(bproj, bproj,
		                            static_cast<size_t>(n_tok) * h.n_layer,
		                            ple_dim, f32(m.per_layer_proj_norm), h.rms_eps);

		if (have_table)
		{
			const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
			for (size_t i = 0; i < ple_all; ++i)
				ple_inputs[i] = (bproj[i] + ple_inputs[i]) * inv_sqrt2;
		}
	}

	for (int li = 0; li < h.n_layer; ++li)
	{
		s.ple_for_layer = have_ple
			                  ? cache.sple_inputs.data() + static_cast<size_t>(li) * ple_dim
			                  : nullptr;
		layer_forward(m, cache, li, pos0, n_tok, xp, s);
	}

	if (m.output_norm)
	{
		nnc_perf_scope ps(NNC_PERF_NORM);
		nnc_rmsnorm_gamma_multi_f32(xp, xp, n_tok, h.n_embd, f32(m.output_norm), h.rms_eps);
	}

	cache.cur_pos = pos0 + n_tok;
	cache.kv_written = std::max(pos0 == 0 ? 0 : cache.kv_written, cache.cur_pos);
	return true;
}

bool model_kv_rewind(model_kv_cache& cache, const int pos)
{
	if (pos < 0 || pos > cache.cur_pos) return false;
	if (pos > 0 && cache.kv_written - pos > cache.max_rewind) return false;
	cache.cur_pos = pos;
	return true;
}

const float* model_last_hidden(const nnc_model& m, const model_kv_cache& cache, const int n_tok)
{
	return cache.sx.data() + static_cast<size_t>(n_tok - 1) * m.hparams.n_embd;
}

bool model_is_recurrent(const nnc_model& m)
{
	return m.hparams.n_layer_rs > 0;
}

void model_state_save(model_kv_cache& cache)
{
	if (cache.conv_state.empty()) return;
	std::copy(cache.conv_state.begin(), cache.conv_state.end(), cache.conv_saved.begin());
	std::copy(cache.ssm_state.begin(), cache.ssm_state.end(), cache.ssm_saved.begin());
	cache.saved_pos = cache.cur_pos;
}

bool model_state_restore(model_kv_cache& cache)
{
	if (cache.conv_state.empty() || cache.saved_pos < 0) return false;
	std::copy(cache.conv_saved.begin(), cache.conv_saved.end(), cache.conv_state.begin());
	std::copy(cache.ssm_saved.begin(), cache.ssm_saved.end(), cache.ssm_state.begin());
	cache.cur_pos = cache.saved_pos;
	return true;
}

void model_logits(const nnc_model& m, const float* hidden, float* logits)
{
	const nnc_tensor* lm_head = m.output ? m.output : m.token_embd;
	nnc_perf_scope ps(NNC_PERF_LM_HEAD, weight_bytes(lm_head));
	model_gemv(lm_head, hidden, logits);
	if (m.hparams.final_logit_softcap > 0)
		nnc_softcap_f32(logits, logits, static_cast<size_t>(m.hparams.n_vocab), m.hparams.final_logit_softcap);
}

int model_argmax(const nnc_model& m, const float* hidden, float* scratch)
{
	const nnc_tensor* lm_head = m.output ? m.output : m.token_embd;
	nnc_perf_scope ps(NNC_PERF_LM_HEAD, weight_bytes(lm_head));
	const uint32_t rows = static_cast<uint32_t>(lm_head->ne[1]);
	// BF16 streams the projection through a running argmax; the quantized
	// formats write the logits (a ~1 MB pass, small next to the gemv).
	if (lm_head->type == NNC_TYPE_BF16)
		return nnc_gemv_bf16w_argmax_f32x(lm_head->data, hidden, rows, static_cast<uint32_t>(lm_head->ne[0]));

	model_gemv(lm_head, hidden, scratch);
	int best = 0;
	for (uint32_t r = 1; r < rows; ++r)
		if (scratch[r] > scratch[best]) best = static_cast<int>(r);
	return best;
}

void model_logits_batch(const nnc_model& m, model_kv_cache& cache, const int n, float* logits, const bool soft_cap)
{
	const nnc_tensor* lm_head = m.output ? m.output : m.token_embd;
	const size_t n_vocab = static_cast<size_t>(m.hparams.n_vocab);
	{
		nnc_perf_scope ps(NNC_PERF_LM_HEAD, weight_bytes(lm_head));
		gemv_batch(lm_head, cache.sx.data(), static_cast<size_t>(m.hparams.n_embd), logits, n_vocab, n,
		           cache.sxsum.data());
	}
	if (soft_cap && m.hparams.final_logit_softcap > 0)
		for (int i = 0; i < n; ++i)
			nnc_softcap_f32(logits + i * n_vocab, logits + i * n_vocab, n_vocab, m.hparams.final_logit_softcap);
}

// ----------------------------------------------------------------------------
// MTP drafting.
// ----------------------------------------------------------------------------

bool model_draft_init(const nnc_model& d, const nnc_model& t, model_kv_cache& tcache, model_kv_cache& dcache)
{
	const auto& dh = d.hparams;
	const auto& th = t.hparams;
	if (dh.n_embd_target <= 0)
		return nnc_fail(NNC_ERR_INVALID_ARGUMENT, "%s is not an MTP drafter", d.name.c_str());
	if (th.arch_id != model_arch::gemma4 || !th.causal || th.n_embd_target > 0 || dh.n_embd_target != th.n_embd
		|| dh.n_vocab != th.n_vocab)
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "the drafter %s was not made for %s (it drafts for a Gemma 4 "
		                "model %d wide)", d.name.c_str(), t.name.c_str(), dh.n_embd_target);

	// The drafter's sliding layers read the cache of the target's last
	// sliding layer, and its global layers that of the last global layer.
	int last_swa = -1, last_global = -1;
	for (int li = 0; li < th.n_layer; ++li) (t.layers[li].sliding_window ? last_swa : last_global) = li;
	if (!model_kv_init(d, dcache, tcache.n_ctx, 1)) return false;
	for (int li = 0; li < dh.n_layer; ++li)
	{
		const model_layer& L = d.layers[li];
		const int tl = L.sliding_window ? last_swa : last_global;
		if (tl < 0 || L.head_dim != t.layers[tl].head_dim || L.n_head_kv != t.layers[tl].n_head_kv
			|| (L.sliding_window && dh.sliding_window != th.sliding_window))
		{
			dcache = model_kv_cache{};
			return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "the drafter %s does not match the attention of %s",
			                d.name.c_str(), t.name.c_str());
		}
		dcache.kv_buf[li] = tcache.kv_buf[tl];
	}
	dcache.kv_ext = &tcache;
	return true;
}

int model_draft_step(const nnc_model& d, const nnc_model& t, model_kv_cache& dc, const int token, const float* h,
                     const int pos, float* h_out, float* p)
{
	const auto& dh = d.hparams;
	const size_t n_tgt = static_cast<size_t>(dh.n_embd_target);
	const size_t n_embd = static_cast<size_t>(dh.n_embd);
	const nnc_tensor* lm_head = d.output ? d.output : d.token_embd;

	// The step is profiled as one phase: its layers' own scopes are off.
	const bool perf = g_nnc_perf_on;
	const int64_t t0 = perf ? nnc_time_ns() : 0;
	g_nnc_perf_on = false;

	// [the target's embedding of the token | the hidden state before it]
	float* xh = dc.sdraft_in.data();
	model_embed_row(t.token_embd, xh, token, n_tgt, t.hparams.embd_scale);
	std::memcpy(xh + n_tgt, h, n_tgt * sizeof(float));
	float* x = dc.sx.data();
	layer_scratch s = scratch_for(dc, 0);
	model_gemv(d.nextn_pre, xh, x);

	// All steps of a round share the position of the first token.
	for (int li = 0; li < dh.n_layer; ++li) layer_forward(d, dc, li, pos, 1, x, s);
	nnc_rmsnorm_gamma_multi_f32(x, x, 1, n_embd, f32(d.output_norm), dh.rms_eps);
	model_gemv(d.nextn_post, x, h_out);

	float* logits = dc.sdraft_logits.data();
	model_gemv(lm_head, x, logits);
	const int n_vocab = dh.n_vocab;
	int best = 0;
	for (int i = 1; i < n_vocab; ++i)
		if (logits[i] > logits[best]) best = i;
	nnc_softmax_f32_inplace(logits, static_cast<size_t>(n_vocab));
	*p = logits[best];

	g_nnc_perf_on = perf;
	if (perf)
	{
		uint64_t bytes = weight_bytes(d.nextn_pre) + weight_bytes(d.nextn_post) + weight_bytes(lm_head);
		for (const model_layer& L : d.layers)
			bytes += weight_bytes(L.attn_q) + weight_bytes(L.attn_output) + weight_bytes(L.ffn_gate)
				+ weight_bytes(L.ffn_up) + weight_bytes(L.ffn_down);
		nnc_perf_add(NNC_PERF_DRAFT, nnc_time_ns() - t0, bytes);
	}
	return best;
}

// ----------------------------------------------------------------------------
// Attention spans and embeddings.
// ----------------------------------------------------------------------------

void model_attn_span(const bool causal, const int window, const int pos, const int n_seq, int* first, int* last)
{
	if (causal)
	{
		*first = (window > 0 && pos + 1 > window) ? pos + 1 - window : 0;
		*last = std::min(pos, n_seq - 1);
		return;
	}
	*first = 0;
	*last = n_seq - 1;
	if (window > 0)
	{
		const int half = window / 2;
		*first = std::max(0, pos - half);
		*last = std::min(n_seq - 1, pos + half);
	}
}

void model_pool(const model_pooling pooling, const float* rows, const int n_tok, const int n, float* out)
{
	const size_t w = static_cast<size_t>(n);
	if (pooling == model_pooling::cls || pooling == model_pooling::last)
	{
		const size_t row = pooling == model_pooling::cls ? 0 : static_cast<size_t>(n_tok - 1);
		std::memcpy(out, rows + row * w, w * sizeof(float));
		return;
	}
	for (size_t i = 0; i < w; ++i)
	{
		double sum = 0.0;
		for (int t = 0; t < n_tok; ++t) sum += rows[static_cast<size_t>(t) * w + i];
		out[i] = static_cast<float>(sum / n_tok);
	}
}

void model_l2_normalize(float* x, const int n)
{
	double ss = 0.0;
	for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
	if (!(ss > 0.0)) return;
	const float inv = static_cast<float>(1.0 / std::sqrt(ss));
	for (int i = 0; i < n; ++i) x[i] *= inv;
}

bool model_embed(const nnc_model& m, const std::string& text, float* out, const int n_out)
{
	const auto& h = m.hparams;
	if (h.n_embd_out <= 0)
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "%s generates text; it is not a text-embedding model",
		                m.name.c_str());
	if (!out || n_out <= 0 || n_out > h.n_embd_out)
		return nnc_fail(NNC_ERR_INVALID_ARGUMENT, "this model's embeddings have up to %d values, not %d",
		                h.n_embd_out, n_out);

	// [BOS] text [EOS], with the text cut so that the whole fits.
	std::vector<int> ids = vocab_tokenize(m.vocab, text, m.vocab.add_bos);
	const bool eos = m.vocab.add_eos && m.vocab.eos_id >= 0;
	const size_t limit = static_cast<size_t>(std::min(h.n_ctx, MODEL_EMBED_MAX_TOKENS)) - (eos ? 1 : 0);
	if (ids.size() > limit)
	{
		nnc_log(NNC_LOG_DEBUG, "embedding the first %zu of %zu tokens", limit, ids.size());
		ids.resize(limit);
	}
	if (eos) ids.push_back(m.vocab.eos_id);
	if (ids.empty()) return nnc_fail(NNC_ERR_INVALID_ARGUMENT, "there is no text to embed");
	const int n_tok = static_cast<int>(ids.size());

	model_kv_cache cache;
	if (!model_kv_init(m, cache, n_tok, n_tok, true) || !model_forward(m, cache, ids.data(), n_tok, 0)) return false;

	size_t width = static_cast<size_t>(h.n_embd);
	for (const nnc_tensor* p : m.embd_proj) width = std::max(width, static_cast<size_t>(p->ne[1]));
	std::vector<float> a(width), b(width);
	model_pool(h.pooling, cache.sx.data(), n_tok, h.n_embd, a.data());
	for (const nnc_tensor* p : m.embd_proj)
	{
		model_gemv(p, a.data(), b.data());
		a.swap(b);
	}
	// Matryoshka models are trained so that a prefix of the vector is itself
	// an embedding; it is normalised on its own.
	model_l2_normalize(a.data(), n_out);
	std::memcpy(out, a.data(), sizeof(float) * static_cast<size_t>(n_out));
	return true;
}
