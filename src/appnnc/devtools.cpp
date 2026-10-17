// nnc app — inspection and smoke-test modes. See devtools.h.

#include "devtools.h"

#include "chat.h"
#include "models.h"
#include "platform.h"
#include "ui.h"

// libnnc internals: these tools look at what the public API hides.
#include "gguf.h"
#include "model.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
	// ---- GGUF printing --------------------------------------------------

	void print_value(const gguf_value& v)
	{
		switch (v.type)
		{
		case GGUF_TYPE_UINT8:
		case GGUF_TYPE_UINT16:
		case GGUF_TYPE_UINT32:
		case GGUF_TYPE_UINT64:
			std::printf("%" PRIu64, v.u64);
			break;
		case GGUF_TYPE_INT8:
		case GGUF_TYPE_INT16:
		case GGUF_TYPE_INT32:
		case GGUF_TYPE_INT64:
			std::printf("%" PRId64, v.i64);
			break;
		case GGUF_TYPE_FLOAT32:
		case GGUF_TYPE_FLOAT64:
			std::printf("%g", v.f64);
			break;
		case GGUF_TYPE_BOOL:
			std::printf("%s", v.u64 ? "true" : "false");
			break;
		case GGUF_TYPE_STRING:
			{
				// Truncate long strings (e.g. chat templates) for readability.
				constexpr size_t MAX = 96;
				if (v.str.size() <= MAX)
					std::printf("\"%s\"", v.str.c_str());
				else
					std::printf("\"%.*s...\" (%zu chars)", static_cast<int>(MAX), v.str.c_str(), v.str.size());
				break;
			}
		case GGUF_TYPE_ARRAY:
			{
				static const char* const names[] = {"u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool",
				                                    "string", "array", "u64", "i64", "f64"};
				const char* etn = (v.arr_type <= GGUF_TYPE_FLOAT64) ? names[v.arr_type] : "?";
				std::printf("array<%s>[%zu]", etn, v.arr.size());
				const size_t show = v.arr.size() < 4 ? v.arr.size() : 4;
				if (show > 0 && v.arr_type != GGUF_TYPE_ARRAY)
				{
					std::printf(" = { ");
					for (size_t i = 0; i < show; ++i)
					{
						if (i) std::printf(", ");
						print_value(v.arr[i]);
					}
					if (v.arr.size() > show) std::printf(", ...");
					std::printf(" }");
				}
				break;
			}
		}
	}

	int gguf_info(const std::string& path)
	{
		gguf_file f{};
		if (!gguf_load(path, f))
		{
			ui_error("%s", nnc_last_error());
			return 1;
		}
		std::printf("gguf: version=%u, kv=%" PRIu64 ", tensors=%" PRIu64 ", alignment=%" PRIu64
		            ", data_offset=0x%" PRIx64 "\n",
		            f.version, f.kv_count, f.tensor_count, f.alignment, f.data_offset);

		std::printf("\n--- metadata (%zu entries) ---\n", f.kv.size());
		for (const auto& kv : f.kv)
		{
			std::printf("  %s = ", kv.key.c_str());
			print_value(kv.value);
			std::printf("\n");
		}

		std::printf("\n--- tensors (%zu) ---\n", f.tensors.size());
		std::printf("  %-48s %-8s %-28s %16s\n", "name", "type", "shape", "offset");
		uint64_t total_elems = 0;
		for (const auto& t : f.tensors)
		{
			std::string shape = "[";
			for (uint32_t d = 0; d < t.n_dims; ++d)
			{
				if (d) shape += ", ";
				shape += std::to_string(t.ne[d]);
			}
			shape += "]";
			total_elems += gguf_tensor_nelements(t);
			std::printf("  %-48s %-8s %-28s 0x%14" PRIx64 "\n", t.name.c_str(), gguf_ggml_type_name(t.ggml_type),
			            shape.c_str(), t.offset);
		}
		std::printf("\n  total elements across all tensors: %" PRIu64 "\n", total_elems);
		return 0;
	}

	// ---- --gguf-stats ---------------------------------------------------

	// One element of an F32 / F16 / BF16 tensor, or NaN for other types.
	float elem_to_f32(const uint32_t type, const uint8_t* p)
	{
		float v = std::numeric_limits<float>::quiet_NaN();
		if (type == 0)
		{
			std::memcpy(&v, p, 4);
		}
		else if (type == 1 || type == 30)
		{
			uint16_t h;
			std::memcpy(&h, p, 2);
			if (type == 30)
			{
				const uint32_t u = static_cast<uint32_t>(h) << 16;
				std::memcpy(&v, &u, 4);
			}
			else
			{
				const uint32_t s = (h & 0x8000u) << 16;
				const uint32_t e = (h >> 10) & 0x1fu;
				const uint32_t m = h & 0x3ffu;
				uint32_t u;
				if (e == 0)
				{
					v = std::ldexp(static_cast<float>(m), -24); // subnormal (or zero)
					return s ? -v : v;
				}
				if (e == 31) u = s | 0x7f800000u | (m << 13);
				else u = s | ((e + 112) << 23) | (m << 13);
				std::memcpy(&v, &u, 4);
			}
		}
		return v;
	}

	bool stats_for_tensor(const gguf_file& f, const size_t idx)
	{
		const auto& t = f.tensors[idx];
		const uint64_t n = gguf_tensor_nelements(t);
		std::printf("  %-48s %-6s n=%-12" PRIu64, t.name.c_str(), gguf_ggml_type_name(t.ggml_type), n);
		if (t.ggml_type != 0 && t.ggml_type != 1 && t.ggml_type != 30)
		{
			std::printf("  (no scalar decode for this type)\n");
			return true;
		}
		const auto base = static_cast<const uint8_t*>(gguf_tensor_data(f, idx));
		if (!base)
		{
			std::printf("  (no data)\n");
			return false;
		}
		const size_t bytes_per = (t.ggml_type == 0) ? 4 : 2;
		double sum = 0.0, sum_sq = 0.0;
		float lo = std::numeric_limits<float>::infinity();
		float hi = -std::numeric_limits<float>::infinity();
		uint64_t nan_count = 0;
		for (uint64_t i = 0; i < n; ++i)
		{
			const float v = elem_to_f32(t.ggml_type, base + i * bytes_per);
			if (std::isnan(v))
			{
				++nan_count;
				continue;
			}
			lo = std::min(lo, v);
			hi = std::max(hi, v);
			sum += v;
			sum_sq += static_cast<double>(v) * v;
		}
		const double mean = n ? sum / static_cast<double>(n) : 0.0;
		const double var = n ? (sum_sq / static_cast<double>(n)) - mean * mean : 0.0;
		std::printf("  min=%+.4g max=%+.4g mean=%+.4g rms=%.4g", static_cast<double>(lo), static_cast<double>(hi),
		            mean, std::sqrt(var > 0 ? var : 0.0));
		if (nan_count) std::printf(" nan=%" PRIu64, nan_count);
		const uint64_t show = n < 6 ? n : 6;
		std::printf("  first=[");
		for (uint64_t i = 0; i < show; ++i)
			std::printf("%s%+.4g", i ? ", " : "", static_cast<double>(elem_to_f32(t.ggml_type, base + i * bytes_per)));
		std::printf("%s]\n", show < n ? ", ..." : "");
		return true;
	}

	int gguf_stats(const std::string& path, const std::string& needle)
	{
		gguf_file f{};
		if (!gguf_mmap(path, f))
		{
			ui_error("%s", nnc_last_error());
			return 1;
		}
		std::printf("gguf: '%s' mapped %" PRIu64 " bytes, %" PRIu64 " tensors\n", path.c_str(), f.mapped_size,
		            f.tensor_count);
		if (!needle.empty()) std::printf("gguf: filter = substring '%s'\n", needle.c_str());
		std::printf("\n");

		int matched = 0, unreadable = 0;
		for (size_t i = 0; i < f.tensors.size(); ++i)
		{
			if (!needle.empty() && f.tensors[i].name.find(needle) == std::string::npos) continue;
			// A tensor whose bytes lie outside the file is a failed check.
			if (!stats_for_tensor(f, i)) ++unreadable;
			++matched;
		}
		if (matched == 0) std::printf("(no tensors matched)\n");
		gguf_unmap(f);
		if (unreadable != 0)
		{
			ui_error("%d tensor(s) in '%s' have no data inside the file", unreadable, path.c_str());
			return 1;
		}
		return 0;
	}

	// ---- --inspect-model ------------------------------------------------

	// Collapse runs of digits so "blk.0.attn_q.weight" and
	// "blk.31.attn_q.weight" share the pattern "blk.*.attn_q.weight".
	std::string tensor_name_pattern(const std::string& name)
	{
		std::string out;
		bool prev_digit = false;
		for (const char c : name)
		{
			const bool digit = c >= '0' && c <= '9';
			if (!digit) out.push_back(c);
			else if (!prev_digit) out.push_back('*');
			prev_digit = digit;
		}
		return out;
	}

	// A digest aimed at "would nnc load this, and if not, why?". Reads only
	// the header, metadata and tensor table; then tries the real loader's
	// verdict without decoding anything heavy.
	int inspect_model(const std::string& path)
	{
		gguf_file f{};
		if (!gguf_load(path, f))
		{
			ui_error("%s", nnc_last_error());
			return 1;
		}

		auto kv_str = [&](const char* key, const char* dflt) -> std::string
		{
			const gguf_value* v = gguf_find_kv(f, key);
			return (v && v->type == GGUF_TYPE_STRING) ? v->str : dflt;
		};
		auto kv_num = [&](const std::string& key, std::string& out) -> bool
		{
			const gguf_value* v = gguf_find_kv(f, key);
			if (!v) return false;
			char buf[64];
			switch (v->type)
			{
			case GGUF_TYPE_FLOAT32:
			case GGUF_TYPE_FLOAT64: std::snprintf(buf, sizeof(buf), "%.6g", v->f64);
				break;
			case GGUF_TYPE_INT8:
			case GGUF_TYPE_INT16:
			case GGUF_TYPE_INT32:
			case GGUF_TYPE_INT64: std::snprintf(buf, sizeof(buf), "%" PRId64, v->i64);
				break;
			case GGUF_TYPE_ARRAY: std::snprintf(buf, sizeof(buf), "array[%zu]", v->arr.size());
				break;
			default: std::snprintf(buf, sizeof(buf), "%" PRIu64, v->u64);
				break;
			}
			out = buf;
			return true;
		};

		std::printf("file:         %s\n", path.c_str());
		std::printf("gguf:         v%u, %" PRIu64 " KVs, %" PRIu64 " tensors, alignment=%" PRIu64 "\n", f.version,
		            f.kv_count, f.tensor_count, f.alignment);
		const std::string arch = kv_str("general.architecture", "(missing)");
		std::printf("architecture: %s\n", arch.c_str());
		std::printf("name:         %s\n", kv_str("general.name", "(none)").c_str());
		if (const gguf_value* vocab = gguf_find_kv(f, "tokenizer.ggml.tokens"); vocab && vocab->type == GGUF_TYPE_ARRAY)
			std::printf("vocab:        %zu tokens (%s)\n", vocab->arr.size(), kv_str("tokenizer.ggml.model", "?").c_str());
		for (const char* suffix : {"context_length", "embedding_length", "block_count", "attention.head_count",
		                           "attention.head_count_kv", "attention.key_length", "feed_forward_length",
		                           "rope.dimension_count", "rope.freq_base", "attention.layer_norm_rms_epsilon",
		                           "expert_count"})
		{
			std::string val;
			if (kv_num(arch + "." + suffix, val)) std::printf("  %-44s = %s\n", (arch + "." + suffix).c_str(), val.c_str());
		}

		std::unordered_map<uint32_t, uint64_t> type_count, type_bytes;
		uint64_t total_bytes = 0;
		for (const auto& t : f.tensors)
		{
			type_count[t.ggml_type] += 1;
			type_bytes[t.ggml_type] += gguf_tensor_nbytes(t);
			total_bytes += gguf_tensor_nbytes(t);
		}
		std::printf("\ntensor types (%zu unique, %s total):\n", type_count.size(), format_bytes(total_bytes).c_str());
		for (const auto& [type, count] : type_count)
			std::printf("  %-10s %6" PRIu64 " tensors  %10s\n", gguf_ggml_type_name(type), count,
			            format_bytes(type_bytes[type]).c_str());

		std::unordered_map<std::string, uint64_t> pattern_count;
		std::vector<std::string> pattern_order;
		for (const auto& t : f.tensors)
		{
			const std::string p = tensor_name_pattern(t.name);
			if (pattern_count.emplace(p, 0).second) pattern_order.push_back(p);
			pattern_count[p] += 1;
		}
		std::printf("\ntensor name patterns (%zu unique):\n", pattern_order.size());
		for (const auto& p : pattern_order) std::printf("  %4" PRIu64 "  %s\n", pattern_count[p], p.c_str());

		const bool arch_ok = arch == "gemma4" || arch == "gemma3" || arch == "llama" || arch == "qwen2"
			|| arch == "qwen3" || arch == "qwen35" || arch == "gemma-embedding" || arch == "gemma-embedding2"
			|| arch == "gemma4-assistant";
		bool types_ok = true;
		for (const auto& [type, count] : type_count)
			types_ok = types_ok && (type == 0 || type == 1 || type == 30 || nnc_quant_block_elems(type) > 0);
		std::printf("\nnnc compatibility:\n");
		std::printf("  architecture supported: %s\n",
		            arch_ok ? "yes" : "no  (nnc runs gemma3, gemma4, llama, qwen2, qwen3, qwen35, gemma-embedding, "
		                              "gemma-embedding2 and gemma4-assistant drafters)");
		std::printf("  tensor types supported: %s\n",
		            types_ok
			            ? "yes"
			            : "no  (nnc reads F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K; see above)");
		return 0;
	}

	// ---- model diagnostics ------------------------------------------------

	struct stats
	{
		float lo, hi;
		double mean, rms;
	};

	stats compute_stats(const float* p, const size_t n)
	{
		stats s{std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 0.0, 0.0};
		double sum = 0.0, sum2 = 0.0;
		for (size_t i = 0; i < n; ++i)
		{
			s.lo = std::min(s.lo, p[i]);
			s.hi = std::max(s.hi, p[i]);
			sum += p[i];
			sum2 += static_cast<double>(p[i]) * p[i];
		}
		s.mean = n ? sum / static_cast<double>(n) : 0.0;
		s.rms = n ? std::sqrt(sum2 / static_cast<double>(n)) : 0.0;
		return s;
	}

	void print_stats(const char* tag, const float* p, const size_t n)
	{
		const stats s = compute_stats(p, n);
		std::printf("  %-26s n=%-6zu min=%+.4g max=%+.4g mean=%+.4g rms=%.4g\n", tag, n,
		            static_cast<double>(s.lo), static_cast<double>(s.hi), s.mean, s.rms);
	}

	std::string token_label(const nnc_model* m, const nnc_token t)
	{
		const char* s = nnc_token_text(m, t);
		return s ? s : "?";
	}

	bool parse_token(const std::string& s, const nnc_model* m, nnc_token& out)
	{
		errno = 0;
		char* end = nullptr;
		const long v = std::strtol(s.c_str(), &end, 10);
		if (errno != 0 || end == s.c_str() || *end != '\0' || v < 0 || v >= nnc_model_get_info(m).n_vocab)
		{
			ui_error("'%s' is not a token id (the vocabulary has %d)", s.c_str(), nnc_model_get_info(m).n_vocab);
			return false;
		}
		out = static_cast<nnc_token>(v);
		return true;
	}

	bool parse_count(const std::string& s, const char* what, int& out)
	{
		errno = 0;
		char* end = nullptr;
		const long v = std::strtol(s.c_str(), &end, 10);
		if (errno != 0 || end == s.c_str() || *end != '\0' || v < 0 || v > INT_MAX)
		{
			ui_error("invalid %s '%s'", what, s.c_str());
			return false;
		}
		out = static_cast<int>(v);
		return true;
	}

	// Top-5 next tokens after `token`, through the public API.
	void print_top5(const nnc_model* m, const float* logits)
	{
		const int n_vocab = nnc_model_get_info(m).n_vocab;
		std::vector<int> idx(static_cast<size_t>(n_vocab));
		for (int i = 0; i < n_vocab; ++i) idx[static_cast<size_t>(i)] = i;
		std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
		                  [&](const int a, const int b) { return logits[a] > logits[b]; });
		std::printf("\ntop 5 next tokens:\n");
		for (int i = 0; i < 5; ++i)
			std::printf("  [%d]  id=%-7d logit=%+8.4f  token='%s'\n", i, idx[i], logits[idx[i]],
			            token_label(m, idx[i]).c_str());
	}

	void print_model_info(const nnc_model& f)
	{
		const auto& h = f.hparams;
		std::printf("%s: arch=%s file_type=%d weights=%s\n", f.name.c_str(), h.arch.c_str(), h.file_type,
		            f.weights == NNC_WEIGHTS_Q4 ? "Q4" : f.weights == NNC_WEIGHTS_BF16 ? "BF16" : "Q8_0");
		std::printf("  n_layer=%d n_embd=%d n_head=%d n_head_kv=%d head_dim=%d (swa=%d)\n", h.n_layer, h.n_embd,
		            h.n_head, h.n_head_kv, h.head_dim, h.head_dim_swa);
		std::printf("  rope: dim=%d (swa=%d) freq_base=%g (swa=%g) scale=%g mode=%s freq_factors=%s\n", h.rope_dim,
		            h.rope_dim_swa, h.rope_freq_base, h.rope_freq_base_swa, h.rope_freq_scale,
		            h.rope_mode == NNC_ROPE_NORM ? "normal" : "neox", f.rope_freqs ? "yes" : "no");
		std::printf("  ctx=%d sliding_window=%d shared_kv_layers=%d ple_dim=%d softcap=%g rms_eps=%g\n", h.n_ctx,
		            h.sliding_window, h.shared_kv_layers, h.ple_dim, h.final_logit_softcap, h.rms_eps);
		std::printf("  attention_scale=%g embd_scale=%g\n", h.attention_scale, h.embd_scale);
		if (h.n_layer_rs > 0)
			std::printf("  gated deltanet: %d layers, conv=%d, %d q/k heads of %d, %d v heads of %d\n", h.n_layer_rs,
			            h.ssm_d_conv, h.ssm_n_k, h.ssm_d_k, h.ssm_n_v, h.ssm_d_v);
		const auto& v = f.vocab;
		std::printf("  vocab=%d tokens, %s (bos=%d eos=%d unk=%d pad=%d), chat=%s, end-of-generation:", h.n_vocab,
		            v.model.c_str(), v.bos_id, v.eos_id, v.unk_id, v.pad_id,
		            v.chat == chat_format::gemma    ? "gemma"
		            : v.chat == chat_format::gemma4 ? "gemma4"
		            : v.chat == chat_format::llama3 ? "llama3"
		            : v.chat == chat_format::chatml ? "chatml"
		                                            : "none");
		for (const int id : v.eog_ids) std::printf(" %d'%s'", id, v.tokens[static_cast<size_t>(id)].c_str());
		if (v.can_think || v.always_thinks)
		{
			std::string open = v.always_thinks ? v.reply_open : v.reply_open_think;
			for (size_t at = open.find('\n'); at != std::string::npos; at = open.find('\n', at + 2))
				open.replace(at, 1, "\\n");
			const std::string system = v.tok_think >= 0 ? "<|think|>"
				: v.think_instructions.empty()          ? "-"
				                                        : "\"" + v.think_instructions + "\"";
			std::printf("\n  thinking: %s; the reply opens \"%s\"; the system turn starts %s",
			            v.always_thinks ? "always" : "with --think", open.c_str(), system.c_str());
		}
		std::printf("\n  %.1f MB decoded, %" PRIu64 " parameters\n",
		            static_cast<double>(f.decoded_bytes) / (1024.0 * 1024.0), f.n_params);

		auto report = [&](const char* name, nnc_tensor* model_layer::* p)
		{
			int n = 0;
			for (const auto& L : f.layers)
				if (L.*p) ++n;
			const char* tag = (n == h.n_layer) ? "ok" : (n == 0 ? "absent" : "partial");
			std::printf("  layer.%-22s : %3d / %d  [%s]\n", name, n, h.n_layer, tag);
		};
		std::printf("\nper-layer tensor presence:\n");
		report("attn_norm", &model_layer::attn_norm);
		report("ffn_norm", &model_layer::ffn_norm);
		report("post_attention_norm", &model_layer::post_attention_norm);
		report("post_ffw_norm", &model_layer::post_ffw_norm);
		report("post_norm", &model_layer::post_norm);
		report("attn_q_norm", &model_layer::attn_q_norm);
		report("attn_k_norm", &model_layer::attn_k_norm);
		report("attn_q", &model_layer::attn_q);
		report("attn_k", &model_layer::attn_k);
		report("attn_v", &model_layer::attn_v);
		report("attn_output", &model_layer::attn_output);
		report("attn_q.bias", &model_layer::attn_q_b);
		report("attn_qkv (deltanet)", &model_layer::ssm_qkv);
		report("ssm_out (deltanet)", &model_layer::ssm_out);
		report("ffn_gate", &model_layer::ffn_gate);
		report("ffn_up", &model_layer::ffn_up);
		report("ffn_down", &model_layer::ffn_down);
		report("inp_gate", &model_layer::inp_gate);
		report("proj", &model_layer::proj);
		report("layer_output_scale", &model_layer::layer_output_scale);
		if (h.n_layer_kv < h.n_layer)
			std::printf("  (layers %d..%d reuse earlier layers' K/V, so load no K/V weights)\n", h.n_layer_kv,
			            h.n_layer - 1);

		std::printf("\ntop-level: token_embd=%s output=%s output_norm=%s rope_freqs=%s per_layer_token_embd=%s\n",
		            f.token_embd ? "ok" : "absent", f.output ? "ok" : "tied", f.output_norm ? "ok" : "absent",
		            f.rope_freqs ? "ok" : "absent", f.per_layer_token_embd ? "ok" : "absent");

		int sw = 0, min_ff = INT_MAX, max_ff = 0, min_hd = INT_MAX, max_hd = 0;
		for (const auto& L : f.layers)
		{
			sw += L.sliding_window ? 1 : 0;
			min_ff = std::min(min_ff, L.n_ff);
			max_ff = std::max(max_ff, L.n_ff);
			if (L.rs_index >= 0) continue; // a Gated DeltaNet layer has no attention heads
			min_hd = std::min(min_hd, L.head_dim);
			max_hd = std::max(max_hd, L.head_dim);
		}
		std::printf("sliding-window layers: %d / %d\n", sw, h.n_layer);
		std::printf("per-layer n_ff: %d..%d, head_dim: %d..%d\n", min_ff, max_ff, min_hd, max_hd);
	}

	// The first attention layer by hand (embed, attn_norm, Q, q_norm, RoPE),
	// then the full forward pass through the public API.
	int probe(nnc_model* model, nnc_token token)
	{
		const nnc_model& f = *model;
		const auto& h = f.hparams;
		int li = 0;
		while (li < h.n_layer && f.layers[li].rs_index >= 0) ++li;
		if (li == h.n_layer)
		{
			std::printf("probe: the model has no attention layers\n");
			return 0;
		}
		const model_layer& L0 = f.layers[li];
		const int n_embd = h.n_embd;
		const int q_dim = L0.n_head * L0.head_dim;
		std::printf("probe: token %d ('%s')\n", token, token_label(model, token).c_str());
		std::printf("  layer %d: sliding_window=%d q_dim=%d head_dim=%d n_head=%d n_rot=%d gated=%d\n", li,
		            L0.sliding_window ? 1 : 0, q_dim, L0.head_dim, L0.n_head, L0.n_rot, L0.attn_gated ? 1 : 0);

		std::vector<float> x(static_cast<size_t>(n_embd)), normed(x.size()),
		                   q(static_cast<size_t>(L0.attn_q->ne[1]));
		model_embed_row(f.token_embd, x.data(), token, x.size(), h.embd_scale);
		print_stats("embed * scale", x.data(), x.size());
		nnc_rmsnorm_gamma_multi_f32(normed.data(), x.data(), 1, x.size(),
		                            static_cast<const float*>(L0.attn_norm->data), h.rms_eps);
		print_stats("rmsnorm * attn_norm", normed.data(), normed.size());
		model_gemv(L0.attn_q, normed.data(), q.data());
		if (L0.attn_q_b) nnc_add_inplace_f32(q.data(), static_cast<const float*>(L0.attn_q_b->data), q.size());
		print_stats("Q = W_q * normed", q.data(), q.size());
		if (L0.attn_gated)
		{
			// Each head's rows are its query, then its output gate: keep the queries.
			const size_t hd = static_cast<size_t>(L0.head_dim);
			for (size_t hh = 0; hh < static_cast<size_t>(L0.n_head); ++hh)
				std::memmove(q.data() + hh * hd, q.data() + 2 * hh * hd, hd * sizeof(float));
			q.resize(static_cast<size_t>(q_dim));
		}
		if (L0.attn_q_norm)
		{
			nnc_rmsnorm_gamma_multi_f32(q.data(), q.data(), static_cast<size_t>(L0.n_head),
			                            static_cast<size_t>(L0.head_dim),
			                            static_cast<const float*>(L0.attn_q_norm->data), h.rms_eps);
			print_stats("Q after q_norm", q.data(), q.size());
		}
		nnc_rope_f32(q.data(), static_cast<uint32_t>(L0.n_head), static_cast<uint32_t>(L0.head_dim),
		             static_cast<uint32_t>(L0.n_rot), 0.0f, L0.sliding_window ? h.rope_freq_base_swa : h.rope_freq_base,
		             h.rope_mode, L0.rope_freqs);
		print_stats("Q after RoPE (pos 0)", q.data(), q.size());
		return 0;
	}

	// One token through every layer; logit statistics and the top 5.
	int forward(nnc_model* model, const nnc_token token)
	{
		nnc_session_params sp = nnc_session_default_params();
		sp.n_ctx = 16;
		nnc_session* s = nullptr;
		if (nnc_session_create(model, &sp, &s) != NNC_OK)
		{
			ui_error("%s", nnc_last_error());
			return 1;
		}
		const nnc_model_info info = nnc_model_get_info(model);
		std::printf("forward: token %d ('%s'), %d layers, n_embd %d, vocab %d\n", token,
		            token_label(model, token).c_str(), info.n_layer, info.n_embd, info.n_vocab);
		const int64_t t0 = nnc_time_us();
		const nnc_status st = nnc_session_eval(s, &token, 1);
		const float* logits = (st == NNC_OK) ? nnc_session_logits(s) : nullptr;
		const double ms = static_cast<double>(nnc_time_us() - t0) / 1000.0;
		if (!logits)
		{
			ui_error("%s", nnc_last_error());
			nnc_session_free(s);
			return 1;
		}
		print_stats("logits", logits, static_cast<size_t>(info.n_vocab));
		print_top5(model, logits);
		std::printf("\nforward pass: %.1f ms\n", ms);
		nnc_session_free(s);
		return 0;
	}

	int tokenize(nnc_model* model, const std::string& text)
	{
		const nnc_token bos = nnc_token_bos(model);
		const int32_t n = nnc_tokenize(model, text.data(), text.size(), bos >= 0, nullptr, 0);
		std::vector<nnc_token> toks(static_cast<size_t>(n > 0 ? n : 0));
		nnc_tokenize(model, text.data(), text.size(), bos >= 0, toks.data(), n);
		std::printf("input  : %s\ntokens : %d\n", text.c_str(), n);
		std::string round;
		for (size_t k = 0; k < toks.size(); ++k)
		{
			char piece[256];
			const int32_t len = nnc_token_to_piece(model, toks[k], piece, sizeof(piece));
			round.append(piece, static_cast<size_t>(std::min<int32_t>(len, sizeof(piece))));
			std::printf("  [%3zu] id=%-7d '%s'\n", k, toks[k], token_label(model, toks[k]).c_str());
		}
		std::printf("detok  : %s\n", round.c_str());
		return 0;
	}

	// Greedy generation with every token printed: the benchmark harness
	// docs/models.md uses. Prefill and decode are timed separately.
	int generate(nnc_model* model, const std::vector<nnc_token>& prompt, const int n_predict, int n_ctx)
	{
		if (n_ctx <= 0) n_ctx = std::max(256, static_cast<int>(prompt.size()) + n_predict + 1);
		nnc_session_params sp = nnc_session_default_params();
		sp.n_ctx = n_ctx;
		nnc_session* s = nullptr;
		if (nnc_session_create(model, &sp, &s) != NNC_OK)
		{
			ui_error("%s", nnc_last_error());
			return 1;
		}

		std::printf("\nprompt (%zu tokens):\n", prompt.size());
		for (size_t i = 0; i < prompt.size(); ++i)
			std::printf("  [%3zu] id=%-7d '%s'\n", i, prompt[i], token_label(model, prompt[i]).c_str());

		nnc_perf_reset();
		const int64_t t_pre = nnc_time_us();
		if (nnc_session_eval(s, prompt.data(), static_cast<int32_t>(prompt.size())) != NNC_OK)
		{
			ui_error("%s", nnc_last_error());
			nnc_session_free(s);
			return 1;
		}
		nnc_sampler_params greedy = nnc_sampler_default_params();
		greedy.temperature = 0.0f;
		nnc_token next = nnc_session_sample(s, &greedy);
		const double pre_ms = static_cast<double>(nnc_time_us() - t_pre) / 1000.0;
		std::printf("\nprefill: %zu tokens in %.1f ms (%.1f ms/tok)\n", prompt.size(), pre_ms,
		            pre_ms / static_cast<double>(prompt.size()));

		std::printf("\ngenerated:\n");
		// n_step counts decode forward passes; the first token printed came out
		// of prefill, so it is not one. Dividing by n_predict would overstate
		// throughput whenever EOS stops us early.
		int n_step = 0;
		const int64_t t_gen = nnc_time_us();
		for (int g = 0; g < n_predict && next >= 0; ++g)
		{
			std::printf("  [%3d] id=%-7d '%s'\n", g, next, token_label(model, next).c_str());
			std::fflush(stdout);
			if (nnc_token_is_eog(model, next) || g + 1 >= n_predict) break;
			if (nnc_session_eval(s, &next, 1) != NNC_OK)
			{
				ui_error("%s", nnc_last_error());
				break;
			}
			next = nnc_session_sample(s, &greedy);
			++n_step;
		}
		const double gen_ms = static_cast<double>(nnc_time_us() - t_gen) / 1000.0;
		std::printf("\ngenerate: %d decode steps in %.1f ms (%.1f ms/tok)\n", n_step, gen_ms,
		            gen_ms / std::max(1, n_step));
		if (nnc_perf_enabled())
			std::fputs(nnc_perf_report("prefill + generate", static_cast<int32_t>(prompt.size()) + n_step), stdout);
		nnc_session_free(s);
		return 0;
	}

	// Perplexity of the first `max_tokens` tokens of a text file: exp of the
	// mean negative log-likelihood the model gives each next token. Lower is
	// better; comparing weight formats or code changes on the same text is
	// the point, not the absolute number.
	int perplexity(nnc_model* model, const std::string& text_path, const int max_tokens)
	{
		std::ifstream in(u8_path(text_path), std::ios::binary);
		if (!in)
		{
			ui_error("cannot open '%s'", text_path.c_str());
			return 1;
		}
		const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

		const int32_t n_all = nnc_tokenize(model, text.data(), text.size(), nnc_token_bos(model) >= 0, nullptr, 0);
		std::vector<nnc_token> toks(static_cast<size_t>(n_all > 0 ? n_all : 0));
		nnc_tokenize(model, text.data(), text.size(), nnc_token_bos(model) >= 0, toks.data(), n_all);
		if (static_cast<int>(toks.size()) > max_tokens) toks.resize(static_cast<size_t>(max_tokens));
		if (toks.size() < 2)
		{
			ui_error("'%s' is too short to score", text_path.c_str());
			return 1;
		}

		nnc_session_params sp = nnc_session_default_params();
		sp.n_ctx = static_cast<int32_t>(toks.size());
		nnc_session* s = nullptr;
		if (nnc_session_create(model, &sp, &s) != NNC_OK)
		{
			ui_error("%s", nnc_last_error());
			return 1;
		}
		const int n_vocab = nnc_model_get_info(model).n_vocab;
		double nll = 0.0;
		const int64_t t0 = nnc_time_us();
		progress_bar bar("Scoring");
		for (size_t i = 0; i + 1 < toks.size(); ++i)
		{
			const float* logits = (nnc_session_eval(s, &toks[i], 1) == NNC_OK) ? nnc_session_logits(s) : nullptr;
			if (!logits)
			{
				bar.finish("failed");
				ui_error("%s", nnc_last_error());
				nnc_session_free(s);
				return 1;
			}
			float top = logits[0];
			for (int v = 1; v < n_vocab; ++v) top = std::max(top, logits[v]);
			double sum = 0.0;
			for (int v = 0; v < n_vocab; ++v) sum += std::exp(static_cast<double>(logits[v] - top));
			nll += (static_cast<double>(top) + std::log(sum)) - logits[toks[i + 1]];
			bar.update(static_cast<double>(i + 1) / static_cast<double>(toks.size() - 1));
		}
		bar.finish("done");
		const double n = static_cast<double>(toks.size() - 1);
		std::printf("perplexity %.4f over %zu tokens (mean NLL %.4f) in %.1f s\n", std::exp(nll / n),
		            toks.size() - 1, nll / n, static_cast<double>(nnc_time_us() - t0) / 1e6);
		nnc_session_free(s);
		return 0;
	}
}

int run_devtool(const app_options& o)
{
	const auto& a = o.args;
	switch (o.mode)
	{
	case app_mode::gguf_info: return gguf_info(a[0]);
	case app_mode::gguf_stats: return gguf_stats(a[0], a.size() > 1 ? a[1] : std::string());
	case app_mode::inspect_model: return inspect_model(a[0]);
	case app_mode::inspect_all:
		{
			const std::vector<std::string> models = find_models();
			if (models.empty())
			{
				ui_error("no .gguf models found");
				return 1;
			}
			for (const std::string& m : models)
			{
				std::printf("\n========== %s ==========\n", m.c_str());
				inspect_model(m);
			}
			return 0;
		}
	default: break;
	}

	// The rest load the model.
	nnc_model* model = load_model_ui(a[0], o.weights);
	if (!model) return 1;
	int rc = 1;
	nnc_token token = nnc_token_bos(model);
	if (token < 0) token = 0;
	int n_predict = 16, n_ctx = 0;
	switch (o.mode)
	{
	case app_mode::model_info:
		print_model_info(*model);
		rc = 0;
		break;
	case app_mode::probe:
		if (a.size() < 2 || parse_token(a[1], model, token)) rc = probe(model, token);
		break;
	case app_mode::forward:
		if (a.size() < 2 || parse_token(a[1], model, token)) rc = forward(model, token);
		break;
	case app_mode::tokenize:
		rc = tokenize(model, a[1]);
		break;
	case app_mode::prompt:
	case app_mode::gen:
		{
			if ((a.size() > 2 && !parse_count(a[2], "token count", n_predict))
				|| (a.size() > 3 && !parse_count(a[3], "context size", n_ctx)))
				break;
			std::vector<nnc_token> prompt;
			if (o.mode == app_mode::prompt)
			{
				const int32_t n = nnc_tokenize(model, a[1].data(), a[1].size(), nnc_token_bos(model) >= 0, nullptr, 0);
				prompt.resize(static_cast<size_t>(n > 0 ? n : 0));
				nnc_tokenize(model, a[1].data(), a[1].size(), nnc_token_bos(model) >= 0, prompt.data(), n);
			}
			else
			{
				// Space- or comma-separated token ids.
				std::string ids = a[1];
				std::replace(ids.begin(), ids.end(), ',', ' ');
				size_t p = 0;
				bool ok = true;
				while (ok && (p = ids.find_first_not_of(" \t", p)) != std::string::npos)
				{
					const size_t e = ids.find_first_of(" \t", p);
					nnc_token t;
					ok = parse_token(ids.substr(p, e == std::string::npos ? std::string::npos : e - p), model, t);
					if (ok) prompt.push_back(t);
					p = e;
				}
				if (!ok) break;
			}
			if (prompt.empty()) prompt.push_back(token);
			rc = generate(model, prompt, n_predict, n_ctx);
			break;
		}
	case app_mode::perplexity:
		{
			int n_tokens = 512;
			if (a.size() > 2 && !parse_count(a[2], "token count", n_tokens)) break;
			rc = perplexity(model, a[1], n_tokens);
			break;
		}
	default:
		break;
	}
	nnc_model_free(model);
	return rc;
}
