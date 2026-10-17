// nnc — tokenizer. See tokenizer.h.

#include "tokenizer.h"

#include "runtime.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <queue>

namespace
{
	const char* const SPACE_MARK = "\xE2\x96\x81"; // ▁ (U+2581), SentencePiece's space

	// Length in bytes of a UTF-8 codepoint starting at the given lead byte.
	int utf8_len(const unsigned char c)
	{
		if (c < 0x80) return 1;
		if ((c & 0xE0) == 0xC0) return 2;
		if ((c & 0xF0) == 0xE0) return 3;
		if ((c & 0xF8) == 0xF0) return 4;
		return 1; // invalid lead -> consume one byte
	}

	void split_utf8(const std::string& s, std::vector<std::string>& out)
	{
		out.clear();
		size_t i = 0;
		while (i < s.size())
		{
			const size_t n = static_cast<size_t>(utf8_len(static_cast<unsigned char>(s[i])));
			const size_t take = (i + n <= s.size()) ? n : (s.size() - i);
			out.emplace_back(s.data() + i, take);
			i += take;
		}
	}

	std::string spaces_to_marks(const std::string& in)
	{
		std::string out;
		out.reserve(in.size() + in.size() / 4);
		for (const char c : in)
		{
			if (c == ' ') out.append(SPACE_MARK, 3);
			else out.push_back(c);
		}
		return out;
	}

	std::string marks_to_spaces(const std::string& in)
	{
		std::string out;
		out.reserve(in.size());
		for (size_t i = 0; i < in.size();)
		{
			if (in.compare(i, 3, SPACE_MARK) == 0)
			{
				out.push_back(' ');
				i += 3;
			}
			else
			{
				out.push_back(in[i]);
				++i;
			}
		}
		return out;
	}

	std::string encode_utf8(const uint32_t cp)
	{
		std::string s;
		if (cp < 0x80) s.push_back(static_cast<char>(cp));
		else if (cp < 0x800)
		{
			s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
			s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
		}
		else
		{
			s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
			s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
			s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
		}
		return s;
	}

	// GPT-2 byte-to-unicode mapping. Each byte 0..255 maps to a printable
	// Unicode codepoint, encoded as UTF-8. This is the same mapping as
	// the original GPT-2 BPE tokenizer (and Llama-3's, which derives from it).
	const std::array<std::string, 256>& byte_to_unicode()
	{
		static const auto table = []
		{
			std::array<std::string, 256> t;
			// Printable ranges that map to themselves: '!'..'~', '¡'..'¬', '®'..'ÿ'.
			std::vector<int> bs;
			for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
			for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
			for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
			std::vector<int> cs = bs;
			int n = 0;
			for (int b = 0; b < 256; ++b)
			{
				if (std::find(bs.begin(), bs.end(), b) == bs.end())
				{
					bs.push_back(b);
					cs.push_back(256 + n);
					++n;
				}
			}
			for (size_t i = 0; i < bs.size(); ++i)
				t[bs[i]] = encode_utf8(static_cast<uint32_t>(cs[i]));
			return t;
		}();
		return table;
	}

	// Inverse of byte_to_unicode over a whole token: each mapped codepoint
	// becomes its byte, anything else passes through.
	std::string unicode_to_bytes(const std::string& in)
	{
		static const auto rev = []
		{
			std::unordered_map<std::string, unsigned char> m;
			m.reserve(512);
			const auto& tab = byte_to_unicode();
			for (int b = 0; b < 256; ++b) m.emplace(tab[b], static_cast<unsigned char>(b));
			return m;
		}();
		std::string out;
		out.reserve(in.size());
		size_t i = 0;
		while (i < in.size())
		{
			const size_t n = static_cast<size_t>(utf8_len(static_cast<unsigned char>(in[i])));
			const size_t take = (i + n <= in.size()) ? n : (in.size() - i);
			const auto it = rev.find(in.substr(i, take));
			if (it != rev.end()) out.push_back(static_cast<char>(it->second));
			else out.append(in, i, take);
			i += take;
		}
		return out;
	}

	// "<0xHH>" -> HH, or -1.
	int parse_byte_token(const std::string& t)
	{
		if (t.size() != 6 || t[0] != '<' || t[1] != '0' || t[2] != 'x' || t[5] != '>') return -1;
		auto hex = [](const char c) -> int
		{
			if (c >= '0' && c <= '9') return c - '0';
			if (c >= 'A' && c <= 'F') return c - 'A' + 10;
			if (c >= 'a' && c <= 'f') return c - 'a' + 10;
			return -1;
		};
		const int hi = hex(t[3]), lo = hex(t[4]);
		return (hi < 0 || lo < 0) ? -1 : hi * 16 + lo;
	}

	// Repeatedly merge the highest-priority adjacent pair of symbols until
	// no pair qualifies; ties go to the leftmost pair. `priority(a, b, p)`
	// returns false when a + b may not merge. A priority queue with stale-
	// entry checks keeps this O(n log n) rather than rescanning per merge.
	template <typename PriorityFn>
	void merge_symbols(std::vector<std::string>& sym, PriorityFn priority)
	{
		const int n = static_cast<int>(sym.size());
		if (n < 2) return;
		std::vector<int> prev(n), next(n), rev(n, 0);
		for (int i = 0; i < n; ++i)
		{
			prev[i] = i - 1;
			next[i] = i + 1;
		}
		next[n - 1] = -1;

		struct cand
		{
			double priority;
			int left;
			int rev_left;
			int rev_right;
		};
		auto lower = [](const cand& a, const cand& b)
		{
			if (a.priority != b.priority) return a.priority < b.priority;
			return a.left > b.left;
		};
		std::priority_queue<cand, std::vector<cand>, decltype(lower)> pq(lower);

		auto push = [&](const int left)
		{
			if (left < 0) return;
			const int right = next[left];
			if (right < 0) return;
			double p = 0;
			if (!priority(sym[left], sym[right], p)) return;
			pq.push({p, left, rev[left], rev[right]});
		};
		for (int i = 0; i + 1 < n; ++i) push(i);

		while (!pq.empty())
		{
			const cand c = pq.top();
			pq.pop();
			const int right = next[c.left];
			if (rev[c.left] != c.rev_left || right < 0 || rev[right] != c.rev_right) continue; // stale

			sym[c.left] += sym[right];
			sym[right].clear();
			rev[c.left]++;
			rev[right] = -1;
			const int rr = next[right];
			next[c.left] = rr;
			if (rr >= 0) prev[rr] = c.left;
			push(prev[c.left]);
			push(c.left);
		}

		// Symbol 0 is never merged away (nothing lies to its left).
		std::vector<std::string> out;
		for (int i = 0; i >= 0; i = next[i]) out.push_back(std::move(sym[i]));
		sym.swap(out);
	}

	// Map final symbols to ids: whole pieces where the vocabulary has them,
	// otherwise one <0xHH> token per byte, otherwise <unk>.
	void emit_symbols(const model_vocab& v, const std::vector<std::string>& sym, std::vector<int>& out)
	{
		for (const auto& s : sym)
		{
			const auto it = v.token_to_id.find(s);
			if (it != v.token_to_id.end())
			{
				out.push_back(it->second);
				continue;
			}
			for (const unsigned char c : s)
			{
				if (v.byte_ids[c] >= 0) out.push_back(v.byte_ids[c]);
				else if (v.unk_id >= 0) out.push_back(v.unk_id);
			}
		}
	}

	// ---- llama-bpe pre-tokenizer ------------------------------------------
	//
	// Byte-level BPE vocabularies are trained on text pre-split by a regex,
	// and their merges never cross those boundaries. This reproduces the
	// llama-bpe split without a regex engine:
	//
	//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,3}
	//   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
	//
	// tried in that order at each position. Qwen2's regex has \p{N} in place
	// of \p{N}{1,3}, so every digit is a piece of its own; Qwen3.5's also
	// counts combining marks (\p{M}) as letters, where the others count them
	// as punctuation. Letters and digits are exact for ASCII. Beyond it,
	// codepoints are letters unless they are whitespace, or digits, marks,
	// punctuation or symbols in the blocks ordinary text draws on (k_cp_ranges
	// and the Indic layout in classify).

	enum cp_class : uint8_t
	{
		CP_LETTER,
		CP_DIGIT,
		CP_SPACE,
		CP_NEWLINE,
		CP_OTHER,
		CP_MARK, // combining mark: becomes CP_LETTER or CP_OTHER, per pre-tokenizer
	};

	struct cp_range
	{
		uint32_t lo, hi;
		cp_class cls;
	};

	// Codepoints past ASCII whose class is not the default, letter. Sorted
	// and disjoint, for a binary search.
	constexpr cp_range k_cp_ranges[] = {
		{0x00A1, 0x00A9, CP_OTHER}, {0x00AB, 0x00B1, CP_OTHER}, {0x00B2, 0x00B3, CP_DIGIT},
		{0x00B4, 0x00B4, CP_OTHER}, {0x00B6, 0x00B8, CP_OTHER}, {0x00B9, 0x00B9, CP_DIGIT},
		{0x00BB, 0x00BB, CP_OTHER}, {0x00BC, 0x00BE, CP_DIGIT}, {0x00BF, 0x00BF, CP_OTHER},
		{0x00D7, 0x00D7, CP_OTHER}, {0x00F7, 0x00F7, CP_OTHER}, {0x0300, 0x036F, CP_MARK},
		{0x0483, 0x0489, CP_MARK}, {0x055A, 0x055F, CP_OTHER}, {0x0589, 0x058A, CP_OTHER},
		{0x0591, 0x05BD, CP_MARK}, {0x05BE, 0x05BE, CP_OTHER}, {0x05BF, 0x05BF, CP_MARK},
		{0x05C0, 0x05C0, CP_OTHER}, {0x05C1, 0x05C2, CP_MARK}, {0x05C3, 0x05C3, CP_OTHER},
		{0x05C4, 0x05C5, CP_MARK}, {0x05C6, 0x05C6, CP_OTHER}, {0x05C7, 0x05C7, CP_MARK},
		{0x05F3, 0x05F4, CP_OTHER}, {0x0600, 0x060F, CP_OTHER}, {0x0610, 0x061A, CP_MARK},
		{0x061B, 0x061F, CP_OTHER}, {0x064B, 0x065F, CP_MARK}, {0x0660, 0x0669, CP_DIGIT},
		{0x066A, 0x066D, CP_OTHER}, {0x0670, 0x0670, CP_MARK}, {0x06D4, 0x06D4, CP_OTHER},
		{0x06D6, 0x06DC, CP_MARK}, {0x06DD, 0x06DE, CP_OTHER}, {0x06DF, 0x06E4, CP_MARK},
		{0x06E7, 0x06E8, CP_MARK}, {0x06E9, 0x06E9, CP_OTHER}, {0x06EA, 0x06ED, CP_MARK},
		{0x06F0, 0x06F9, CP_DIGIT}, {0x06FD, 0x06FE, CP_OTHER}, {0x0E31, 0x0E31, CP_MARK},
		{0x0E34, 0x0E3A, CP_MARK}, {0x0E3F, 0x0E3F, CP_OTHER}, {0x0E47, 0x0E4E, CP_MARK},
		{0x0E4F, 0x0E4F, CP_OTHER}, {0x0E50, 0x0E59, CP_DIGIT}, {0x0E5A, 0x0E5B, CP_OTHER},
		{0x0EB1, 0x0EB1, CP_MARK}, {0x0EB4, 0x0EBC, CP_MARK}, {0x0EC8, 0x0ECE, CP_MARK},
		{0x0ED0, 0x0ED9, CP_DIGIT}, {0x0F20, 0x0F33, CP_DIGIT}, {0x1040, 0x1049, CP_DIGIT},
		{0x17E0, 0x17E9, CP_DIGIT}, {0x1810, 0x1819, CP_DIGIT}, {0x1AB0, 0x1AFF, CP_MARK},
		{0x1DC0, 0x1DFF, CP_MARK}, {0x200B, 0x205E, CP_OTHER}, {0x2060, 0x206F, CP_OTHER},
		{0x2070, 0x2070, CP_DIGIT}, {0x2074, 0x2079, CP_DIGIT}, {0x207A, 0x207E, CP_OTHER},
		{0x2080, 0x2089, CP_DIGIT}, {0x208A, 0x208E, CP_OTHER}, {0x20A0, 0x20C0, CP_OTHER},
		{0x20D0, 0x20F0, CP_MARK}, {0x2150, 0x2189, CP_DIGIT}, {0x218A, 0x218B, CP_OTHER},
		{0x2190, 0x245F, CP_OTHER}, {0x2460, 0x249B, CP_DIGIT}, {0x249C, 0x24E9, CP_OTHER},
		{0x24EA, 0x24FF, CP_DIGIT}, {0x2500, 0x2775, CP_OTHER}, {0x2776, 0x2793, CP_DIGIT},
		{0x2794, 0x2BFF, CP_OTHER}, {0x2E00, 0x2E7F, CP_OTHER}, {0x3001, 0x3004, CP_OTHER},
		{0x3007, 0x3007, CP_DIGIT}, {0x3008, 0x3020, CP_OTHER}, {0x3021, 0x3029, CP_DIGIT},
		{0x302A, 0x302F, CP_MARK}, {0x3030, 0x3030, CP_OTHER}, {0x3036, 0x3037, CP_OTHER},
		{0x3038, 0x303A, CP_DIGIT}, {0x303D, 0x303F, CP_OTHER}, {0x3099, 0x309A, CP_MARK},
		{0x309B, 0x309C, CP_OTHER}, {0x30A0, 0x30A0, CP_OTHER}, {0x30FB, 0x30FB, CP_OTHER},
		{0x3190, 0x3191, CP_OTHER}, {0x3192, 0x3195, CP_DIGIT}, {0x3196, 0x319F, CP_OTHER},
		{0x3200, 0x321E, CP_OTHER}, {0x3220, 0x3229, CP_DIGIT}, {0x322A, 0x3247, CP_OTHER},
		{0x3248, 0x324F, CP_DIGIT}, {0x3250, 0x3250, CP_OTHER}, {0x3251, 0x325F, CP_DIGIT},
		{0x3260, 0x327F, CP_OTHER}, {0x3280, 0x3289, CP_DIGIT}, {0x328A, 0x32B0, CP_OTHER},
		{0x32B1, 0x32BF, CP_DIGIT}, {0x32C0, 0x33FF, CP_OTHER}, {0xFE00, 0xFE0F, CP_MARK},
		{0xFE10, 0xFE19, CP_OTHER}, {0xFE20, 0xFE2F, CP_MARK}, {0xFE30, 0xFE6B, CP_OTHER},
		{0xFEFF, 0xFEFF, CP_OTHER}, {0xFF01, 0xFF0F, CP_OTHER}, {0xFF10, 0xFF19, CP_DIGIT},
		{0xFF1A, 0xFF20, CP_OTHER}, {0xFF3B, 0xFF40, CP_OTHER}, {0xFF5B, 0xFF65, CP_OTHER},
		{0xFFE0, 0xFFEE, CP_OTHER}, {0xFFF9, 0xFFFD, CP_OTHER}, {0x1D7CE, 0x1D7FF, CP_DIGIT},
		{0x1F000, 0x1F0FF, CP_OTHER}, {0x1F100, 0x1F10C, CP_DIGIT}, {0x1F10D, 0x1FBEF, CP_OTHER},
		{0x1FBF0, 0x1FBF9, CP_DIGIT}, {0xE0000, 0xE007F, CP_OTHER}, {0xE0100, 0xE01EF, CP_MARK},
	};

	constexpr bool ranges_sorted()
	{
		for (size_t i = 0; i < std::size(k_cp_ranges); ++i)
			if (k_cp_ranges[i].lo > k_cp_ranges[i].hi || (i > 0 && k_cp_ranges[i].lo <= k_cp_ranges[i - 1].hi))
				return false;
		return true;
	}

	static_assert(ranges_sorted(), "k_cp_ranges must be sorted and disjoint");

	cp_class classify(const uint32_t cp)
	{
		if (cp == '\n' || cp == '\r') return CP_NEWLINE;
		if (cp == ' ' || cp == '\t' || cp == '\v' || cp == '\f' || cp == 0x85 || cp == 0xA0
			|| (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029
			|| cp == 0x202F || cp == 0x205F || cp == 0x3000)
			return CP_SPACE;
		if (cp < 0x80)
		{
			if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return CP_LETTER;
			if (cp >= '0' && cp <= '9') return CP_DIGIT;
			return CP_OTHER;
		}
		// Devanagari to Malayalam share one layout per 128-codepoint block:
		// vowel signs, virama and other marks, then the digits.
		if (cp >= 0x0900 && cp < 0x0D80)
		{
			const uint32_t off = cp & 0x7F;
			if (off >= 0x66 && off <= 0x6F) return CP_DIGIT;
			if (off <= 0x03 || (off >= 0x3A && off <= 0x3C) || (off >= 0x3E && off <= 0x4F)
				|| (off >= 0x51 && off <= 0x57) || off == 0x62 || off == 0x63)
				return CP_MARK;
			return (cp == 0x0964 || cp == 0x0965 || cp == 0x0970) ? CP_OTHER : CP_LETTER;
		}
		const cp_range* r = std::upper_bound(std::begin(k_cp_ranges), std::end(k_cp_ranges), cp,
		                                     [](const uint32_t c, const cp_range& e) { return c < e.lo; });
		if (r != std::begin(k_cp_ranges) && cp <= (r - 1)->hi) return (r - 1)->cls;
		return CP_LETTER;
	}

	struct codepoint
	{
		uint32_t offset; // byte offset in the text
		uint32_t cp;
		cp_class cls;
	};

	std::vector<codepoint> decode_codepoints(const std::string& s)
	{
		std::vector<codepoint> out;
		out.reserve(s.size());
		size_t i = 0;
		while (i < s.size())
		{
			const auto c0 = static_cast<unsigned char>(s[i]);
			size_t n = static_cast<size_t>(utf8_len(c0));
			if (i + n > s.size()) n = 1;
			uint32_t cp = c0;
			if (n == 2) cp = ((c0 & 0x1Fu) << 6) | (s[i + 1] & 0x3Fu);
			else if (n == 3) cp = ((c0 & 0x0Fu) << 12) | ((s[i + 1] & 0x3Fu) << 6) | (s[i + 2] & 0x3Fu);
			else if (n == 4)
				cp = ((c0 & 0x07u) << 18) | ((s[i + 1] & 0x3Fu) << 12) | ((s[i + 2] & 0x3Fu) << 6)
					| (s[i + 3] & 0x3Fu);
			out.push_back({static_cast<uint32_t>(i), cp, (n == 1 && c0 >= 0x80) ? CP_OTHER : classify(cp)});
			i += n;
		}
		return out;
	}

	std::vector<std::string> pretokenize_bpe(const std::string& text, const size_t max_digits,
	                                         const bool marks_are_letters)
	{
		std::vector<codepoint> cps = decode_codepoints(text);
		for (codepoint& c : cps)
			if (c.cls == CP_MARK) c.cls = marks_are_letters ? CP_LETTER : CP_OTHER;
		const size_t n = cps.size();
		std::vector<std::string> chunks;
		auto cls = [&](const size_t k) { return cps[k].cls; };
		auto is_ws = [&](const size_t k) { return cps[k].cls == CP_SPACE || cps[k].cls == CP_NEWLINE; };
		auto lower = [&](const size_t k) -> uint32_t
		{
			const uint32_t c = cps[k].cp;
			return (c >= 'A' && c <= 'Z') ? c + 32 : c;
		};
		size_t i = 0;
		auto emit = [&](const size_t end)
		{
			const size_t a = cps[i].offset;
			const size_t b = (end < n) ? cps[end].offset : text.size();
			chunks.emplace_back(text, a, b - a);
			i = end;
		};

		while (i < n)
		{
			if (cps[i].cp == '\'' && i + 1 < n)
			{
				const uint32_t c1 = lower(i + 1);
				const uint32_t c2 = (i + 2 < n) ? lower(i + 2) : 0;
				if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'))
				{
					emit(i + 3);
					continue;
				}
				if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
				{
					emit(i + 2);
					continue;
				}
			}
			{
				size_t j = i;
				if (cls(j) != CP_LETTER && cls(j) != CP_DIGIT && cls(j) != CP_NEWLINE
					&& j + 1 < n && cls(j + 1) == CP_LETTER)
					++j;
				if (cls(j) == CP_LETTER)
				{
					size_t k = j;
					while (k < n && cls(k) == CP_LETTER) ++k;
					emit(k);
					continue;
				}
			}
			if (cls(i) == CP_DIGIT)
			{
				size_t k = i;
				while (k < n && k - i < max_digits && cls(k) == CP_DIGIT) ++k;
				emit(k);
				continue;
			}
			{
				size_t j = i;
				if (cps[j].cp == ' ' && j + 1 < n && cls(j + 1) == CP_OTHER) ++j;
				if (cls(j) == CP_OTHER)
				{
					size_t k = j;
					while (k < n && cls(k) == CP_OTHER) ++k;
					while (k < n && cls(k) == CP_NEWLINE) ++k;
					emit(k);
					continue;
				}
			}
			size_t k = i;
			size_t last_nl = n;
			while (k < n && is_ws(k))
			{
				if (cls(k) == CP_NEWLINE) last_nl = k;
				++k;
			}
			if (last_nl < n) emit(last_nl + 1); // \s*[\r\n]+
			else if (k > i && k < n && k - i >= 2) emit(k - 1); // \s+(?!\S): leave one space for the word
			else if (k > i) emit(k); // \s+
			else emit(i + 1);
		}
		return chunks;
	}

	std::string render_piece(const model_vocab& v, const int id)
	{
		const std::string& t = v.tokens[id];
		int32_t type = v.types.empty() ? VOCAB_TOKEN_NORMAL : v.types[id];
		if (v.types.empty())
		{
			if (id == v.bos_id || id == v.eos_id || id == v.pad_id || id == v.unk_id) type = VOCAB_TOKEN_CONTROL;
			else if (parse_byte_token(t) >= 0) type = VOCAB_TOKEN_BYTE;
		}
		switch (type)
		{
		case VOCAB_TOKEN_CONTROL:
		case VOCAB_TOKEN_UNUSED:
		case VOCAB_TOKEN_UNKNOWN:
			return {};
		case VOCAB_TOKEN_BYTE:
			{
				const int b = parse_byte_token(t);
				return b >= 0 ? std::string(1, static_cast<char>(b)) : std::string();
			}
		case VOCAB_TOKEN_USER_DEFINED:
			return v.kind == vocab_kind::gpt2 ? t : marks_to_spaces(t);
		default:
			return v.kind == vocab_kind::gpt2 ? unicode_to_bytes(t) : marks_to_spaces(t);
		}
	}

	bool kv_int(const gguf_file& f, const char* key, int& out)
	{
		const gguf_value* val = gguf_find_kv(f, key);
		if (!val) return false;
		switch (val->type)
		{
		case GGUF_TYPE_UINT8:
		case GGUF_TYPE_UINT16:
		case GGUF_TYPE_UINT32:
		case GGUF_TYPE_UINT64:
			out = static_cast<int>(val->u64);
			return true;
		case GGUF_TYPE_INT8:
		case GGUF_TYPE_INT16:
		case GGUF_TYPE_INT32:
		case GGUF_TYPE_INT64:
			out = static_cast<int>(val->i64);
			return true;
		default: return false;
		}
	}

	bool kv_bool(const gguf_file& f, const char* key, bool& out)
	{
		const gguf_value* val = gguf_find_kv(f, key);
		if (!val || val->type != GGUF_TYPE_BOOL) return false;
		out = val->u64 != 0;
		return true;
	}

	// TranslateGemma's chat template opens with a Jinja dictionary of the
	// languages it knows: {%- set languages = {"en": "English", ...} -%}.
	void parse_languages(const std::string& tpl, std::unordered_map<std::string, std::string>& out)
	{
		const size_t at = tpl.find("set languages");
		const size_t open = at == std::string::npos ? at : tpl.find('{', at);
		const size_t close = open == std::string::npos ? open : tpl.find('}', open);
		if (close == std::string::npos) return;
		std::string key;
		bool have_key = false;
		for (size_t i = open + 1; i < close; ++i)
		{
			if (tpl[i] != '"') continue;
			const size_t end = tpl.find('"', i + 1);
			if (end == std::string::npos || end > close) break;
			std::string s = tpl.substr(i + 1, end - i - 1);
			if (have_key) out.emplace(std::move(key), std::move(s));
			else key = std::move(s);
			have_key = !have_key;
			i = end;
		}
	}

	// A Jinja string literal's text: "\n" (two characters in the template)
	// becomes a newline.
	std::string jinja_unescape(const std::string& s)
	{
		std::string out;
		for (size_t i = 0; i < s.size(); ++i)
		{
			if (s[i] == '\\' && i + 1 < s.size() && s[i + 1] == 'n')
			{
				out.push_back('\n');
				++i;
			}
			else
			{
				out.push_back(s[i]);
			}
		}
		return out;
	}

	// The instruction a template with a reasoning_effort setting (Qwen3.8)
	// puts at the top of the system turn when thinking is on, for its default
	// effort:
	//   reasoning_effort|default('xhigh') ...
	//   {%- if resolved_reasoning_effort == 'xhigh' %}
	//       {%- set reasoning_instructions = '...' %}
	// Empty when the template has none.
	std::string chatml_think_instructions(const std::string& tpl)
	{
		const std::string def = "reasoning_effort|default('";
		const size_t d = tpl.find(def);
		if (d == std::string::npos) return {};
		const size_t name = d + def.size();
		const size_t name_end = tpl.find('\'', name);
		if (name_end == std::string::npos) return {};
		const std::string test = "resolved_reasoning_effort == '" + tpl.substr(name, name_end - name) + "'";
		const size_t branch = tpl.find(test, name_end);
		if (branch == std::string::npos) return {};
		const std::string set = "set reasoning_instructions = '";
		const size_t at = tpl.find(set, branch);
		// The literal must belong to this branch, not to a later effort's.
		if (at == std::string::npos || tpl.find("resolved_reasoning_effort ==", branch + 1) < at) return {};
		const size_t text = at + set.size();
		const size_t end = tpl.find('\'', text);
		if (end == std::string::npos) return {};
		return jinja_unescape(tpl.substr(text, end - text));
	}

	// What a ChatML template writes on its own: the system prompt it falls
	// back to - Qwen2.5's '<|im_start|>system\nYou are Qwen, ...<|im_end|>\n',
	// a literal with nothing spliced in - and how its generation prompt opens
	// the reply. Templates with an enable_thinking switch write an empty
	// <think> block to turn thinking off, which nnc does unless asked to let
	// the model think, as it does Gemma 4's empty thought channel; with
	// thinking on, Qwen3.5 and 3.8 write "<think>\n" and Qwen3 nothing. A
	// model that always thinks has its reply opened with "<think>\n".
	void parse_chatml_template(const std::string& tpl, model_vocab& v)
	{
		const std::string open = "'<|im_start|>system\\n";
		const std::string close = "<|im_end|>\\n'";
		for (size_t at = tpl.find(open); at != std::string::npos; at = tpl.find(open, at + 1))
		{
			const size_t text = at + open.size();
			const size_t end = tpl.find(close, text);
			if (end == std::string::npos) break;
			const std::string body = tpl.substr(text, end - text);
			if (!body.empty() && body.find_first_of("'\"{}+") == std::string::npos)
			{
				v.default_system = jinja_unescape(body);
				break;
			}
		}
		const size_t gen = tpl.rfind("add_generation_prompt");
		if (gen == std::string::npos) return;
		const std::string tail = tpl.substr(gen);
		const bool markers = v.tok_thought_open >= 0 && v.tok_thought_close >= 0;
		if (tail.find("<think>\\n\\n</think>\\n\\n") != std::string::npos)
		{
			v.reply_open = "<think>\n\n</think>\n\n";
			v.can_think = markers;
			if (tail.find("'<think>\\n'") != std::string::npos) v.reply_open_think = "<think>\n";
			v.think_instructions = chatml_think_instructions(tpl);
		}
		else if (tail.find("<|im_start|>assistant\\n<think>\\n") != std::string::npos)
		{
			v.reply_open = "<think>\n";
			v.always_thinks = markers;
		}
	}
}

bool vocab_load(const gguf_file& f, model_vocab& v)
{
	const gguf_value* tokens = gguf_find_kv(f, "tokenizer.ggml.tokens");
	if (!tokens || tokens->type != GGUF_TYPE_ARRAY || tokens->arr_type != GGUF_TYPE_STRING
		|| tokens->arr.empty())
		return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "the file has no tokenizer vocabulary");
	const size_t n = tokens->arr.size();
	if (n > static_cast<size_t>(INT32_MAX))
		return nnc_fail(NNC_ERR_BAD_FORMAT, "the vocabulary is implausibly large");

	v.tokens.resize(n);
	for (size_t i = 0; i < n; ++i) v.tokens[i] = tokens->arr[i].str;

	if (const gguf_value* sc = gguf_find_kv(f, "tokenizer.ggml.scores");
		sc && sc->type == GGUF_TYPE_ARRAY && sc->arr.size() == n)
	{
		v.scores.resize(n);
		for (size_t i = 0; i < n; ++i) v.scores[i] = static_cast<float>(sc->arr[i].f64);
	}
	if (const gguf_value* tt = gguf_find_kv(f, "tokenizer.ggml.token_type");
		tt && tt->type == GGUF_TYPE_ARRAY && tt->arr.size() == n)
	{
		v.types.resize(n);
		for (size_t i = 0; i < n; ++i) v.types[i] = static_cast<int32_t>(tt->arr[i].i64);
	}

	auto special = [&](const char* key, int& out)
	{
		int id = -1;
		if (kv_int(f, key, id) && id >= 0 && id < static_cast<int>(n)) out = id;
	};
	special("tokenizer.ggml.bos_token_id", v.bos_id);
	special("tokenizer.ggml.eos_token_id", v.eos_id);
	special("tokenizer.ggml.unknown_token_id", v.unk_id);
	special("tokenizer.ggml.padding_token_id", v.pad_id);

	if (const gguf_value* tm = gguf_find_kv(f, "tokenizer.ggml.model"); tm && tm->type == GGUF_TYPE_STRING)
		v.model = tm->str;

	if (const gguf_value* mg = gguf_find_kv(f, "tokenizer.ggml.merges");
		mg && mg->type == GGUF_TYPE_ARRAY && mg->arr_type == GGUF_TYPE_STRING)
	{
		v.merge_rank.reserve(mg->arr.size() * 2);
		for (size_t i = 0; i < mg->arr.size(); ++i)
			v.merge_rank.emplace(mg->arr[i].str, static_cast<int>(i));
	}

	if (v.model == "gpt2")
	{
		if (v.merge_rank.empty())
			return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "the byte-level BPE tokenizer has no merges");
		v.kind = vocab_kind::gpt2;
		if (const gguf_value* pre = gguf_find_kv(f, "tokenizer.ggml.pre"); pre && pre->type == GGUF_TYPE_STRING)
		{
			if (pre->str == "qwen2") v.pre = vocab_pre::qwen2;
			else if (pre->str == "qwen35") v.pre = vocab_pre::qwen35;
		}
	}
	else
	{
		v.kind = v.merge_rank.empty() ? vocab_kind::spm : vocab_kind::bpe;
	}

	// SentencePiece Llama models prepend a space unless the file says
	// otherwise; Gemma files always say.
	std::string arch;
	if (const gguf_value* a = gguf_find_kv(f, "general.architecture"); a && a->type == GGUF_TYPE_STRING)
		arch = a->str;
	v.add_space_prefix = (v.kind == vocab_kind::spm && arch == "llama");
	kv_bool(f, "tokenizer.ggml.add_space_prefix", v.add_space_prefix);
	kv_bool(f, "tokenizer.ggml.add_bos_token", v.add_bos);
	kv_bool(f, "tokenizer.ggml.add_eos_token", v.add_eos);

	v.token_to_id.reserve(n * 2);
	for (size_t i = 0; i < n; ++i) v.token_to_id.emplace(v.tokens[i], static_cast<int>(i));

	if (!v.types.empty())
	{
		for (size_t i = 0; i < n; ++i)
			if (v.types[i] == VOCAB_TOKEN_USER_DEFINED && !v.tokens[i].empty())
				v.user_defined.emplace_back(v.tokens[i], static_cast<int>(i));
		std::sort(v.user_defined.begin(), v.user_defined.end(), [](const auto& a, const auto& b)
		{
			const auto fa = static_cast<unsigned char>(a.first[0]), fb = static_cast<unsigned char>(b.first[0]);
			if (fa != fb) return fa < fb;
			if (a.first.size() != b.first.size()) return a.first.size() > b.first.size();
			return a.second < b.second;
		});
	}
	{
		size_t j = 0;
		for (unsigned b = 0; b < 256; ++b)
		{
			v.user_defined_at[b] = static_cast<uint32_t>(j);
			while (j < v.user_defined.size() && static_cast<unsigned char>(v.user_defined[j].first[0]) == b) ++j;
		}
		v.user_defined_at[256] = static_cast<uint32_t>(j);
	}

	std::fill(std::begin(v.byte_ids), std::end(v.byte_ids), -1);
	for (size_t i = 0; i < n; ++i)
	{
		const int b = parse_byte_token(v.tokens[i]);
		if (b >= 0 && (v.types.empty() || v.types[i] == VOCAB_TOKEN_BYTE) && v.byte_ids[b] < 0)
			v.byte_ids[b] = static_cast<int>(i);
	}

	v.pieces.resize(n);
	for (size_t i = 0; i < n; ++i) v.pieces[i] = render_piece(v, static_cast<int>(i));

	auto find = [&](const std::initializer_list<const char*> names) -> int
	{
		for (const char* name : names)
		{
			const auto it = v.token_to_id.find(name);
			if (it != v.token_to_id.end()) return it->second;
		}
		return -1;
	};

	// Chat turn format, recognised by its control tokens. Gemma 4 spells
	// Gemma's turn markers "<|turn>" and "<turn|>" and adds a system role
	// and a thought channel.
	const int sot = find({"<start_of_turn>", "<|turn>"});
	const int eot = find({"<end_of_turn>", "<turn|>"});
	const int nl = find({"\n"});
	const int sh = find({"<|start_header_id|>"});
	const int eh = find({"<|end_header_id|>"});
	const int eot_id = find({"<|eot_id|>"});
	const int im_start = find({"<|im_start|>"});
	const int im_end = find({"<|im_end|>"});
	const gguf_value* tpl = gguf_find_kv(f, "tokenizer.chat_template");
	const bool have_tpl = tpl && tpl->type == GGUF_TYPE_STRING;
	if (sot >= 0 && eot >= 0 && nl >= 0)
	{
		v.chat = v.tokens[sot] == "<|turn>" ? chat_format::gemma4 : chat_format::gemma;
		v.tok_turn_start = sot;
		v.tok_turn_end = eot;
		v.tok_newline = nl;
		v.tok_channel_start = find({"<|channel>"});
		v.tok_channel_end = find({"<channel|>"});
		v.developer_turn = v.chat == chat_format::gemma && find({"<start_function_declaration>"}) >= 0;
		// The template is Jinja source, so its newline is the two characters "\n".
		v.empty_thought = v.tok_channel_start >= 0 && v.tok_channel_end >= 0 && have_tpl
			&& tpl->str.find("<|channel>thought\\n<channel|>") != std::string::npos;
		if (v.chat == chat_format::gemma4 && v.tok_channel_start >= 0 && v.tok_channel_end >= 0)
		{
			v.tok_thought_open = v.tok_channel_start;
			v.tok_thought_close = v.tok_channel_end;
			v.thought_label = true;
			v.tok_think = find({"<|think|>"});
			v.can_think = v.tok_think >= 0 && have_tpl && tpl->str.find("<|think|>") != std::string::npos;
		}
		if (have_tpl && tpl->str.find("source_lang_code") != std::string::npos) parse_languages(tpl->str, v.languages);
	}
	else if (sh >= 0 && eh >= 0 && eot_id >= 0)
	{
		v.chat = chat_format::llama3;
		v.tok_turn_start = sh;
		v.tok_header_end = eh;
		v.tok_turn_end = eot_id;
	}
	else if (im_start >= 0 && im_end >= 0)
	{
		v.chat = chat_format::chatml;
		v.tok_turn_start = im_start;
		v.tok_turn_end = im_end;
		const int think_open = find({"<think>"});
		const int think_close = find({"</think>"});
		if (think_open >= 0 && think_close >= 0)
		{
			v.tok_thought_open = think_open;
			v.tok_thought_close = think_close;
		}
		if (have_tpl) parse_chatml_template(tpl->str, v);
	}

	// End of generation: EOS, the turn terminator, and the other common
	// end markers when the file marks them as control tokens.
	auto add_eog = [&](const int id)
	{
		if (id >= 0 && std::find(v.eog_ids.begin(), v.eog_ids.end(), id) == v.eog_ids.end())
			v.eog_ids.push_back(id);
	};
	add_eog(v.eos_id);
	add_eog(v.tok_turn_end);
	for (const char* name : {"<end_of_turn>", "<turn|>", "<|eot_id|>", "<|eom_id|>", "<|end_of_text|>",
	                         "<|endoftext|>", "<|im_end|>", "<eos>", "</s>"})
	{
		const auto it = v.token_to_id.find(name);
		if (it == v.token_to_id.end()) continue;
		if (v.types.empty() || v.types[it->second] == VOCAB_TOKEN_CONTROL) add_eog(it->second);
	}
	// FunctionGemma ends its turn early after a function call, where the
	// caller is to supply the function's response.
	if (v.developer_turn) add_eog(find({"<start_function_response>"}));
	return true;
}

namespace
{
	// Encode text that holds no user-defined token. `space_prefix` is the
	// SentencePiece leading space, which llama.cpp gives the start of the
	// text and the text after each user-defined token.
	void encode_fragment(const model_vocab& v, const std::string& text, const bool space_prefix,
	                     std::vector<int>& out)
	{
		std::vector<std::string> sym;
		if (v.kind == vocab_kind::gpt2)
		{
			const auto& tab = byte_to_unicode();
			auto by_rank = [&](const std::string& a, const std::string& b, double& p)
			{
				std::string key;
				key.reserve(a.size() + 1 + b.size());
				key.append(a).append(1, ' ').append(b);
				const auto it = v.merge_rank.find(key);
				if (it == v.merge_rank.end()) return false;
				p = -static_cast<double>(it->second);
				return true;
			};
			for (const std::string& chunk : vocab_pretokenize(v, text))
			{
				sym.clear();
				for (const unsigned char c : chunk) sym.push_back(tab[c]);
				merge_symbols(sym, by_rank);
				emit_symbols(v, sym, out);
			}
			return;
		}

		const std::string marked = spaces_to_marks(space_prefix ? " " + text : text);
		split_utf8(marked, sym);
		if (v.kind == vocab_kind::spm)
		{
			// SentencePiece: merge the pair whose result is the highest-scoring
			// vocabulary entry.
			merge_symbols(sym, [&](const std::string& a, const std::string& b, double& p)
			{
				const auto it = v.token_to_id.find(a + b);
				if (it == v.token_to_id.end()) return false;
				p = v.scores.empty() ? 0.0 : static_cast<double>(v.scores[it->second]);
				return true;
			});
		}
		else
		{
			merge_symbols(sym, [&](const std::string& a, const std::string& b, double& p)
			{
				std::string key;
				key.reserve(a.size() + 1 + b.size());
				key.append(a).append(1, ' ').append(b);
				const auto it = v.merge_rank.find(key);
				if (it == v.merge_rank.end()) return false;
				p = -static_cast<double>(it->second);
				return true;
			});
		}
		emit_symbols(v, sym, out);
	}
}

std::vector<int> vocab_tokenize(const model_vocab& v, const std::string& text, const bool add_bos)
{
	std::vector<int> out;
	if (add_bos && v.bos_id >= 0) out.push_back(v.bos_id);

	// Split the text at user-defined tokens, leftmost and then longest
	// first; the pieces between them are encoded on their own (each with
	// the leading space, if the vocabulary adds one).
	size_t start = 0;
	for (size_t i = 0; i < text.size();)
	{
		const auto b = static_cast<unsigned char>(text[i]);
		int id = -1;
		size_t len = 0;
		for (uint32_t c = v.user_defined_at[b]; c < v.user_defined_at[b + 1]; ++c)
		{
			const std::string& s = v.user_defined[c].first;
			if (text.compare(i, s.size(), s) == 0)
			{
				id = v.user_defined[c].second;
				len = s.size();
				break;
			}
		}
		if (id < 0)
		{
			++i;
			continue;
		}
		if (i > start) encode_fragment(v, text.substr(start, i - start), v.add_space_prefix, out);
		out.push_back(id);
		i += len;
		start = i;
	}
	if (start < text.size()) encode_fragment(v, text.substr(start), v.add_space_prefix, out);
	return out;
}

const std::string& vocab_piece(const model_vocab& v, const int token)
{
	static const std::string empty;
	return v.valid(token) ? v.pieces[token] : empty;
}

std::vector<std::string> vocab_pretokenize(const model_vocab& v, const std::string& text)
{
	return pretokenize_bpe(text, v.pre == vocab_pre::llama3 ? 3 : 1, v.pre == vocab_pre::qwen35);
}

std::string vocab_detokenize(const model_vocab& v, const int* tokens, const size_t n)
{
	std::string out;
	for (size_t i = 0; i < n; ++i) out += vocab_piece(v, tokens[i]);
	return out;
}

bool vocab_is_eog(const model_vocab& v, const int token)
{
	return token >= 0 && std::find(v.eog_ids.begin(), v.eog_ids.end(), token) != v.eog_ids.end();
}
