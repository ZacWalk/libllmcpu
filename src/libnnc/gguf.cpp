// nnc — minimal GGUF v2/v3 reader. See gguf.h for the format reference.
//
// The file is memory-mapped and parsed in place: no stream, no copy of the
// tensor data. Every length and count read from the file is checked
// against the bytes that remain, and the parsed metadata against a memory
// budget, so a corrupt or hostile file is refused with a message rather
// than allowed to read past the mapping or allocate far more than its own
// size.

#include "gguf.h"

#include "runtime.h"

#include <cinttypes>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace
{
	constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" little-endian

	// Bound the per-allocation sizes we accept from the file. Anything
	// larger almost certainly means a corrupt or hostile input. The
	// limits are generous enough for real models (256k-vocab string
	// arrays etc.).
	constexpr uint64_t MAX_STRING_LEN = 1ull << 24; // 16 MiB
	constexpr uint64_t MAX_ARRAY_LEN = 1ull << 28; // 256 M elements
	constexpr uint64_t MAX_TENSOR_COUNT = 1ull << 20; // 1 M tensors
	constexpr uint64_t MAX_KV_COUNT = 1ull << 20;
	// Real files nest at most one level (an array of strings or numbers).
	// Each level costs 12 bytes of file but a recursion frame, so without a
	// bound a megabyte of nested array headers overflows the stack.
	constexpr int MAX_ARRAY_DEPTH = 8;
	// A metadata value takes sizeof(gguf_value) of memory however few bytes
	// it has in the file, so the values are budgeted too: together they may
	// take about as much memory as the file is long, or 256 MiB for a small
	// file. Real vocabularies (a few hundred thousand entries) fit easily.
	constexpr uint64_t MIN_METADATA_BUDGET = 256ull << 20;

	struct reader
	{
		const uint8_t* p; // read position
		const uint8_t* end;
		const std::string& path;
		int depth = 0; // arrays currently being read
		uint64_t values_left = 0; // metadata values still within budget
		bool ok = true;

		// Bytes left between the read position and the end of the file.
		uint64_t remaining() const { return static_cast<uint64_t>(end - p); }

		bool fail(const char* what)
		{
			if (ok) nnc_fail(NNC_ERR_BAD_FORMAT, "'%s' is not a valid GGUF file: %s", path.c_str(), what);
			ok = false;
			return false;
		}

		bool read_raw(void* dst, const size_t n)
		{
			if (!ok) return false;
			if (n > remaining()) return fail("it ends early (truncated?)");
			std::memcpy(dst, p, n);
			p += n;
			return true;
		}

		template <typename T>
		bool read_pod(T& v)
		{
			return read_raw(&v, sizeof(T));
		}

		bool read_string(std::string& out)
		{
			uint64_t len = 0;
			if (!read_pod(len)) return false;
			if (len > MAX_STRING_LEN) return fail("a string is implausibly long");
			if (len > remaining()) return fail("it ends early (truncated?)");
			out.assign(reinterpret_cast<const char*>(p), static_cast<size_t>(len));
			p += len;
			return true;
		}
	};

	bool read_value(reader& r, gguf_value_type type, gguf_value& out);

	// The fewest bytes one value of `type` occupies in the file.
	uint64_t min_encoded_size(const gguf_value_type type)
	{
		switch (type)
		{
		case GGUF_TYPE_UINT16:
		case GGUF_TYPE_INT16: return 2;
		case GGUF_TYPE_UINT32:
		case GGUF_TYPE_INT32:
		case GGUF_TYPE_FLOAT32: return 4;
		case GGUF_TYPE_UINT64:
		case GGUF_TYPE_INT64:
		case GGUF_TYPE_FLOAT64:
		case GGUF_TYPE_STRING: return 8; // the length prefix
		case GGUF_TYPE_ARRAY: return 12; // element type + length
		default: return 1;
		}
	}

	template <typename T>
	bool read_unsigned(reader& r, gguf_value& out)
	{
		T v = 0;
		if (!r.read_pod(v)) return false;
		out.u64 = v;
		return true;
	}

	template <typename T>
	bool read_signed(reader& r, gguf_value& out)
	{
		T v = 0;
		if (!r.read_pod(v)) return false;
		out.i64 = v;
		return true;
	}

	bool read_scalar(reader& r, const gguf_value_type type, gguf_value& out)
	{
		out.type = type;
		switch (type)
		{
		case GGUF_TYPE_UINT8: return read_unsigned<uint8_t>(r, out);
		case GGUF_TYPE_INT8: return read_signed<int8_t>(r, out);
		case GGUF_TYPE_UINT16: return read_unsigned<uint16_t>(r, out);
		case GGUF_TYPE_INT16: return read_signed<int16_t>(r, out);
		case GGUF_TYPE_UINT32: return read_unsigned<uint32_t>(r, out);
		case GGUF_TYPE_INT32: return read_signed<int32_t>(r, out);
		case GGUF_TYPE_UINT64: return read_unsigned<uint64_t>(r, out);
		case GGUF_TYPE_INT64: return read_signed<int64_t>(r, out);
		case GGUF_TYPE_FLOAT32:
			{
				float v = 0;
				if (!r.read_pod(v)) return false;
				out.f64 = v;
				return true;
			}
		case GGUF_TYPE_FLOAT64:
			{
				double v = 0;
				if (!r.read_pod(v)) return false;
				out.f64 = v;
				return true;
			}
		case GGUF_TYPE_BOOL:
			{
				uint8_t v = 0;
				if (!r.read_pod(v)) return false;
				out.u64 = (v != 0) ? 1 : 0;
				return true;
			}
		case GGUF_TYPE_STRING:
			return r.read_string(out.str);
		default:
			return r.fail("a metadata value has an unknown type");
		}
	}

	bool read_array(reader& r, gguf_value& out)
	{
		out.type = GGUF_TYPE_ARRAY;

		uint32_t arr_type_u32 = 0;
		uint64_t len = 0;
		if (!r.read_pod(arr_type_u32) || !r.read_pod(len))
			return false;
		out.arr_type = static_cast<gguf_value_type>(arr_type_u32);
		// Every element takes at least this many bytes of the file, so a
		// length the rest of the file cannot hold is corrupt or hostile.
		// Checking first keeps a few bytes from claiming gigabytes.
		const uint64_t min_bytes = min_encoded_size(out.arr_type);
		if (len > MAX_ARRAY_LEN || len > r.remaining() / min_bytes)
			return r.fail("an array is longer than the file can hold");
		if (len > r.values_left)
			return r.fail("its metadata would take implausibly much memory");

		// Grow as elements are actually read rather than trusting `len`
		// up front: each value is far larger in memory than on disk.
		out.arr.clear();
		out.arr.reserve(static_cast<size_t>(len < 65536 ? len : 65536));
		for (uint64_t i = 0; i < len; ++i)
		{
			gguf_value v{};
			if (!read_value(r, out.arr_type, v))
				return false;
			out.arr.push_back(std::move(v));
		}
		return true;
	}

	bool read_value(reader& r, const gguf_value_type type, gguf_value& out)
	{
		if (r.values_left == 0)
			return r.fail("its metadata would take implausibly much memory");
		--r.values_left;
		if (type == GGUF_TYPE_ARRAY)
		{
			if (r.depth >= MAX_ARRAY_DEPTH)
				return r.fail("arrays are nested too deeply");
			++r.depth;
			const bool ok = read_array(r, out);
			--r.depth;
			return ok;
		}

		return read_scalar(r, type, out);
	}

	// Parse the header, metadata and tensor table from the mapped bytes.
	bool parse(const uint8_t* data, const uint64_t size, const std::string& path, gguf_file& out)
	{
		reader r{data, data + size, path};
		r.values_left = (size > MIN_METADATA_BUDGET ? size : MIN_METADATA_BUDGET) / sizeof(gguf_value);

		uint32_t magic = 0;
		if (!r.read_pod(magic)) return false;
		if (magic != GGUF_MAGIC)
			return nnc_fail(NNC_ERR_BAD_FORMAT, "'%s' is not a GGUF file", path.c_str());

		if (!r.read_pod(out.version)) return false;
		if (out.version != 2 && out.version != 3)
			return nnc_fail(NNC_ERR_UNSUPPORTED_MODEL, "'%s' is GGUF version %u; nnc reads versions 2 and 3",
			                path.c_str(), out.version);

		if (!r.read_pod(out.tensor_count) || !r.read_pod(out.kv_count))
			return false;
		// Besides the hard caps, each KV needs at least 13 bytes (key length,
		// value type, a 1-byte value) and each tensor descriptor 32, so counts
		// the rest of the file cannot hold are refused before anything is sized.
		if (out.tensor_count > MAX_TENSOR_COUNT || out.kv_count > MAX_KV_COUNT
			|| out.kv_count * 13 + out.tensor_count * 32 > r.remaining())
			return r.fail("its tensor and metadata counts do not fit in the file");

		out.kv.resize(out.kv_count);
		for (auto& kv : out.kv)
		{
			uint32_t type_u32 = 0;
			if (!r.read_string(kv.key) || !r.read_pod(type_u32))
				return false;
			if (!read_value(r, static_cast<gguf_value_type>(type_u32), kv.value))
				return false;
		}

		out.alignment = 32;
		for (const auto& kv : out.kv)
		{
			if (kv.key == "general.alignment" && kv.value.type == GGUF_TYPE_UINT32)
			{
				out.alignment = kv.value.u64 ? kv.value.u64 : 32;
				break;
			}
		}

		out.tensors.resize(out.tensor_count);
		for (auto& t : out.tensors)
		{
			if (!r.read_string(t.name) || !r.read_pod(t.n_dims))
				return false;
			if (t.n_dims == 0 || t.n_dims > 4)
				return r.fail("a tensor has an invalid number of dimensions");
			t.ne[0] = t.ne[1] = t.ne[2] = t.ne[3] = 1;
			for (uint32_t d = 0; d < t.n_dims; ++d)
			{
				if (!r.read_pod(t.ne[d]))
					return false;
			}
			// nnc keeps tensor dimensions as int, and byte counts are derived
			// from the element count, so a dimension beyond INT_MAX or a count
			// whose byte size (up to 8 bytes an element) would wrap lets a small
			// file describe a huge tensor.
			constexpr uint64_t max_elements = static_cast<uint64_t>(INT64_MAX) / 8;
			uint64_t elements = 1;
			for (uint32_t d = 0; d < t.n_dims; ++d)
			{
				if (t.ne[d] > static_cast<uint64_t>(INT32_MAX) ||
				    (t.ne[d] != 0 && elements > max_elements / t.ne[d]))
					return r.fail("a tensor has an impossible shape");
				elements *= t.ne[d];
			}
			if (!r.read_pod(t.ggml_type) || !r.read_pod(t.offset))
				return false;
		}

		// Align to `alignment` past the descriptor table; that absolute
		// offset is where the tensor data block starts.
		const uint64_t pos = static_cast<uint64_t>(r.p - data);
		const uint64_t align = out.alignment;
		out.data_offset = (pos + align - 1) / align * align;
		return true;
	}

	// Block layout for the ggml types we know about.
	// {bytes per block, elements per block}; {0, 0} = unknown to this build.
	struct ggml_block_info
	{
		uint32_t bytes;
		uint32_t elems;
	};

	ggml_block_info ggml_block_for(const uint32_t t)
	{
		switch (t)
		{
		case 0: return {4, 1}; // F32
		case 1: return {2, 1}; // F16
		case 2: return {18, 32}; // Q4_0   (2 + 16)
		case 3: return {20, 32}; // Q4_1   (2+2+16)
		case 6: return {22, 32}; // Q5_0
		case 7: return {24, 32}; // Q5_1
		case 8: return {34, 32}; // Q8_0
		case 9: return {40, 32}; // Q8_1
		// K-quants (block size = 256 elements). Sizes match ggml's
		// `block_q*_k` structs; Q4_K/Q5_K/Q6_K are decoded at load, the
		// rest are used for byte accounting only.
		case 10: return {84, 256}; // Q2_K
		case 11: return {110, 256}; // Q3_K
		case 12: return {144, 256}; // Q4_K
		case 13: return {176, 256}; // Q5_K
		case 14: return {210, 256}; // Q6_K
		case 15: return {292, 256}; // Q8_K
		case 24: return {1, 1}; // I8
		case 25: return {2, 1}; // I16
		case 26: return {4, 1}; // I32
		case 27: return {8, 1}; // I64
		case 28: return {8, 1}; // F64
		case 30: return {2, 1}; // BF16
		default: return {0, 0};
		}
	}
}

const char* gguf_ggml_type_name(const uint32_t t)
{
	// Mirrors enum ggml_type as of llama.cpp / ggml master. Covers the
	// subset we are likely to see in modern model dumps.
	switch (t)
	{
	case 0: return "F32";
	case 1: return "F16";
	case 2: return "Q4_0";
	case 3: return "Q4_1";
	case 6: return "Q5_0";
	case 7: return "Q5_1";
	case 8: return "Q8_0";
	case 9: return "Q8_1";
	case 10: return "Q2_K";
	case 11: return "Q3_K";
	case 12: return "Q4_K";
	case 13: return "Q5_K";
	case 14: return "Q6_K";
	case 15: return "Q8_K";
	case 16: return "IQ2_XXS";
	case 17: return "IQ2_XS";
	case 18: return "IQ3_XXS";
	case 19: return "IQ1_S";
	case 20: return "IQ4_NL";
	case 21: return "IQ3_S";
	case 22: return "IQ2_S";
	case 23: return "IQ4_XS";
	case 24: return "I8";
	case 25: return "I16";
	case 26: return "I32";
	case 27: return "I64";
	case 28: return "F64";
	case 29: return "IQ1_M";
	case 30: return "BF16";
	case 31: return "Q4_0_4_4";
	case 32: return "Q4_0_4_8";
	case 33: return "Q4_0_8_8";
	case 34: return "TQ1_0";
	case 35: return "TQ2_0";
	}
	thread_local char buf[24];
	snprintf(buf, sizeof(buf), "?(%u)", t);
	return buf;
}

bool gguf_mmap(const std::string& path, gguf_file& out)
{
	out = gguf_file{};
	if (!sys_mmap_file_ro(path.c_str(), out.mapping)) return false;
	out.mapped_base = static_cast<const uint8_t*>(out.mapping.base);
	out.mapped_size = out.mapping.size;
	if (!parse(out.mapped_base, out.mapped_size, path, out))
	{
		gguf_unmap(out);
		return false;
	}
	return true;
}

bool gguf_load(const std::string& path, gguf_file& out)
{
	if (!gguf_mmap(path, out)) return false;
	gguf_unmap(out);
	return true;
}

void gguf_unmap(gguf_file& out)
{
	sys_munmap(out.mapping);
	out.mapped_base = nullptr;
	out.mapped_size = 0;
}

void gguf_prefetch(const gguf_file& f)
{
	if (f.mapped_base) sys_prefetch_ro(f.mapped_base, f.mapped_size);
}

uint64_t gguf_tensor_nelements(const gguf_tensor_info& t)
{
	uint64_t n = 1;
	for (uint32_t d = 0; d < t.n_dims; ++d) n *= t.ne[d];
	return n;
}

uint64_t gguf_tensor_nbytes(const gguf_tensor_info& t)
{
	const auto bi = ggml_block_for(t.ggml_type);
	if (bi.bytes == 0) return 0;
	const uint64_t n = gguf_tensor_nelements(t);
	return (n / bi.elems) * bi.bytes;
}

size_t gguf_find_tensor(const gguf_file& f, const std::string& name)
{
	for (size_t i = 0; i < f.tensors.size(); ++i)
		if (f.tensors[i].name == name) return i;
	return SIZE_MAX;
}

const gguf_value* gguf_find_kv(const gguf_file& f, const std::string& key)
{
	for (const auto& kv : f.kv)
		if (kv.key == key) return &kv.value;
	return nullptr;
}

const void* gguf_tensor_data(const gguf_file& f, const size_t i)
{
	if (!f.mapped_base || i >= f.tensors.size()) return nullptr;
	const uint64_t rel = f.tensors[i].offset;
	// Guard against overflow on data_offset + rel before bounds-checking
	// against mapped_size; a malformed file can otherwise wrap to a
	// small offset and pass the >= mapped_size test.
	if (rel > UINT64_MAX - f.data_offset) return nullptr;
	const uint64_t off = f.data_offset + rel;
	if (off >= f.mapped_size) return nullptr;
	// Validate the entire payload fits within the mapping. A malformed
	// GGUF can specify a small offset with a huge tensor and otherwise
	// hand back a pointer whose tail walks past the mmap.
	const uint64_t nbytes = gguf_tensor_nbytes(f.tensors[i]);
	if (nbytes > f.mapped_size - off) return nullptr;
	return f.mapped_base + off;
}