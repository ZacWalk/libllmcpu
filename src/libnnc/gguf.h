// nnc — minimal GGUF v2/v3 reader.
//
// Maps the file and parses the header, every metadata key/value pair and
// the tensor descriptor table in place. Tensor data is never copied:
// gguf_tensor_data() returns pointers into the mapping.
//
// Spec reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

#pragma once

#include "sys.h"

#include <cstdint>
#include <string>
#include <vector>

enum gguf_value_type : uint32_t
{
	GGUF_TYPE_UINT8 = 0,
	GGUF_TYPE_INT8 = 1,
	GGUF_TYPE_UINT16 = 2,
	GGUF_TYPE_INT16 = 3,
	GGUF_TYPE_UINT32 = 4,
	GGUF_TYPE_INT32 = 5,
	GGUF_TYPE_FLOAT32 = 6,
	GGUF_TYPE_BOOL = 7,
	GGUF_TYPE_STRING = 8,
	GGUF_TYPE_ARRAY = 9,
	GGUF_TYPE_UINT64 = 10,
	GGUF_TYPE_INT64 = 11,
	GGUF_TYPE_FLOAT64 = 12,
};

struct gguf_value
{
	gguf_value_type type;

	// scalar storage (whichever matches `type`)
	uint64_t u64; // also used for u8/u16/u32/bool
	int64_t i64; // also used for i8/i16/i32
	double f64; // also used for f32
	std::string str;

	// array storage (when type == GGUF_TYPE_ARRAY)
	gguf_value_type arr_type;
	std::vector<gguf_value> arr;
};

struct gguf_kv
{
	std::string key;
	gguf_value value;
};

struct gguf_tensor_info
{
	std::string name;
	uint32_t n_dims;
	uint64_t ne[4]; // GGUF caps at 4 dims
	uint32_t ggml_type;
	uint64_t offset; // offset within the tensor data block
};

struct gguf_file
{
	uint32_t version = 0;
	uint64_t tensor_count = 0;
	uint64_t kv_count = 0;
	uint64_t alignment = 32; // from general.alignment, default 32
	uint64_t data_offset = 0; // absolute file offset of the tensor data block
	std::vector<gguf_kv> kv;
	std::vector<gguf_tensor_info> tensors;

	// Set by gguf_mmap(); zero/null after gguf_load() or gguf_unmap().
	sys_mmap mapping{};
	const uint8_t* mapped_base = nullptr; // == mapping.base, kept for convenience
	uint64_t mapped_size = 0; // == mapping.size
};

// Map `path` (UTF-8) read-only and parse the header into `out`. The mapping
// stays valid until gguf_unmap(); gguf_tensor_data() points into it. On
// failure records the reason with nnc_fail() and returns false.
bool gguf_mmap(const std::string& path, gguf_file& out);
void gguf_unmap(gguf_file& out);

// Parse the header, metadata and tensor descriptors only; the file is not
// left mapped, so gguf_tensor_data() returns nullptr afterwards.
bool gguf_load(const std::string& path, gguf_file& out);

// Ask the OS to start reading the whole mapped file into memory.
void gguf_prefetch(const gguf_file& f);

// Pointer to the start of tensor `i`'s raw bytes within the mapped data
// block, or nullptr when the file is not mapped or the tensor's bytes do
// not lie inside it.
const void* gguf_tensor_data(const gguf_file& f, size_t i);

// Find a tensor by exact name. Returns SIZE_MAX if not found.
size_t gguf_find_tensor(const gguf_file& f, const std::string& name);

// Total element count for a tensor (product of all `ne`). Uses uint64
// since some Gemma tables (e.g. per_layer_token_embd) exceed 2^31.
uint64_t gguf_tensor_nelements(const gguf_tensor_info& t);

// Storage size in bytes (post-block-quantization aware for Q* types);
// 0 for types this build does not know.
uint64_t gguf_tensor_nbytes(const gguf_tensor_info& t);

// Human-readable name for a GGML tensor type code. Returns "?(<n>)" for
// values we don't recognise.
const char* gguf_ggml_type_name(uint32_t t);

// Find a metadata KV by exact key. Returns nullptr if not present.
const gguf_value* gguf_find_kv(const gguf_file& f, const std::string& key);