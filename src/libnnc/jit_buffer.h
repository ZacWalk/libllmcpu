// nnc — executable-memory allocator.
//
// Backed by a process-wide pool of code pages (jit_code_pool, internal).
// Each `commit()` copies the staged bytes into fresh pages, seals them
// read+execute and returns a pointer to the code; sealed pages are never
// written again. `jit_buffer` itself only owns the staging vector; pool
// pages are process-lifetime and never freed.

#pragma once

#include <cstdint>
#include <vector>

class jit_buffer
{
public:
	jit_buffer();
	~jit_buffer();

	jit_buffer(const jit_buffer&) = delete;
	jit_buffer& operator=(const jit_buffer&) = delete;

	// Append raw bytes to the staging vector. Nothing is executable yet.
	void append(const void* bytes, size_t len);
	void emit_u8(const uint8_t b) { append(&b, 1); }
	void emit_u32(const uint32_t v) { append(&v, 4); }
	void emit_u64(const uint64_t v) { append(&v, 8); }

	size_t size() const { return staging_.size(); }
	const uint8_t* staging_data() const { return staging_.data(); }

	// Copy the staging bytes into the shared executable-page pool and
	// return a pointer to the start of the placed code, or nullptr if the
	// OS refused the memory. Previously-returned pointers remain valid
	// (and executable) for the lifetime of the process.
	void* commit();

private:
	std::vector<uint8_t> staging_;
};
