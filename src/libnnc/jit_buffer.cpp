// nnc — executable-memory allocator implementation.
//
// Process-wide pool: each `jit_buffer::commit()` copies its staging bytes
// into the next whole pages of the current block, flips exactly those pages
// to read+execute, and returns them. Blocks are kept for the lifetime of the
// process; kernels are never freed.
//
// W^X discipline: code pages are written once, while still RW and before
// anyone can call into them, then sealed RX for good. A page that holds a
// published kernel is never made writable again, so committing a new
// kernel cannot fault a thread that is running an old one (a model loaded
// on one thread while another generates). The price is page granularity:
// each kernel occupies at least one 4 KB page, a few dozen pages per model.
//
// Blocks are reserved 64 KB at a time so most commits cost one mprotect /
// VirtualProtect rather than a fresh allocation.

#include "jit_buffer.h"

#include "sys.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

jit_buffer::jit_buffer() = default;
jit_buffer::~jit_buffer() = default;

void jit_buffer::append(const void* bytes, const size_t len)
{
	const auto p = static_cast<const uint8_t*>(bytes);
	staging_.insert(staging_.end(), p, p + len);
}

namespace
{
	constexpr size_t POOL_BLOCK_BYTES = 64 * 1024;

	struct jit_code_pool
	{
		std::mutex mu;
		uint8_t* block = nullptr; // current block; [used, size) is still RW and empty
		size_t size = 0;
		size_t used = 0; // always a whole number of pages

		void* commit_bytes(const void* bytes, const size_t len)
		{
			std::lock_guard<std::mutex> lk(mu);

			const size_t page = sys_page_size();
			const size_t need = (len + page - 1) & ~(page - 1);
			if (block == nullptr || used + need > size)
			{
				// The old block's tail stays RW and unused; it never held code.
				size_t want = (need > POOL_BLOCK_BYTES) ? need : POOL_BLOCK_BYTES;
				want = (want + page - 1) & ~(page - 1);
				auto* p = static_cast<uint8_t*>(sys_alloc_exec_pages(want));
				if (!p) return nullptr;
				block = p;
				size = want;
				used = 0;
			}

			uint8_t* dst = block + used;
			std::memcpy(dst, bytes, len);
			if (!sys_protect_rx(dst, need)) return nullptr;
			sys_flush_icache(dst, len);
			used += need;
			return dst;
		}
	};

	jit_code_pool& pool()
	{
		static jit_code_pool p;
		return p;
	}
}

void* jit_buffer::commit()
{
	if (staging_.empty()) return nullptr;
	return pool().commit_bytes(staging_.data(), staging_.size());
}