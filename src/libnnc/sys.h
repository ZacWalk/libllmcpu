// nnc — libnnc's thin OS abstraction layer.
// All Windows / Linux specific calls used by nnc live behind this header:
// executable memory, file mapping, CPU detection, console/input, interrupts,
// and model downloads. Implementations are in sys_win.cpp (Windows) and
// sys_linux.cpp (Linux); each is #if-guarded so the other compiles to an
// empty object.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// ---- Console and process setup ------------------------------------------
//
// libnnc itself never calls these; they are the native half of the app's
// platform.h and live here so that no file outside sys_win.cpp /
// sys_linux.cpp needs an OS header.

// One-time console setup. Forces UTF-8 code pages on Windows (so BPE pieces
// with U+2581 etc. render) and puts the console's own code pages back at
// exit and on an unhandled Ctrl-C; enables ANSI sequences where the terminal
// supports them. On Linux only decides whether colour is wanted.
void sys_init_console();

// MSVC debug builds: report asserts and aborts on stderr instead of in a
// modal dialog. No-op elsewhere.
void sys_init_crash_handlers();

// Replace argv with the arguments spelled in UTF-8, the encoding every other
// path and string in nnc uses. Windows hands main() the ANSI code page, which
// mangles a prompt or a model path outside it. No-op on Linux.
void sys_utf8_args(int& argc, char**& argv);

enum sys_stream
{
	SYS_STDIN = 0,
	SYS_STDOUT = 1,
	SYS_STDERR = 2,
};

bool sys_is_terminal(sys_stream stream);

// True when sys_init_console() found a terminal that takes ANSI colour and
// NO_COLOR is not set.
bool sys_console_color();

// One line of UTF-8 from stdin without its line ending; false at end of
// input. A Windows console is read as UTF-16 so non-ASCII input survives.
bool sys_read_line(std::string& line);

// The user's home directory (Windows: %USERPROFILE%; POSIX: $HOME), or "".
std::string sys_home_dir();

// Route Ctrl-C (and Ctrl-Break) to `on_interrupt`, which runs on a signal
// handler or a console thread and so must be async-signal-safe. Returning
// true swallows the interrupt; false ends the process as usual.
void sys_install_interrupt_handler(bool (*on_interrupt)());

// ---- Downloads and files ------------------------------------------------

// Bytes received so far and the expected total (0 if unknown). Return false
// to stop the transfer.
using sys_progress_callback = bool (*)(uint64_t done, uint64_t total, void* user);

// Fetch `url` into the file `dest` (UTF-8), following redirects and
// resuming a partial `dest` where the server allows. `total` is the expected
// size, passed to `progress` whenever the server's own figure is unknown.
// Windows uses WinHTTP; Linux runs curl, or wget when curl is missing. On
// failure fills `error` ("cancelled" when `progress` stopped it) and returns
// false.
bool sys_download(const std::string& url, const std::string& dest, uint64_t total,
                  sys_progress_callback progress, void* user, std::string& error);

// Atomically move `source` over `destination` on the same volume, replacing
// any existing file. On failure fills `error` and returns false.
bool sys_replace_file(const std::string& source, const std::string& destination, std::string& error);

// ---- Executable memory --------------------------------------------------

// OS page size (e.g. 4096 on x86-64 Windows / Linux).
size_t sys_page_size();

// Reserve and commit `bytes` of writable memory suitable for later being
// flipped to executable. Returns nullptr on failure.
void* sys_alloc_exec_pages(size_t bytes);

// Flip an existing region between writable and executable. `bytes` should
// be a whole number of pages starting at `p`. Returns false on failure.
bool sys_protect_rw(void* p, size_t bytes);
bool sys_protect_rx(void* p, size_t bytes);

// Release a region from sys_alloc_exec_pages.
void sys_free_exec_pages(void* p, size_t bytes);

// Notify the CPU that the bytes at `[p, p+bytes)` were just written and
// will be executed shortly. Required after publishing JITted code.
void sys_flush_icache(const void* p, size_t bytes);

// ---- Read-only file mapping -------------------------------------------

struct sys_mmap
{
	const void* base = nullptr; // start of mapped region (page-aligned)
	uint64_t size = 0; // bytes
	// Opaque OS handles. Windows uses both (file + mapping HANDLE);
	// Linux only needs the file descriptor.
	void* handle = nullptr;
	void* handle2 = nullptr;
};

// mmap the UTF-8 `path` read-only into the process. On failure records the
// reason with nnc_fail(NNC_ERR_IO, ...) and returns false.
bool sys_mmap_file_ro(const char* path, sys_mmap& out);
void sys_munmap(sys_mmap& m);

// Hint the OS to start paging the entire mapping into RAM. Best-effort;
// failure is silently ignored.
void sys_prefetch_ro(const void* base, uint64_t bytes);

// ---- CPUID / XGETBV ----------------------------------------------------

// Cross-compiler CPUID wrappers. `regs` is filled as {eax, ebx, ecx, edx}.
void sys_cpuid(int regs[4], int leaf);
void sys_cpuidex(int regs[4], int leaf, int subleaf);

// Read an extended control register (used to confirm OS YMM state save).
uint64_t sys_xgetbv(unsigned int xcr);
