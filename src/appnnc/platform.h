// nnc app — the command-line app's view of the OS: console, UTF-8 arguments
// and input, Ctrl-C, and HTTP downloads. platform.cpp forwards to libnnc's
// sys.h, where the native code lives. Plain filesystem work uses
// std::filesystem through the UTF-8 helpers at the bottom of this header.

#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// UTF-8 console output (Windows code pages, restored at exit) and ANSI
// colour where the terminal supports it.
void platform_init_console();

// No modal dialogs on asserts in debug builds (Windows); no-op on Linux.
void platform_init_crash_handlers();

// Replace argv with the arguments spelled in UTF-8, the encoding every
// path and string in nnc uses. Windows hands main() the ANSI code page,
// which mangles a prompt or a model path outside it. No-op on Linux.
void platform_utf8_args(int& argc, char**& argv);

bool platform_stdin_is_terminal();
bool platform_stdout_is_terminal();
bool platform_stderr_is_terminal();

// True when platform_init_console() found a terminal that understands ANSI
// colour sequences.
bool platform_color();

// Read one line of UTF-8 from stdin, without the line ending. False at end
// of input (Ctrl-D, Ctrl-Z, a closed pipe). On a Windows console this reads
// UTF-16 and converts, so non-ASCII input survives.
bool platform_read_line(std::string& line);

// The user's home directory (USERPROFILE / HOME), or "".
std::string platform_home_dir();

// ---- Ctrl-C -------------------------------------------------------------
//
// Ctrl-C normally ends the process. While an interrupt_scope is alive it
// instead sets platform_interrupted() and calls the scope's hook (from a
// signal handler or a console thread, so the hook must be async-safe, e.g.
// nnc_session_cancel). A second Ctrl-C in the same scope ends the process.

using interrupt_hook = void (*)(void* user);

// Install the Ctrl-C handler. Call once, early.
void platform_install_interrupt_handler();

namespace platform_detail
{
	inline std::atomic<interrupt_hook> hook{nullptr};
	inline std::atomic<void*> hook_user{nullptr};
	inline std::atomic<int> scopes{0};
	inline std::atomic<int> interrupts{0};

	// Called by the OS handler. True when a scope handled the interrupt;
	// false means "end the process as usual".
	inline bool on_interrupt()
	{
		if (scopes.load() == 0) return false;
		if (interrupts.fetch_add(1) > 0) return false;
		if (const interrupt_hook h = hook.load()) h(hook_user.load());
		return true;
	}
}

struct interrupt_scope
{
	explicit interrupt_scope(const interrupt_hook hook = nullptr, void* user = nullptr)
	{
		platform_detail::interrupts = 0;
		platform_detail::hook_user = user;
		platform_detail::hook = hook;
		++platform_detail::scopes;
	}

	~interrupt_scope()
	{
		--platform_detail::scopes;
		platform_detail::hook = nullptr;
	}

	interrupt_scope(const interrupt_scope&) = delete;
	interrupt_scope& operator=(const interrupt_scope&) = delete;
};

// True once Ctrl-C was pressed during the current (or last) scope.
inline bool platform_interrupted() { return platform_detail::interrupts.load() > 0; }

// ---- downloads ------------------------------------------------------------

// Bytes received so far and the expected total (0 when unknown). Return
// false to stop the download.
using download_progress = bool (*)(uint64_t done, uint64_t total, void* user);

// Fetch `url` into the file `dest`, following redirects. If `dest` already
// holds a partial download it is resumed where the server allows. `total`
// is the size to expect when the server does not say. Windows uses WinHTTP;
// Linux runs curl, or wget when curl is missing. On failure fills `error`.
bool platform_download(const std::string& url, const std::string& dest, uint64_t total,
                       download_progress progress, void* user, std::string& error);

// Atomically move `source` over `destination` on the same volume.
bool platform_replace_file(const std::string& source, const std::string& destination, std::string& error);

// ---- UTF-8 paths ------------------------------------------------------------

inline std::filesystem::path u8_path(const std::string& s)
{
	return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

inline std::string path_u8(const std::filesystem::path& p)
{
	const std::u8string u = p.u8string();
	return std::string(reinterpret_cast<const char*>(u.data()), u.size());
}
