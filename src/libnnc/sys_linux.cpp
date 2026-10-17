// nnc — Linux (POSIX) implementation of libnnc's sys.h.
// Built only when not _WIN32.

#include "sys.h"

#if !defined(_WIN32)

#include "runtime.h"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <cpuid.h>

extern char** environ;

namespace
{
	bool g_color = false;
	bool (*g_on_interrupt)() = nullptr;

	void on_sigint(int)
	{
		if (g_on_interrupt && g_on_interrupt()) return;
		std::signal(SIGINT, SIG_DFL);
		std::raise(SIGINT);
	}

	void sleep_ms(const long ms)
	{
		timespec ts{ms / 1000, (ms % 1000) * 1000000L};
		nanosleep(&ts, nullptr);
	}

	uint64_t file_size(const std::string& path)
	{
		struct stat st{};
		return stat(path.c_str(), &st) == 0 ? static_cast<uint64_t>(st.st_size) : 0;
	}

	pid_t spawn(const std::vector<std::string>& args)
	{
		std::vector<char*> argv;
		for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
		argv.push_back(nullptr);
		pid_t pid = -1;
		const int rc = posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), environ);
		if (rc != 0)
		{
			errno = rc;
			return -1;
		}
		return pid;
	}
}

void sys_init_console()
{
	const char* term = std::getenv("TERM");
	g_color = isatty(STDOUT_FILENO) && term && std::strcmp(term, "dumb") != 0 && !std::getenv("NO_COLOR");
}

void sys_init_crash_handlers() {}
void sys_utf8_args(int&, char**&) {}
bool sys_is_terminal(const sys_stream stream) { return isatty(static_cast<int>(stream)) != 0; }
bool sys_console_color() { return g_color; }

bool sys_read_line(std::string& line)
{
	if (!std::getline(std::cin, line)) return false;
	if (!line.empty() && line.back() == '\r') line.pop_back();
	return true;
}

std::string sys_home_dir()
{
	const char* home = std::getenv("HOME");
	return home ? std::string(home) : std::string();
}

void sys_install_interrupt_handler(bool (*on_interrupt)())
{
	g_on_interrupt = on_interrupt;
	struct sigaction sa{};
	sa.sa_handler = on_sigint;
	sigemptyset(&sa.sa_mask);
	sigaction(SIGINT, &sa, nullptr);
}

bool sys_download(const std::string& url, const std::string& dest, const uint64_t total,
                  const sys_progress_callback progress, void* user, std::string& error)
{
	pid_t pid = spawn({"curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3",
	                   "--continue-at", "-", "--output", dest, url});
	if (pid < 0 && errno == ENOENT)
		pid = spawn({"wget", "--quiet", "--continue", "--output-document", dest, url});
	if (pid < 0)
	{
		error = (errno == ENOENT)
			        ? "downloading needs curl or wget; install one, or fetch the file yourself"
			        : std::string("cannot start curl: ") + std::strerror(errno);
		return false;
	}

	int status = 0;
	for (;;)
	{
		const pid_t r = waitpid(pid, &status, WNOHANG);
		if (r == pid) break;
		if (r < 0 && errno != EINTR)
		{
			error = std::string("lost track of the downloader: ") + std::strerror(errno);
			return false;
		}
		if (progress && !progress(file_size(dest), total, user))
		{
			kill(pid, SIGTERM);
			waitpid(pid, &status, 0);
			error = "cancelled";
			return false;
		}
		sleep_ms(200);
	}
	if (progress && !progress(file_size(dest), total, user))
	{
		error = "cancelled";
		return false;
	}
	if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return true;
	if (WIFEXITED(status))
		error = "the download failed (downloader exit code " + std::to_string(WEXITSTATUS(status)) + ")";
	else
		error = "the download was interrupted";
	return false;
}

bool sys_replace_file(const std::string& source, const std::string& destination, std::string& error)
{
	std::error_code ec;
	std::filesystem::rename(source, destination, ec);
	if (ec)
	{
		error = ec.message();
		return false;
	}
	return true;
}

size_t sys_page_size()
{
	static const size_t cached = []
	{
		const long ps = sysconf(_SC_PAGESIZE);
		return ps > 0 ? static_cast<size_t>(ps) : size_t{4096};
	}();
	return cached;
}

void* sys_alloc_exec_pages(size_t bytes)
{
	void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
	               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	return p == MAP_FAILED ? nullptr : p;
}

bool sys_protect_rw(void* p, size_t bytes) { return mprotect(p, bytes, PROT_READ | PROT_WRITE) == 0; }
bool sys_protect_rx(void* p, size_t bytes) { return mprotect(p, bytes, PROT_READ | PROT_EXEC) == 0; }

void sys_free_exec_pages(void* p, size_t bytes)
{
	if (p) munmap(p, bytes);
}

void sys_flush_icache(const void* p, size_t bytes)
{
	// On x86 the icache is coherent with stores from the same core, so
	// this is effectively a no-op. We still call the GCC builtin to keep
	// the contract clean (and to be correct on other archs in future).
	auto* s = const_cast<char*>(static_cast<const char*>(p));
	__builtin___clear_cache(s, s + bytes);
}

// ---- File mapping ------------------------------------------------------

bool sys_mmap_file_ro(const char* path, sys_mmap& out)
{
	out = {};

	const int fd = open(path, O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return nnc_fail(NNC_ERR_IO, "cannot open '%s': %s", path, std::strerror(errno));

	struct stat st{};
	if (fstat(fd, &st) != 0)
	{
		const int err = errno;
		close(fd);
		return nnc_fail(NNC_ERR_IO, "cannot stat '%s': %s", path, std::strerror(err));
	}
	if (st.st_size == 0)
	{
		close(fd);
		return nnc_fail(NNC_ERR_BAD_FORMAT, "'%s' is empty", path);
	}
	const size_t len = static_cast<size_t>(st.st_size);
	void* base = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
	if (base == MAP_FAILED)
	{
		const int err = errno;
		close(fd);
		return nnc_fail(NNC_ERR_IO, "cannot map '%s': %s", path, std::strerror(err));
	}

	out.base = base;
	out.size = static_cast<uint64_t>(st.st_size);
	out.handle = reinterpret_cast<void*>(static_cast<intptr_t>(fd));
	out.handle2 = nullptr;
	return true;
}

void sys_munmap(sys_mmap& m)
{
	if (m.base && m.size)
	{
		munmap(const_cast<void*>(m.base), static_cast<size_t>(m.size));
	}
	if (m.handle)
	{
		const int fd = static_cast<int>(reinterpret_cast<intptr_t>(m.handle));
		close(fd);
	}
	m.base = nullptr;
	m.size = 0;
	m.handle = nullptr;
	m.handle2 = nullptr;
}

void sys_prefetch_ro(const void* base, uint64_t bytes)
{
#if defined(POSIX_MADV_WILLNEED)
	posix_madvise(const_cast<void*>(base), static_cast<size_t>(bytes), POSIX_MADV_WILLNEED);
#else
	madvise(const_cast<void*>(base), static_cast<size_t>(bytes), MADV_WILLNEED);
#endif
}

// ---- CPUID -------------------------------------------------------------

void sys_cpuid(int regs[4], int leaf)
{
	unsigned int a = 0, b = 0, c = 0, d = 0;
	__get_cpuid(static_cast<unsigned int>(leaf), &a, &b, &c, &d);
	regs[0] = static_cast<int>(a);
	regs[1] = static_cast<int>(b);
	regs[2] = static_cast<int>(c);
	regs[3] = static_cast<int>(d);
}

void sys_cpuidex(int regs[4], int leaf, int sub)
{
	unsigned int a = 0, b = 0, c = 0, d = 0;
	__cpuid_count(static_cast<unsigned int>(leaf),
	              static_cast<unsigned int>(sub), a, b, c, d);
	regs[0] = static_cast<int>(a);
	regs[1] = static_cast<int>(b);
	regs[2] = static_cast<int>(c);
	regs[3] = static_cast<int>(d);
}

uint64_t sys_xgetbv(unsigned int xcr)
{
	unsigned int eax = 0, edx = 0;
	__asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(xcr));
	return (static_cast<uint64_t>(edx) << 32) | eax;
}

#endif // !_WIN32