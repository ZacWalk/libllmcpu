// nnc — Windows implementation of libnnc's sys.h.
// Built only on _WIN32; sys_linux.cpp covers POSIX.

#include "sys.h"

#if defined(_WIN32)

#include "runtime.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <winhttp.h>

#include <intrin.h>

#if defined(_MSC_VER)
#include <crtdbg.h>
#endif

namespace
{
	UINT g_saved_output_cp = 0;
	UINT g_saved_input_cp = 0;
	bool g_color = false;
	bool (*g_on_interrupt)() = nullptr;

	void restore_console_code_pages()
	{
		if (g_saved_output_cp) SetConsoleOutputCP(g_saved_output_cp);
		if (g_saved_input_cp) SetConsoleCP(g_saved_input_cp);
	}

	BOOL WINAPI console_event(const DWORD event)
	{
		if ((event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT) && g_on_interrupt && g_on_interrupt())
			return TRUE;
		restore_console_code_pages();
		return FALSE;
	}

	std::wstring to_wide(const char* utf8)
	{
		const int n = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
		if (n <= 0) return {};
		std::wstring w(static_cast<size_t>(n), L'\0');
		if (!MultiByteToWideChar(CP_UTF8, 0, utf8, -1, w.data(), n)) return {};
		w.pop_back();
		return w;
	}

	std::wstring to_wide(const std::string& utf8)
	{
		if (utf8.empty()) return {};
		const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
		if (n <= 0) return {};
		std::wstring w(static_cast<size_t>(n), L'\0');
		if (!MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), w.data(), n))
			return {};
		return w;
	}

	std::string to_utf8(const wchar_t* wide, const size_t len)
	{
		if (len == 0) return {};
		const int n = WideCharToMultiByte(CP_UTF8, 0, wide, static_cast<int>(len), nullptr, 0, nullptr, nullptr);
		if (n <= 0) return {};
		std::string utf8(static_cast<size_t>(n), '\0');
		if (!WideCharToMultiByte(CP_UTF8, 0, wide, static_cast<int>(len), utf8.data(), n, nullptr, nullptr))
			return {};
		return utf8;
	}

	bool is_console(const DWORD which)
	{
		DWORD mode = 0;
		return GetConsoleMode(GetStdHandle(which), &mode) != 0;
	}

	bool enable_vt(const DWORD which)
	{
		const HANDLE h = GetStdHandle(which);
		DWORD mode = 0;
		if (!GetConsoleMode(h, &mode)) return false;
		return SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
	}

	// The system's own wording for a Win32 error code, in UTF-8.
	std::string error_text(const DWORD code)
	{
		wchar_t* buf = nullptr;
		const HMODULE winhttp = GetModuleHandleW(L"winhttp.dll");
		DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS;
		if (winhttp && code >= 12000 && code < 13000) flags |= FORMAT_MESSAGE_FROM_HMODULE;
		const DWORD n = FormatMessageW(
			flags, winhttp, code, 0, reinterpret_cast<wchar_t*>(&buf), 0, nullptr);
		std::string out = (n > 0 && buf) ? to_utf8(buf, n) : std::string();
		if (buf) LocalFree(buf);
		while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ' || out.back() == '.'))
			out.pop_back();
		if (out.empty()) out = "error " + std::to_string(code);
		return out;
	}

	uint64_t file_size(const std::wstring& path)
	{
		WIN32_FILE_ATTRIBUTE_DATA data{};
		if (!GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &data)) return 0;
		return (static_cast<uint64_t>(data.nFileSizeHigh) << 32) | data.nFileSizeLow;
	}

	struct winhttp_handle
	{
		HINTERNET h = nullptr;
		~winhttp_handle()
		{
			if (h) WinHttpCloseHandle(h);
		}
	};
}

void sys_init_console()
{
	g_saved_output_cp = GetConsoleOutputCP();
	g_saved_input_cp = GetConsoleCP();
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
	std::atexit(restore_console_code_pages);
	const bool out_vt = enable_vt(STD_OUTPUT_HANDLE);
	enable_vt(STD_ERROR_HANDLE);
	g_color = out_vt && !std::getenv("NO_COLOR");
}

void sys_init_crash_handlers()
{
#if defined(_MSC_VER) && defined(_DEBUG)
	_set_error_mode(_OUT_TO_STDERR);
	_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
	_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
}

void sys_utf8_args(int& argc, char**& argv)
{
	int wargc = 0;
	LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
	if (!wargv) return;

	static std::vector<std::string> storage;
	static std::vector<char*> pointers;
	storage.clear();
	for (int i = 0; i < wargc; ++i) storage.push_back(to_utf8(wargv[i], wcslen(wargv[i])));
	LocalFree(wargv);

	pointers.clear();
	for (auto& s : storage) pointers.push_back(s.data());
	pointers.push_back(nullptr);
	argc = wargc;
	argv = pointers.data();
}

bool sys_is_terminal(const sys_stream stream)
{
	switch (stream)
	{
	case SYS_STDIN: return is_console(STD_INPUT_HANDLE);
	case SYS_STDOUT: return is_console(STD_OUTPUT_HANDLE);
	case SYS_STDERR: return is_console(STD_ERROR_HANDLE);
	default: return false;
	}
}

bool sys_console_color() { return g_color; }

bool sys_read_line(std::string& line)
{
	line.clear();
	const HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
	DWORD mode = 0;
	if (!GetConsoleMode(in, &mode))
	{
		if (!std::getline(std::cin, line)) return false;
		if (!line.empty() && line.back() == '\r') line.pop_back();
		return true;
	}

	std::wstring wide;
	wchar_t buf[512];
	for (;;)
	{
		DWORD got = 0;
		if (!ReadConsoleW(in, buf, static_cast<DWORD>(std::size(buf)), &got, nullptr) || got == 0) return false;
		wide.append(buf, got);
		if (wide.back() == L'\n') break;
	}
	while (!wide.empty() && (wide.back() == L'\n' || wide.back() == L'\r')) wide.pop_back();
	if (!wide.empty() && wide[0] == 0x1A) return false;
	line = to_utf8(wide.data(), wide.size());
	return true;
}

std::string sys_home_dir()
{
	wchar_t buf[MAX_PATH];
	const DWORD n = GetEnvironmentVariableW(L"USERPROFILE", buf, MAX_PATH);
	if (n == 0 || n >= MAX_PATH) return {};
	return to_utf8(buf, n);
}

void sys_install_interrupt_handler(bool (*on_interrupt)())
{
	g_on_interrupt = on_interrupt;
	SetConsoleCtrlHandler(console_event, TRUE);
}

bool sys_download(const std::string& url, const std::string& dest, const uint64_t total_hint,
                  const sys_progress_callback progress, void* user, std::string& error)
{
	const std::wstring wurl = to_wide(url);
	URL_COMPONENTS uc{};
	uc.dwStructSize = sizeof(uc);
	uc.dwHostNameLength = static_cast<DWORD>(-1);
	uc.dwUrlPathLength = static_cast<DWORD>(-1);
	uc.dwExtraInfoLength = static_cast<DWORD>(-1);
	if (wurl.empty() || !WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc))
	{
		error = "invalid URL: " + url;
		return false;
	}
	const std::wstring host(uc.lpszHostName, uc.dwHostNameLength);
	const std::wstring object = std::wstring(uc.lpszUrlPath, uc.dwUrlPathLength)
		+ std::wstring(uc.lpszExtraInfo ? uc.lpszExtraInfo : L"",
		               uc.lpszExtraInfo ? uc.dwExtraInfoLength : 0);

	winhttp_handle session{WinHttpOpen(L"nnc", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
	                                   WINHTTP_NO_PROXY_BYPASS, 0)};
	if (!session.h)
		session.h = WinHttpOpen(L"nnc", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME,
		                        WINHTTP_NO_PROXY_BYPASS, 0);
	if (!session.h)
	{
		error = "cannot start WinHTTP: " + error_text(GetLastError());
		return false;
	}
	WinHttpSetTimeouts(session.h, 0, 30000, 30000, 60000);
	winhttp_handle connect{WinHttpConnect(session.h, host.c_str(), uc.nPort, 0)};
	winhttp_handle request{
		connect.h
			? WinHttpOpenRequest(connect.h, L"GET", object.c_str(), nullptr, WINHTTP_NO_REFERER,
			                     WINHTTP_DEFAULT_ACCEPT_TYPES,
			                     uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0)
			: nullptr
	};
	if (!request.h)
	{
		error = "cannot reach " + to_utf8(host.data(), host.size()) + ": " + error_text(GetLastError());
		return false;
	}

	const std::wstring wdest = to_wide(dest);
	if (wdest.empty())
	{
		error = "invalid destination path: " + dest;
		return false;
	}
	uint64_t offset = file_size(wdest);
	const std::wstring range = L"Range: bytes=" + std::to_wstring(offset) + L"-";
	if (!WinHttpSendRequest(request.h, offset ? range.c_str() : WINHTTP_NO_ADDITIONAL_HEADERS,
	                        offset ? static_cast<DWORD>(-1L) : 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0)
		|| !WinHttpReceiveResponse(request.h, nullptr))
	{
		error = "cannot download from " + to_utf8(host.data(), host.size()) + ": " + error_text(GetLastError());
		return false;
	}

	DWORD status_code = 0;
	DWORD status_size = sizeof(status_code);
	if (!WinHttpQueryHeaders(request.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
	                         WINHTTP_HEADER_NAME_BY_INDEX, &status_code, &status_size, WINHTTP_NO_HEADER_INDEX))
	{
		error = "cannot read the HTTP response status: " + error_text(GetLastError());
		return false;
	}
	if (status_code != 200 && status_code != 206)
	{
		error = "the server returned HTTP status " + std::to_string(status_code);
		return false;
	}
	if (offset && status_code != 206) offset = 0;

	uint64_t length = 0;
	wchar_t len_buf[32];
	DWORD len_size = sizeof(len_buf);
	if (WinHttpQueryHeaders(request.h, WINHTTP_QUERY_CONTENT_LENGTH, WINHTTP_HEADER_NAME_BY_INDEX, len_buf,
	                        &len_size, WINHTTP_NO_HEADER_INDEX))
		length = _wcstoui64(len_buf, nullptr, 10);
	const uint64_t total = length ? offset + length : total_hint;

	HANDLE file = CreateFileW(wdest.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (file == INVALID_HANDLE_VALUE)
	{
		error = "cannot write '" + dest + "': " + error_text(GetLastError());
		return false;
	}
	LARGE_INTEGER pos{};
	pos.QuadPart = static_cast<LONGLONG>(offset);
	bool ok = SetFilePointerEx(file, pos, nullptr, FILE_BEGIN) && SetEndOfFile(file);
	if (!ok) error = "cannot position '" + dest + "': " + error_text(GetLastError());

	uint64_t done = offset;
	std::vector<char> buf(1 << 20);
	while (ok)
	{
		DWORD got = 0;
		if (!WinHttpReadData(request.h, buf.data(), static_cast<DWORD>(buf.size()), &got))
		{
			error = "download interrupted: " + error_text(GetLastError());
			ok = false;
			break;
		}
		if (got == 0) break;
		DWORD written = 0;
		if (!WriteFile(file, buf.data(), got, &written, nullptr) || written != got)
		{
			error = "cannot write '" + dest + "': " + error_text(GetLastError());
			ok = false;
			break;
		}
		done += got;
		if (progress && !progress(done, total, user))
		{
			error = "cancelled";
			ok = false;
		}
	}
	CloseHandle(file);
	if (ok && length && done != offset + length)
	{
		error = "the connection closed early";
		ok = false;
	}
	return ok;
}

bool sys_replace_file(const std::string& source, const std::string& destination, std::string& error)
{
	const std::wstring from = to_wide(source);
	const std::wstring to = to_wide(destination);
	if (from.empty() || to.empty())
	{
		error = "source or destination path is not valid UTF-8";
		return false;
	}
	if (!MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
	{
		error = error_text(GetLastError());
		return false;
	}
	return true;
}

size_t sys_page_size()
{
	static const size_t cached = []
	{
		SYSTEM_INFO si;
		GetSystemInfo(&si);
		return si.dwPageSize
			       ? static_cast<size_t>(si.dwPageSize)
			       : size_t{4096};
	}();
	return cached;
}

void* sys_alloc_exec_pages(const size_t bytes)
{
	return VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}

bool sys_protect_rw(void* p, const size_t bytes)
{
	DWORD old = 0;
	return VirtualProtect(p, bytes, PAGE_READWRITE, &old) != 0;
}

bool sys_protect_rx(void* p, const size_t bytes)
{
	DWORD old = 0;
	return VirtualProtect(p, bytes, PAGE_EXECUTE_READ, &old) != 0;
}

void sys_free_exec_pages(void* p, size_t)
{
	if (p) VirtualFree(p, 0, MEM_RELEASE);
}

void sys_flush_icache(const void* p, const size_t bytes)
{
	FlushInstructionCache(GetCurrentProcess(), p, bytes);
}

// ---- File mapping ------------------------------------------------------

bool sys_mmap_file_ro(const char* path, sys_mmap& out)
{
	out = {};

	const std::wstring wpath = to_wide(path);
	const HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
	                                 OPEN_EXISTING,
	                                 FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
	                                 nullptr);
	if (hFile == INVALID_HANDLE_VALUE)
		return nnc_fail(NNC_ERR_IO, "cannot open '%s': %s", path, error_text(GetLastError()).c_str());

	LARGE_INTEGER sz{};
	if (!GetFileSizeEx(hFile, &sz))
	{
		const DWORD err = GetLastError();
		CloseHandle(hFile);
		return nnc_fail(NNC_ERR_IO, "cannot size '%s': %s", path, error_text(err).c_str());
	}
	if (sz.QuadPart == 0)
	{
		CloseHandle(hFile);
		return nnc_fail(NNC_ERR_BAD_FORMAT, "'%s' is empty", path);
	}
	const HANDLE hMap = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
	if (!hMap)
	{
		const DWORD err = GetLastError();
		CloseHandle(hFile);
		return nnc_fail(NNC_ERR_IO, "cannot map '%s': %s", path, error_text(err).c_str());
	}
	const void* base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
	if (!base)
	{
		const DWORD err = GetLastError();
		CloseHandle(hMap);
		CloseHandle(hFile);
		return nnc_fail(NNC_ERR_IO, "cannot map '%s': %s", path, error_text(err).c_str());
	}

	out.base = base;
	out.size = static_cast<uint64_t>(sz.QuadPart);
	out.handle = hFile;
	out.handle2 = hMap;
	return true;
}

void sys_munmap(sys_mmap& m)
{
	if (m.base)
	{
		UnmapViewOfFile(m.base);
		m.base = nullptr;
	}
	if (m.handle2)
	{
		CloseHandle(m.handle2);
		m.handle2 = nullptr;
	}
	if (m.handle)
	{
		CloseHandle(m.handle);
		m.handle = nullptr;
	}
	m.size = 0;
}

void sys_prefetch_ro(const void* base, const uint64_t bytes)
{
	using PFV = BOOL (WINAPI*)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
	const HMODULE hk = GetModuleHandleW(L"kernel32.dll");
	if (!hk) return;
	const auto pfv = reinterpret_cast<PFV>(GetProcAddress(hk, "PrefetchVirtualMemory"));
	if (!pfv) return;
	WIN32_MEMORY_RANGE_ENTRY r;
	r.VirtualAddress = const_cast<void*>(base);
	r.NumberOfBytes = bytes;
	pfv(GetCurrentProcess(), 1, &r, 0);
}

// ---- CPUID -------------------------------------------------------------

void sys_cpuid(int regs[4], const int leaf) { __cpuid(regs, leaf); }
void sys_cpuidex(int regs[4], const int leaf, const int sub) { __cpuidex(regs, leaf, sub); }
uint64_t sys_xgetbv(const unsigned int xcr) { return _xgetbv(xcr); }

#endif // _WIN32