// nnc app — terminal output helpers. See ui.h.

#include "ui.h"

#include "platform.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <string>

namespace
{
	// A progress bar has drawn a line that does not end in a newline yet.
	bool g_bar_open = false;

	bool color() { return platform_color() && platform_stderr_is_terminal(); }

	int64_t now_us()
	{
		return std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count();
	}

	void vline(const char* prefix, const char* fmt, va_list args)
	{
		std::fflush(stdout); // keep model text and messages in order
		if (g_bar_open)
		{
			std::fputc('\n', stderr); // don't write over a half-drawn bar
			g_bar_open = false;
		}
		std::fputs(prefix, stderr);
		std::vfprintf(stderr, fmt, args);
		std::fputs(ui_reset(), stderr);
		std::fputc('\n', stderr);
	}
}

std::string format_bytes(const uint64_t bytes)
{
	char buf[32];
	const double b = static_cast<double>(bytes);
	if (b >= 1024.0 * 1024.0 * 1024.0) std::snprintf(buf, sizeof(buf), "%.1f GB", b / (1024.0 * 1024.0 * 1024.0));
	else if (b >= 1024.0 * 1024.0) std::snprintf(buf, sizeof(buf), "%.0f MB", b / (1024.0 * 1024.0));
	else if (b >= 1024.0) std::snprintf(buf, sizeof(buf), "%.0f KB", b / 1024.0);
	else std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(bytes));
	return buf;
}

std::string format_duration(const double seconds)
{
	const long long s = static_cast<long long>(seconds + 0.5);
	char buf[32];
	if (s >= 3600) std::snprintf(buf, sizeof(buf), "%lld:%02lld:%02lld", s / 3600, (s / 60) % 60, s % 60);
	else std::snprintf(buf, sizeof(buf), "%lld:%02lld", s / 60, s % 60);
	return buf;
}

std::string tidy_path(const std::string& path)
{
	const std::string home = platform_home_dir();
	if (!home.empty() && path.size() > home.size() && path.compare(0, home.size(), home) == 0
		&& (path[home.size()] == '/' || path[home.size()] == '\\'))
		return "~" + path.substr(home.size());
	return path;
}

const char* ui_dim() { return color() ? "\x1b[2m" : ""; }
const char* ui_bold() { return color() ? "\x1b[1m" : ""; }
const char* ui_red() { return color() ? "\x1b[31m" : ""; }
const char* ui_reset() { return color() ? "\x1b[0m" : ""; }

void ui_info(const char* fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	vline("", fmt, args);
	va_end(args);
}

void ui_error(const char* fmt, ...)
{
	const std::string prefix = std::string(ui_red()) + "nnc: error: ";
	va_list args;
	va_start(args, fmt);
	vline(prefix.c_str(), fmt, args);
	va_end(args);
}

bool ui_confirm(const char* question, const bool fallback)
{
	std::fflush(stdout);
	std::fprintf(stderr, "%s %s ", question, fallback ? "[Y/n]" : "[y/N]");
	std::fflush(stderr);
	std::string line;
	if (!platform_read_line(line)) return fallback;
	for (const char c : line)
	{
		if (c == ' ' || c == '\t') continue;
		if (c == 'y' || c == 'Y') return true;
		if (c == 'n' || c == 'N') return false;
		break;
	}
	return fallback;
}

progress_bar::progress_bar(std::string label)
	: label_(std::move(label)), live_(platform_stderr_is_terminal())
{
}

void progress_bar::update(const double fraction, const std::string& detail)
{
	if (!live_) return;
	const double f = fraction < 0 ? 0 : (fraction > 1 ? 1 : fraction);
	const int64_t t = now_us();
	// At most ten redraws a second, plus the moment it completes.
	if (drawn_ && f < 1.0 && (t - last_us_ < 100000 || f - last_ < 0.001)) return;
	last_ = f;
	last_us_ = t;
	constexpr int width = 24;
	const int filled = static_cast<int>(f * width + 0.5);
	std::string bar(static_cast<size_t>(filled), '#');
	bar.append(static_cast<size_t>(width - filled), '-');
	std::fprintf(stderr, "\r%s [%s] %3d%%%s%s%s", label_.c_str(), bar.c_str(), static_cast<int>(f * 100.0 + 0.5),
	             detail.empty() ? "" : "  ", detail.c_str(), color() ? "\x1b[K" : "   ");
	std::fflush(stderr);
	drawn_ = true;
	g_bar_open = true;
}

void progress_bar::finish(const std::string& summary)
{
	if (!live_)
	{
		if (!summary.empty()) std::fprintf(stderr, "%s %s\n", label_.c_str(), summary.c_str());
		return;
	}
	if (drawn_ && g_bar_open)
		std::fprintf(stderr, "\r%s %s%s\n", label_.c_str(), summary.c_str(),
		             color() ? "\x1b[K" : "                                                  ");
	else if (!summary.empty()) std::fprintf(stderr, "%s %s\n", label_.c_str(), summary.c_str());
	std::fflush(stderr);
	drawn_ = false;
	g_bar_open = false;
}
