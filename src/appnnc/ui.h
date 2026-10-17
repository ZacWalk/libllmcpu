// nnc app — terminal output helpers: messages, colour, progress bars and
// questions. Status goes to stderr so stdout carries only model output.

#pragma once

#include <cstdint>
#include <string>

std::string format_bytes(uint64_t bytes); // "1.0 GB", "512 MB"
std::string format_duration(double seconds); // "0:42", "1:05:09"

// `path` with the home directory written as "~".
std::string tidy_path(const std::string& path);

// ANSI styles, or "" when colour is off.
const char* ui_dim();
const char* ui_bold();
const char* ui_red();
const char* ui_reset();

// printf-style lines on stderr: plain, and "nnc: error: ..." in red.
void ui_info(const char* fmt, ...);
void ui_error(const char* fmt, ...);

// Ask a yes/no question on stderr and read the answer; `fallback` is the
// answer for an empty line, and for end of input.
bool ui_confirm(const char* question, bool fallback);

// A one-line progress bar on stderr, redrawn in place. Silent when stderr
// is not a terminal.
class progress_bar
{
public:
	explicit progress_bar(std::string label);
	// `fraction` in [0, 1]; `detail` follows the bar ("456 / 1013 MB ...").
	void update(double fraction, const std::string& detail = {});
	// Finish the line, replacing the bar with `summary` when one is given.
	void finish(const std::string& summary = {});

private:
	std::string label_;
	bool live_;
	bool drawn_ = false;
	double last_ = -1.0;
	int64_t last_us_ = 0;
};
