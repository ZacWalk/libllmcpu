// nnc app — platform.h on top of libnnc's OS layer (sys.h). The native code
// is in src/libnnc/sys_win.cpp and sys_linux.cpp; this file only adapts it
// to the app's interrupt scopes and progress callbacks.

#include "platform.h"
#include "sys.h"

void platform_init_console() { sys_init_console(); }
void platform_init_crash_handlers() { sys_init_crash_handlers(); }
void platform_utf8_args(int& argc, char**& argv) { sys_utf8_args(argc, argv); }

bool platform_stdin_is_terminal() { return sys_is_terminal(SYS_STDIN); }
bool platform_stdout_is_terminal() { return sys_is_terminal(SYS_STDOUT); }
bool platform_stderr_is_terminal() { return sys_is_terminal(SYS_STDERR); }
bool platform_color() { return sys_console_color(); }
bool platform_read_line(std::string& line) { return sys_read_line(line); }
std::string platform_home_dir() { return sys_home_dir(); }

void platform_install_interrupt_handler()
{
	sys_install_interrupt_handler(platform_detail::on_interrupt);
}

namespace
{
	struct progress_context
	{
		download_progress callback;
		void* user;
	};

	bool dispatch_progress(const uint64_t done, const uint64_t total, void* context)
	{
		if (platform_interrupted()) return false;
		auto* state = static_cast<progress_context*>(context);
		return !state->callback || state->callback(done, total, state->user);
	}
}

bool platform_download(const std::string& url, const std::string& dest, const uint64_t total,
                       const download_progress progress, void* user, std::string& error)
{
	progress_context context{progress, user};
	return sys_download(url, dest, total, dispatch_progress, &context, error);
}

bool platform_replace_file(const std::string& source, const std::string& destination, std::string& error)
{
	return sys_replace_file(source, destination, error);
}
