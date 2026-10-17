// nnc — command-line entry point.
//
// Checks the CPU, sets up the console, parses the options and hands over to
// the chat (chat.cpp), the model downloader (models.cpp), the diagnostic
// modes (devtools.cpp) or the self-tests (tests.cpp). Everything that runs
// a model goes through libnnc's public API, nnc.h.

#include "chat.h"
#include "cli.h"
#include "devtools.h"
#include "models.h"
#include "platform.h"
#include "ui.h"

#include "nnc.h"

#include <cstdio>

int run_tests(); // tests.cpp

namespace
{
	bool g_verbose = false;

	void on_log(const nnc_log_level level, const char* message, void*)
	{
		if (level >= NNC_LOG_WARN) ui_info("nnc: %s%s", level == NNC_LOG_ERROR ? "error: " : "", message);
		else if (g_verbose) ui_info("%s%s%s", ui_dim(), message, ui_reset());
	}
}

int main(int argc, char** argv)
{
	// First, before anything that might use AVX: nnc's kernels need AVX2 and
	// FMA, and this file is built without them so the check itself is safe.
	if (!nnc_cpu_supported())
	{
		std::fputs("nnc: this CPU is not supported: nnc needs an x86-64 processor with AVX2 and FMA "
		           "(Intel Haswell / AMD Excavator or newer).\n", stderr);
		return 2;
	}

	platform_init_crash_handlers();
	platform_utf8_args(argc, argv);
	platform_init_console();
	platform_install_interrupt_handler();

	app_options o;
	if (!parse_options(argc, argv, o)) return 1;
	g_verbose = o.verbose;
	nnc_set_log_callback(on_log, nullptr);
	if (o.n_threads > 0) nnc_set_num_threads(o.n_threads);
	if (o.perf) nnc_perf_enable(true);

	switch (o.mode)
	{
	case app_mode::help:
		print_usage(argv[0]);
		return 0;
	case app_mode::version:
		std::printf("nnc %s\n", nnc_version());
		return 0;
	case app_mode::test: return run_tests();
	case app_mode::list_models: return run_list_models();
	case app_mode::download: return run_download(o.args.empty() ? std::string() : o.args[0], o.yes);
	case app_mode::chat: return run_chat(o);
	default: return run_devtool(o);
	}
}