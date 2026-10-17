// nnc — the part of libnnc that runs on any x86-64 CPU.
//
// This file is compiled WITHOUT the AVX2/FMA flags the rest of libnnc uses
// (see CMakeLists.txt). It holds the CPU check and every public function
// that needs no model: errors and logging, the version and the default
// parameters, and the gates in front of model loading, the worker pool and
// the profiler. So on a CPU nnc cannot use, nnc_model_load() fails with
// NNC_ERR_UNSUPPORTED_CPU instead of faulting on the first AVX instruction.
//
// Keep it free of templates, standard containers and other headers' inline
// functions: their out-of-line copies are shared with the AVX-compiled
// files, and the linker may keep an AVX one.

#include "runtime.h"
#include "sys.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>

#ifndef NNC_VERSION
#define NNC_VERSION "0.0.0"
#endif

// ---- CPU ------------------------------------------------------------------

namespace
{
	cpu_features detect()
	{
		cpu_features f{};

		int regs[4] = {0, 0, 0, 0};
		sys_cpuid(regs, 0);
		const int max_leaf = regs[0];

		sys_cpuid(regs, 1);
		const bool osxsave = (regs[2] & (1 << 27)) != 0;
		const bool avx = (regs[2] & (1 << 28)) != 0;
		f.fma = (regs[2] & (1 << 12)) != 0;
		f.f16c = (regs[2] & (1 << 29)) != 0;

		bool avx2_bit = false;
		if (max_leaf >= 7)
		{
			sys_cpuidex(regs, 7, 0);
			avx2_bit = (regs[1] & (1 << 5)) != 0;
		}

		// The OS must also save the YMM registers across context switches,
		// which XCR0 bits 1 (SSE) and 2 (AVX) report.
		bool ymm_enabled = false;
		if (osxsave && avx)
		{
			const uint64_t xcr0 = sys_xgetbv(0);
			ymm_enabled = (xcr0 & 0x6) == 0x6;
		}

		f.avx2 = avx2_bit && ymm_enabled;
		f.fma = f.fma && ymm_enabled;
		return f;
	}
}

const cpu_features& nnc_cpu_features()
{
	static const cpu_features f = detect();
	return f;
}

// ---- errors and logging -------------------------------------------------

namespace
{
	constexpr size_t MESSAGE_MAX = 2048;

	// Plain storage: nothing runs to construct it.
	struct thread_error
	{
		nnc_status status;
		char message[MESSAGE_MAX];
	};

	thread_local thread_error t_error;

	// vsnprintf into `buf`. Text too long for it is cut short, and then so
	// is a UTF-8 character the cut split, so the result stays valid UTF-8.
	void format(char* buf, const size_t size, const char* fmt, va_list args)
	{
		const int n = std::vsnprintf(buf, size, fmt, args);
		if (n < 0)
		{
			buf[0] = '\0';
			return;
		}
		if (static_cast<size_t>(n) < size) return;
		size_t i = size - 1;
		size_t cont = 0;
		while (i > 0 && cont < 3 && (static_cast<unsigned char>(buf[i - 1]) & 0xC0) == 0x80)
		{
			--i;
			++cont;
		}
		if (i == 0) return;
		const auto lead = static_cast<unsigned char>(buf[i - 1]);
		const size_t need = (lead & 0xE0) == 0xC0 ? 2 : (lead & 0xF0) == 0xE0 ? 3 : (lead & 0xF8) == 0xF0 ? 4 : 1;
		if (cont + 1 < need) buf[i - 1] = '\0';
	}

	void default_log(const nnc_log_level level, const char* message, void*)
	{
		if (level < NNC_LOG_WARN) return;
		std::fprintf(stderr, "nnc: %s%s\n", level == NNC_LOG_ERROR ? "error: " : "", message);
	}

	nnc_log_callback g_log_fn = default_log;
	void* g_log_user = nullptr;
}

bool nnc_fail(const nnc_status status, const char* fmt, ...)
{
	// Formatted aside first, so a message may quote nnc_last_error().
	char text[MESSAGE_MAX];
	va_list args;
	va_start(args, fmt);
	format(text, sizeof(text), fmt, args);
	va_end(args);
	std::memcpy(t_error.message, text, std::strlen(text) + 1);
	t_error.status = status;
	return false;
}

nnc_status nnc_last_status() { return t_error.status; }

void nnc_clear_error()
{
	t_error.status = NNC_OK;
	t_error.message[0] = '\0';
}

void nnc_log(const nnc_log_level level, const char* fmt, ...)
{
	char text[MESSAGE_MAX];
	va_list args;
	va_start(args, fmt);
	format(text, sizeof(text), fmt, args);
	va_end(args);
	g_log_fn(level, text, g_log_user);
}

// ---- the public functions that need no model ----------------------------

extern "C" {

bool nnc_cpu_supported(void)
{
	const cpu_features& f = nnc_cpu_features();
	return f.avx2 && f.fma;
}

const char* nnc_last_error(void) { return t_error.message; }

void nnc_set_log_callback(const nnc_log_callback callback, void* user)
{
	g_log_fn = callback ? callback : default_log;
	g_log_user = callback ? user : nullptr;
}

const char* nnc_status_string(const nnc_status status)
{
	switch (status)
	{
	case NNC_OK: return "ok";
	case NNC_ERR_INVALID_ARGUMENT: return "invalid argument";
	case NNC_ERR_UNSUPPORTED_CPU: return "unsupported CPU";
	case NNC_ERR_IO: return "I/O error";
	case NNC_ERR_BAD_FORMAT: return "bad file format";
	case NNC_ERR_UNSUPPORTED_MODEL: return "unsupported model";
	case NNC_ERR_OUT_OF_MEMORY: return "out of memory";
	case NNC_ERR_CONTEXT_FULL: return "context full";
	case NNC_ERR_CANCELLED: return "cancelled";
	}
	return "unknown status";
}

const char* nnc_version(void) { return NNC_VERSION; }

nnc_model_params nnc_model_default_params(void)
{
	nnc_model_params p{};
	p.weights = NNC_WEIGHTS_Q8_0;
	return p;
}

nnc_session_params nnc_session_default_params(void)
{
	return nnc_session_params{};
}

nnc_sampler_params nnc_sampler_default_params(void)
{
	nnc_sampler_params p{};
	p.temperature = 0.7f;
	p.top_k = 40;
	p.top_p = 0.95f;
	p.min_p = 0.05f;
	return p;
}

nnc_generate_params nnc_generate_default_params(void)
{
	nnc_generate_params p{};
	p.sampler = nnc_sampler_default_params();
	return p;
}

// The gates: past these, everything is AVX2 code.

nnc_status nnc_model_load(const char* path, const nnc_model_params* params, nnc_model** model)
{
	if (!nnc_cpu_supported())
	{
		if (model) *model = nullptr;
		nnc_fail(NNC_ERR_UNSUPPORTED_CPU, "this CPU cannot run nnc: it needs an x86-64 processor with AVX2 "
		         "and FMA (Intel Haswell / AMD Excavator or newer)");
		return NNC_ERR_UNSUPPORTED_CPU;
	}
	return nnc_model_load_avx(path, params, model);
}

void nnc_set_num_threads(const int n_threads)
{
	if (nnc_cpu_supported()) nnc_set_num_threads_avx(n_threads);
}

int nnc_num_threads(void) { return nnc_cpu_supported() ? nnc_num_threads_avx() : 1; }

void nnc_perf_enable(const bool enabled)
{
	if (nnc_cpu_supported()) nnc_perf_enable_avx(enabled);
}

void nnc_perf_reset(void)
{
	if (nnc_cpu_supported()) nnc_perf_reset_avx();
}

const char* nnc_perf_report(const char* title, const int32_t n_tokens)
{
	return nnc_cpu_supported() ? nnc_perf_report_avx(title, n_tokens) : "";
}

} // extern "C"
