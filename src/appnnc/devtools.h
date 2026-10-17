// nnc app — inspection and smoke-test modes (--gguf-info, --inspect-model,
// --gemma-*). These look inside the library, so unlike the chat they use
// libnnc's internal headers as well as nnc.h.

#pragma once

#include "cli.h"

// Run a diagnostic mode (any app_mode from gguf_info onwards).
int run_devtool(const app_options& options);
