# nnc — Neural Net Compiler

**Read [docs/design.md](docs/design.md) first.** It is the single source of
truth for how nnc works: layering, dtypes and the Q8_0 split layout, the JIT
and its ABI, threading, the forward pass, the profiler, and the current
performance picture. This file covers only the rules for *changing* the code.
When you change behaviour, update `docs/design.md`; `README.md` and this file
link to it rather than restating it. When you add, remove or repurpose a source
file or change a flow, also update `docs/nnc-diagram.py` and re-run it to
redraw `nnc-diagram.png`.

## Project identity

- **Name:** `nnc` (neural net compiler). Use this name in all new docs, comments, log prefixes, CLI help text, and identifiers.
- **Executable:** one binary, `nnc.exe` (Debug: `nnc-d.exe`) on Windows; `nnc` / `nnc-d` on Linux. No second exe for tests, no second exe for tooling.
- **Platforms:** Windows x64 (MSVC v143+) and Linux x64 (g++ 10+ / clang++ 12+, including WSL2). C++20, C17. Build system is CMake + Ninja on both.
- **CPU baseline:** AVX2 + FMA. Do **not** use AVX-512 intrinsics or codegen. Detect AVX2/FMA at startup; refuse to run on older CPUs.
- **No new dependencies** without an explicit ask. Must build with only MSVC + the Windows SDK, or g++/clang + libstdc++.

## Repository layout

Keep the two source components in their current directories; keep each
directory internally flat. The stable library interface is the plain-C
`src/libnnc/nnc.h`; other library headers are private implementation details.

```
CMakeLists.txt               CMake build, targets libnnc and nnc
CMakePresets.json            `release` / `debug` presets
dd.ps1                       Windows dev driver: run / test / download
.github/workflows/           CI: linux.yml, windows.yml
docs/design.md               how it works — keep this current
docs/models.md               per-model measured speed / footprint / what won't load
docs/nnc-diagram.py          draws nnc-diagram.png, the README figure (Python + Pillow)
docs/nnc-demo.py             records a real chat (Linux/WSL) and renders ncc-prompt.gif, the README animation
src/libnnc/                  model/runtime, tokenizer/session, GGUF, kernels, JIT, OS abstraction
src/appnnc/                  CLI, chat + translate, embeddings, model picker/downloader, UI/platform, tests
examples/                    plain-C consumers of the public API: minimal.c, embed.c, function_call.c
cmake/                       CMake package config and dd project command
models/                      weight files (.gguf)
build/                       CMake trees, one per host+preset (gitignored)
exe/                         app output (nnc(.exe), nnc-d(.exe))
```

**OS isolation rule:** no `windows.h`, `<intrin.h>`, `<sys/mman.h>`,
`<unistd.h>` etc. outside `src/libnnc/sys_win.cpp` /
`src/libnnc/sys_linux.cpp`. New OS functionality goes through a new
`sys_*` function declared in `src/libnnc/sys.h` and implemented in both
backends.

**New source files must be added to the appropriate explicit source list in
`CMakeLists.txt`** — a new `.cpp` silently does nothing until it is listed.

**No computation graph.** `src/libnnc/model.cpp` calls kernels directly. Do
not add a graph or op-dispatch layer without a concrete need.

## CLI conventions

The executable is one mode-dispatched app. Bare `nnc` searches `./models`,
`$HOME/.lmstudio/models` and `$HOME/models`; it uses the only model found or
asks the user to choose when there are several. If none are found, it offers
the download catalogue and confirms before downloading. `-m` accepts either a
GGUF path or a catalogue name. Downloads use `./models` when it exists, else
`$HOME/models` (or `./models` when no home directory is available). `-y` /
`--yes` skips the download confirmation.

Q8_0 is the default in-memory weight format. `--bf16` preserves BF16 values
where possible, and `--q4` selects the lower-quality smaller layout.
Sampling defaults to temperature 0.7, top-k 40, top-p 0.95 and min-p 0.05;
temperature zero selects greedy decoding. `-n` defaults to zero (generate
until end-of-turn or context full).

- `nnc --download [NAME]` — choose from the catalogue or download a named entry.
- `nnc --list-models` — list discovered local GGUF files.
- `nnc --test` — run all self-tests; tests live in `src/appnnc/tests.cpp`.
- `--translate FROM:TO` — translate each message (TranslateGemma's prompt; any
  chat model). `/translate` switches it in the REPL.
- `--task NAME`, `--dim N` — for text-embedding models, which `run_chat` hands
  to `src/appnnc/embed.cpp`.
- `--draft FILE`, `--no-draft`, `--draft-n N` — the MTP drafter. A Gemma 4
  model looks for `mtp-<model>*.gguf` on its own; an automatically found
  drafter that does not fit is dropped with a note, a `--draft` one is an
  error.
- `--kv-f32` — keep the K/V cache in F32 rather than the default BF16
  (`nnc_session_params.kv_type`).
- `--think` — let a model with a thinking mode (Gemma 4; Qwen 3, 3.5, 3.6,
  3.8) reason before it answers (`nnc_session_params.thinking`). The app
  shows the reasoning dimmed on stderr, so stdout carries only the answer.
- Diagnostic modes are declared in `src/appnnc/cli.cpp`; update its parser
  and help text together when adding a CLI option.
- On Windows, the app converts argv and file paths through the platform layer;
  paths exposed by the C API are UTF-8.

Quantization belongs to `libnnc`'s model loader. Callers select a
`nnc_weight_format` in `nnc_model_params`; no app code should add a separate
quantization pass.

### Chat app

The app's REPL is implemented in `src/appnnc/chat.cpp`. The console,
interrupt, and download code is native, so it lives in
`src/libnnc/sys_win.cpp` / `sys_linux.cpp`; `src/appnnc/platform.cpp`
adapts it to the app's interrupt scopes and progress callbacks.
Chat templates — including FunctionGemma's developer turn,
TranslateGemma's prompt and each family's thinking switch — are applied by
`libnnc` based on the vocabulary, not by CLI code; so is the split of a
reply into its reasoning (`on_thinking`) and its answer (`on_text`), which
must never carry reasoning. Keep the C API's one-session/one-driving-thread
contract and its UTF-8 callback guarantee intact. Drafters (`mtp-*`) are
catalogue entries but stay out of the first-run menu: they cannot run on
their own.

## Tests (`src/appnnc/tests.cpp`)

- Single translation unit. No framework. Plain functions returning `bool`, collected in a static array, run sequentially.
- Each test prints `[PASS]` / `[FAIL] reason` with a stable name. Final line: `nnc: N passed, M failed`. Exit code = number of failures.
- Cover both app-level helpers and JIT/SIMD primitives (allocator, emitter encodings, dot/gemv numerical equivalence to a scalar reference, gelu/softmax/layernorm/rmsnorm/swiglu/rope/softcap, bf16-W and Q8_0-W gemv, three-token batch gemvs bit-identical to single-vector ones, Q8_0/legacy/K-quant dequant, both embedding-row lookups, MoE top-k, attention spans, blocked attention with F32/BF16 K/V, sliding-window rings and parallel decode attention, embedding pooling, the tokenizer's user-defined tokens, pre-tokenizer splits and chat-format detection, chat turns with thinking on and off and the split of a reply into reasoning and answer, the Gated DeltaNet kernels, and a synthetic Qwen3.5 model's batching and recurrent-state handling).
- JIT numerical tests compare against a scalar reference within a relative tolerance (start at `1e-4`); FMA rounding will differ from naive scalar order — that is expected, not a bug.
- Tests must not require model files. Generate synthetic inputs (a tiny GGUF in the temp directory is fine).

Current test count: **63 tests, all passing**.

## Design invariants worth restating

These are the ones most easily broken by a well-meaning change. The reasoning
behind each lives in [docs/design.md](docs/design.md).

- **No computation graph.** Call kernels directly from `src/libnnc/model.cpp`.
- **`vzeroupper` before every `ret`** in emitted code. Non-negotiable.
- **Kernels assume the Win64 ABI**; every builder starts with
  `emit_win64_arg_shuffle(n_int_args)` so Linux works with the same encoders.
- **Decode is DRAM-bandwidth bound, against a *measured* roof.** This
  machine tops out near 28 GB/s and reaches 25.3 GB/s on a **single**
  thread, so threads are there to cover dequantisation, not to accumulate
  bandwidth. A kernel that needs many threads to approach the roof has a
  kernel problem. Check `--perf` before and after, and compare GB/s per
  phase.
- **Benchmark by interleaving the two binaries run-by-run, with an
  untouched control** (`--bf16` shares no kernel with Q8_0/Q4). Session-to-
  session variance is ~20%, which is bigger than most wins worth having;
  measuring all of A then all of B will invent results.
- **Every gemv kernel uses 4 independent FMA accumulators.** One
  accumulator chains four 4-cycle FMAs per block and caps a thread at
  ~5 GB/s. To afford the registers, x is a memory operand on the FMA and
  Q4 isolates low nibbles with a `vpslld 28`/`vpsrld 28` pair rather than
  holding a `0x0F` mask. Kernels needing `ymm8`–`ymm15` use the `g_*`
  encoders and save `xmm6`–`xmm15` (Win64 callee-saved).
- **Block metadata (Q8_0 / Q4 scales and biases) is BF16, never FP16.** A
  block scale can be arbitrarily small; FP16 would flush it to subnormal.
  BF16 also decodes with `vpbroadcastw` + `vpslld 16`, needing no F16C.
- **Query heads sharing a KV head are processed together** so the K/V cache
  is walked once per KV head, not once per query head.
- **Prefill is compute-bound** — the opposite regime, since a batch of 16
  tokens shares each weight pass. Do not assume a change helps both.
- **`layer_forward` is one code path for decode and prefill** (`n_tok` 1 or
  16), for embedding models (bidirectional, the whole sequence as one batch)
  and for MTP drafters (queries over the target's cache via `kv_ext`). Do
  not fork it; a batched copy will drift.
- **Attention spans come only from `model_attn_span`.** Causal, sliding,
  bidirectional and drafter spans are all one function, tested on its own.
- **MTP must not change the output.** Verification samples the target's own
  token at every position from logits computed with the same bits as plain
  decoding, so with Q8_0/Q4 weights greedy and seeded output are
  identical with or without a drafter. Check that after any change to the
  session loop, the batched lm_head, the batch gemv kernels or attention.
- **A batch row gives the bits decoding gives.** The `_3tok` prefill
  kernels follow their 1-row kernel's operation order exactly, and
  attention works in 256-key blocks aligned to absolute positions, merged
  in order, whichever thread or path runs them. A "faster" reduction order
  in either breaks MTP; keep the `memcmp` tests.
- **Moving `cur_pos` back goes through `model_kv_rewind`.** A sliding
  layer's buffer is a ring of `round_up(window + n_batch, 256)` rows, so
  keys more than `max_rewind` behind `kv_written` are gone; when the
  rewind is refused, start again from 0.
- **A recurrent state is restored, never cut back.** Gated DeltaNet layers
  (Qwen3.5) keep a state that sums up the whole history, so it describes
  exactly `cur_pos` tokens: `model_forward` at position 0 clears it, and
  anything that moves `cur_pos` back must restore a saved copy
  (`model_state_restore`) or start again from 0, never just lower `cur_pos`
  or re-evaluate a token the state has already taken in.
- **Forward scratch lives in the session cache** and is sized once by
  `model_kv_init`; preserve the no-reallocation hot path after setup.
- **`nnc_gemv_pool` thread count is capped at `MAX_THREADS`** because dispatch
  and the argmax reduction index fixed-size stack arrays by thread id.
- **Packed weights are decoded exactly once, at load**, straight into the
  target layout — never via an intermediate whole-tensor BF16 or F32 image.
- **`src/libnnc/baseline.cpp` is built without AVX flags** and holds the CPU
  check plus every public function that needs no model, so an old CPU gets
  `NNC_ERR_UNSUPPORTED_CPU` rather than a crash. Keep templates, standard
  containers and other headers' inline functions out of it (the linker may
  keep their AVX-compiled copies), and keep `/GL` off libnnc.

## Runtime (`src/libnnc/runtime.cpp` / `runtime.h`)

The assert handler, dtype table, tensor descriptors and sizes, timers, and the
`--perf` phase profiler. Nothing else: errors and logging live in
`baseline.cpp`, which must run before the CPU check. The Q8_0 split layout, the
load-time decode path and the `token_embd` handling are described in
[docs/design.md §3](docs/design.md#3-data-types).

Model-owned tensor descriptors live in `nnc_model::tensors`; their weight
bytes live in the GGUF mapping or in model-owned decode buffers. Preserve that
ownership split.

## JIT design rules

Full detail (kernel list, encodings, W^X pool, cache) is in
[docs/design.md §4](docs/design.md#4-the-jit). The rules for changing it:

- **ABI:** kernels assume Windows x64 (int args in RCX, RDX, R8, R9). Every builder calls `x64_emitter::emit_win64_arg_shuffle(n_int_args)` as its first instruction — a no-op on Windows, the SysV→Win64 moves on Linux. Document register usage at the top of each builder. Preserve every nonvolatile GPR and the low 128 bits of XMM6–XMM15 if used. Kernels make no calls, so outgoing shadow space is not needed.
- **`vzeroupper` before every `ret`.** Non-negotiable.
- **No third-party JIT libs.** Hand-rolled encoders only.
- **Specialization:** bake shape constants as 32-bit immediates. Use 4-accumulator FMA unrolling whenever `cols % 32 == 0` to break the FMA latency chain.
- **Fallback:** anything off the JIT fast path runs through the scalar tail in the corresponding `nn_ops.cpp` wrapper. Keep that tail correct — it is what the tests compare against.
- Add a numerical test against a scalar reference for every new kernel.

## Before claiming a performance win

Decode is DRAM-bandwidth bound (see
[docs/design.md §7](docs/design.md#7-profiler)). Run `--perf` before and
after and compare the effective GB/s per phase, not just wall time. A change
that removes instructions without removing bytes read will measure as noise.

## Build & run (reference)

One CMake + Ninja build for both platforms. Presets `release` and `debug`;
both write their binary to `exe/`, and the build trees are kept apart per
host OS (`build/Windows-release`, `build/Linux-debug`, …) so a Windows and a
WSL build of the same tree can coexist.

```bash
cmake --preset release && cmake --build --preset release   # -> exe/nnc[.exe]
cmake --preset debug   && cmake --build --preset debug     # -> exe/nnc-d[.exe]
ctest --preset debug                                       # runs --test
```

`dd.ps1` is the Windows dev driver — it enters the VS x64 environment (Ninja
calls `cl.exe` directly, so this is required) before invoking CMake:

```powershell
.\dd.ps1 build                        # build Debug and Release
.\dd.ps1 test                         # build, then run the self-tests
.\dd.ps1 run -- --help                # run nnc with arguments (stdin is closed)
.\dd.ps1 download --dry-run           # list the downloadable models
.\dd.ps1 download --name all --yes    # fetch them all into .\models
```

### Windows (raw CMake)

Run from a **x64 Native Tools Command Prompt** or a VS Developer PowerShell:

```powershell
cmake --preset debug
cmake --build --preset debug
.\exe\nnc-d.exe --test
```

### Linux / WSL

```bash
sudo apt-get install -y cmake ninja-build g++
cmake --preset debug && cmake --build --preset debug
./exe/nnc-d --test
```

CI (`.github/workflows/linux.yml`, `windows.yml`) builds both presets and
runs `--test` on every push and PR to `master`; the Linux job also builds
the C examples so `nnc.h` stays valid C. Keep both green.

## House style

- C++20, exceptions allowed but not used in hot paths.
- **Asserts:** use `NNC_ASSERT(expr)` from `runtime.h`. It prints `nnc: ASSERT failed: <expr> at <file>:<line>` to stderr and exits via `std::_Exit` — no modal message box. Plain `assert(...)` is acceptable inside the emitters for build-time invariants but `NNC_ASSERT` is preferred in any new runtime / op code.
- Tabs for indentation in existing files (matches current code); do not reformat unrelated lines.
- New files: same brace style as `main.cpp` (Allman, tabs).
- No new dependencies without an explicit ask. The project must build with only MSVC + the Windows SDK on Windows, and only g++/clang + libstdc++ on Linux.
- Do not add docstrings, comments, or refactors to code you are not otherwise changing.
- When concatenating C/C++ files on Windows, strip any 0x1A (Ctrl+Z) bytes — MSVC treats them as EOF in text mode.
- Standard-library portability: explicitly include `<cstddef>` for `size_t`, `<cstring>` for `memcpy`/`memset`, `<cmath>` for math — don't rely on transitive includes that happen to work on MSVC.

## dd build system

This repository vendors the [dd](https://github.com/ZacWalk/dd) build driver as
plain upstream files: `dd.ps1` plus `.dd/`, with hashes recorded in
[docs/dd-upstream.json](docs/dd-upstream.json). Project settings live in
`dd.psd1`, dependency pins live in `cmake/dd-dependencies.json` with
`dependencies.owner = 'dd'`, and `.\dd.ps1 commands --json` lists nnc's
project-specific commands. dd's
[two modes](https://github.com/ZacWalk/dd/blob/main/docs/mcp.md#two-modes)
and [commands](https://github.com/ZacWalk/dd/blob/main/docs/cli.md#commands)
are documented in its own repository.

For normal verification:

```pwsh
.\dd.ps1 doctor --json
.\dd.ps1 build
.\dd.ps1 test --label nnc
```

### Project commands

`dd download` fetches GGUF weight files into `./models`. It declares
`effects = 'write'`, so dd requires an explicit opt-in:

```pwsh
.\dd.ps1 download --dry-run              # list the known models
.\dd.ps1 download --name all --dry-run   # plan a fetch
.\dd.ps1 download --name gemma-3-1b-q6k --yes
```
