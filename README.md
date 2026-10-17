# Run LLMs on your CPU

[![Linux](https://github.com/ZacWalk/nnc/actions/workflows/linux.yml/badge.svg)](https://github.com/ZacWalk/nnc/actions/workflows/linux.yml)
[![Windows](https://github.com/ZacWalk/nnc/actions/workflows/windows.yml/badge.svg)](https://github.com/ZacWalk/nnc/actions/workflows/windows.yml)

**nnc** runs large language models (LLMs), the AI models behind today's
chatbots, on your computer's ordinary processor. It is a C library you can
embed in your own apps, and it comes with a command-line app so you can use
it straight away. You don't need a graphics card, a cloud account or Python:
download a model once, then chat with it, have it write code, translate text
or search documents by meaning, all on your own PC. It works offline, and
nothing you type leaves your machine.

![nnc prompt demo](ncc-prompt.gif)

- **What you need:** a 64-bit Intel or AMD processor with AVX2 (most made
  since 2013), Windows or Linux, and enough free memory for the model: under
  1 GB for a small one, about 2–4 GB for a capable 4B model. See
  [requirements](#requirements).
- **How fast it is:** on a desktop PC, a 1-billion-parameter model writes
  23–45 tokens a second (a token is about three-quarters of a word), faster
  than you can read; 4B models manage 6–12 tokens a second, and a 27B model
  1–2. See [choosing a model](#choosing-a-model).
- **Why a CPU can do it:** writing each token means reading every weight of
  the model from memory once, so the speed is set mostly by memory
  bandwidth, not by raw arithmetic. nnc shrinks the weights to 8 or 4 bits
  as it loads them, and runs them through machine code generated for your
  processor's vector instructions, which keeps the CPU up with its memory.

nnc comes in two parts:

- **`libnnc`**, the library to embed in your own apps: a static library with
  a plain-C API ([nnc.h](src/libnnc/nnc.h)) that works from C, C++ and any
  language that can call C. Link it in, load a model and talk to it from your
  own code, with no server and no network. See
  [using libnnc in your own program](#using-libnnc-in-your-own-program).
- **`nnc`**, the command-line app built on it: chat with a model, translate
  text, or turn text into embeddings, offline, on your own PC.

nnc (neural net compiler) is also a learning project. I have always been
fascinated by all the memory shuffling that happens to run a model on a GPU.
What happens if you just run it on the CPU, with code optimized for its SIMD
instructions? nnc JIT-compiles the hot operations of a transformer into
machine code tuned for the host CPU, then uses them to run **Gemma, Llama and
Qwen inference end-to-end** on its own minimal tensor runtime. No GPU, no
Python, no ML framework and no third-party JIT: just a C++20 compiler and
hand-rolled x86-64 AVX2/FMA encoders, on Windows x64 and Linux x64 (including
WSL).

## Requirements

- **CPU:** x86-64 with AVX2 and FMA: Intel Haswell (4th-generation Core, 2013)
  or newer, or AMD Excavator / any Ryzen. Some Pentium, Celeron and Atom chips
  lack AVX2; nnc checks at startup and tells you. ARM (including Apple
  silicon) is not supported, and no GPU is used.
- **OS:** 64-bit Windows 10 or 11, or 64-bit Linux (WSL2 works).
- **Memory:** under 1 GB for the small models, up to 16–24 GB for the largest;
  see [choosing a model](#choosing-a-model).
- **To build on Windows:** Visual Studio 2022 (any edition, or its Build
  Tools) with the *Desktop development with C++* workload, CMake 3.24+, Ninja
  and Git (Visual Studio's *C++ CMake tools* component provides CMake and
  Ninja), and [PowerShell 7.4+](https://learn.microsoft.com/powershell/scripting/install/installing-powershell-on-windows)
  for the `dd.ps1` build driver. `.\dd.ps1 doctor` checks all of these, and
  `.\dd.ps1 toolchain --yes`, run in an elevated PowerShell, installs any that
  are missing with winget.
- **To build on Linux:** g++ 10+ or clang++ 12+, CMake 3.21+ and Ninja, plus
  `curl` or `wget` for model downloads.

## Quick start

Windows (in PowerShell 7):

```powershell
git clone https://github.com/ZacWalk/nnc
cd nnc
.\dd.ps1 build      # Release and Debug builds, into .\exe
.\exe\nnc.exe       # offers to download a model, then chats
```

Linux or WSL:

```bash
sudo apt-get install -y git cmake ninja-build g++ curl
git clone https://github.com/ZacWalk/nnc && cd nnc
cmake --preset release && cmake --build --preset release
./exe/nnc           # offers to download a model, then chats
```

With no model on disk, nnc shows its download catalogue and asks before
downloading anything. The first entry, Gemma 3 1B (1 GB), is the
recommended first model. Then type a message at the `>` prompt. Ctrl-C stops
a reply, `/reset` starts a new conversation and `/exit` quits.

`nnc --test` runs the self-tests (they need no model), which is a quick
check that the build works on your machine.

## Choosing a model

Pass any of these names to `-m` (nnc offers to download it if needed), or
download one with `nnc --download NAME`:

| name | download | default: RAM, speed | `--q4`: RAM, speed | notes |
| --- | --- | --- | --- | --- |
| `gemma-3-270m-q8` | 0.3 GB | 0.3 GB, 97 tok/s | 0.2 GB, 159 tok/s | tiny; fine for simple tasks |
| `gemma-3-1b-q6k` | 1.0 GB | 1.0 GB, 28 tok/s | 0.6 GB, 45 tok/s | **start here**: small, quick, capable |
| `llama-3.2-1b-q6k` | 1.0 GB | 1.2 GB, 23 tok/s | 0.7 GB, 38 tok/s | Meta's small model |
| `qwen3.5-0.8b-q8` | 0.8 GB | 0.7 GB, 35 tok/s | 0.4 GB, 52 tok/s | Qwen's small hybrid model |
| `gemma-4-e2b-q4km` | 3.1 GB | 4.6 GB, 13 tok/s | 2.7 GB, 21 tok/s | much smarter; faster still with its drafter |
| `gemma-3-4b-q4km` | 2.5 GB | 3.8 GB, 7.6 tok/s | 2.3 GB, 12 tok/s | |
| `qwen3-4b-2507-q4km` | 2.5 GB | 4.0 GB, 6.9 tok/s | 2.3 GB, 11 tok/s | |
| `qwen3.5-4b-q4km` | 2.7 GB | 4.2 GB, 6.0 tok/s | 2.5 GB, 10 tok/s | |
| `gemma-4-e4b-q4km` | 5.0 GB | 7.4 GB, 6.1 tok/s | 4.3 GB, 9.9 tok/s | |
| `gemma-4-26b-a4b-q4km` | 16.8 GB | ~27 GB | 14.7 GB, 9.2 tok/s | mixture of experts: a 26B model at 4B speed; use `--q4` |
| `qwen3.8-27b-q4km` | 19.0 GB | 27 GB, 1.1 tok/s | 16 GB, 1.8 tok/s | the strongest here, and slow; use `--q4` |
| `translategemma-4b-q4km` | 2.5 GB | 3.8 GB, 7.6 tok/s | 2.3 GB, 12 tok/s | translation: `--translate en:de` |
| `functiongemma-270m-q8` | 0.3 GB | 0.3 GB, 102 tok/s | 0.2 GB, 165 tok/s | function calling ([examples/function_call.c](examples/function_call.c)) |
| `embeddinggemma-300m-q8` | 0.3 GB | 0.3 GB | | text embeddings, texts up to 2K tokens |
| `embeddinggemma-2-q8` | 0.3 GB | 0.3 GB | | text embeddings, texts up to 8K tokens |
| `mtp-gemma-4-e2b-q8`, `mtp-gemma-4-e4b-q8`, `mtp-gemma-4-26b-a4b-q8` | 0.1–0.5 GB | | | drafters, fetched with `--download` rather than run: the Gemma 4 model of the same name then replies 1.2–2× faster, with the same text |

- **Speed** is the reply speed measured on a Ryzen 9 5900XT with dual-channel
  DDR4-2400. Generating a token reads every weight once, so speed follows
  your memory bandwidth more than your core count.
- **RAM** is the weights as nnc holds them. nnc re-encodes the weights when it
  loads them, to 8-bit by default, so a 4-bit download takes about 1.5× its
  file size in RAM. `--q4` keeps it near the file size and is about 1.6×
  faster, at some cost in quality.
- **The conversation's K/V cache** comes on top of the RAM figure, and `-v`
  prints its size. At the default 4096-token context it is, for example,
  about 33 MB for Gemma 3 1B and Gemma 4 E2B, 128 MB for Llama 3.2 1B,
  225 MB for Gemma 3 4B and 576 MB for Qwen3 4B. It grows with `-c`, apart
  from Gemma's sliding-window layers and Qwen 3.5's recurrent layers, which
  stay the same size.
- **Other GGUF files** of a supported architecture work too: pass the path to
  `-m`. That covers Gemma 3 and 4 (with FunctionGemma, TranslateGemma and
  EmbeddingGemma 1 and 2), Llama 3, and Qwen2, Qwen2.5, Qwen3 and Qwen 3.5,
  3.6 and 3.8. The common Q4_K_M, Q6_K, Q8_0, F16 and BF16 files load;
  IQ-quantised and Q2_K/Q3_K files do not. `nnc --inspect-model FILE` tells
  you whether a file will load.
- nnc looks for models in `./models`, `~/.lmstudio/models` (so LM Studio's
  downloads are found) and `~/models`. Downloads go to `./models` when that
  directory exists, and to `~/models` otherwise.

[docs/models.md](docs/models.md) has the full measurements.

## Using the app

```powershell
.\exe\nnc.exe                                       # choose a model found on disk (or download one), then chat
.\exe\nnc.exe -m gemma-4-e2b-q4km                   # a catalogue name; offers to download it if needed
.\exe\nnc.exe -m D:\models\some-model-Q4_K_M.gguf   # or any GGUF file
.\exe\nnc.exe -m qwen3.5-4b-q4km -p "Write a haiku about memory bandwidth"   # one reply, to stdout
.\exe\nnc.exe -m gemma-4-e4b-q4km --think -c 16384     # reason before answering: better at code and maths
.\exe\nnc.exe -m translategemma-4b-q4km --translate en:de -p "Good morning!"
.\exe\nnc.exe -m embeddinggemma-300m-q8             # type sentences, see how alike they are
Get-Content docs.txt | .\exe\nnc.exe -m embeddinggemma-300m-q8 --task document > vectors.jsonl
```

On Linux the binary is `./exe/nnc`; everything else is the same.

In the chat, Ctrl-C stops a reply. `/reset` starts a new conversation,
`/translate en:de` translates each message instead of answering it
(`/translate off` to chat again), `/help` lists the commands and `/exit` quits.
An embedding model gets its own mode instead: typed lines are compared with
the ones before, and piped lines print one JSON vector each.

The options you are most likely to want (`--help` lists them all):

| option | what it does |
| --- | --- |
| `-m FILE\|NAME` | the model: a `.gguf` path, or a name from the table above |
| `-p TEXT` | answer one message, print the reply to stdout and exit |
| `-sys TEXT` | instructions for the model, given before the first message |
| `-c N` | the context window in tokens (default: up to 4096) |
| `-n N` | the most tokens per reply (default: until the reply ends) |
| `--temp X`, `-s N` | sampling temperature (default 0.7; 0 always picks the likeliest token), and a seed for repeatable replies |
| `--q4`, `--bf16` | 4-bit weights (less RAM, faster, lower quality), or 16-bit weights (more RAM, slower) |
| `--kv-f32` | keep the K/V cache in F32 rather than BF16, at twice the memory |
| `--think` | let the model reason before it answers (Gemma 4; Qwen 3, 3.5, 3.6, 3.8). It is better at hard problems but writes many more tokens; the reasoning shows dimmed on stderr and takes up context, so a larger `-c` helps |
| `-t N` | worker threads (default: `$NNC_THREADS`, else up to 8) |
| `--no-draft` | don't use a Gemma 4 drafter (`--draft FILE` picks one explicitly) |
| `--perf`, `-v` | a timing breakdown after each reply; what the library is doing |
| `--download [NAME]`, `--list-models` | download a model; list the models nnc can see |
| `-y` | download without asking |

For looking inside models there are `--inspect-model`, `--inspect-all`,
`--gguf-info`, `--gguf-stats`, `--gemma-info`, `--gemma-probe`,
`--gemma-forward`, `--gemma-tokenize`, `--gemma-prompt`, `--gemma-gen` and
`--perplexity`; `--help` gives their arguments.

## Using libnnc in your own program

### Adding it to your build

With CMake, add nnc as a subdirectory. The app is not built when nnc is a
subproject:

```cmake
add_subdirectory(path/to/nnc)
target_link_libraries(myapp PRIVATE nnc::nnc)
```

Or install it once and use `find_package`:

```powershell
cmake --preset release
cmake --build --preset release --target libnnc
cmake --install build/Windows-release --prefix C:/path/to/nnc-install   # build/Linux-release on Linux
```

```cmake
find_package(nnc CONFIG REQUIRED)
target_link_libraries(myapp PRIVATE nnc::nnc)
```

The library is C++ inside, so link a C program with the C++ linker (in
CMake, `set_target_properties(myapp PROPERTIES LINKER_LANGUAGE CXX)`). The
`nnc::nnc` target brings its own dependencies. Without CMake, link `nnc.lib`
with `winhttp.lib` and `shell32.lib` on Windows, or `libnnc.a` with
`-pthread` on Linux. Your own code needs no AVX2 compiler flags.
`nnc_cpu_supported()` tells you at run time whether the CPU can run nnc, so
your program can fall back gracefully.

### A complete program

```c
#include "nnc.h"
#include <stdio.h>

/* Receives the reply as it is generated, in whole UTF-8 characters. */
static bool print_text(const char* text, size_t length, void* user)
{
	(void)user;
	fwrite(text, 1, length, stdout);
	fflush(stdout);
	return true; /* false would stop the reply */
}

int main(void)
{
	nnc_model* model = NULL;
	if (nnc_model_load("models/gemma-3-1b-it-Q6_K.gguf", NULL, &model) != NNC_OK)
	{
		fprintf(stderr, "cannot load the model: %s\n", nnc_last_error());
		return 1;
	}

	nnc_session_params sp = nnc_session_default_params();
	sp.system_prompt = "Answer in two sentences.";
	nnc_session* chat = NULL;
	if (nnc_session_create(model, &sp, &chat) != NNC_OK)
	{
		fprintf(stderr, "cannot start a session: %s\n", nnc_last_error());
		nnc_model_free(model);
		return 1;
	}

	nnc_generate_params gen = nnc_generate_default_params();
	gen.on_text = print_text;
	nnc_session_chat(chat, "Why is the sky blue?", &gen, NULL);
	printf("\n");
	nnc_session_chat(chat, "Then why are sunsets red?", &gen, NULL); /* the session remembers the conversation */
	printf("\n");

	nnc_session_free(chat);
	nnc_model_free(model);
	return 0;
}
```

[examples/](examples) has three complete programs:

- [minimal.c](examples/minimal.c): one chat message.
- [embed.c](examples/embed.c): semantic search with an embedding model.
- [function_call.c](examples/function_call.c): a FunctionGemma call and its result.

Configure with `-DNNC_BUILD_EXAMPLES=ON` to build them as `nnc-minimal`,
`nnc-embed` and `nnc-function-call`.

### The API at a glance

| functions | for |
| --- | --- |
| `nnc_model_load`, `nnc_model_free`, `nnc_model_get_info` | loading a GGUF file; `nnc_model_params` picks the weight format and a progress callback |
| `nnc_session_create`, `nnc_session_free`, `nnc_session_reset` | a conversation; `nnc_session_params` sets the context size, seed, system prompt, drafter, K/V type and thinking |
| `nnc_session_chat` | one chat turn in the model's own template: the answer streams to `on_text` and, with thinking on, the reasoning to `on_thinking` |
| `nnc_session_translate` | one translation turn (TranslateGemma, or any chat model) |
| `nnc_session_generate`, `nnc_session_eval`, `nnc_session_logits`, `nnc_session_sample` | working with raw tokens: completion, scoring, your own sampling |
| `nnc_tokenize`, `nnc_token_to_piece`, `nnc_token_is_eog` | the tokenizer |
| `nnc_embed` | a unit-length embedding vector from an embedding model; no session needed |
| `nnc_session_cancel` | stopping a running call, from another thread or a signal handler |
| `nnc_set_num_threads`, `nnc_set_log_callback`, `nnc_last_error`, `nnc_cpu_supported` | process-wide settings and error details |

The rules that matter:

- A loaded model is read-only. Any number of sessions on any threads can
  share it, and it must outlive them, as must a session's drafter.
- Drive each session from one thread at a time. `nnc_session_cancel()` is
  the exception, and `nnc_embed()` may run on any number of threads at once.
- Failures are reported through return values (an `nnc_status`, or -1,
  NULL or `NNC_TOKEN_NULL`), and `nnc_last_error()` describes the last one
  on the calling thread.
- Strings are UTF-8, including file paths on Windows. Streamed text always
  arrives in whole characters.
- All sessions share one worker pool, so two sessions running at once is
  safe but not faster.

[nnc.h](src/libnnc/nnc.h) documents every function. It is the only public
header; the other headers in `src/libnnc` are internal and may change.

## Getting the best speed

- **Generation is limited by memory bandwidth.** Each new token reads every
  weight once, so a smaller model or `--q4` is faster. Dual-channel memory
  matters more than core count, and more than about 8 threads rarely helps.
- **Gemma 4 has drafters.** With `mtp-gemma-4-e2b-q8` (or the E4B or 26B one)
  downloaded, nnc picks it up automatically. Replies come 1.2–2× faster, with
  exactly the same text.
- **Prompts are cheaper than replies.** nnc reads a prompt 16 tokens per pass
  over the weights, so a long pasted document costs far less per token than
  the reply.
- **Thinking is paid for at reply speed.** With `--think`, a model may write
  hundreds or thousands of tokens of reasoning before its answer. On a CPU
  that takes minutes for the larger models, so turn it on for the questions
  that need it.
- **Long contexts are cheapest on Gemma.** Each reply token reads the K/V
  cache as well as the weights. The cache is BF16, and Gemma's
  sliding-window layers keep only their window, so Gemma 3 1B holds an 8K
  context (`-c 8192`) in under 50 MB. An all-attention model such as Qwen3
  4B needs over 1 GB for the same.
- **Load from a local disk.** Loading decodes the weights once, which takes a
  second or two for a small model on an SSD. Under WSL, keep models on the
  Linux file system: reading them from `/mnt/c` is several times slower.
- `--perf` shows where the time goes, phase by phase, with the bandwidth each
  phase reached.

## Limitations

- x86-64 CPUs with AVX2 and FMA only: no GPU, ARM or macOS support.
- Text only. Vision projector files (`mmproj-*`) are skipped, so models take
  no image input.
- Thinking is off unless you ask for it (`--think`, or
  `nnc_session_params.thinking`), and a model's earlier reasoning stays in the
  context rather than being dropped between turns.
- Qwen3.8-Flash-Next, a 125B mixture of experts with n-gram embeddings, is
  not supported.
- One conversation per session. There is no server mode: nnc is a library
  and a command-line app.

## Troubleshooting

| problem | what to do |
| --- | --- |
| "this CPU is not supported" | The CPU lacks AVX2 or FMA (see [requirements](#requirements)), so nnc cannot run on it. |
| A model will not load | `nnc --inspect-model FILE` says why. Prefer a Q4_K_M, Q6_K or Q8_0 file; IQ-quantised and Q2_K/Q3_K ones are not supported. |
| The model forgot the conversation | It filled the context window (4096 tokens by default), so nnc started afresh. Use a larger `-c`. |
| Replies are slow | Try `--q4` or a smaller model, close memory-hungry programs, and look at `--perf`. |
| `dd.ps1` will not run | It needs PowerShell 7.4+ (`pwsh`, not Windows PowerShell 5.1). A repository downloaded as a ZIP may need `Get-ChildItem -Recurse \| Unblock-File` first. |
| The Windows build cannot find the compiler | `.\dd.ps1 doctor` lists what is missing. Without dd, run CMake from an *x64 Native Tools Command Prompt*, because Ninja calls `cl.exe` directly. |

## How it works

- **Loading.** The GGUF file is memory-mapped and parsed in place. Packed
  weights (F16, BF16, Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 and Q4_K / Q5_K /
  Q6_K) are decoded once, at load time, into nnc's own layout: 8-bit with
  BF16 block scales by default, 4-bit under `--q4`, or BF16 under `--bf16`.
  F32 tensors, and BF16 ones under `--bf16`, are used in place.
- **The JIT.** Matrix-vector kernels are emitted as x86-64 machine code
  specialised to each matrix shape, then cached and reused. Each keeps four
  independent FMA accumulators, so FMA latency does not hold back the stream
  of weights. Softmax, the norms, RoPE, soft-capping, SwiGLU and GELU are
  in-house AVX2/FMA kernels too.
- **Batched prefill.** One pass over the weights serves 16 prompt tokens, and
  the 8-bit and 4-bit kernels work through them three at a time. That makes
  prompt processing compute-bound rather than bandwidth-bound.
- **Threads.** A static worker pool (by default `min(8, CPUs)`) splits every
  large matrix by rows. It also splits attention across positions for
  prompts, embedded texts and long-context replies.
- **Compact context.** The K/V cache is BF16, and sliding-window layers keep
  a ring of their recent rows. Gemma 3 1B at an 8K context holds 48.5 MB of
  K/V, where F32 buffers of the whole context would take 416 MB.
- **Model families.** These include:
  - Gemma 4, including the 26B A4B mixture of experts (128 experts, 8 per
    token).
  - Qwen 3.5, 3.6 and 3.8, hybrids whose Gated DeltaNet layers keep a
    recurrent state, with gated attention every fourth layer.
  - EmbeddingGemma, FunctionGemma and TranslateGemma, each with the prompt
    format it was trained on.

  The Qwen models' logits, EmbeddingGemma's vectors and the tokenizer are
  checked against llama.cpp.
- **Speculative decoding.** A Gemma 4 model uses its `mtp-*.gguf` drafter
  to guess several tokens, then checks them all in one pass. With 8-bit or
  4-bit weights, the output is exactly what it would be without the drafter.
- **No computation graph.** The forward pass calls the kernels directly. All
  OS calls sit behind a small `sys.h` layer. The kernels use the Windows x64
  calling convention, with a register shuffle on Linux, so one set of
  encoders serves both platforms.

![nnc architecture](nnc-diagram.png)

[docs/design.md](docs/design.md) covers all of this in depth: the data
layouts, the JIT and its ABI, threading, the forward pass and the measured
performance.

## Building and testing

```powershell
.\dd.ps1 build        # Release and Debug
.\dd.ps1 test         # build, then run the self-tests
.\exe\nnc.exe         # run the REPL directly: dd run closes stdin
```

On Linux, or on Windows from an x64 Native Tools prompt:

```bash
cmake --preset release && cmake --build --preset release   # -> exe/nnc
cmake --preset debug && cmake --build --preset debug       # -> exe/nnc-d
ctest --preset debug                                       # runs nnc-d --test
```

Build trees live in `build/<HostSystem>-<preset>/`, so a Windows build and a
WSL build of the same working tree can coexist; the binaries land in `exe/`.
CI builds both configurations on both platforms and runs the self-tests.
[AGENTS.md](AGENTS.md) lists the conventions for contributors and coding
agents.

## Documentation

- [docs/design.md](docs/design.md): how nnc works. It covers the layering, the
  Q8_0 split layout, the JIT and its ABI, threading, the forward pass (mixture
  of experts, Gated DeltaNet, embeddings, MTP drafting, the tokenizer) and
  where the remaining headroom is.
- [docs/models.md](docs/models.md): the models run end-to-end, with measured
  speed and memory, plus the ones that do not load and why.
- [docs/nnc-diagram.py](docs/nnc-diagram.py): draws the architecture figure
  above (`python docs/nnc-diagram.py`; needs Pillow).
- [docs/nnc-demo.py](docs/nnc-demo.py): makes the animation at the top from a
  real chat. `record` runs nnc on a terminal (Linux or WSL), and `render`
  draws the GIF (needs Pillow).

## License

[MIT](LICENSE)
