# nnc — design

This is the single source of truth for how nnc is put together. `README.md`
is the front page (what it is, how to build, how to run) and `AGENTS.md`
holds the house rules for making changes; both defer here for design detail.

---

## 1. What nnc is

nnc JIT-compiles the hot operations of a transformer into machine-code blobs
specialized to the host CPU and the model's exact shapes, then runs Gemma,
Llama and Qwen GGUF inference end-to-end on its own minimal tensor runtime:
chat and completion (Gemma 3 and 4, including the 26B A4B mixture of
experts, FunctionGemma and TranslateGemma; Qwen 2, 2.5 and 3; and Qwen 3.5,
3.6 and 3.8, whose Gated DeltaNet layers carry a recurrent state), text
embeddings (EmbeddingGemma 1 and 2), and speculative decoding with Gemma 4's
multi-token-prediction drafters.

Deliberate constraints, because this is a learning project:

- **No third-party JIT library.** No AsmJit, no xbyak, no LLVM. Every byte of
  emitted code comes from the hand-rolled encoders in `emitter_x64.cpp` and
  `emitter_avx2.cpp`.
- **No ML framework, no BLAS.** All kernels are ours.
- **AVX2 + FMA only.** No AVX-512, in codegen or intrinsics. Detected at
  startup; the process refuses to run on older CPUs. The check, and every
  public function that needs no model, live in `baseline.cpp`, which is built
  without AVX flags, so on an older CPU `nnc_model_load()` fails with
  `NNC_ERR_UNSUPPORTED_CPU` instead of faulting. libnnc is built without
  `/GL`, both so an installed `nnc.lib` links with other MSVC versions and so
  link-time inlining cannot carry AVX2 code into that gate.
- **Two source components, one app binary.** The reusable library is in
  `src/libnnc`; the CLI app and its tests are in `src/appnnc`. The app binaries
  are `nnc(.exe)` release and `nnc-d(.exe)` debug.
- **Windows x64 and Linux x64 from one tree.** All OS calls sit behind
  `sys.h`, implemented twice (`sys_win.cpp`, `sys_linux.cpp`).

The OS layer also owns app services that require native APIs: console code
pages and input, UTF-8 arguments, interrupt handling, model downloads and
atomic file replacement. `appnnc/platform.cpp` adapts the app's
interrupt-scope and progress callbacks to `sys.h`; it calls no native API
itself.

## 2. Layer cake

```
appnnc: main / CLI / chat + translate / embeddings / model picker / downloader / platform UI / tests
   |
libnnc: public C API (nnc.h)
   |
model.cpp + tokenizer.cpp + session.cpp    model, tokenization, conversation
   |
nn_ops.cpp             SIMD kernels + sleeping worker pool + JIT dispatch
   |
jit_ops.cpp            specialized kernel builders
   |
emitter_avx2.cpp / emitter_x64.cpp         instruction encoders
   |
jit_buffer.cpp + sys_win.cpp / sys_linux.cpp   executable memory + OS services
```

`runtime.cpp` provides the dtype table, tensor descriptors and sizes, timers,
and the profiler; errors and logging are in `baseline.cpp`. `gguf.cpp` parses
GGUF v2/v3 directly from its mapping.

`libnnc` exposes one plain-C header, `nnc.h`; other headers under `src/libnnc`
are implementation details. It can be consumed with `add_subdirectory` or
installed for `find_package(nnc CONFIG)`. Models are immutable after load and
can be shared across sessions; each session owns its conversation and KV
cache, and is single-threaded except for cancellation. Sessions share one
process-wide worker pool. `nnc_embed()` needs no session: it reads the model
and allocates its own working memory, so any number of threads may embed at
once. The plain-C examples show the API end to end: `examples/minimal.c` (one
chat message), `examples/embed.c` (semantic search) and
`examples/function_call.c` (a FunctionGemma call and its result).

When no model is present, the app offers its download catalogue interactively.
It searches `./models`, `~/.lmstudio/models` and `~/models`; downloads go to
`./models` if it exists, otherwise `~/models` (or `./models` if no home
directory is available). Downloads are checked for exact size and GGUF magic,
and interrupted `.part` files can be resumed. The picker leaves out vision
projectors (`mmproj-*`) and drafters (`mtp-*`, `dflash-*`), which cannot run
on their own; a Gemma 4 model picks up its `mtp-<model>*.gguf` drafter
automatically when one is beside it or in those folders, and the chat
suggests the catalogue entry when one is not.

The app adapts to the model it loaded: a text-embedding model gets the
embedding mode (`embed.cpp`: a similarity explorer when typing, one JSON
array per line when piped), TranslateGemma gets a hint for `--translate`, and
everything else gets the chat REPL.

**There is no computation graph.** `model.cpp` calls kernels directly, in
order, against scratch buffers pre-allocated in the session's KV cache. Do
not add a graph or op-dispatch layer without a concrete need.

## 3. Data types

| enum | from the file | in memory | used for |
| --- | --- | --- | --- |
| `NNC_TYPE_F32` | F32, else decoded | mmap in place, else loader-owned buffer | every 1-D tensor (norms, biases, RoPE factors), and small matrices whose precision steers the model: the MoE router, Gated DeltaNet's decay and write-strength rows and its convolution taps |
| `NNC_TYPE_BF16` | BF16, else decoded | mmap in place, else loader-owned buffer | weights when `--bf16`; matrices whose rows are not whole 32-element blocks |
| `NNC_TYPE_Q8_0` | — | loader-owned buffer | weights by default |
| `NNC_TYPE_Q4_S` | — | loader-owned buffer | weights when `--q4` |

F16 tensors are always decoded: `NNC_TYPE_F16` is in the dtype table, but the
loader never produces it.

Activations and the residual stream are always F32. The KV cache is BF16 by
default (`nnc_session_params.kv_type`, `--kv-f32` for F32): keys and values
are rounded once as they are appended and widened with the same
`vpmovzxwd` + `vpslld 16` pair as BF16 weights when attention reads them.
That halves the cache and attention's read traffic. Against llama.cpp's
logits for Qwen3.5-0.8B, the mean KL divergence is 1.2e-6 with a BF16 cache
(1e-12 with F32), and every top-1 token is the same. Embedding models keep an
F32 cache — the whole sequence is one batch there, so the cache is scratch,
not state.

Bytes per weight, which is the only number that matters for decode:

| format | layout per 32 weights | bytes/weight |
| --- | --- | --- |
| BF16 | 64 B | 2.0 |
| Q8_0 | 32 B quants + 2 B scale | 1.0625 |
| Q4_S | 16 B nibbles + 2 B scale + 2 B bias | 0.625 |

**All block metadata is BF16, not FP32 or FP16.** FP32 would cost 11% of
Q8_0's bytes and 33% of Q4's. FP16 would be the obvious replacement,
but a block scale can be arbitrarily small — K-quant super-block scales
routinely land in FP16's subnormal range, which is exactly what the
`fp16_subnormal_dequant` regression test exists to catch. BF16 keeps FP32's
exponent range so it cannot flush, and it decodes with two instructions
(`vpbroadcastw` + `vpslld 16`) rather than needing F16C.

BF16 carries 8 significand bits, so a scale is exact to ~0.4%. Against Q4's
3.3% quantization step that is invisible, and even for Q8_0 the per-block
scale error is random across blocks and averages out over a dot product
— a Q8_0 model with BF16 scales continues a prompt token for token as one
with F32 scales does.

### Q8_0 split layout (nnc-specific)

Standard ggml Q8_0 interleaves a fp16 scale with each 32-byte quant block.
nnc instead keeps one allocation per tensor holding

```
int8  qs[rows * cols]                 <- tensor->data points here
bf16  scales[rows * cols / 32]        <- at (uint8_t*)data + rows*cols
```

Row-major, one BF16 scale per 32-element block. The split keeps the JIT
inner loop free of per-block unpacking: it can broadcast the scale and
stream `qs` contiguously.

Quantization is absmax per block: `scale = max(|w|)/127`,
`q = round(w/scale)` clamped to `[-127, 127]`. The scale is rounded to
BF16 *before* the values are quantized against it, so the encoder targets
exactly the number the kernel will multiply by.

### Q4_S split layout

Same idea one step further, and asymmetric because 4 bits cannot afford to
waste a sign:

```
uint8 qs[rows * cols / 2]             <- tensor->data; nibbles
bf16  scales[rows * cols / 32]        <- at data + rows*cols/2
bf16  biases[rows * cols / 32]        <- after scales
```

Reconstruction is `w = scale * q + bias` with `q` an unsigned nibble in
`[0, 15]`; the quantizer sets `scale = (max-min)/15`, `bias = min` per
32-element block. An absmax/symmetric scheme would throw away a bit of
range for no saving.

Nibble order inside a block: byte `i` holds element `i` in its low nibble
and element `i+16` in its high nibble. That is what lets one `vpmovzxbd`
and a shift or two (`vpslld 28` + `vpsrld 28` for the low nibble, `vpsrld 4`
for the high) yield two independent 8-element groups with no shuffles.

The bias term is deliberately **not** in the JIT kernel. Expanding the dot
product,

```
sum_k w_k x_k = sum_b [ scale_b * sum_{k in b} q_k x_k  +  bias_b * S_b ]
                                                            ^^^^^^^^^^^
         where S_b = sum_{k in b} x[k] depends only on x, not on the row
```

so the whole correction is one f32 dot of the row's biases against a
`cols/32`-element prefix computed once per gemv (`nnc_block_sums_f32`).
That keeps the kernel to four arguments and off the awkward path of
horizontally reducing per block.

Quality: 4-bit output stays fluent and on-topic but does diverge from the
Q8_0 token stream within a few tokens. It is a real accuracy trade, not a
free win — which is why Q8_0 remains the default.

### Load-time decode

Packed on-disk types are decoded exactly once, at load, into the Q8_0 split
layout (default), Q4_S (`--q4`) or BF16 (`--bf16`):

| ggml type | block | path |
| --- | --- | --- |
| F16 (1) | 1 elem / 2 B | → F32 → nnc Q8_0, Q4_S or BF16 |
| Q4_0, Q4_1 (2, 3) | 32 elems / 18, 20 B | → F32 → nnc Q8_0, Q4_S or BF16 |
| Q5_0, Q5_1 (6, 7) | 32 elems / 22, 24 B | → F32 → nnc Q8_0, Q4_S or BF16 |
| Q8_0 (8) | 32 elems / 34 B | → F32 → nnc Q8_0, Q4_S or BF16 |
| Q4_K (12) | 256 elems / 144 B | → F32 → nnc Q8_0, Q4_S or BF16 |
| Q5_K (13) | 256 elems / 176 B | → F32 → nnc Q8_0, Q4_S or BF16 |
| Q6_K (14) | 256 elems / 210 B | → F32 → nnc Q8_0, Q4_S or BF16 |

The 32-element types matter even in K-quant files: a matrix whose rows are
not whole 256-element super-blocks cannot be K-quantized, so converters fall
back to them (Gemma 4 26B A4B's 704- and 2112-wide expert and MLP rows are
Q5_0 and Q8_0 inside a Q4_K_M file).

The decode runs in 16384-element chunks (a multiple of both 32 and 256, so a
source block never straddles a chunk). This matters: a whole-tensor F32
staging buffer for `token_embd` on a 4B model is ~2.7 GB.

Crucially the intermediate BF16 image is **never materialised** when
quantizing — the chunk goes F32 → Q8_0 directly. Loading a 2.3 GB Q4_K_M
4B model peaks at about 6.5 GB resident with Q8_0 weights and 4.9 GB with
`--q4`, the file's mapped pages included; staging a whole-tensor BF16 image
first would take about 13 GB.

BF16 tensors are mapped in place only under `--bf16`. Otherwise they take the
same chunked decode into Q8_0 or Q4_S as every other source type, so there is
no separate quantization pass.

### `token_embd` is quantized too

`token_embd` doubles as the input embedding table and (unless the model ships
a separate `output`) the lm_head. Both readers are dtype-aware —
`model_embed_row` for the row lookup, `model_gemv` for the projection — so it is
quantized like any other weight. This matters a lot: the lm_head is ~30% of
decode time on Gemma (vocab = 262144), and leaving it BF16 would forfeit most
of the Q8_0 win.

## 4. The JIT

### Why JIT at all

Every gemv in a transformer has fixed `rows` and `cols` for the life of the
process. Baking them as 32-bit immediates removes the loop bounds check,
lets the inner loop be unrolled to exactly the right shape, and lets the
row-stride add be a constant. There is no runtime shape dispatch in the
emitted code — the kernel *is* the shape.

### ABI

Kernels are written assuming the **Windows x64** convention: integer args in
RCX, RDX, R8, R9. On Linux every builder calls
`x64_emitter::emit_win64_arg_shuffle(n_int_args)` first, which emits the
SysV → Win64 register moves (and is a no-op on Windows). One set of encoders
serves both platforms.

Register discipline:

- Prefer volatile registers: RAX, RCX, RDX, R8–R11, XMM0–XMM5.
- Builders that use RSI/RDI save and restore them. Kernels do not make calls,
  so they do not need to create outgoing shadow space or align RSP for a call.
  Kernels that use XMM6–XMM15 save and restore their low 128 bits, which
  Win64 treats as nonvolatile: XMM6/XMM7 in the Q4 single-row kernel, all ten
  in the three-token kernels.
- **`vzeroupper` before every `ret`.** Non-negotiable — otherwise the
  caller's SSE code eats a transition penalty.

Document register usage in a comment block at the top of each builder.

### Kernels

| builder | shape | notes |
| --- | --- | --- |
| `nnc_build_dot_f32` | runtime `n` | reference/test kernel |
| `nnc_build_gemv_f32` | rows, cols baked | reference/test kernel |
| `nnc_build_gemv_bf16w_f32x` | rows, cols baked | 4 accumulators when `cols % 32 == 0` |
| `nnc_build_gemv_bf16w_f32x_4row` | rows (×4), cols baked | shared x-tile across 4 rows |
| `nnc_build_gemv_q8_0_f32x_1row` | cols baked | caller walks rows; 4 accumulators |
| `nnc_build_gemv_q4_s_f32x_1row` | cols baked | caller walks rows; 4 accumulators |
| `nnc_build_gemv_q8_0_f32x_3tok` | cols baked | one row × 3 tokens; prefill |
| `nnc_build_gemv_q4_s_f32x_3tok` | cols baked | one row × 3 tokens; prefill |

BF16 → F32 in JIT is a free shift, not a lookup:

```
vpmovzxwd ymm, [m128]    ; 8 u16 -> 8 u32, zero-extended
vpslld    ymm, ymm, 16   ; bf16 bits become the high half of the f32
```

Q8_0 → F32 is `vpmovsxbd` (8 i8 → 8 i32) + `vcvtdq2ps` + `vmulps` by the
broadcast block scale, then `vfmadd231ps` against x.

Q4_S → F32 widens one 8-byte load into two 8-element groups:

```
vpmovzxbd y5, [rcx + rax]      ; bytes 0..7 -> 8 u32
vpslld    y7, y5, 28
vpsrld    y7, y7, 28           ; low nibbles  = elements 0..7
vpsrld    y5, y5, 4            ; high nibbles = elements 16..23
```

One counter drives both streams in the Q4 kernel: `rax` walks the packed
byte offset, and since 16 quant bytes cover 32 floats, the x address is
exactly `[rdx + rax*8]`.

### Four accumulators, and why the register budget is tight

Each block issues four FMAs. Pointed at a single accumulator they form a
dependency chain, and FMA latency on Zen 3 is 4 cycles, so a block costs
~16 cycles no matter how much memory bandwidth is spare. For Q4 that is
20 bytes per 16 cycles ≈ **5 GB/s per thread**; measured, a one-accumulator
Q4 kernel manages 4.0, against 25.3 for a plain load loop
([§8](#the-roofline)). So every gemv kernel keeps four independent
accumulators, one per FMA of a block.

Splitting across four independent accumulators costs registers, and the
low-register encoders those kernels use only reach `ymm0`–`ymm7`. Two moves buy the room:

- **x becomes a memory operand on the FMA** (`vfmadd231ps ymm, ymm, m256`)
  instead of a separate `vmovups` into a register — one fewer instruction
  and one fewer live register per group.
- **Q4 drops the `0x0F` mask register**, isolating low nibbles with a
  `vpslld 28` / `vpsrld 28` pair instead. That trades one held register for
  one extra instruction, which is the right way round when the constraint
  is registers.

Q4 then lands exactly on eight: `y0`–`y3` accumulators, `y4` scale,
`y5`/`y6` the two loaded byte groups, `y7` scratch. Reusing `y7` (and later
`y5`/`y6`, once each is dead) across groups is free — register renaming
breaks the write-after-write, so the four FMA chains stay independent.

Measured against one accumulator, interleaved A/B on otherwise identical
binaries:

| | prefill | decode @2 threads | decode @8 threads |
| --- | --- | --- | --- |
| Q8_0 | +40% | +28% | +3.4% |
| Q4 | +31% | +40% | 0% |

The shape of that table is the point. Prefill is compute-bound, so it takes
the gain in full; decode at 8 threads on a 16-core machine sits on the
memory roof either way, so it cannot. What four accumulators buy decode is
reaching that roof with about half the threads.

The 4-row BF16 variant loads the `x[k..k+7]` tile once and FMAs it into four
independent row accumulators, cutting x-side bandwidth 4×. Its tail reduces
four ymm partial sums to four contiguous floats with a 3-`vhaddps` tree plus
one `vextractf128` + `vaddps`, and stores them with a single 16-byte write.

### Three tokens per row (prefill)

Prefill is compute-bound, and in the 1-row kernels most of the work per
8 weights is widening them (`vpmovsxbd` / nibble shifts, `vcvtdq2ps`,
`vmulps` by the scale), not the one FMA. The `_3tok` kernels widen each
group once and issue its FMA for three tokens, taking `(qs, xs[3], ys[3],
scales)`. They need all sixteen ymm registers — twelve accumulators, the
scale, and one weight register (Q8_0) or two byte groups plus scratch (Q4)
— so they use the emitter's register-general `g_*` encoders (VEX3 for every
operand, any GPR as base or index) and save `xmm6`–`xmm15` for Win64.

**Each token's sum is bit-identical to the 1-row kernel's**: the same
`q × scale` product, the same four accumulators fed in the same order, the
same reduction tree. A batch row therefore gives exactly what decoding that
token alone gives, which is what keeps MTP verification
([§6](#speculative-decoding-mtp)) exact; `gemv_batch_matches_rows` checks it
with `memcmp`. The batch workers run tokens three at a time and finish the
remainder with the 1-row kernel. They are kept in emitted code rather than
intrinsics because a compiler under `-ffast-math` may re-associate the
reduction and break the identity.

### Kernel cache

`jit_kernel_cache` holds one `std::unordered_map` per kernel family, keyed on
`pack(rows, cols)` (or just `cols` for the 1-row and three-token kernels). Lookups take a
`std::shared_lock`; only first-time codegen takes the exclusive lock, and the
map is re-checked under it so a race builds the kernel once. Entries are
never evicted — a model has a few dozen distinct shapes.

### Executable memory (W^X)

Each `jit_buffer::commit()` allocates dedicated executable pages, writes the
staged code while writable, then seals the pages read/execute. Published code
is never made writable again. This avoids permission changes racing with
concurrent calls, at the cost of page-sized allocation overhead per kernel.

## 5. Threading

A single static pool (`nnc_gemv_pool`) of `min(8, CPUs) - 1` workers plus the
calling thread. Override the total with `NNC_THREADS`; it is hard-capped at
`MAX_THREADS` (64) because dispatch and the argmax reduction index
fixed-size stack arrays by thread id.

Dispatch is serialized by a mutex and uses a ticket counter. Idle workers spin
for up to 2 ms to make back-to-back gemvs cheap, then park on a condition
variable; the caller wakes sleepers, runs its own share, and waits for every
worker's completion counter. This prevents busy-spinning while the app waits
at its prompt.

Parallelised along the row axis when `rows >= 256 && cols >= 256` — that
threshold catches every significant weight matrix while leaving small PLE
gemvs on the single-threaded path where dispatch would dominate.

Attention is the other parallel phase, and both kinds of batch use the same
unit: a block of 256 keys aligned to absolute positions (`MODEL_ATTN_BLOCK`,
[§6](#attention-and-gqa)). Decode with a long span (≥ 256 KB of K/V to read
for the layer) is shared out by (KV head × block) with `nnc_parallel_for`;
each item writes its block's partial — unnormalised output, max and sum —
and the calling thread merges the partials in block order. A batch with long
spans (a prefill well into the context, or an embedding model's whole
sequence, which is quadratic) is shared out in interleaved chunks of its
tokens (chunk `c` takes tokens `c`, `c + n_chunks`, ...), so that a causal
batch's longer spans are spread evenly; each token then walks its blocks
itself. Each slot's scores and partials are sized in `model_kv_init` to the
thread count, so the per-token path still allocates nothing. Because blocks
are fixed by position and always merged in the same order, a token's result
is the same bits whichever path, thread count or batch computes it.

The Gated DeltaNet recurrence ([§6](#gated-deltanet-qwen-35-36-38)) is the
third. Each value head's state depends only on its own inputs, so the heads
are shared out with `nnc_parallel_for`, each running through the batch's
tokens in order, once the batch's state work passes 64K floats (256 KB);
the result is the same whichever thread runs a head.

## 6. Model forward pass

Per batch of tokens, `model_forward`:

1. Embed: row lookup from `token_embd`, scaled by `sqrt(n_embd)` (Gemma) or
   1.0 (Llama, Qwen).
2. PLE prep (when present): project the hidden state through
   `per_layer_model_proj` and RMSNorm each layer slice; when the model has a
   per-layer token table (Gemma 4 E2B/E4B, not EmbeddingGemma 2), average
   that with the token's table row.
3. For each layer, `layer_forward`:
   - RMSNorm × gamma → Q/K/V projections, plus their biases (Qwen2); a
     Qwen3.5 layer's Q projection also yields a gate per head
   - per-head Q/K norms, per-head V RMSNorm (gemma4 only)
   - RoPE on Q, and on K when freshly computed
   - append K/V to the layer's cache buffer (rounded to BF16 by default)
   - per KV head, in blocks of 256 keys: `dot(Q, K[t])` scores for the
     query heads sharing it, softmax within the block, × V; blocks merged
     online; times sigmoid(gate) for Qwen3.5
   - or, in three of every four Qwen3.5 layers, a Gated DeltaNet in place of
     all of the above (below)
   - output projection, post-attention norm, residual add
   - RMSNorm → gate/up projections → `gelu(gate)*up` (Gemma) or
     `silu(gate)*up` (Llama, Qwen) → down projection; for a mixture-of-experts
     layer, plus the routed experts (below); post-FFW norm, residual add
   - PLE gate/proj block, residual add
   - per-layer output scale
4. Final `output_norm`.

The session API requests logits for sampled generation or uses a streaming
argmax path when temperature is zero. Chat formatting is selected from the
model vocabulary (known Gemma, Gemma 4, Llama 3 and ChatML formats; see
[tokenizer](#tokenizer-and-chat-formats)); unsupported templates use plain text
rather than pretending to apply a model-specific format.

### Batching

The layer forward pass takes an `n_tok` and a token-major `[n_tok, n_embd]`
residual buffer. Decode passes 1; prefill passes up to
`MODEL_PREFILL_BATCH` (16). There is exactly one code path — no separate
batched implementation to drift out of sync.

What batching buys: without it, prefill would read every weight once per
token; with it, one pass over the weights serves 16 tokens. The batched
gemv keeps the loop order row-outer / batch-inner so each weight row is
pulled from DRAM once and then hit in L1 for the rest of the batch. The
quantised formats run a batch three tokens per kernel call
([§4](#three-tokens-per-row-prefill)); BF16 calls its 4-row kernel `n_tok`
times per row group.

Most of the surrounding ops are indifferent to the batch, because the batch
dimension is just more groups:

- `nnc_rmsnorm_gamma_multi_f32(y, x, n_tok, n_embd, gamma, eps)` — the
  per-token norms are already an `n_groups` loop.
- Per-head Q/K norms become `n_tok * n_head` groups in a single call.
- Activations and residual adds run over the whole `n_tok * dim` block.

Only three things stay per-token: RoPE (each position has its own angle),
the KV append, and attention itself (each query attends to its own span;
the tokens of a batch are shared across the pool, [§5](#5-threading)).

### Attention and GQA

Query heads that share a KV head are processed **together**, not one at a
time. Gemma 3 1B is 4 query heads to 1 KV head, so the naive per-head loop
re-streams the entire K and V cache four times. The loop is instead:

```
for kv_head:
    for each block of 256 keys in the span:
        for t:  load K[t] once, dot against all 4 query heads
        softmax all 4 score rows within the block (max m, sum l)
        for t:  load V[t] once, FMA into all 4 output accumulators
        merge (o, m, l) into the running result
```

`scores` is therefore sized `[n_head][256]` per slot, and attention traffic
is divided by the GQA ratio.

Blocks (`nnc_attn_block`, `nnc_attn_merge`, `nnc_attn_finish`) are the
online-softmax form: each block is exp'd against its own max, and merging
rescales by `exp(m_old − m_new)`. They are aligned to absolute positions —
block `b` is keys `[256b, 256b + 255]` clipped to the span — and merged in
increasing order, so the arithmetic depends only on the query's position and
span. That is what lets decode split one query across threads, and still
give the bits a batch gives ([§5](#5-threading)); the
`attn_decode_matches_batch` test checks it.

Batch width is a cache trade: the batch's activations (`n_tok * cols`
floats) must stay resident while the row loop sweeps. 16 tokens × a few
thousand columns is tens of KB — comfortably L2.

Notable details:

- **Shared KV** (`attention.shared_kv_layers`): layers at or past
  `n_layer - count` reuse the cache of an earlier layer and skip K/V
  projection, K norm and K RoPE entirely. `model_kv_cache.kv_buf` maps each
  layer to the buffer it reads and writes.
- **Attention spans** come from one function, `model_attn_span`. A causal
  model's query at `pos` sees `[0, pos]`, and a sliding layer only the last
  `sliding_window` of those. A bidirectional (embedding) model's query sees
  the whole sequence, and a sliding layer `|key - pos| <= window / 2`
  (llama.cpp's "symmetric" window; Hugging Face's mask works out the same).
  An MTP drafter's query is clamped to the keys its target has evaluated.
- **Sliding window**: per-layer flag from the GGUF pattern array, or a
  period rule (`sliding_window_pattern` scalar, default 6) for Gemma 3 and the
  embedding models.
- **`head_dim` is derived from the `attn_q_norm` tensor shape**, not the
  `key_length` hparam — Gemma 3n GGUFs disagree with themselves there.
- All forward scratch buffers live in the session's `model_kv_cache`, sized
  once to the maxima across layers. Sampling scratch belongs to the session
  and is reused across generated tokens.

### Sliding-window ring buffer

A sliding layer only ever reads its last `sliding_window` keys, so its
buffer need not grow with the context. `model_kv_init` gives it
`round_up(window + n_batch, 256)` rows — enough for a whole batch's writes
plus the window behind its first token — and key `p` lives at row
`p % rows`; attention reads row `k % rows` and nothing else changes. A
layer becomes a ring only if that is smaller than the context and every
layer reading the buffer (shared-KV layers included) slides the same way.
Gemma 3 1B at an 8K context holds 48.5 MB of K/V this way (BF16, five of
every six layers in rings of 768 rows), where F32 buffers of the whole
context would take 416 MB; and the ring is bit-identical to a full buffer
(`sliding_ring_matches_full`).

The price is that a ring has forgotten keys more than `rows − window`
behind the highest position written. `model_kv_cache.kv_written` tracks that
high-water mark, and `model_kv_rewind` refuses to move `cur_pos` back further
than `max_rewind` (the smallest `rows − window` over the rings); the session
then starts again from 0 instead. Every rewind the session makes — a
cancelled prefill, the next message sharing a prefix, MTP's rejected guesses
(at most `n_draft`) — goes through it.

### Mixture of experts

Gemma 4 26B A4B keeps a dense MLP in every layer — the shared expert — and
adds 128 small gated MLPs of which a router picks 8 per token. Following
llama.cpp's `gemma4.cpp`:

- The dense MLP's output gets its own norm (`post_ffw_norm_1`).
- The router reads `rms_norm(x) / sqrt(n_embd) * ffn_gate_inp.scale` (no
  gamma) against F32 router rows, softmaxes over all experts, and keeps the
  top 8 (`nnc_moe_top_k`: ties to the lower index), renormalised to sum to 1.
- Each chosen expert runs on `rms_norm(x, pre_ffw_norm_2)`: its `[gate | up]`
  rows, `gelu(gate) * up`, its down rows, times the expert's
  `ffn_down_exps.scale` and its routing weight. The weighted sum gets
  `post_ffw_norm_2`, is added to the dense output, and the existing
  `post_ffw_norm` and residual follow.

The stacked `[cols, rows, n_expert]` tensors are decoded at load into one
split-layout matrix per expert, back to back, so `expert_matrix(T, e)` is a
plain 2-D view and every gemv kernel works unchanged. Within a batch, each
chosen expert runs once over every token that picked it, so its weights are
streamed once per batch, not once per token. `--perf` shows the router and the
experts as `moe_route` and `moe_experts`.

### Gated DeltaNet (Qwen 3.5, 3.6, 3.8)

Qwen3.5 and its successors (GGUF architecture `qwen35`, which includes
Qwen3.8-27B) are hybrids: three of every four layers replace attention with a
Gated DeltaNet, a linear-attention layer whose memory is a fixed-size state
rather than a K/V cache. Following llama.cpp's `qwen35.cpp`, per token:

- One projection gives q and k (16 heads of 128) and v (16 to 48 heads of
  128), through a depthwise causal convolution over the last four positions
  and SiLU. q and k are scaled to unit length per head, and q by 1/√d_k too.
- Each value head gets a decay, `exp(softplus(x·w_α + dt) · A)` (the file
  stores A as −exp(A_log)), and a write strength, `sigmoid(x·w_β)`. These
  few rows stay F32, like a router.
- The delta rule updates the head's `[d_v × d_k]` state S:
  `S = decay·S;  S += β (v − S k) kᵀ;  o = S q` (`nnc_delta_rule_f32`, which
  reads, updates and rereads each row while it is in L1).
- `o` is RMS-normed per head, scaled by gamma and by silu(z), where z has a
  projection of its own, and projected back.

The converter orders value heads so that head h reads key head h mod n_k
(ggml's broadcast), not h / (n_v / n_k) as in the Hugging Face weights.

The attention layers are gated too: their query projection gives each head
its query and then a gate, and the attention output is multiplied by
sigmoid(gate) before the output projection. Their RoPE is interleaved
multimodal RoPE over the first 64 of each head's 256 lanes; for text every
position component is the token's position, which makes it NeoX RoPE on those
lanes. The pre-MLP norm is stored as `post_attention_norm`, and a file may end
with a multi-token-prediction block (`nextn_predict_layers`), which nnc skips.

**The recurrent state is the one thing in nnc that cannot be cut back.** K/V
past `cur_pos` are simply overwritten later; the state sums up the whole
history. So `model_forward` at position 0 clears it, and the session saves a
copy (with its hidden state) at the start of every call that evaluates more
than one prefill batch: about 20 MB for the 0.8B and 160 MB for the 27B, a
few milliseconds to copy. One batch needs no copy, since a cancel is only
seen between batches and `model_forward` fails, if at all, before it changes
anything; so decoding a token at a time copies nothing. A call that is
cancelled between batches puts the copy back (`model_state_restore`) instead
of cutting `cur_pos` back. When the prediction for the last token is lost
some other way, the session goes back to the last copy, or failing that to
the start, and evaluates forward again. It never re-evaluates a token over a
state that has already taken it in. MTP drafting, which cuts rejected guesses
back, stays Gemma 4's.

Batching changes nothing: `delta_net_forward` runs a batch's tokens through
the recurrence in order, and the convolution reads `[saved history | batch]`,
so a 16-token prefill gives each token the state that one token at a time
does (the `qwen35_batch_matches_steps` test).

Checked against llama.cpp with the same BF16 weights (nnc `--bf16`, and
llama.cpp on an F32 copy so that it does not round activations): Qwen3.5-0.8B's
logits match within 1.4e-5 over a 64-token prompt and 48 generated tokens, and
Qwen3-0.6B's and Qwen2.5-0.5B's match as closely. Qwen3.8-27B, from its
Q4_K_M file, agrees with llama.cpp on every greedy token with the default
Q8_0 weights.

### Embedding models

EmbeddingGemma (`gemma-embedding`, a Gemma 3 body) and EmbeddingGemma 2
(`gemma-embedding2`, a Gemma 4 body with per-layer inputs but no per-layer
token table) run through the same `layer_forward`, with three differences:

- **Bidirectional attention** (see the spans above). `model_embed` therefore
  runs the whole sequence as one batch from position 0, with a context sized
  to it; and because each layer rewrites the whole sequence before it
  attends, every layer shares one K/V buffer, so a long text costs O(tokens)
  K/V memory rather than O(layers × tokens).
- **No lm_head.** The final hidden states are mean-pooled (the GGUF's
  `pooling_type`; first- and last-token pooling are also supported), run
  through the model's projections — EmbeddingGemma's `dense_2` (768 → 3072)
  and `dense_3` (3072 → 768), EmbeddingGemma 2's `output.weight` (512 → 768)
  — and scaled to unit length. A shorter `n_out` keeps a Matryoshka prefix,
  normalised on its own.
- **Tokens**: `[BOS] text [EOS]` (`add_eos_token`), cut to the context
  (2048 / 8192 tokens). The task prefix ("task: search result | query: ")
  is the caller's; `nnc.h` lists EmbeddingGemma's.

Validated against Google's published similarities (the STS matrix within
0.0007, the "Red Planet" search scores within 0.0004) and against
llama.cpp on the same GGUFs (cosine ≥ 0.9994, including texts past the
sliding windows).

### Speculative decoding (MTP)

Gemma 4 ships multi-token-prediction drafters ("assistant" models,
`gemma4-assistant`): four small layers that predict the next few tokens from
the target's own state, so the target can check several tokens in one weight
pass. Per round, with token `t` sampled but not yet evaluated at position
`p`:

1. **Draft.** The drafter's input is `[target.token_embd(t) * sqrt(n_embd) |
   h]`, where `h` is the target's final (post-norm) hidden state at `p - 1`,
   projected down by `nextn.pre_projection`. Its layers are Gemma 4 layers
   with no K/V of their own: their queries attend to the target's cache —
   sliding layers to the target's last sliding layer, global layers to its
   last global layer — through `model_kv_cache.kv_ext`, so `layer_forward`
   is reused unchanged (they take its shared-KV path). Every step of a round
   uses position `p`. The drafter's guess is the argmax of its own lm_head;
   `nextn.post_projection` of its final state becomes the next step's `h`.
   A round stops at `n_draft` guesses (4; 2 for a mixture of experts, where
   each extra token checked reads experts of its own), at an end-of-turn
   token, or when the drafter's probability for its guess is under 0.7.
2. **Verify.** One `model_forward` over `[t, d1, ..., dk]` gives the
   target's hidden state after each; one batched lm_head pass gives their
   logits.
3. **Accept.** Walking the rows in order, the target samples its own token
   with the session's sampler. While it equals the next guess, the guess is
   kept (its K/V are already in the cache); the first token that differs —
   or the one after the last guess — becomes the next round's `t`. The
   cache position is cut back past the rejected guesses.

Because every emitted token is the target's own sample from logits computed
by the same per-row kernels as plain decoding, the output is exactly what
plain decoding gives — bit for bit with greedy decoding, and with sampling
too, since the sampler consumes one draw per emitted token either way
(checked with a fixed seed). That holds for Q8_0 and Q4 weights, whose
single-vector and batched gemvs give the same bits per row
([§4](#three-tokens-per-row-prefill)), and for the BF16 K/V cache, whose
blocked attention does too; with `--bf16`, a
small matrix or a single thread takes a whole-matrix kernel that may round
differently, so a near-tie can go the other way. The drafter only changes
the speed. On this machine, with the Q8_0 drafters: Gemma 4 E2B goes from
11 to 15 tok/s on prose and from 11 to 23 on code (62% and 58% of guesses
kept), and 26B A4B (`--q4`, two guesses) from ~7.5 to ~9 on prose and ~11.5
on code ([docs/models.md](models.md#with-an-mtp-drafter)). The draft step is
one `mtp_draft` phase in `--perf`.

### Tokenizer and chat formats

Three encoders, chosen by `tokenizer.ggml.model`: SentencePiece by score
(Gemma 3, Llama 2), byte-level BPE (Llama 3, Qwen) and ▁-for-space BPE by
merge rank (Gemma 4). Before any of them, the text is split at **user-defined
tokens** — FunctionGemma's `<start_function_call>` and `<escape>`, Gemma 4's
`<|tool_call>` and `<|channel>`, Qwen's `<think>` and `<tool_call>`, and
Gemma's runs of newlines, spaces and tabs — which are always one token,
leftmost and then longest first, as in llama.cpp and Hugging Face. Control
tokens (`<start_of_turn>`) typed in the text stay text. Checked
token-for-token against llama.cpp on all four vocabulary kinds.

Byte-level BPE then cuts the text with a regex, reproduced without a regex
engine, whose pieces the merges never cross. `tokenizer.ggml.pre` picks the
regex: llama-bpe's takes digits in runs of up to three; Qwen's (`qwen2`) one
at a time; and Qwen3.5's (`qwen35`) also keeps combining marks, such as
Devanagari's vowel signs, inside words, where the others split them off like
punctuation. Beyond ASCII, a sorted table of ranges says which codepoints
are digits, marks, or punctuation and symbols (Latin-1, Hebrew, Arabic, the
Indic scripts, Thai, the symbol blocks, CJK and full-width forms, emoji);
everything else is a letter. With it, Llama 3's and both Qwen pre-tokenizers
match llama.cpp token for token on a multilingual test text (Chinese,
Japanese, Korean, Hindi, Bengali, Tamil, Thai, vowelled Arabic, pointed
Hebrew, emoji sequences, mathematical letters).

Turn formats come from the vocabulary:

- **gemma**: `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`;
  no system role, so the system prompt opens the first user turn —
  except for **FunctionGemma** (recognised by its function tokens), whose
  system prompt holding the function declarations is a `developer` turn of
  its own. `<start_function_response>` also ends FunctionGemma's turn: the
  caller runs the function and continues the turn with
  `response:NAME{...}<end_function_response>`.
- **gemma4**: `<|turn>` / `<turn|>`, with a system role. With thinking off,
  an empty thought channel opens each reply when the file's template has one
  (26B, 31B); with it on, `<|think|>` heads the system turn, which is written
  for it alone when there is no system prompt.
- **llama3**: header ids and `<|eot_id|>`.
- **chatml** (Qwen): `<|im_start|>role\n...<|im_end|>\n`, with a system role.
  The template decides the rest: Qwen2.5's supplies a system prompt when
  none is given; where thinking can be switched off (Qwen3, 3.5, 3.8), each
  reply opens with the empty block `<think>\n\n</think>\n\n`, as Gemma 4's
  opens with an empty thought channel; a template that always thinks
  (Qwen3 Thinking-2507) has the reply open with `<think>\n`. With thinking
  on, the reply opens as the template then opens it — `<think>\n` for
  Qwen3.5 and 3.8, nothing for Qwen3, whose model writes the `<think>`
  itself — and Qwen3.8's system turn starts with the instruction its
  template gives for the default reasoning effort ("xhigh"). Each run of text
  between markers is encoded whole, so the prompt is token-for-token what
  llama.cpp makes of the rendered template.

**TranslateGemma** is a Gemma 3 model whose template takes a source and
target language; nnc reads the template's 581-entry code → name table at
load, and `nnc_session_translate` writes the prompt the template would. Other
chat models get the same request as a user message (with a small built-in
table of language names).

### Thinking

`nnc_session_params.thinking` (the app's `--think`) lets a model that can
reason do so before it answers: Gemma 4, and Qwen models whose template has
an `enable_thinking` switch (Qwen3, 3.5, 3.6, 3.8). Each chat turn is then
written as the template writes it with `enable_thinking` set (above), and
otherwise as with it cleared. It is off by default: reasoning improves hard
problems — code, maths — and multiplies the tokens a reply takes, which a
CPU pays for at decode speed.

The reasoning is told apart from the answer by the model's thought markers,
`<think>` / `</think>` or Gemma 4's `<|channel>` / `<channel|>` (whose first
line names the channel). They are user-defined tokens, so they are matched
by id; otherwise they would stream as text. `thought_split` (session.h)
routes each generated token to `on_thinking` or `on_text`. The markers,
Gemma's name line and the newlines that open the reasoning and the answer
show nowhere. Every reply goes through it, so a model that thinks whatever
it is told (Qwen3 Thinking-2507) has its reasoning split off too. A call that
carries a reply on (after `max_tokens`) picks the state up by scanning the
current turn back to its last marker.

The reasoning stays in the context. The templates drop earlier turns'
reasoning from the history they render (Qwen3.8's keeps it by default, its
`preserve_thinking`). In nnc, though, the context *is* the K/V cache, so
dropping a finished reply's reasoning would mean evaluating the answer after
it again. A sliding-window ring or a Gated DeltaNet state cannot be cut back
that far, so in practice it would mean evaluating the whole conversation
again, at prefill speed, before every turn. Keeping it costs context
instead, so a thinking session wants a larger `n_ctx`.

## 7. Profiler

`--perf` enables a fixed-slot wall-clock accumulator (`nnc_perf_scope`, RAII)
and prints a breakdown after generation: ms, % of measured time, call count,
µs/call, and effective GB/s for the phases that stream weights.

Off by default so the hot path pays nothing. When on, the overhead is two
`steady_clock` reads per scope — roughly 0.05% of decode.

Besides the phases below, a mixture-of-experts model reports `moe_route` (the
router and top-k) and `moe_experts` (the chosen experts' MLPs, with the bytes
of the experts a batch actually used), and an MTP drafter's steps appear
whole as `mtp_draft` — its own layers' scopes are paused during a step so
they are not counted twice. A Qwen3.5 model's Gated DeltaNet layers report
`dn_proj` (their input projections), `dn_core` (the convolution, the
recurrence and the gated norm, with the state's traffic as its bytes; the
state mostly stays in cache, so its GB/s runs past the DRAM roof) and
`dn_out` (the output projection). On Qwen3.5-0.8B decode, `dn_core` is about
3% of the time; the weight-streaming phases sit at 23–26 GB/s, and the
248,320-token lm_head alone is about a third.

Representative decode profile (Gemma 3 1B, Q8_0, 8 threads; one prompt
token, then 20 decode steps):

```
phase                ms       %      calls    us/call       GB/s
------------ ---------- ------- ---------- ---------- ----------
embed               0.0    0.0%         21        0.9
norm                2.8    0.3%       2751        1.0
attn_qkv           38.9    4.8%        546       71.2       26.4
attn_rope           1.3    0.2%        546        2.4
attn_core           1.5    0.2%        546        2.7        4.2
attn_out           24.4    3.0%        546       44.8       28.0
ffn_gate_up       322.1   40.0%        546      590.0       28.7
ffn_act             1.8    0.2%        546        3.4
ffn_down          160.7   20.0%        546      294.3       28.8
lm_head           250.8   31.2%         21    11944.6       26.9
TOTAL             804.4  100.0%
weight traffic: 20.78 GB @ 27.7 GB/s effective
38.30 ms/token over 21 tokens
```

Read: **decode is entirely DRAM-bandwidth bound.** Every weight-streaming
phase lands within a few percent of the same ~28 GB/s, which is the roof
this machine's memory system delivers ([§8](#the-roofline)). Arithmetic is
free by comparison — attention, RoPE, norms and the activation together are
under 1%.

The consequence for optimization: nothing that reduces *instructions* will
help. Only reducing *bytes read per token* will. That is why Q8_0 is the
default and `--q4` the faster choice, and why the remaining headroom
([§9](#9-where-the-remaining-headroom-is)) is mostly about bytes.

## 8. Measured performance

Gemma 3 1B, Release, 8 threads: a 34-token prompt, then 32 decode steps.
Best of 7 runs, the three formats interleaved run by run (see "measuring"
below):

| weights | prefill (ms/tok) | decode (ms/tok) | resident |
| --- | --- | --- | --- |
| BF16 (`--bf16`) | 8.3 | 67.2 | mmap only |
| Q8_0 (default) | 4.8 | 36.4 | 1013 MB |
| Q4 (`--q4`) | 4.5 | 23.4 | 596 MB |

[docs/models.md](models.md) carries the per-model figures — every model
run end-to-end, both formats, with resident memory and the ones that do not
load.

Two things worth reading off these numbers:

- **Prefill is compute-bound.** Q8_0 and Q4 prefill at almost the same
  rate, although Q4 reads 40% fewer bytes: the work is in widening the
  weights, which the three-token kernels share between tokens
  ([§4](#three-tokens-per-row-prefill)). BF16 prefill, which has no such
  kernel, is slower.
- **Decode tracks bytes per weight.** 2.0, 1.0625 and 0.625 bytes give
  67.2, 36.4 and 23.4 ms/tok. Decode has a batch of one and nothing to
  amortise, so it is a bandwidth problem.

### The roofline

"Decode is bandwidth-bound" is only meaningful against a known ceiling, so
measure the ceiling rather than inferring it. A standalone probe that walks
a 1 GB buffer with 256-bit loads and four accumulators — the same shape as
a weight stream, no dequantisation — gives this machine (Ryzen 9 5900XT,
2×32 GB DDR4-2400 dual channel, 38.4 GB/s theoretical):

| threads | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| GB/s | 25.3 | 27.8 | 27.9 | 26.1 | 29.2 | 28.3 |

Two facts fall out, and both matter more than the peak itself:

- The practical roof is **~28 GB/s**, about 73% of theoretical.
- **One thread already reaches 25.3 GB/s.** Threads exist here to cover
  dequantisation work, *not* to accumulate bandwidth. If a kernel needs
  many threads to approach the roof, that is a property of the kernel.

The second point is the useful diagnostic: a kernel thread that streams far
below 25 GB/s is limited by its own instruction chain, as a one-accumulator
Q4 kernel is at 4.0 GB/s
([§4](#four-accumulators-and-why-the-register-budget-is-tight)).

### Measuring

Session-to-session variance on this machine is around ±20% — larger than
most of the effects worth chasing. Two rules keep conclusions honest:

- **Interleave A and B run-by-run** and take best-of-N per binary, rather
  than measuring all of A then all of B. Thermal and boost state drift over
  minutes and will otherwise be attributed to the change.
- **Keep an untouched control in the comparison.** `--bf16` shares no
  kernel with the Q8_0/Q4 paths, so when a Q8_0 change reads +3.4% and the
  BF16 control reads +0.4%, the noise floor is visible in the same table.
  A control that moves as much as the treatment means the run is worthless.

### Long context

A 2.7–2.8k-token prompt with `-c 8192`, then 64 decoded tokens; best of 8
runs:

| model, weights | prefill (ms/tok) | decode (tok/s) | K/V cache at 8K |
| --- | --- | --- | --- |
| Gemma 3 1B, Q8_0 | 4.4 | 25.1 | 48.5 MB |
| Gemma 3 1B, BF16 | 4.8 | 14.2 | 48.5 MB |
| Qwen3 0.6B, Q8_0 | 5.2 | 25.5 | 896 MB |
| Qwen3 0.6B, BF16 | 5.7 | 16.5 | 896 MB |

Gemma 3 1B reads a long prompt faster per token than a short one — every
batch is full, and its sliding layers attend to at most their window — and
decodes nearly as fast as after a short prompt: attention over a long span is
shared across the pool in 256-key blocks ([§5](#5-threading)) and reads BF16
keys and values, and the sliding layers keep only a ring of their window
([§6](#sliding-window-ring-buffer)). Every layer of Qwen3 0.6B attends to
everything before it and keeps it all, so its prefill pays for the longer
spans, and it decodes at a little over half its short-prompt speed with
eighteen times the cache.

Runs this long vary more than short ones on this machine: the slowest of the
eight took up to 1.6× as long as the fastest.

A note on reading `--perf` during prefill: the GB/s column counts weight
bytes actually streamed, which is once per batch, not once per token. A low
figure there (say 15 GB/s on `ffn_gate_up`, against 27 in decode) is the
*good* outcome: the phase is not limited by memory.

## 9. Where the remaining headroom is

In decode, the weight-streaming phases reach 26–29 GB/s with Q8_0 weights
and 25–27 GB/s with Q4 ([§7](#7-profiler)) against a ~28 GB/s roof, so
decode is close to the memory system and the list is mostly about bytes:

1. **A native Q4_K gemv over the on-disk blocks.** Q4_S is 0.625
   bytes/weight; ggml's Q4_K is 0.5625 because its 8 sub-block scales share
   one FP16 multiplier and are stored as 6-bit integers. Reading those
   blocks in place would cut another 10% of decode traffic *and* remove the
   load-time decode pass entirely (instant load, no extra RAM). The cost is
   unpacking 6-bit scales in emitted code, which is materially harder than
   the current nibble path — likely scalar GPR code inside the kernel.
   Note that `_M`/`_S` files are mixtures, so this also implies a Q6_K
   kernel before any real model loads zero-copy.
2. **Fold the Q4 bias term into the emitted kernel.** `nnc_gemv_q4_s_f32x`
   calls `nnc_dot_bf16_f32_simd` once per row for the
   `Σ bias_b · xsum_b` correction — a separate call, a third input stream
   and a horizontal reduction per row. Folding it in needs a spare
   register, which the 4-accumulator layout does not have; dropping the
   asymmetric bias entirely (symmetric Q4, values in [-8,7]) would remove
   both the pass and 2 bytes per block, at some accuracy cost.
3. **Speculative decode beyond Gemma 4.** MTP drafting
   ([§6](#speculative-decoding-mtp)) uses the batched machinery to check
   several tokens per weight pass, and is the one item here worth more than
   bandwidth tweaks. It needs a drafter made for the target; a generic path
   with an ordinary small draft model (Gemma 3 270M for Gemma 3) would bring
   it to the other families, at the cost of the draft model's own KV cache.
   Qwen3.5 and 3.8 files carry an MTP block of their own (which nnc skips);
   using it means rewinding the Gated DeltaNet state past rejected guesses,
   so a copy of the state per drafted token.
   Rejection sampling (Leviathan et al.) instead of sample-and-match would
   keep more guesses when sampling, and an expert-aware draft length would
   help the mixture of experts.
4. **Sampling cost.** Sampling is enabled by default (temperature 0.7,
   top-k 40, top-p 0.95, min-p 0.05). Candidate selection scans the
   vocabulary each token; optimize it only if profiling shows it matters
   relative to the weight-streaming phases.

## 10. Testing

`tests.cpp` is one translation unit, no framework: plain `bool` functions in
a static list, run by `nnc --test`, exit code = failure count.

Rules:

- Tests must not require model files — generate synthetic inputs.
- JIT numerical tests compare against a scalar reference within a relative
  tolerance (start at `1e-4`). FMA rounding differs from naive scalar
  summation order; that is expected, not a bug.
- Silent on success beyond `[PASS]` lines and the final summary.

Coverage: CPU detection, public C API defaults and invalid-handle guards,
`nnc_last_error()` truncation that never splits a UTF-8 character, the
streamed-text UTF-8 splitter (characters split across tokens, U+FFFD for
invalid bytes), atomic replacement of an existing model download, JIT
round-trips (immediate return, arithmetic, dot, gemv, cache reuse), the
register-general encoders against hand-assembled bytes, the three-token
Q8_0/Q4 gemvs against single-vector gemvs (bit for bit), blocked attention
(F32 and BF16 K/V, across block boundaries) against a scalar softmax, gelu,
softmax (including `-inf` masking through both the
scalar and AVX2 paths), layernorm, rmsnorm, bf16 round-trip and row convert,
BF16 gemv and streaming argmax, Q8_0 quantize round-trip and gemv, Q8_0,
legacy (Q4_0/Q4_1/Q5_0/Q5_1) and K-quant dequantizers, fp16 subnormal decode,
swiglu, rope, softcap, both embedding-row lookups, the mixture-of-experts
top-k, attention spans (causal, sliding, bidirectional, symmetric window,
drafter clamp), embedding pooling and Matryoshka normalisation, the
tokenizer's user-defined-token split and FunctionGemma detection and the
TranslateGemma language table (on tokenizer-only GGUFs written to the temp
directory), Qwen's pre-tokenizer splits (digits, CJK and full-width
punctuation, combining marks) and ChatML template reading (thinking
switches, openers and Qwen3.8's reasoning instruction), chat turns with
thinking on and off rendered exactly as the Qwen and Gemma 4 templates
render them, the split of a reply into reasoning and answer (markers, a
Gemma 4 channel's name line, carrying on mid-reply, turn boundaries), the Gated
DeltaNet kernels (sigmoid gate, L2 normalisation, causal convolution, delta
rule), a synthetic Qwen3.5 model (three Gated DeltaNet layers, a gated
attention layer and an MTP block to skip, written to the temp directory) on
which a batch gives each token the state that step-by-step decoding does, a
saved state comes back exactly, and a session rebuilds a lost prediction from
its saved state and is unchanged by a cancelled call, a synthetic dense
Qwen3 with a sliding and a global layer on which a sliding layer's ring
buffer matches a full buffer bit for bit through wraps and rewinds (and
refuses a rewind past what it holds) and parallel decode attention matches
a batch bit for bit, the refusal of Qwen
files holding unknown tensors and of the Qwen architectures nnc does not
implement, and `gguf_load` through a UTF-8 path outside the
Windows ANSI code page (a tiny GGUF written to the temp directory) and its
refusal of an array length the rest of the file cannot hold, of metadata
whose in-memory size would far exceed the file's, of arrays nested
deeper than eight levels, and of tensor shapes whose dimensions exceed
`INT_MAX` or whose byte count could overflow.

What the model files themselves need is checked by hand against references:
EmbeddingGemma against Google's published similarities and llama.cpp, the
tokenizer token-for-token against llama.cpp's `llama-tokenize`, Qwen's
forward passes and chat prompts against llama.cpp's logits and rendered
templates ([Gated DeltaNet](#gated-deltanet-qwen-35-36-38)), and MTP by
comparing its output with plain decoding (identical, greedy and seeded).
