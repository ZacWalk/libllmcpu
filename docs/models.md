# Models

What nnc can load, and how fast each one actually runs.
For *why* the numbers look like this, see
[docs/design.md](design.md) — particularly
[the roofline](design.md#the-roofline) and
[measured performance](design.md#8-measured-performance).

## What loads

The loader accepts the `gemma3`, `gemma4` and `llama` architectures (chat
and completion, including Gemma 4's 26B A4B mixture of experts, FunctionGemma
and TranslateGemma, which are Gemma 3 fine-tunes), Qwen's dense architectures
`qwen2` (Qwen2, Qwen2.5), `qwen3` (Qwen3, including the 2507 updates) and
`qwen35` (Qwen 3.5, 3.6 and 3.8, including Qwen3.8-27B: hybrid Gated DeltaNet
and attention layers), the text-embedding architectures `gemma-embedding` and
`gemma-embedding2` (EmbeddingGemma 1 and 2), and `gemma4-assistant` — Gemma
4's MTP drafters, which only run alongside the model they were made for. It
reads F32 / F16 / BF16, Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 and Q4_K / Q5_K /
Q6_K tensors. Mixed quantizations need the 32-element formats: a row that is
not whole 256-element K-quant blocks (a 2112- or 704-wide down projection,
say) falls back to Q5_0 or Q8_0. Anything packed is decoded once at load time
into nnc's own layout, so the *source* quantization affects load time and
disk size, not decode speed. That rules out the `IQ*` and `UD-IQ*` files and
the lowest `Q2_K`/`Q3_K` ones; the plain `Q4_K_M`, `Q6_K` and `Q8_0` files load.

`nnc --inspect-model <file>` will tell you whether a given file is loadable;
`nnc --gemma-info <file>` prints the architecture, dimensions and the
resident size it would occupy.

## Measured performance

Release build, 8 threads, on the machine described under
[the roofline](design.md#the-roofline) (Ryzen 9 5900XT, dual-channel
DDR4-2400), with nothing else running. Each model decoded 32 tokens after a
32–33-token prompt asking why the sky is blue (`--gemma-prompt`). Every model
and format was measured in one session, in three interleaved rounds, and the
table gives the best round. `resident` is what nnc holds after its load-time
decode; decode is in ms/token. Sorted fastest first.

| model | arch | file | `--q4` resident | `--q4` decode | `--q8` resident | `--q8` decode |
| --- | --- | --- | --- | --- | --- | --- |
| functiongemma-270m-it | gemma3 | Q8_0, 0.29 GB | 160 MB | 6.1 (165 tok/s) | 272 MB | 9.9 (102 tok/s) |
| gemma-3-270m-it | gemma3 | Q8_0, 0.29 GB | 160 MB | 6.3 (159 tok/s) | 272 MB | 10.3 (97 tok/s) |
| qwen2.5-0.5b-instruct | qwen2 | Q8_0, 0.68 GB | 376 MB | 11.0 (91 tok/s) | 639 MB | 17.9 (56 tok/s) |
| Qwen3-0.6B | qwen3 | Q8_0, 0.64 GB | 355 MB | 13.6 (73 tok/s) | 604 MB | 22.6 (44 tok/s) |
| Qwen3.5-0.8B | qwen35 | Q8_0, 0.83 GB | 452 MB | 19.1 (52 tok/s) | 765 MB | 28.5 (35 tok/s) |
| gemma-3-1b-it | gemma3 | Q6_K, 1.01 GB | 596 MB | 22.0 (45 tok/s) | 1013 MB | 35.2 (28 tok/s) |
| Llama-3.2-1B-Instruct | llama | Q6_K, 1.02 GB | 737 MB | 26.5 (38 tok/s) | 1252 MB | 43.4 (23 tok/s) |
| gemma-4-E2B-it | gemma4 | Q4_K_M, 3.11 GB | 2759 MB | 48.3 (20.7 tok/s) | 4690 MB | 77.5 (12.9 tok/s) |
| translategemma-4b-it | gemma3 | Q4_K_M, 2.49 GB | 2313 MB | 82.4 (12.1 tok/s) | 3931 MB | 132.4 (7.6 tok/s) |
| Qwen3-4B-Instruct-2507 | qwen3 | Q4_K_M, 2.50 GB | 2398 MB | 89.6 (11.2 tok/s) | 4076 MB | 144.0 (6.9 tok/s) |
| Qwen3.5-4B | qwen35 | Q4_K_M, 2.74 GB | 2522 MB | 99.7 (10.0 tok/s) | 4275 MB | 167.6 (6.0 tok/s) |
| gemma-4-E4B-it | gemma4 | Q4_K_M, 4.98 GB | 4448 MB | 101.3 (9.9 tok/s) | 7562 MB | 164.8 (6.1 tok/s) |
| Qwen3.8-27B | qwen35 | Q4_K_M, 18.97 GB | 16113 MB | 544 (1.8 tok/s) | 27324 MB | 888 (1.1 tok/s) |

TranslateGemma 4B has Gemma 3 4B's shapes, so its row stands for
`gemma-3-4b-it` too. Session-to-session drift on this machine (see
[measuring](design.md#measuring)) is larger than most of the gaps between
neighbouring rows, so read the table as a ranking rather than as constants.

Given "Rome is the capital of Italy, [...] The capital of France is", every
model continues with Paris in Q8_0. With `--q4` all of them still do except
the smallest: Qwen3-0.6B says Lyon, and the two 270M models wander off. The
4-bit format's accuracy cost shows first in small models.

### Measured separately

Two models that were not on this machine for the run above, measured the
same way in another session (best of 2):

| model | arch | file | `--q4` resident | `--q4` decode | `--q8` resident | `--q8` decode |
| --- | --- | --- | --- | --- | --- | --- |
| gemma-4-26B-A4B-it | gemma4, MoE | Q4_K_M, 16.80 GB | 14.7 GB | 109.2 (9.2 tok/s) | ~27 GB | — |
| Llama-3-8B (Finance-RAG) | llama | Q4_K_S, 4.69 GB | 4786 MB | 169.6 (5.9 tok/s) | 8137 MB | 284.6 (3.5 tok/s) |

The 26B A4B holds 26B weights but reads only ~3.8B per token — the dense
part, and the 8 of 128 experts its router picks — so it decodes about as fast
as a 4B dense model while needing the memory of a 26B one. Use `--q4` unless
the machine has well over 27 GB free.

### Qwen

Every Qwen model here is checked against llama.cpp (see
[Gated DeltaNet](design.md#gated-deltanet-qwen-35-36-38)).

The Gated DeltaNet layers barely show: their state is a few megabytes per
layer and mostly stays in cache, so the Qwen3.5 models stream their weights
at much the same rate as the all-attention Qwen3 ones (27 against 28 GB/s at
4B). Qwen3.8-27B is a dense 27B model and decodes like one, streaming nearly
all 16 GB of its `--q4` weights per token; use `--q4` unless the machine has
well over 28 GB free.

### Prefill

Reading a prompt costs far less per token than writing a reply, because a
batch of 16 prompt tokens shares each pass over the weights (see
[batching](design.md#batching)). Milliseconds per prompt token with a
34–35-token prompt, best of 3 interleaved rounds, measured alongside the
table above:

| model | `--q4` | `--q8` |
| --- | --- | --- |
| functiongemma-270m-it | 1.0 | 1.2 |
| gemma-3-270m-it | 1.3 | 1.1 |
| qwen2.5-0.5b-instruct | 2.6 | 2.9 |
| Qwen3-0.6B | 3.0 | 3.5 |
| Qwen3.5-0.8B | 4.4 | 4.3 |
| gemma-3-1b-it | 4.5 | 5.3 |
| Llama-3.2-1B-Instruct | 6.9 | 6.5 |
| gemma-4-E2B-it | 14.5 | 14.3 |
| translategemma-4b-it | 21.8 | 22.0 |
| Qwen3-4B-Instruct-2507 | 22.8 | 25.6 |
| Qwen3.5-4B | 23.7 | 24.8 |
| gemma-4-E4B-it | 36.2 | 33.3 |
| Qwen3.8-27B | 269 | 279 |

Prefill is compute-bound, so `--q4` is no faster than `--q8` here: its
weights take as much work to widen, though fewer bytes to read.

### With an MTP drafter

Gemma 4 models use their `mtp-*.gguf` drafter automatically when it is found
(see [speculative decoding](design.md#speculative-decoding-mtp)). The output
is identical; only the speed changes, and it depends on how predictable the
text is. Greedy, 120 tokens of a short story (prose) and of a Python function
(code), each the best of 3 runs interleaved with the same model without its
drafter; the 26B A4B row was measured the same way in a separate session:

| model | prose: without → with | code: without → with | guesses kept (prose / code) |
| --- | --- | --- | --- |
| gemma-4-E2B-it (Q4_K_M → Q8_0) | 11.1 → 14.8 tok/s | 11.0 → 22.8 tok/s | 62% / 58% |
| gemma-4-26B-A4B-it (`--q4`) | ~7.5 → ~9.2 tok/s | ~7.6 → ~11.5 tok/s | 68% / 81% |

On code the drafter is confident more often, so a round checks more guesses,
and E2B's replies come twice as fast even though a smaller share is kept.

On a prose prompt, llama.cpp's own MTP keeps 29% of E2B's guesses with its
default of three guesses and no confidence cut-off; nnc keeps 30% under
those settings, which is how the default cut-off of 0.7 was chosen.

### Embedding models

| model | arch | out | resident | one sentence | ~1500-token text |
| --- | --- | --- | --- | --- | --- |
| embeddinggemma-300M (Q8_0) | gemma-embedding | 768 | ~320 MB | 14 ms | 2.4 s |
| embeddinggemma-2 (Q8_0) | gemma-embedding2 | 768 | ~300 MB | 18 ms | 3.8 s |

Timed with piped input, as the difference between runs with many inputs and
with one, so loading cancels out; best of 3 or more runs.

Both match Google's published similarities and llama.cpp's vectors
(cosine ≥ 0.9994); see [embedding models](design.md#embedding-models).

## Reading the table

Decode is bandwidth-bound, so the ranking is essentially resident bytes and
almost nothing else. Dividing the bytes a token reads by decode time gives
the rate the memory system is actually sustaining. That is the resident
size, except for Qwen3.8-27B, which keeps its lm_head apart from its
input-embedding table, so the table (0.8 GB) is only looked up:

| model | bytes read ÷ decode (`--q4`) |
| --- | --- |
| gemma-3-1b-it | ~28 GB/s |
| Llama-3.2-1B-Instruct | ~29 GB/s |
| translategemma-4b-it (Gemma 3 4B) | ~29 GB/s |
| Qwen3-4B-Instruct-2507 | ~28 GB/s |
| Qwen3.5-4B | ~27 GB/s |
| Qwen3.8-27B | ~30 GB/s |
| gemma-4-E2B-it | ~60 GB/s (?) |
| gemma-4-E4B-it | ~46 GB/s (?) |

The first six land on this machine's ~28 GB/s roof, which is the whole
story for them: they stream every weight once per token and the DRAM bus
sets the pace.

The two gemma-4 figures are *above* the measured roof, which is impossible
and therefore a sign the model doesn't work the way the arithmetic assumes.
It doesn't: Gemma 4 E2B and E4B, like Gemma 3n, keep per-layer-input
embedding tables (`ple_dim=256`) that are **row lookups, not streamed
matrices**, and E2B shares KV across 20 of its 35 layers
(`shared_kv_layers=20`). A large slice of
resident memory is therefore never read on a given token. So for gemma-4,
resident size overstates the per-token traffic — it is a memory-footprint
number, not a speed predictor.

Two practical consequences:

- **`--q4` is ~1.6× faster than `--q8`** (1.5–1.7× in the table) across the
  board, at a real accuracy cost. It is the same weights at 0.625 vs 1.0625
  bytes each.
- **Source quantization does not affect decode speed.**
  `gemma-3-1b-it-BF16.gguf` and `gemma-3-1b-it-Q6_K.gguf` are the same
  model and both decode to **595.9 MB** resident under `--q4`, benchmarking
  at 23.4 vs 22.9 ms/tok (best of 7, interleaved) — the same number within
  noise. The 2.01 GB vs 1.01 GB file size only changes load time, and
  `--bf16` is the one mode where it matters.

## What does not load

| file | why |
| --- | --- |
| `mmproj-*.gguf` | Vision projectors, not standalone language models. The local-model picker skips files whose name contains `mmproj`. |
| `dflash-*.gguf` | Block-diffusion drafters (a different speculative technique from MTP). The picker skips them. |
| `mtp-*.gguf` on their own | MTP drafters load, but only help the model they were made for (`--draft`, or automatically); the picker skips them. |
| other mixture-of-experts architectures | Only Gemma 4's experts are implemented; others are refused at load. That includes Qwen's (`qwen2moe`, `qwen3moe`, `qwen35moe`, `qwen3next`: Qwen3-30B-A3B, Qwen3.5/3.6-35B-A3B, Qwen3-Next). |
| `qwen4exp` (Qwen3.8-Flash-Next) | A 125B mixture of experts (6B active) plus 51B of n-gram embeddings, with sparse attention, hyper-connections and 512 experts that nnc does not implement; in nnc's formats it would also need about 80–110 GB of RAM. Refused with a message that says so. Qwen3.8-27B is the Qwen 3.8 model nnc runs. |
| `mtp-Qwen*.gguf` | llama.cpp's Qwen3.5/3.8 MTP drafter on its own; refused. The MTP block inside a full Qwen3.5 file is skipped. |
| Qwen3-Embedding, Qwen-VL | Qwen's text-embedding models (a `pooling_type`) and vision models (`qwen2vl`, `qwen3vl`, …) are refused. |

## Reproducing

```powershell
.\exe\nnc.exe --q4 --gemma-prompt <file.gguf> "<prompt>" 33
```

`--gemma-prompt` runs the raw prompt, without a chat template, greedily, and
reports prefill and decode separately. 33 tokens are the one the prompt
predicts and 32 decode steps. The decode line counts actual decode steps,
so a run that stops early on EOS still divides by the right number, though
a model that answers in a word leaves too few steps to time; a prompt that
asks for an explanation avoids that. Add `-v` for the exact resident size
and `--perf` for the per-phase breakdown.

Note that session-to-session variance on a desktop is around ±20%, which is
larger than most changes worth measuring — see
[measuring](design.md#measuring) before drawing conclusions from a single
run.
