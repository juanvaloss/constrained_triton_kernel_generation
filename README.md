# Constrained Decoding for Triton GPU Kernels

*Does grammar-guided generation plus a few-shot pattern library help a small LLM write valid Triton kernels — and how far does it close the gap to frontier models?*

This repository contains an end-to-end experiment that evaluates **five generation regimes** on the **TritonBench-T** benchmark (PyTorch → Triton operator translation). The central question is whether **XGrammar-constrained decoding** and a small **few-shot pattern library** can lift the kernel-generation quality of a local, quantized 7B coder model (`Qwen2.5-Coder-7B-Instruct`) closer to two frontier APIs (Claude Sonnet 4.6 and DeepSeek-V3).

---

## TL;DR results

50 randomly sampled TritonBench-T operators, fixed seed (`SEED=42`), single greedy decode per item.

| Regime | Model | Decoding | `call@1` | `exe@1` | speedup* |
|---|---|---|---|---|---|
| `qwen_free` | Qwen2.5-Coder-7B (int4) | unconstrained | 4.0% | 4.0% | — |
| `qwen_constrained` | Qwen2.5-Coder-7B (int4) | XGrammar + keyword mask | 30.0% | 30.0% | 5.30× |
| `qwen_patterns` | Qwen2.5-Coder-7B (int4) | XGrammar + keyword mask + 1 few-shot | 30.0% | 30.0% | 7.58× |
| `claude` | Claude Sonnet 4.6 (API) | unconstrained | 68.0% | 68.0% | 1.66× |
| `deepseek` | DeepSeek-V3 (API) | unconstrained | 56.0% | 56.0% | 1.16× |

\* **Speedup is *not* comparable across regimes.** Each value is averaged over the kernels that *that* regime passed, and those operator subsets differ in both count and difficulty. See [Interpreting the results](#interpreting-the-results-read-this).

**Main takeaway that holds up:** grammar-constrained decoding rescues the small model from near-total failure (4% → 30% `call@1`), but frontier models still lead on correctness by a wide margin.

---

## What this measures

The benchmark uses TritonBench-T's official three-phase evaluator:

- **`call@1`** — the generated kernel compiles and runs without crashing.
- **`exe@1`** — the kernel's output matches the PyTorch reference numerically.
- **speedup** — wall-clock vs. the PyTorch reference (`triton.testing.do_bench`), measured only on kernels that pass `exe@1`.

All five regimes share the **same system prompt**, so the only variables are the decoding strategy and the optional few-shot pattern.

---

## Method

### 1. EBNF grammar (`TRITON_EBNF`)

A deliberately **shallow** context-free grammar in Extended Backus–Naur Form, compiled by XGrammar into a pushdown automaton. At each decoding step XGrammar computes a token mask: any token that would push the prefix outside the language gets `-inf` logits.

The grammar encodes the *shape* of a Triton kernel translation (required imports → `@triton.jit` kernel(s) → Python wrapper) rather than full Python. It guarantees:

- output starts with the correct imports;
- output contains an `@triton.jit`-decorated function;
- identifiers come from a restricted alphabet;
- a curated `tl.*` / `triton.*` vocabulary (loads, stores, reductions, math, atomics, shape ops).

It does **not** guarantee correct masking, correct reduction axes, or correct math — those are semantic concerns left to `exe@1`.

> Design note: XGrammar has no negative lookahead, so Python keywords that are syntactically valid identifiers but illegal in a kernel body (`class`, `lambda`, `yield`, …) are filtered outside the grammar by a `LogitsProcessor` (see below).

### 2. Forbidden-tokens processor

`ForbiddenTokensProcessor` is a Hugging Face `LogitsProcessor` that masks 26 Python-keyword token IDs the grammar can't exclude on its own. A sanity check confirms none of the masked IDs collide with tokens the grammar needs.

### 3. Pattern library + picker

Five validated kernels distilled from the official Triton tutorials, with **intentionally abstract identifiers** (`example_kernel`, `arg_tensor`, `scalar_factor`, …) so the model adapts the structure rather than copying names:

`scale_mask`, `reduce_two_stage`, `softmax_row_reduce`, `layernorm_single_pass`, `dropout_with_mask`.

`pick_pattern_for_instruction()` is a keyword heuristic that selects one pattern per operator and injects it into the prompt as a single few-shot example (only in the `qwen_patterns` regime).

### 4. Output cleaning

`process_code()` mirrors TritonBench-T's own extractor: it strips markdown fences and chat-template artefacts, then iteratively truncates from the end until `ast.parse()` succeeds.

---

## Repository / notebook layout

The experiment is a single self-contained notebook organized into sections:

| Section | Contents |
|---|---|
| 1 | Environment + GPU check, dependency install |
| 2 | Load `Qwen2.5-Coder-7B-Instruct` in int4 (NF4) |
| 3 | `TRITON_EBNF` grammar + XGrammar compilation |
| 4 | Pattern library + `pick_pattern_for_instruction` |
| 5 | `SYSTEM_PROMPT`, `process_code`, `ForbiddenTokensProcessor` |
| 6 | Clone TritonBench-T, install frontier SDKs, sample 50 ops |
| 7–8 | Generation for all five regimes (with resume-from-disk) |
| 9 | Phase 1 (`call@1`) + Phase 2 (`exe@1`) |
| 10 | Phase 3 (speedup) |
| 11 | Final comparison table |
| 12–13 | Backup + discussion |

---

## Requirements

- **GPU.** Developed/run on an NVIDIA RTX PRO 6000 (Blackwell, ~96 GB), CUDA 12.8. The int4 model fits in ~5.6 GB VRAM, so a 16 GB card (e.g. T4) is enough for the Qwen regimes; Phase 3 benchmarking is the heaviest step.
- **Python 3.12**, `torch 2.11`, `transformers 5.0`, `triton 3.6`, `xgrammar 0.2.1`, `bitsandbytes 0.49.2`.
- **API keys** (optional) for the frontier regimes: `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`. A missing key skips that regime silently.

## Reproducing

1. Run sections 1–6 to set up the model, grammar, helpers, and operator subset.
2. Run section 7–8 to generate predictions for all regimes (resumable — re-running picks up from the last saved index).
3. Run sections 9–10 for evaluation. Set `RUN_PHASE_3 = False` to skip the expensive speedup phase.
4. Section 11 prints the final table.

To run the full benchmark instead of the 50-op subset, set `N_OPS = 166` in section 6.4.

---

## Interpreting the results

The headline qualitative finding is sound, but three properties of the current numbers mean they should **not** be over-interpreted:

1. **`call@1` equals `exe@1` in every regime.** In the TritonBench-T literature, `exe@1` is normally strictly lower than `call@1` (plenty of kernels run but produce wrong numbers). The fact that every kernel which compiled also passed numeric verification (`28/28 = 100%` for DeepSeek, etc.) suggests the Phase-2 correctness check is not discriminating as expected. Treat `exe@1` as unverified until the reference comparison and tolerances are audited.

2. **Cross-regime speedup is confounded.** Speedup is averaged over each regime's *own* surviving kernels. The small model only survives on easy elementwise operators (where a naive Triton kernel beats PyTorch eager overhead handily), while the frontier models also solve harder operators (where well-optimized PyTorch is hard to beat). The higher Qwen speedup therefore reflects *operator selection*, not better code generation.

3. **Single greedy run, n = 50, no variance estimate.** There are no confidence intervals or significance tests. The 4% → 30% jump is large enough to be real signal; the identical 30%/30% for `constrained` vs `patterns` indicates the pattern library did **not** improve correctness on this sample.

A fairer next iteration would: audit the Phase-2 numeric check; report speedup only on the operators **all** regimes solved (paired comparison); and run `pass@k` with multiple seeds to get error bars.

---

## Limitations

- 50/166 operators, fixed seed; bump to 166 for the full benchmark.
- Single-shot greedy decoding (no `pass@k`, no temperature sweep) — by design, since the variable under study is the decoding constraint.
- One five-entry pattern library and a keyword picker; some patterns (softmax, layernorm, dropout) overlap directly with benchmark operators, a mild contamination concern worth a failure analysis.
- The EBNF grammar is a hand-built subset and can structurally exclude valid solutions for operators it did not anticipate.

## License

Specify a license before publishing (e.g. MIT). TritonBench-T and the Triton tutorials retain their own licenses.

## Acknowledgements

Built on [TritonBench](https://github.com/thunlp/TritonBench), [XGrammar](https://github.com/mlc-ai/xgrammar), and the official [Triton](https://github.com/triton-lang/triton) tutorials.
