# DeepSeek V4 Flash on Strix Halo: adaptive DSpark and q8 sparse attention

> [!WARNING]
> **This fork-specific work is LLM-generated experimental “slop.”** It may be useful, but it is provided as-is without guarantees. Review the implementation and reproduce the tests before relying on it.

## Scope

This branch targets DeepSeek V4 Flash inference through the Vulkan backend on AMD Ryzen AI Max+ 395 / Radeon 8060S (Strix Halo, RADV, wave64). The tested deployment used:

- a quantized DeepSeek V4 Flash main model;
- q8_0 K and V caches;
- the Q8_0 DSpark drafter;
- one server slot;
- adaptive speculative depth from 2 through 5;
- a 409,600-token production context.

The changes are opt-in or narrowly shape-gated. Other models and unsupported attention shapes continue through the existing paths.

## Changes

### Adaptive DSpark depth

`common/speculative.cpp` supports request-local speculative-depth adaptation when:

```bash
export LLAMA_DSPARK_ADAPTIVE=1
```

The controller:

- starts each request at depth 2;
- permits depths 2 through the configured `--spec-draft-n-max` value (tested through 5);
- maintains an exponential moving average of draft acceptance;
- increases depth at high acceptance and backs off at low acceptance;
- leaves the original static behavior unchanged when the environment variable is absent.

The tested Q8 DSpark artifact declares `dflash.block_size=5`. Requests for depths 6–8 are therefore clamped to 5 by the drafter even though the sparse-attention backend itself is correctness-tested through batch 8.

### q8_0 sparse decode gathering

DeepSeek V4 compressed sparse attention carries a dense raw prefix plus top-k compressed-cache indices. Before this change, the Vulkan gather-to-compact decode path accepted only f16 K/V. A production q8_0 cache therefore fell back to flash attention across the complete compressed cache despite most rows being masked.

The new q8_0 gather shader:

1. reads the raw prefix and selected q8_0 cache rows;
2. dequantizes only those active rows into a compact f16 scratch;
3. preserves the exact source mask values;
4. reuses the ordinary dense f16 flash-attention implementation on the compact set.

The path requires the DeepSeek V4 CSA shape already recognized by the backend: 512-wide K/V latent, 64 query heads, MQA K==V storage, f16 mask, and attached top-k indices.

Disable it for A/B tests or rollback with:

```bash
export GGML_VK_FA_TOPK_GATHER=0
```

### Batched speculative verification

DSpark verifies multiple candidate tokens in one target-model batch. The compact path therefore supports query batches from 1 through 8.

The initial batched implementation concatenated a shared raw prefix with every query's top-k segment. Each query then attended across that whole rectangle while masking the other queries' segments. This was much faster than scanning the deep cache, but its attention work grew approximately quadratically with speculative depth.

The retained query-private layout instead gathers one independently padded `raw prefix + top-k` segment per query. The host issues one single-query flash-attention dispatch per segment, with correctly offset Q, output, mask, and multi-stream scratch views. This duplicates the raw-prefix gather for tiny batches but substantially reduces attention work. Split-K is disabled for these private dispatches because 64 query-head workgroups already fill the tested GPU and independent split temporaries would add overhead.

The path supports batches 1 through 8 and engages when the complete source cache is at least twice one padded private active set. Otherwise it keeps the existing dense path.

### q8_0 decode projection row coarsening

A bounded non-speculative Vulkan profile found ordinary q8_0 matrix-vector projections consuming about 29.5 ms, or 48.5% of the measured 60.9 ms single-token GPU graph. The `m=32768, n=1, k=1024` projection was the strongest isolated row-reuse candidate.

`GGML_VK_Q8_DMMV_ROWS4=1` enables a shape-gated pipeline that computes four output rows per workgroup only for that exact q8_0/f32 projection. Other dimensions, types, and generic matvec heuristics are unchanged. Setting the variable to `0` or leaving it unset is the same-binary rollback control.

Global row coarsening, forced integer-dot MMVQ, wave32, and 256-thread decode-vector workgroups were screened and rejected because they were mixed or slower at full-model level.

### Long-context cooperative-matrix indexer batches

DeepSeek V4's lightning indexer chooses compressed-cache rows for sparse attention. The existing cooperative-matrix decode shader was used only for a one-token graph; speculative verification batches of 2–5 fell back to a much slower scalar/subgroup shader even though the cooperative-matrix shader already supports an arbitrary token index.

`GGML_VK_LIGHTNING_DECODE_CM_BATCH=N` extends the cooperative-matrix shader through batch `N`, clamped to 0–8. Leaving the variable unset or setting it to `0` preserves the prior selector. The production DSpark configuration uses `N=5`; testing through 8 covers the backend's supported verification range.

This path is numerically different from the scalar kernel because it converts query inputs to f16 cooperative-matrix operands and changes floating-point evaluation. It is therefore accepted on semantic correctness, perplexity, and task accuracy rather than byte-identical generated text. Keep the environment control available for workload-specific rollback.

### Experimental q8_0 indexer cache and fused top-k

Two additional long-context paths are implemented but are **not production-promoted** because fixed 32K-prefix model throughput was mixed/neutral despite strong bounded-operation gains.

`LLAMA_DSV4_LID_Q8_0=1` stores the independent 128-wide lightning-indexer K cache as q8_0 when the public K cache is also q8_0. Vulkan scalar, prefill cooperative-matrix, and decode cooperative-matrix shaders dequantize one complete q8_0 block per lane directly into shared f16 tiles. Unset or `0` retains f16. The q8 and f16 cache-state files are not mutually restorable, so deployments testing this switch should use a separate slot-save directory.

`GGML_VK_LIGHTNING_TOPK_FUSE=1` combines decode indexer scoring with the first top-k reduction. Four wave64 subgroups score 1,024 keys per workgroup, retain 512 block-local candidates, and feed those `(index, score)` pairs into the existing top-k merge passes. It defaults to compressed KV lengths of at least 32,768; `GGML_VK_LIGHTNING_TOPK_MIN_KV` overrides that threshold. Unset or `0` preserves the separate indexer and top-k operations.

The fusion deliberately remains off by default. It is numerically different at top-k boundaries, and safe full-model profiling/filling at the production 400K allocation was not repeated after the earlier UMA overcommit.

## Build

```bash
cmake -S . -B build-vulkan \
  -DGGML_VULKAN=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-vulkan --config Release \
  --target llama-server test-backend-ops -j "$(nproc)"
```

## Example server

Replace model paths with compatible GGUF files:

```bash
LLAMA_DSPARK_ADAPTIVE=1 \
GGML_VK_FA_TOPK_GATHER=1 \
GGML_VK_Q8_DMMV_ROWS4=1 \
GGML_VK_LIGHTNING_DECODE_CM_BATCH=5 \
./build-vulkan/bin/llama-server \
  -m /path/to/deepseek-v4-flash.gguf \
  -md /path/to/dspark-q8_0.gguf \
  -ngl all -ngld all \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -c 409600 -np 1 \
  -b 3072 -ub 3072 \
  --spec-type draft-dspark \
  --spec-draft-n-max 5 \
  --fit off --jinja --cache-prompt
```

Large contexts on unified-memory systems can destabilize the desktop. The tested 128 GiB machine used `b3072/ub3072`; `ub4096` with a 400K context caused severe memory pressure and should not be treated as safe merely because allocation succeeds.

## Correctness tests

Focused q8 sparse cases are registered in `tests/test-backend-ops.cpp`, including masks, sinks, invalid indices, and speculative batch widths.

```bash
./build-vulkan/bin/test-backend-ops test \
  -b Vulkan0 -o FLASH_ATTN_EXT \
  -p 'n_kv_raw='
```

A targeted example:

```bash
./build-vulkan/bin/test-backend-ops test \
  -b Vulkan0 -o FLASH_ATTN_EXT \
  -p 'kv=8192,nb=5,n_kv_raw=1024,n_top_k=512,sinks=1,type_K=q8_0'
```

The focused q8 cases passed against the CPU reference. A broader Vulkan q8 flash-attention filter passed 1328/1328 cases in the development worktree.

The decode-projection case can be checked separately:

```bash
GGML_VK_Q8_DMMV_ROWS4=1 ./build-vulkan/bin/test-backend-ops test \
  -b Vulkan0 -o MUL_MAT \
  -p 'type_a=q8_0,type_b=f32,m=32768,n=1,k=1024.*'
```

This passed against the CPU reference, and the fixed DSpark quality gate retained 10/10 semantic passes and 10/10 exact reference SHA-256 hashes.

## Performance

### Operation-level ABBA

At 32,768 compressed K rows, `n_kv_raw=1024`, and `n_top_k=512`:

| Query batch | Dense q8_0 | Concatenated compact | Query-private compact | Private vs concatenated |
|---:|---:|---:|---:|---:|
| 1 | 2878.375 us | 71.320 us | ~72 us | neutral |
| 2 | 17556.780 us | 524.455 us | 143.150 us | 3.66x faster |
| 3 | 17756.890 us | 665.190 us | 246.300 us | 2.70x faster |
| 5 | 18248.440 us | 976.045 us | 474.025 us | 2.06x faster |
| 6 | 18394.755 us | 1123.485 us | 614.730 us | 1.83x faster |

The concatenated/private columns for batches 2–6 are same-session ABBA means. Query-private latency reductions versus the concatenated implementation were 72.70%, 62.97%, 51.43%, and 45.28%, respectively.

Single-query context sweep:

| Compressed K rows | Dense q8_0 | Compact gather | Speedup |
|---:|---:|---:|---:|
| 8,192 | 695.170 us | 70.840 us | 9.81x |
| 32,768 | 2844.300 us | 74.235 us | 38.31x |
| 65,536 | 3248.345 us | 73.375 us | 44.27x |

These are attention-operation measurements, not whole-model token rates.

### Full-model 32K-prefix tests

Same binary, with only `GGML_VK_FA_TOPK_GATHER` changed between candidate and dense control:

| Workload | Dense control | q8 sparse gather | Change |
|---|---:|---:|---:|
| Low-acceptance prose | 9.22 tok/s | 13.90 tok/s | +50.8% |
| High-acceptance code | 17.57 tok/s | 18.91 tok/s | +7.6% |
| Short long-context retrieval | 15.41 tok/s | 24.33 tok/s | +57.9% |
| Ten-case deep gate mean | 13.51 tok/s | 18.27 tok/s | +35.3% |

Quality checks:

- ordinary fixed gate: 10/10 semantically valid and 10/10 exact retained hashes;
- fixed gate behind a 32K prefix: candidate and dense control matched SHA-256 on 10/10 outputs;
- the synthetic deep gate produced 9/10 semantic passes in both arms, with the same archive-prefix-confounded arithmetic response;
- long retrieval returned the exact expected key in every arm.

Free-form 256-token prose/code generations were coherent but not byte-stable across repeated prefix-cache runs, including the dense control. They should not be interpreted as an exact-output gate.

### Shape-gated q8_0 decode projection

Isolated same-session ABBA for `m=32768, n=1, k=1024` measured:

| Path | Mean time |
|---|---:|
| Generic q8_0 dequantize-matvec | 155.495 us |
| Four-row shape-gated pipeline | 87.375 us |

That is a 43.8% operation-latency reduction (1.78x speedup). Two 512-token full-model campaigns, one ABBA and one reverse BAAB, retained identical output SHA-256 hashes in every arm. Candidate latency improved by 0.8% in the first campaign and 5.2% in the reverse campaign; pooling all eight arms gave a 3.1% mean latency reduction. Firmware power drift was substantial, so the campaign range is more honest than treating the pooled figure as a guaranteed token-rate gain.

### Query-private follow-up

A same-session 32K-prefix comparison against the previous concatenated compact implementation measured:

| Workload | Concatenated compact | Query-private compact | Change |
|---|---:|---:|---:|
| Low-acceptance prose | 10.56 tok/s | 10.79 tok/s | +2.2% |
| High-acceptance code | 15.64 tok/s | 17.88 tok/s | +14.3% |
| Ten-case deep gate mean | 14.11 tok/s | 16.51 tok/s | +17.0% |

The deep gate remained byte-identical on 10/10 outputs and semantically identical at 9/10 versus 9/10; the shared arithmetic miss was again caused by the synthetic archive prefix. Focused q8 correctness passed batches 1, 2, 3, 5, and 6, including sinks, invalid indices, per-query masks, and a two-stream batch-5 case.

## 128K follow-up profiling and rejected indexer experiments

A full 409,600-context server with the Vulkan per-node performance logger enabled destabilized the desktop during a 32K request and required a restart. No retained result depends on that run. Full-model performance logging at the 400K allocation is therefore not considered desktop-safe; the follow-up used bounded backend-operation graphs at 128K and ordinary, non-profiled full-model validation at `ctx=131072`, `b1024/ub1024`.

The retained q8 query-private attention operation stayed effectively constant as the compressed cache grew. With `n_kv_raw=1024` and `n_top_k=512`:

| Compressed K rows | Batch 1 | Batch 2 | Batch 5 |
|---:|---:|---:|---:|
| 8,192 | 73.87 us | 146.46 us | 463.56 us |
| 32,768 | 73.05 us | 148.42 us | 457.33 us |
| 65,536 | 72.48 us | 145.57 us | 451.20 us |
| 131,072 | 72.62 us | 147.05 us | 454.89 us |

The long-context growth instead appeared in `LIGHTNING_INDEXER`. The existing cooperative-matrix decode kernel is selected for one token, while batches 2–5 use the exact scalar/subgroup path:

| K rows | Batch 1 | Batch 2 | Batch 5 |
|---:|---:|---:|---:|
| 4,096 | 13.60 us | 90.69 us | 219.61 us |
| 65,536 | 137.65 us | 1,378.06 us | 3,465.73 us |
| 131,072 | 293.36 us | 2,769.65 us | 7,070.51 us |

The cooperative-matrix selector reduced operation latency by roughly 80–82% at 64K and 80% at 128K (about 4.9–5.6x). Under the later semantic-quality policy it was promoted after these additional comparisons:

| Quality benchmark | Scalar control | Cooperative-matrix batch | Result |
|---|---:|---:|---|
| WikiText-2, 512 context, 1 chunk | 4.9479 ± 0.85308 PPL | 4.9479 ± 0.85308 PPL | identical reported estimate |
| WikiText-2, 4096 context, 4 chunks | 4.6142 ± 0.12718 PPL | 4.6146 ± 0.12717 PPL | +0.0004 PPL; negligible versus uncertainty |
| Serial HellaSwag subset | 83/100 | 83/100 | 100/100 answers matched |
| Serial Winogrande subset | 73/100 | 73/100 | 100/100 answers matched |
| Fixed 32K-prefix semantic gate | 9/10 | 10/10 | candidate corrected the prefix-confounded arithmetic case |

All 200 serial multiple-choice outputs matched byte-for-byte despite exact identity not being required. Mean decode throughput was 4.4% higher on the HellaSwag subset and 3.9% higher on Winogrande. The ordinary fixed gate remained 10/10 semantically valid and 10/10 exact; the deep gate matched 8/10 hashes, demonstrating why exact hashes remain useful diagnostics but are not a quality metric by themselves.

A later q8-cache and fused-top-k campaign produced these bounded results:

| Path | 65K K, batch 1 | 65K K, batch 5 | 131K K, batch 1 | 131K K, batch 5 |
|---|---:|---:|---:|---:|
| f16 cooperative-matrix indexer | 140.54 us | 684.32 us | 298.49 us | 1,425.47 us |
| q8_0 cooperative-matrix indexer | 128.77 us | 615.10 us | 252.62 us | 1,258.18 us |
| q8 improvement | 8.4% | 10.1% | 15.4% | 11.7% |

The q8 row size is 136 bytes versus 256 bytes for f16, a 46.9% reduction in indexer-cache storage. Focused q8 scalar/prefill/decode tests passed 18/18 against CPU reference. WikiText-2 remained 4.9479 at 512 context; at 4096 context the estimate changed from 4.6146 ± 0.12717 to 4.6183 ± 0.12736 (+0.08%). HellaSwag and Winogrande serial subsets remained 83/100 and 73/100 with all 200 outputs matching the f16-indexer candidate.

At 131K compressed K, fused q8 scoring plus top-k took about 99 us / 162 us / 391 us for batches 1/2/5. The corresponding separate q8 indexer plus top-k budget was roughly 338 us / 644 us / 1,595 us, a 71–76% operation reduction. Structured block-candidate tests passed for f16 and q8_0 at batches 1/2/5; ordinary and deep server outputs matched the unfused q8 candidate on the fixed gates.

Whole-model 32K-prefix ABBA/BAAB was neutral and workload-dependent:

- q8 cache alone: +5.0% prose, -2.7% code, +0.3% pooled;
- fused top-k over q8: -1.7% prose, +2.0% code, +0.5% pooled.

Those prompts contain only about 8K compressed indexer rows and are below the retained fusion threshold. The paths remain opt-in experiments pending safe validation at materially deeper cache residency.

Two earlier follow-up experiments remained rejected:

1. **Two-query cooperative-matrix workgroups sharing K tiles.** This was 23–48% slower than independent one-query workgroups.
2. **Exact-order scalar groups widened from 8 to 16 keys.** Isolated gains fell from about 2.2% at 64K to 0.4–1.2% at 128K, below a meaningful full-model threshold.

The production service remained inactive and disabled throughout testing.

## General-TPS follow-up: rejected tree and packed GEMV candidates

Two broader decode ideas were screened after the long-context campaign. Neither was retained.

### DSpark two-branch tree

A width-two DSpark prototype generated an ordinary Markov top-1 branch and a second branch taking the position-zero top-2 candidate, then continuing through its own Markov chain. The target and draft contexts reserved a second sequence and verified both branches together for deterministic requests.

The prototype preserved the test output (`703` for `37*19`) but was substantially slower in a same-binary restart comparison:

| Path | Decode | Draft acceptance |
|---|---:|---:|
| Linear adaptive DSpark | 28.13 tok/s | 39/47 |
| Width-two tree | 14.98 tok/s | 36/48 |

The tree was about 46.7% slower. DSpark already produces an entire block in one semi-autoregressive pass, so a second branch duplicated draft and target work without enough additional acceptance. Reserving a second non-unified sequence also halved the reported per-sequence context; preserving 409,600 tokens would require an unsafe doubled KV allocation or a more invasive unified-memory design. The prototype was fully removed.

### Exact-shape q8_0 packed GEMV

The exact DeepSeek V4 q8_0 projection shapes were screened with:

- forced q8_1 integer-dot activation packing;
- 256-thread reduction workgroups;
- two-, four-, and eight-row output coarsening;
- a K16 packed kernel consuming 16 q8 values per lane instead of eight.

The K16 variants passed CPU-reference checks for rows 1/2/4 across five representative shapes, but operation ABBA averaged +0.03%, +1.31%, and +0.30% latency respectively. Large workgroups were about 32% slower. Broad row coarsening showed only about a 1.1% isolated mean at best and had already failed to improve whole-model throughput in the preceding campaign. All new selector and shader code was removed; the previously validated `32768x1x1024` rows4 gate remains the only retained q8 decode coarsening.

## Internet-inspired follow-up: exact indexer bounds and Markov tails

Two additional ideas were implemented or bounded after surveying recent long-context and speculative-decoding work. Both were rejected and removed.

### Exact lightning-indexer page bounds

A Quest-inspired exact variant used per-page coordinate minima and maxima to upper-bound

`sum_h weight_h * relu(dot(query_h, key))`.

Unlike approximate page ranking, a page would be skipped only if its upper bound could not reach the current exact top-512 threshold. Synthetic screens covered 128-dimensional normalized keys, 64 query heads, page sizes 1–1,024, and autoregressive temporal correlations from 0 through 0.999.

The dimensional box bound was too loose. Pages of four or more keys pruned nothing for correlations through 0.99. Even the deliberately extreme `rho=0.999` case pruned only 0.54% with four-key pages. Two-key pages pruned materially only at unrealistic extreme correlation, while computing page bounds plus exact scores for surviving pages required more work than the original cooperative-matrix scan. Page size one degenerates to the original exact score computation. No metadata cache or Vulkan selector was retained.

Raw screen: `/home/canavar/benchmarks/dsv4-inference-research/campaign-1-2/exact-page-bound-screen.json`.

### DSpark Markov-tail cascade

Several lossless-at-the-target speculative variants replaced the end of a five-token DSpark block with pure Markov-head proposals:

- separate 2+3, 3+2, and 4+1 continuation graphs;
- an inline 4+1 graph that avoided a second dispatch;
- an inline 4+1 path gated by the request-local acceptance EMA.

The separate continuations were generally slower. Ungated inline 4+1 was initially mixed: a 192-token coding run improved while arithmetic and prose regressed. DeepSeek emitted substantial `reasoning_content` even with `enable_thinking=false`, so subsequent comparisons included those tokens and used enough output budget for every coding answer to complete.

| Workload | EMA-gated 4+1 | Linear control | Delta |
|---|---:|---:|---:|
| Arithmetic/reasoning stream | 28.40 tok/s | 28.54 tok/s | -0.50% |
| Prose reasoning stream | 21.45 tok/s | 21.13 tok/s | +1.52% |
| Coding reasoning + answer | 26.15 tok/s | 26.67 tok/s | -1.96% |
| Pigeonhole reasoning stream | 20.89 tok/s | 21.01 tok/s | -0.59% |

Arithmetic, prose, and coding streams were identical between paths. The final reasoning case shared the same semantic prefix and diverged only at the maximum-token truncation boundary. A conservative EMA threshold never activated and was merely baseline behavior; a threshold that activated was neutral-to-slower overall.

The ungated inline 4+1 path was then retested directly with a 1,536-token budget. All three coding tasks reached their final answers. The planned ABBA was stopped after the first control/candidate pair because every candidate task showed a large regression:

| Completed coding task | Ungated 4+1 | Linear control | Delta |
|---|---:|---:|---:|
| Generator-safe longest run | 23.17 tok/s | 25.61 tok/s | -9.52% |
| O(1) LRU cache | 23.25 tok/s | 26.06 tok/s | -10.78% |
| Normalized interval merge | 21.95 tok/s | 24.83 tok/s | -11.60% |

Mean throughput changed by -10.63%, and draft acceptance fell on all three tasks. This establishes that the earlier truncated coding gain was not durable. All Markov-tail graph/API/driver changes were removed.

Raw runs and JSON: `/home/canavar/benchmarks/dsv4-inference-research/campaign-1-2/`.

### Direct indexed sparse attention

A follow-up attempted to remove the compact f16 scratch entirely. The existing direct f16 top-k shader was enabled for decode batches, then extended with a q8_0 variant that loads each selected q8 tile once, dequantizes it into LDS, and shares it across eight query heads. Key tiles of 8, 16, and 32 and four-head/256-thread versus eight-head/512-thread workgroups were screened.

Correctness passed the focused f16 case and all 7/7 q8 cases, including sinks, batches 1/2/3/5/6/8, and a two-stream batch. Performance did not justify retaining it. At 32,768 compressed rows:

| Query batch | Compact q8 gather | Direct q8 | Direct delta |
|---:|---:|---:|---:|
| 1 | 75.07 us | 403.71 us | +437.81% |
| 2 | 155.56 us | 445.82 us | +186.60% |
| 3 | 265.77 us | 488.32 us | +83.74% |
| 5 | 501.61 us | 576.14 us | +14.86% |
| 8 | 855.64 us | 820.69 us | -4.09% |

Direct access became slightly faster only at batch eight, while the deployed DSpark model is limited to five proposals by `dflash.block_size=5`. The gather path wins every production batch because dequantize-once scratch creation is cheap and the ordinary dense FA kernels exploit small batches much better than the barrier-heavy direct shader. The direct pipelines, selector, and test expansion were removed.

Raw ABBA logs: `/home/canavar/benchmarks/dsv4-inference-research/direct-indexed-attention/`.

### Q8 hot-shape attribution and vocabulary-head bound

A metadata inspection and bounded `GGML_VK_PERF_LOGGER=1` run corrected an initial hypothesis about the hot `m=32768, k=1024` q8_0 projection. It is `blk.<layer>.attn_q_b.weight`, not a DSpark vocabulary head. The Q8 DSpark file has three such tensors and no embedded vocabulary projection. Its decoder explicitly borrows the target model's `output.weight` through `ctx_other`; that tensor is Q8_0 with logical shape `129280 x 4096`.

In a 58-token reasoning-aware decode, the shared vocabulary projection took 2.46–2.63 ms per graph, 52 calls and 129.89 ms total, or 4.49% of the 2,894.20 ms decode wall time. Therefore even a free perfect replacement is bounded near 4.5% on this workload. A real hierarchical shortlist would add lookup/scoring cost and could lower draft acceptance, so vocabulary pruning was deprioritized without implementation.

For comparison, a representative target batch-three graph spent 21.80 ms in IQ2/Q2_K `MUL_MAT_ID`, 14.28 ms in the q8 expert batch-eight operation, 6.67 ms across forty `attn_q_b` projections, and 2.49 ms in the vocabulary head. This moves the next high-upside investigation toward route-coalesced MoE work.

Raw profile and metadata: `/home/canavar/benchmarks/dsv4-inference-research/dspark-head-attribution/`.

## Commits

The fork-specific sequence is:

- `5b6443d4` — adaptive DSpark depth;
- `109292da` — q8_0 sparse decode gathering;
- `2ba5970d` — batched speculative gathering;
- `05b82c24` — query-private sparse verification segments;
- `7c3b67b3` — shape-gated four-row q8_0 decode projection;
- `28633d8c` — opt-in cooperative-matrix lightning-indexer batches for long-context speculative verification;
- `0283af58` — experimental q8_0 indexer-cache and block-local indexer/top-k fusion, retained opt-in and not production-promoted.

The untouched non-sparse adaptive runtime remains a straightforward rollback target, and `GGML_VK_FA_TOPK_GATHER=0` provides a same-binary control.
