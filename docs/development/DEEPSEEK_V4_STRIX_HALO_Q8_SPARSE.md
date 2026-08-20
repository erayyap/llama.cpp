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

### Query-private follow-up

A same-session 32K-prefix comparison against the previous concatenated compact implementation measured:

| Workload | Concatenated compact | Query-private compact | Change |
|---|---:|---:|---:|
| Low-acceptance prose | 10.56 tok/s | 10.79 tok/s | +2.2% |
| High-acceptance code | 15.64 tok/s | 17.88 tok/s | +14.3% |
| Ten-case deep gate mean | 14.11 tok/s | 16.51 tok/s | +17.0% |

The deep gate remained byte-identical on 10/10 outputs and semantically identical at 9/10 versus 9/10; the shared arithmetic miss was again caused by the synthetic archive prefix. Focused q8 correctness passed batches 1, 2, 3, 5, and 6, including sinks, invalid indices, per-query masks, and a two-stream batch-5 case.

## Commits

The fork-specific sequence is:

- `5b6443d4` — adaptive DSpark depth;
- `109292da` — q8_0 sparse decode gathering;
- `2ba5970d` — batched speculative gathering;
- `05b82c24` — query-private sparse verification segments.

The untouched non-sparse adaptive runtime remains a straightforward rollback target, and `GGML_VK_FA_TOPK_GATHER=0` provides a same-binary control.
