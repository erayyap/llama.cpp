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

For a batch of `N` queries, scratch K contains:

- the shared raw prefix;
- one top-k segment for each query;
- alignment padding.

Each compact mask exposes the raw prefix and only that query's selected segment. Selections are intentionally not deduplicated across queries: batches are small, this avoids a GPU hash/union pass, and it remains much cheaper than scanning a deep full cache.

The path engages only when the complete source cache is at least twice the padded compact set. Otherwise it keeps the existing dense path.

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

| Query batch | Dense q8_0 | Compact gather | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|
| 1 | 2878.375 us | 71.320 us | 40.36x | 97.52% |
| 2 | 17556.780 us | 522.435 us | 33.61x | 97.02% |
| 3 | 17756.890 us | 673.615 us | 26.36x | 96.21% |
| 5 | 18248.440 us | 953.985 us | 19.13x | 94.77% |
| 6 | 18394.755 us | 1095.935 us | 16.78x | 94.04% |

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

## Commits

The fork-specific sequence is:

- `5b6443d4` — adaptive DSpark depth;
- `109292da` — q8_0 sparse decode gathering;
- `2ba5970d` — batched speculative gathering.

The untouched non-sparse adaptive runtime remains a straightforward rollback target, and `GGML_VK_FA_TOPK_GATHER=0` provides a same-binary control.
