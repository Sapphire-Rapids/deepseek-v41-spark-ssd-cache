# SSD KV cache for DeepSeek V4.1 on two DGX Sparks

[中文说明](README.zh-CN.md) · [Setup and replay](SETUP.md) · [Sources](SOURCES.md)

This recipe adds node-local SSD storage to an existing two-node V4.1 EXL3 deployment. Each node keeps its own KV shards. The cache does not require a shared filesystem. Repeated prompts can reuse saved KV after the serving processes restart.

It combines hughmadden's disk offloading spec with community fixes for scratch-cache groups and DSpark, plus bounded disk admission and adaptations to the tested V4.1 runtime. The intent is to leave enough code, configuration, and evidence for someone else to reproduce the result.

## Measured result

Two GB10 nodes, TP=2, DSpark K=3, a 256K context limit. Each node used a 128 GiB SSD budget and 512 MiB CPU staging pool, including a 128 MiB streaming ring.

| Prompt tokens | First-token wait, cold | After process restart | Cached tokens |
| ---: | ---: | ---: | ---: |
| 32,624 | 44.79 s | 1.12 s | 32,512 |
| 130,932 | 179.00 s | 2.20 s | 130,816 |

Both replays returned the exact identifier placed at the beginning of the prompt. Timings start after the server is ready; model loading is excluded. These are individual measurements, not latency distributions. The comparison is cold versus restored requests with SSD caching enabled. A separate 32K run before enabling SSD took 41.77 s.

GPU → disk → fresh CPU staging → GPU byte checks passed on both nodes. The test restores 40 keys through an eight-row ring. Tools, image requests, a follow-up to the long prompt, and four short concurrent requests also passed. Details and coding-test failures are in [results/validation.json](results/validation.json).

## Reuse

Start with a working V4.1 deployment using **vLLM `0.1.dev20904+g179dd0fa9`**. [SETUP.md](SETUP.md) covers building a derived image, adding the persistent mount and connector configuration on both nodes, checking the transfer code, and replaying a saved prompt after restart. The installer checks all eight original source hashes before changing anything.

The serving code here follows the tested implementation; provenance comments were added for distribution. The derived-image installer and configurable probe are packaging helpers. Their checks are recorded separately from the original serving measurements in [results/package-checks.json](results/package-checks.json).

## Scope

This is a version-specific recipe. Latest vLLM `main`, other model families, and other cache layouts have not been validated. The longest SSD replay was about 128K tokens; the configured ceiling was 256K. Four full-size contexts and a full 128 GiB fill were not tested.

Clean process restart was tested. Crash recovery, missing shards on one node, and prolonged eviction stress remain untested. Disk errors propagate; there is no demonstrated automatic recovery from a missing shard. The upstream module's append-only metadata can grow independently of the payload budget. Use a new cache directory on **both** nodes when changing weights, runtime, TP layout, or cache geometry.

Coding checks were mixed at temperature 0: the baseline passed 3/3; the first SSD set passed 1/3, with two outputs exhausting a 4K reasoning/output budget. One repeated reasoning. Isolation checks included an uncached code failure and a cached pass. Three checks at temperature 1, top-p 0.95 and an 8K budget passed. These small samples do not establish equal generation quality.

The code is shared under Apache-2.0. See [SOURCES.md](SOURCES.md) for the original authors, pinned revisions, and local changes. No weights or cached prompt payloads are included. AI assistance was used to adapt the implementation, run checks, and prepare this recipe.
