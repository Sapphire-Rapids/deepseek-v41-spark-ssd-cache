# Sources and changes

The disk tier and the compatibility fixes build on existing community work. Their attribution and Apache-2.0 notices are retained.

| Source | Pinned revision | Use |
| --- | --- | --- |
| [hughmadden: NodeLocalDiskOffloadingSpec](https://gist.github.com/hughmadden/03f369339b709a80fd594f4083702722) | `9c365a4fa5cf42164ef9b40d9528eeb334afed49` | Per-rank disk storage, CPU transfer path, streaming ring |
| [vLLM PR #54743, nood-co1](https://github.com/vllm-project/vllm/pull/54743) | `899699c74ae2b8e8adc8726e5c9d0e355935076a` | Restrict offloading to prefix-cacheable groups |
| [vLLM PR #47891, phamduchuy1892004-gif](https://github.com/vllm-project/vllm/pull/47891) | `a9eb2db3460fc10c30b0b9bcc49252b87ae35e7b` | DSpark draft-cache lookup exemption |
| [vLLM contributors](https://github.com/vllm-project/vllm) | `0.1.dev20904+g179dd0fa9` serving image | Eight source files in `overlay/`, all retaining their vLLM copyright notices |
| [MiaAI-Lab V4.1 dual-Spark deployment](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) | V4.1 EXL3 runtime family | Working deployment used as the starting point; its launcher is not redistributed here |

The original disk gist declares Apache-2.0. The vLLM source and upstream fixes are Apache-2.0. `LICENSE` contains the Apache-2.0 text. New packaging helpers and synthetic tests are distributed under the same license.

## Adaptations carried here

- Disk-capacity admission, LRU eviction, in-flight load protection, and coordinated file deletion, carried forward from an earlier local DeepSeek V4 deployment.
- V4.1 scratch-group filtering while preserving original group IDs in stored keys and GPU transfer vectors.
- Backport of DSpark lookup behavior to this runtime's normalized offloading configuration and cache coordinator. Ordinary EAGLE misses still prevent a hit.
- Per-process CPU buffers and a small streaming ring for the two-node setup.
- Fixed runtime settings, source hashes, synthetic transfer tests, and restart measurements.

`overlay-manifest.json` records the original installed hash, the tested adapted hash, and the distributed hash for every overlaid file. Distribution adds a provenance comment without changing executable code. The tested disk module SHA-256 was `34796b9c629d0e337ba2ba67482f3a4cb7562be6deb38d24545eb8b3f1339ac6`; its distributed copy likewise adds an attribution comment.

## Reference test configuration

The weight repository was [manateelazycat/DeepSeek-V4.1-Flash-Uncensored-TensorFold-EXL3](https://huggingface.co/manateelazycat/DeepSeek-V4.1-Flash-Uncensored-TensorFold-EXL3), revision `c0ce542ae7e4f61d86cbb0225276392912552f1b`. Model weights are not included. Other checkpoints have not been tested by this recipe.

The two upstream fixes above were open and unmerged when checked on 2026-10-12. [PR #48459](https://github.com/vllm-project/vllm/pull/48459) separately tracks the DSpark core-cache change. [PR #46556](https://github.com/vllm-project/vllm/pull/46556) addresses worker-executed transfers to shared storage; it was considered but is not used here.
