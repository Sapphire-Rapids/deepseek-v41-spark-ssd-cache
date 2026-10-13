# SPDX-License-Identifier: Apache-2.0
"""Verify actual GPU -> SSD -> fresh CPU staging -> GPU byte equality."""
import gc
import json
import tempfile
import time

import torch

from node_local_disk_offload import NodeLocalDiskManager, NodeLocalDiskWorker
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef, CanonicalKVCaches, CanonicalKVCacheTensor,
    GPULoadStoreSpec, LookupResult, ReqContext, make_offload_key,
)

torch.manual_seed(871)
gpu = torch.randint(-128, 128, (128, 4096), dtype=torch.int8, device='cuda')
expected = gpu[:40].cpu()
gpu[64:104].fill_(42)
torch.cuda.synchronize()
canonical = CanonicalKVCaches(
    tensors=[CanonicalKVCacheTensor(gpu, 4096)],
    group_data_refs=[[CanonicalKVCacheRef(0, 1024)],
                     [CanonicalKVCacheRef(0, 8)],
                     [CanonicalKVCacheRef(0, 2048)]])
keys = [make_offload_key(bytes([i + 1]) * 32, 0 if i < 20 else 2) for i in range(40)]
context = ReqContext('synthetic-gpu-ssd-io')


def finish(worker, job):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        for result in worker.get_finished():
            assert result.job_id == job and result.success, result
            return result
        time.sleep(0.002)
    raise TimeoutError(('GPU disk transfer', job))


with tempfile.TemporaryDirectory(prefix='dsv41-ssd-io-') as directory:
    def manager():
        return NodeLocalDiskManager(num_blocks=24, disk_root=directory,
            rows_per_group=(1, 1, 1), pack_groups=(), stream_loads=True,
            disk_max_bytes=1048576, reserved_bytes_per_key=4096)
    def worker():
        return NodeLocalDiskWorker(kv_caches=canonical, blocks_per_chunk=1,
            num_cpu_blocks=32, mmap_region=None, disk_root=directory,
            disk_io_threads=2, ring_first_row=24, ring_rows=8)
    m, w = manager(), worker()
    for start in range(0, 40, 10):
        subset = keys[start:start + 10]
        prepared = m.prepare_store(subset, context)
        assert list(prepared.keys_to_store) == subset
        sizes = [10, 0, 0] if start < 20 else [0, 0, 10]
        src = GPULoadStoreSpec(list(range(start, start + 10)),
                               group_sizes=sizes, block_indices=[0, 0, 0])
        assert w.submit_store(start, src, prepared.store_spec)
        finish(w, start)
        m.complete_store(subset, context)
    w.shutdown()
    m.shutdown()
    del w, m
    gc.collect()
    # Fresh objects have no hot CPU cache. All 40 keys must come from disk,
    # passing through an eight-row ring, smaller than the requested restore.
    m, w = manager(), worker()
    assert all(m.lookup(k, context) is LookupResult.HIT for k in keys)
    assert m._dbg_counts['disk_hit'] == 40 and m._dbg_counts['cpu_hit'] == 0
    source = m.prepare_load(keys, context)
    assert all(row == -1 for row in source.block_ids)
    destination = GPULoadStoreSpec(list(range(64, 104)),
                                   group_sizes=[20, 0, 20], block_indices=[0, 0, 0])
    assert w.submit_load(100, source, destination)
    finish(w, 100)
    m.complete_load(keys, context)
    torch.cuda.synchronize()
    restored = gpu[64:104].cpu()
    for index in range(40):
        size = 1024 if index < 20 else 2048
        assert torch.equal(restored[index, :size], expected[index, :size]), index
        assert bool(torch.all(restored[index, size:] == 42)), ('padding overwritten', index)
    w.shutdown()
    m.shutdown()
print(json.dumps({'gpu_disk_roundtrip': 'passed', 'keys': 40, 'ring_rows': 8,
                  'disk_only_loads': 40, 'scratch_loads': 0,
                  'payload_bytes_equal': True, 'padding_preserved': True}))
