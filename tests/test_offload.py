# SPDX-License-Identifier: Apache-2.0
"""Focused checks for the community node-local tier and its two backports."""
import json
import runpy
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import MagicMock, Mock, patch

import torch

from node_local_disk_offload import NodeLocalDiskOffloadingSpec
from vllm.config import KVTransferConfig, ParallelConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import build_offloading_config
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig, OffloadingConnectorScheduler, RequestOffloadState,
)
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator, SpecGroup
from vllm.v1.core.kv_cache_utils import _annotate_eagle_groups
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec, FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import OffloadPolicy, ReqContext, get_offload_group_idx

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / 'test_bounded_disk.py'), run_name='__main__')

# Adapted from the scratch-group regression in community PR #54743.
with tempfile.TemporaryDirectory() as cache_dir:
    config = MagicMock()
    config.cache_config.block_size = 16
    config.cache_config.enable_prefix_caching = True
    config.cache_config.prefix_match_unit = None
    config.cache_config.prefix_cache_retention_interval = 4096
    config.cache_config.cache_dtype = torch.float16
    config.cache_config.kv_cache_layout = None
    config.model_config.model = 'test-model'
    config.model_config.use_mla = False
    config.model_config.get_total_num_kv_heads.return_value = 4
    config.speculative_config = None
    config.kv_events_config = None
    config.use_v2_model_runner = False
    with patch.object(current_platform, 'device_count', return_value=2):
        config.parallel_config = ParallelConfig(tensor_parallel_size=2,
                                                 distributed_executor_backend='mp')
    config.kv_transfer_config = KVTransferConfig(
        kv_connector='OffloadingConnector', kv_role='kv_both',
        kv_connector_extra_config={
            'cpu_bytes_to_use': 65536, 'load_ring_bytes': 16384,
            'disk_root': cache_dir, 'disk_max_bytes': 131072,
            'blocks_per_chunk': 1, 'pack_groups': [],
        })
    full = FullAttentionSpec(block_size=16, num_kv_heads=4, head_size=128,
                             dtype=torch.float32)
    scratch = CircularBufferSpec(block_size=4, num_kv_heads=1, head_size=1,
                                  dtype=torch.float32)
    kv = KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(['full0'], full),
        KVCacheGroupSpec(['scratch'], scratch),
        KVCacheGroupSpec(['full2'], full),
    ])
    normalized = build_offloading_config(config, kv)
    assert [g.group_idx for g in normalized.groups] == [0, 2]
    assert normalized.cache.tokens_per_hash == 16
    normalized = replace(normalized, worker_kv_bytes_per_block=1024)
    spec = NodeLocalDiskOffloadingSpec(normalized)
    assert spec.blocks_per_chunk_per_group == (1, 1, 1)
    assert not spec._uses_shared_region()
    scheduler = OffloadingConnectorScheduler(spec, config, kv)
    assert scheduler._lookup_groups == (0, 2)
    assert scheduler.config.num_kv_cache_groups == 3
    request = N(request_id='scratch-test', num_tokens=32, num_prompt_tokens=32,
                kv_transfer_params=None, block_hashes=[bytes([i]) * 32 for i in (1, 2)])
    state = RequestOffloadState(scheduler.config, request, ReqContext('scratch-test'),
                                N(policy=OffloadPolicy.BLOCK_LEVEL))
    state.update_offload_keys()
    assert not state.group_states[1].offload_keys
    assert {get_offload_group_idx(k) for s in state.group_states
            for k in s.offload_keys} == {0, 2}
    scheduler._req_status[request.request_id] = state
    manager = scheduler.manager
    scheduler.manager = N(prepare_load=Mock(return_value=None))
    def block(index):
        return N(block_id=index, is_null=False, block_hash=None)
    blocks = N(blocks=([block(11), block(12)], [block(31)], [block(21), block(22)]))
    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=32)
    job = next(iter(scheduler._current_batch_load_jobs.values()))
    assert job.dst_spec.group_sizes == [2, 0, 2]
    assert job.dst_spec.block_indices == [0, 0, 0]
    assert job.dst_spec.block_ids.tolist() == [11, 12, 21, 22]
    assert {get_offload_group_idx(k) for k in
            scheduler.manager.prepare_load.call_args.args[0]} == {0, 2}
    manager.shutdown()
print('PASS: scratch group skipped, original group IDs and load destinations preserved')

# Same DSpark hit/miss cases used for the earlier verified V4 deployment,
# updated for this image's normalized offloading interface.
for exempt in (False, True):
    scheduler = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    groups = tuple(GroupOffloadConfig(
        group_idx=i, tokens_per_block=4, tokens_per_chunk=4, hashes_per_chunk=1,
        kv_event_group_spec=None, sliding_window_size_in_chunks=None,
        kv_cache_spec=None, manager_cls=None, is_eagle_group=i == 1,
        eagle_group_is_veto_exempt=exempt and i == 1) for i in range(2))
    scheduler.config = N(kv_group_configs=groups, num_kv_cache_groups=2,
                         blocks_per_chunk=1, num_workers=2)
    scheduler._group_config_by_idx = {g.group_idx: g for g in groups}
    scheduler._lookup_groups = (0, 1)
    scheduler._sliding_window_groups = ()
    scheduler._chunks_being_loaded = set()
    scheduler._maximal_prefix_lookup = lambda keys, *args: len(keys) if keys and keys[0] >= 10 else 0
    request = N(request_id='draft-test', num_tokens=12)
    state = N(req=request, num_locally_computed_tokens=0, req_context=None,
              group_states=[N(offload_keys=[10, 11, 12]), N(offload_keys=[1, 2, 3])],
              partial_tail_boundary=None, offloading_context=N(policy=OffloadPolicy.BLOCK_LEVEL),
              transfer_jobs=set())
    assert scheduler._lookup_complete_chunks(state) == (12 if exempt else 0)
    if exempt:
        assert state.lookup_excluded_groups == frozenset({1})
        scheduler._req_status = {request.request_id: state}
        scheduler._current_batch_allocated_block_ids = set()
        scheduler._generate_job_id = lambda: 1
        scheduler._current_batch_load_jobs = {}
        scheduler._jobs = {}
        scheduler.manager = N(prepare_load=Mock(return_value=None))
        blocks = N(blocks=tuple([block(i + 1) for i in range(3)] for _ in range(2)))
        scheduler.update_state_after_alloc(request, blocks, 12)
        assert scheduler.manager.prepare_load.call_args.args[0] == [10, 11, 12]
        assert scheduler._current_batch_load_jobs[1].dst_spec.group_sizes == [3, 0]

for exempt in (False, True):
    coordinator = HybridKVCacheCoordinator.__new__(HybridKVCacheCoordinator)
    target_spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32)
    draft_spec = SlidingWindowSpec(block_size=16, num_kv_heads=1, head_size=1,
                                   dtype=torch.float32, sliding_window=32)
    target = N(find_longest_cache_hit=lambda **kw: ([[1, 2, 3]], 48),
               supports_fine_grained_hash_lookup=False)
    draft = N(find_longest_cache_hit=lambda **kw: ([[]], 0),
              supports_fine_grained_hash_lookup=False)
    coordinator.attention_groups = [SpecGroup(target_spec, [0], target, False, False),
                                    SpecGroup(draft_spec, [1], draft, True, exempt)]
    coordinator.kv_cache_config = N(kv_cache_groups=[None, None])
    coordinator.single_type_managers = [N(block_size=16, dcp_world_size=1, pcp_world_size=1)
                                         for _ in range(2)]
    coordinator.block_pool = None
    coordinator.scheduler_block_size = coordinator.hash_block_size = 16
    coordinator.enable_partial_hash_hits = False
    assert coordinator.find_longest_cache_hit([], 48)[1] == (48 if exempt else 0)
    assert coordinator.find_longest_cache_hit_per_group([], 48)[1] == (48, 0)

groups = [KVCacheGroupSpec(['target'], target_spec), KVCacheGroupSpec(['draft'], draft_spec)]
_annotate_eagle_groups(
    N(speculative_config=N(use_eagle_block_drop=lambda: True,
                            has_ephemeral_draft_context=lambda: True)),
    {'target': target_spec, 'draft': draft_spec}, groups, use_deepseek_v4_fallback=True)
assert not groups[0].is_eagle_group
assert groups[1].is_eagle_group and groups[1].eagle_group_is_veto_exempt
print('PASS: DSpark target-cache reuse, normal eagle miss, excluded draft load, group annotation')
print(json.dumps({'preflight': 'passed'}))
