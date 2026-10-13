# SPDX-License-Identifier: Apache-2.0
# Adapted from hughmadden's node-local spec for V4.1, 2026-10-12; see SOURCES.md.
"""Node-local disk KV offloading spec for multi-node vLLM engines.

TieringOffloadingSpec's fs tier moves L1 bytes through a scheduler-side view
of a shared mmap region, which requires every rank on one host. This spec
keeps the (multi-node-safe) CPU offloading data path and adds a disk tier
whose bytes never leave the owning rank.

Design points:

- CPU staging rows mirror GPU blocks 1:1 (worker runs with blocks_per_chunk=1
  everywhere), so tensors shared by KV groups with different logical
  blocks-per-chunk (e.g. GLM-5.3's indexer k_cache + 4-token tail_cache
  sharing one tensor) need no special handling. The *manager* allocates
  blocks_per_chunk[group] CPU rows per offload key instead.
- Manager (scheduler process): pure metadata. Tracks disk-resident keys,
  derives deterministic per-key file paths, embeds them in load/store specs,
  and persists an append-only index so disk residency survives restarts.
- Worker (per rank): after each GPU->CPU store completes, writes each key's
  CPU rows (all canonical tensors, row-major) to node-local disk via a thread
  pool; loads of disk-resident keys read disk -> CPU rows, then chain the
  normal CPU->GPU transfer under the same job id.

Every rank writes the same relative paths on its own node's disk; shards
stay node-local with no cross-node traffic.

Config (kv_connector_extra_config):
    spec_name: "NodeLocalDiskOffloadingSpec"
    spec_module_path: "nodelocal_disk_spec"
    cpu_bytes_to_use: CPU staging/hot-tier bytes (per rank)
    disk_root: directory for block files (default /root/.cache/vllm/kv-offload-l2)
    disk_io_threads: per-rank IO threads (default 4)
    pack_groups: groups whose per-job chunk files are concatenated into one
        pack file (default [6], the DFlash2 drafter: 56 small files per 3584
        tokens otherwise). A group is silently demoted to whole files if its
        per-key byte size cannot be derived from the group config.

Pack files: when one store job carries >= 2 not-yet-on-disk keys of a pack
group, they are written as a single `gNN/pack/<first-key-hash>.pack` file
(concatenated per-key payloads). The index then records
`g:hash:relpath:slot:0` per key (slot-addressed: the worker derives the
byte range as slot * its actual per-key bytes from its own tensor refs, so
writer and reader share one source of truth; a nonzero final field is read
as a legacy byte offset+length). Legacy `g:hash` lines still parse as
whole-file entries, so mixed old+new index logs keep working. Layout is
verified byte-exactly at write AND read (loud assert, no silent
corruption).

Streaming ring (load_ring_bytes, default 1 GiB): disk loads are pumped
disk -> ring rows -> GPU in bounded waves, so restore size is limited only
by GPU block allocation (full-context restores). load_ring_bytes: 0 falls
back to up-front staging, where a lookup budget cap bounds the granted hit
at half the CPU pool (without it, an over-pool-sized cold restore used to
assert-kill the engine in prepare_load).

v1 limitations: the disk index grows unboundedly (prune only while the engine
is down); disk IO failures fail the transfer job. Pack files are GC'd whole
(delete by atime like any file); a later missing-entry read is covered by the
separate graceful-miss track.
"""

import os
from collections import OrderedDict, Counter
import threading
import time
from collections.abc import Collection, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    LookupResult,
    OffloadingManager,
    OffloadKey,
    OffloadingWorker,
    PrepareStoreOutput,
    ReqContext,
    TransferResult,
    get_offload_block_hash,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

logger = init_logger(__name__)

INDEX_LOG_NAME = "index.log"


def _key_relpath(key: OffloadKey) -> str:
    """Deterministic rank-agnostic relative path for one offload key."""
    h = get_offload_block_hash(key).hex()
    g = get_offload_group_idx(key)
    return f"g{g:02d}/{h[:2]}/{h}.kv"


def _pack_relpath(key: OffloadKey) -> str:
    """Relative path of a pack file, named after its first packed key."""
    h = get_offload_block_hash(key).hex()
    g = get_offload_group_idx(key)
    return f"g{g:02d}/pack/{h}.pack"


class MultiRowBlockStatus(BlockStatus):
    """BlockStatus whose key owns a run of CPU rows (block_id is the first)."""

    def __init__(self, block_ids: list[int]):
        super().__init__(block_ids[0])
        self.row_ids: list[int] = block_ids


class DiskCPULoadStoreSpec(CPULoadStoreSpec):
    """Flat CPU row ids plus per-key (row_count, disk relpath | None).

    Store direction: relpaths the worker writes after the GPU->CPU copy.
    Load direction: relpaths the worker reads into the rows before CPU->GPU.

    key_offsets[i] is None for a whole-file key, or (offset, length): the
    byte range of key i inside the (shared) pack file at key_relpaths[i].
    Packed keys sharing one relpath in a store spec form one pack write.
    """

    def __init__(
        self,
        block_ids: list[int],
        key_row_counts: list[int],
        key_relpaths: "list[str | None]",
        key_group_idxs: "list[int] | None" = None,
        key_offsets: "list[tuple[int, int] | None] | None" = None,
    ):
        super().__init__(block_ids)
        self.delete_relpaths = []
        self.key_row_counts = key_row_counts
        self.key_relpaths = key_relpaths
        self.key_group_idxs = key_group_idxs or [0] * len(key_row_counts)
        self.key_offsets = (
            key_offsets
            if key_offsets is not None
            else [None] * len(key_row_counts)
        )

    def __repr__(self) -> str:
        n_disk = sum(1 for p in self.key_relpaths if p)
        n_packed = sum(1 for o in self.key_offsets if o is not None)
        return (
            f"DiskCPULoadStoreSpec({len(self.key_row_counts)} keys, "
            f"{len(self.key_relpaths)} paths, {n_disk} disk, "
            f"{n_packed} packed)"
        )


class NodeLocalDiskManager(CPUOffloadingManager):
    """CPU manager with per-key multi-row allocation and a disk tier.

    Rows are GPU-block-sized; a key of group g owns blocks_per_chunk[g]
    rows. Disk residency is tracked per key and persisted to an index log.
    """

    def __init__(
        self,
        *args,
        disk_root: str = "",
        rows_per_group: "tuple[int, ...]" = (),
        always_hit_groups: "tuple[int, ...]" = (),
        key_bytes_per_group: "tuple[int, ...]" = (),
        pack_groups: "tuple[int, ...]" = (),
        stream_loads: bool = False,
        stat_on_lookup: bool = True,
        disk_max_bytes: int,
        reserved_bytes_per_key: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert disk_root and rows_per_group
        self._always_hit_groups = frozenset(always_hit_groups)
        # Rows promised to restores in the current scheduling step: several
        # requests can pass lookup before any prepare_load runs, so lookup
        # must account for rows already committed this step.
        self._step_committed_rows = 0
        # Streaming (ring) mode: disk loads bypass CPU-pool staging entirely
        # (the worker pumps disk -> ring rows -> GPU in waves), so lookup
        # never has to decline for capacity and the restore ceiling is GPU
        # allocation itself.
        self._stream_loads = stream_loads
        # Non-streaming fallback: total rows granted to disk hits this step.
        # Without this cap, lookup's per-key check grants an unbounded hit
        # and prepare_load's up-front allocation asserts once the pool is
        # exhausted (engine death on the first >pool-sized cold restore).
        self._step_granted_rows = 0
        # Streaming mode: ref-count bumps we actually made in prepare_load's
        # CPU-cached branch, so complete_load never decrements a ref it did
        # not take (a streamed key can appear in the policy later via a
        # concurrent re-store).
        self._load_ref_bumped: "dict[OffloadKey, int]" = {}
        self._disk_root = disk_root
        self._rows_per_group = rows_per_group
        # Per-key payload bytes per group (per rank), used to lay out pack
        # files scheduler-side. Derived from group config; the worker verifies
        # the layout byte-exactly at write time.
        self._key_bytes_per_group = key_bytes_per_group
        # Packs are slot-addressed: the manager assigns slot indices and the
        # worker derives byte offsets from its own tensor refs (slot *
        # actual per-key bytes), so no scheduler-side byte derivation is
        # needed. (A first version assigned byte offsets from
        # config-derived sizes; the config value was ~18x off for the
        # drafter group and the worker's layout assert killed the engine on
        # the first pack store.)
        self._pack_groups = frozenset(int(g) for g in pack_groups)
        # Value: None = whole file at _key_relpath(key);
        # (relpath, offset, length) = byte range inside a pack file.
        self._disk_index: "dict[OffloadKey, tuple[str, int, int] | None]" = {}
        # Same record shape, with (relpath, None, None) meaning whole file.
        self._pending_disk_writes: "dict[OffloadKey, tuple]" = {}
        # Suffix of the last prepare_store's submitted keys declined under
        # CPU-pool pressure; the (patched) scheduler reads this to avoid
        # advancing next_stored_chunk_idx past never-stored chunks
        # (partial-store hole fix).
        self.last_store_declined_keys: "frozenset[OffloadKey]" = frozenset()
        # Live pruning + graceful miss:
        # - pack relpath -> keys (so a tombstoned pack retires all its keys)
        self._pack_members: "dict[str, list[OffloadKey]]" = {}
        # - janitor tombstone log, tailed once per engine step
        self._deletes_path = os.path.join(disk_root, "deletes.log")
        self._deletes_offset = 0
        # - lookup-time stat verdict cache (TTL) for out-of-band deletions
        self._stat_on_lookup = bool(stat_on_lookup)
        self._stat_ok: "dict[OffloadKey, float]" = {}
        self._index_path = os.path.join(disk_root, INDEX_LOG_NAME)
        self._load_index()
        self._index_fh = open(self._index_path, "a", buffering=1)
        assert not self._pack_groups, "Bounded mode uses one file per key"
        self._disk_max_bytes = disk_max_bytes
        self._reserved_bytes_per_key = reserved_bytes_per_key
        self._disk_lru = OrderedDict(
            (key, os.path.getsize(os.path.join(disk_root, _key_relpath(key))))
            for key in self._disk_index
        )
        self._disk_bytes = sum(self._disk_lru.values())
        assert self._disk_bytes <= disk_max_bytes, "Existing cache exceeds disk budget"
        self._disk_load_pins = Counter()
        self._step_disk_hits = set()
        self._disk_evictions = 0

        try:
            self._deletes_offset = os.path.getsize(self._deletes_path)
        except OSError:
            self._deletes_offset = 0

    # --- persistent index ---

    def _load_index(self) -> None:
        """Rebuild the disk index at boot.

        Whole-file entries: the FILESYSTEM is the source of truth — the
        filename encodes group + full hash, so a directory scan reconstructs
        them exactly. Files pruned while the engine was down simply aren't
        found; nothing goes stale. (index.log used to be trusted blind here,
        which made prune-while-down a boot-time crash class.)

        Packed entries carry (relpath, slot) that filenames don't encode, so
        those still come from index.log 5-field lines — validated against the
        filesystem per pack relpath, so pruned packs drop out. Legacy 2-field
        lines are ignored (redundant with the scan). Later lines win.
        """
        n = n_packed = n_stale = 0
        t0 = time.monotonic()
        try:
            for gdir in os.scandir(self._disk_root):
                if not (gdir.is_dir() and gdir.name.startswith("g")):
                    continue
                try:
                    g = int(gdir.name[1:])
                except ValueError:
                    continue
                for shard in os.scandir(gdir.path):
                    if not shard.is_dir():
                        continue
                    for f in os.scandir(shard.path):
                        if not f.name.endswith(".kv"):
                            continue
                        try:
                            key = make_offload_key(
                                bytes.fromhex(f.name[:-3]), g
                            )
                        except ValueError:
                            continue
                        self._disk_index[key] = None
                        n += 1
        except FileNotFoundError:
            pass
        pack_present: "dict[str, bool]" = {}
        try:
            with open(self._index_path) as fh:
                for line in fh:
                    line = line.strip()
                    parts = line.split(":")
                    if len(parts) != 5:
                        continue
                    try:
                        g_str, h_str, relpath, off_str, len_str = parts
                        ok = pack_present.get(relpath)
                        if ok is None:
                            ok = os.path.isfile(
                                os.path.join(self._disk_root, relpath)
                            )
                            pack_present[relpath] = ok
                        key = make_offload_key(bytes.fromhex(h_str), int(g_str))
                        if not ok:
                            n_stale += 1
                            continue
                        record = (relpath, int(off_str), int(len_str))
                        self._disk_index[key] = record
                        self._pack_members.setdefault(relpath, []).append(key)
                        n_packed += 1
                    except Exception:
                        logger.warning("disk index: bad line %r", line[:60])
        except FileNotFoundError:
            pass
        logger.warning(
            "NodeLocalDiskManager: %d disk-resident keys (scan) + %d packed "
            "(%d stale dropped) in %.1fs from %s",
            n,
            n_packed,
            n_stale,
            time.monotonic() - t0,
            self._disk_root,
        )
        self._dbg_counts = {"cpu_hit": 0, "disk_hit": 0, "miss": 0, "declined": 0}

    def _append_index(
        self, key: OffloadKey, record: "tuple[str, int, int] | None" = None
    ) -> None:
        # Whole-file entries need no log line (the filename is the index);
        # only packed entries persist their (relpath, slot).
        if record is None:
            return
        h = get_offload_block_hash(key).hex()
        g = get_offload_group_idx(key)
        relpath, offset, length = record
        self._index_fh.write(f"{g}:{h}:{relpath}:{offset}:{length}\n")

    def _apply_tombstones(self) -> None:
        """Tail the janitor's delete log; retire pruned entries in-step.

        The janitor appends relpaths BEFORE unlinking (and skips files
        younger than a grace period), so the index only ever under-promises.
        Runs in on_schedule_end — after this step's lookups AND
        prepare_loads — so it can never invalidate an intra-step promise.
        """
        try:
            size = os.path.getsize(self._deletes_path)
        except OSError:
            return
        if size <= self._deletes_offset:
            return
        try:
            with open(self._deletes_path) as fh:
                fh.seek(self._deletes_offset)
                for line in fh:
                    if not line.endswith("\n"):
                        break  # torn tail; re-read next step
                    self._deletes_offset += len(line)
                    rel = line.strip()
                    if not rel:
                        continue
                    for key in self._pack_members.pop(rel, ()):
                        if self._disk_index.get(key) is not None:
                            self._disk_index.pop(key, None)
                        self._stat_ok.pop(key, None)
                    base = os.path.basename(rel)
                    if base.endswith(".kv") and rel[:1] == "g":
                        try:
                            g = int(rel[1:3])
                            key = make_offload_key(bytes.fromhex(base[:-3]), g)
                        except ValueError:
                            continue
                        if self._disk_index.get(key, "absent") is None:
                            self._disk_index.pop(key, None)
                        self._stat_ok.pop(key, None)
        except OSError:
            return

    # --- helpers ---

    _STAT_TTL_S = 5.0

    def _stat_key_ok(self, key: OffloadKey) -> bool:
        """Verify the entry's backing file still exists (TTL-cached).

        Catches out-of-band deletions between janitor passes on THIS node.
        A pass here is not a guarantee (the other rank's shard is invisible,
        and TOCTOU remains) — the tombstone protocol is the primary defense;
        this is cheap hardening. On failure the entry is retired so the walk
        misses instead of promising a dead hit."""
        now = time.monotonic()
        ts = self._stat_ok.get(key)
        if ts is not None and now - ts < self._STAT_TTL_S:
            return True
        record = self._disk_index.get(key)
        rel = record[0] if record is not None else _key_relpath(key)
        if os.path.isfile(os.path.join(self._disk_root, rel)):
            self._stat_ok[key] = now
            return True
        self._disk_index.pop(key, None)
        self._stat_ok.pop(key, None)
        return False

    def _rows_for_key(self, key: OffloadKey) -> int:
        return self._rows_per_group[get_offload_group_idx(key)]

    def _alloc_rows(self, n: int) -> "list[int] | None":
        """Allocate n CPU rows, evicting idle keys if needed. None on failure."""
        while self._get_num_free_blocks() < n:
            if self._num_evictable_cache_blocks <= 0:
                return None
            evicted = self._policy.evict(1, set())
            if not evicted:
                return None
            self._num_evictable_cache_blocks -= len(evicted)
            for _ekey, eblock in evicted:
                self._free_rows(eblock)
        rows: list[int] = []
        num_fresh = min(n, self._num_blocks - self._num_allocated_blocks)
        for _ in range(num_fresh):
            rows.append(self._num_allocated_blocks)
            self._num_allocated_blocks += 1
        for _ in range(n - num_fresh):
            rows.append(self._free_list.pop())
        return rows

    def _free_rows(self, block: BlockStatus) -> None:
        if isinstance(block, MultiRowBlockStatus):
            self._free_list.extend(block.row_ids)
        else:
            self._free_list.append(block.block_id)

    # keep base-class internal frees consistent with multi-row blocks
    def _free_block(self, block: BlockStatus) -> None:
        self._free_rows(block)

    # --- lookup: CPU first, then disk ---

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        # Recent-window-state groups (e.g. GLM-5.3 indexer tail_cache) are
        # never stored and always report HIT; the hit-rollback recompute
        # window rebuilds their true state after a restore.
        if get_offload_group_idx(key) in self._always_hit_groups:
            return LookupResult.HIT
        if key in self._disk_lru:
            self._disk_lru.move_to_end(key)
            self._step_disk_hits.add(key)
        res = super().lookup(key, req_context)
        if res is not LookupResult.MISS:
            self._dbg_counts["cpu_hit"] += 1
            return res
        if key in self._disk_index:
            if self._stat_on_lookup and not self._stat_key_ok(key):
                self._dbg_counts["miss"] += 1
                return LookupResult.MISS
            if self._stream_loads:
                # Ring streaming: no staging reservation needed; the worker
                # pumps this key through a bounded row window.
                self._dbg_counts["disk_hit"] += 1
                if self._dbg_counts["disk_hit"] % 200 == 1:
                    logger.warning("NodeLocalDisk lookup: %s", self._dbg_counts)
                return LookupResult.HIT
            rows = self._rows_for_key(key)
            available = (
                self._get_num_free_blocks()
                + self._num_evictable_cache_blocks
                - self._step_committed_rows
            )
            # Budget cap: bound the TOTAL granted this step (all requests,
            # all keys) at half the available pool, so prepare_load's
            # up-front allocation can never exhaust staging after a HIT.
            if (self._step_granted_rows + rows) * 2 <= available:
                self._step_granted_rows += rows
                self._dbg_counts["disk_hit"] += 1
                if self._dbg_counts["disk_hit"] % 200 == 1:
                    logger.warning("NodeLocalDisk lookup: %s", self._dbg_counts)
                return LookupResult.HIT
            self._dbg_counts["declined"] += 1
            if self._dbg_counts["declined"] % 200 == 1:
                logger.warning(
                    "NodeLocalDisk lookup DECLINED: %s", self._dbg_counts
                )
            return LookupResult.MISS
        self._dbg_counts["miss"] += 1
        if self._dbg_counts["miss"] % 500 == 1:
            logger.warning("NodeLocalDisk lookup: %s", self._dbg_counts)
        return LookupResult.MISS

    # --- load ---

    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        for key in keys:
            if key in self._disk_index:
                self._disk_load_pins[key] += 1
        block_ids: list[int] = []
        row_counts: list[int] = []
        relpaths: "list[str | None]" = []
        group_idxs: list[int] = []
        key_offsets: "list[tuple[int, int] | None]" = []
        for key in keys:
            group_idxs.append(get_offload_group_idx(key))
            block = self._policy.get(key)
            if block is not None:
                assert block.is_ready, f"Block {key!r} not ready"
                if block.ref_cnt == 0:
                    self._policy.mark_non_evictable(key)
                    self._num_evictable_cache_blocks -= 1
                    assert self._num_evictable_cache_blocks >= 0
                block.ref_cnt += 1
                if self._stream_loads:
                    self._load_ref_bumped[key] = (
                        self._load_ref_bumped.get(key, 0) + 1
                    )
                rows = (
                    block.row_ids
                    if isinstance(block, MultiRowBlockStatus)
                    else [block.block_id]
                )
                block_ids.extend(rows)
                row_counts.append(len(rows))
                relpaths.append(None)
                key_offsets.append(None)
                continue

            # disk-resident
            assert key in self._disk_index, f"{key!r} neither in CPU nor disk"
            n = self._rows_for_key(key)
            record = self._disk_index[key]
            if record is None:
                rel = _key_relpath(key)
                off_len = None
            else:
                rel, offset, length = record
                off_len = (offset, length)

            if self._stream_loads:
                # Sentinel rows: the worker assigns ring rows per wave;
                # nothing is allocated or inserted scheduler-side, so the
                # restore size is unbounded by the CPU pool.
                block_ids.extend([-1] * n)
                row_counts.append(n)
                relpaths.append(rel)
                key_offsets.append(off_len)
                continue

            # non-streaming fallback: allocate CPU staging rows up front
            logger.warning(
                "NodeLocalDisk prepare_load: staging g%d key from disk",
                get_offload_group_idx(key),
            ) if len(relpaths) < 3 else None
            self._step_committed_rows += n
            rows_alloc = self._alloc_rows(n)
            assert rows_alloc is not None, "staging alloc failed after HIT"
            new_block = MultiRowBlockStatus(rows_alloc)
            self._policy.insert(key, new_block)
            new_block.ref_cnt = 1
            block_ids.extend(rows_alloc)
            row_counts.append(n)
            relpaths.append(rel)
            key_offsets.append(off_len)

        return DiskCPULoadStoreSpec(
            block_ids, row_counts, relpaths, group_idxs, key_offsets
        )

    # --- store ---

    def _reserve_disk(self, key, deletes):
        size = self._reserved_bytes_per_key
        while self._disk_bytes + size > self._disk_max_bytes:
            victim = next((k for k in self._disk_lru
                           if not self._disk_load_pins[k]
                           and k not in self._step_disk_hits
                           and k not in self._pending_disk_writes), None)
            if victim is None:
                return False
            self._disk_bytes -= self._disk_lru.pop(victim)
            self._disk_index.pop(victim, None)
            self._stat_ok.pop(victim, None)
            deletes.append(_key_relpath(victim))
            self._disk_evictions += 1
        self._disk_lru[key] = size
        self._disk_bytes += size
        return True

    def prepare_store(self, keys, req_context):
        result = self._prepare_store_unbounded(keys, req_context)
        if result is None:
            return None
        spec = result.store_spec
        for i, key in enumerate(result.keys_to_store):
            if key not in self._pending_disk_writes:
                continue
            if not self._reserve_disk(key, spec.delete_relpaths):
                # CPU offload can still complete, but no disk hit is advertised.
                self._pending_disk_writes.pop(key)
                spec.key_relpaths[i] = None
        if spec.delete_relpaths:
            logger.info("NodeLocalDisk evictions=%d reserved_bytes=%d limit=%d",
                        self._disk_evictions, self._disk_bytes, self._disk_max_bytes)
        return result

    def _prepare_store_unbounded(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        self.last_store_declined_keys = frozenset()
        keys_to_store = [
            k
            for k in keys
            if get_offload_group_idx(k) not in self._always_hit_groups
            and self._policy.get(k) is None
        ]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=DiskCPULoadStoreSpec([], [], []),
                evicted_keys=[],
            )

        self.allocation_sizes_in_current_batch.append(len(keys_to_store))

        # Pack planning: >= 2 not-yet-on-disk keys of a pack group in this
        # store job are concatenated into one pack file. Membership is
        # per-store-job; a re-stored key may land in a second pack later —
        # fine, the last index entry wins. Offsets are assigned in submission
        # order; an allocation break below simply truncates the pack, keeping
        # the assigned offsets contiguous and valid.
        pack_counts: "dict[int, int]" = {}
        for k in keys_to_store:
            g = get_offload_group_idx(k)
            if g in self._pack_groups and k not in self._disk_index:
                pack_counts[g] = pack_counts.get(g, 0) + 1
        # group -> (pack relpath, next byte offset)
        pack_state: "dict[int, tuple[str, int]]" = {}

        block_ids: list[int] = []
        row_counts: list[int] = []
        relpaths: "list[str | None]" = []
        group_idxs: list[int] = []
        key_offsets: "list[tuple[int, int] | None]" = []
        stored: list[OffloadKey] = []
        for i, key in enumerate(keys_to_store):
            g = get_offload_group_idx(key)
            group_idxs.append(g)
            n = self._rows_for_key(key)
            rows = self._alloc_rows(n)
            if rows is None:
                # partial success: store what we could; expose the declined
                # suffix so the scheduler does not advance past it.
                self.last_store_declined_keys = frozenset(keys_to_store[i:])
                break
            block = MultiRowBlockStatus(rows)
            self._policy.insert(key, block)
            stored.append(key)
            block_ids.extend(rows)
            row_counts.append(n)
            if key in self._disk_index:
                relpaths.append(None)
                key_offsets.append(None)
            elif pack_counts.get(g, 0) > 1:
                if g not in pack_state:
                    pack_state[g] = (_pack_relpath(key), 0)
                pack_rel, slot = pack_state[g]
                pack_state[g] = (pack_rel, slot + 1)
                relpaths.append(pack_rel)
                # (slot, 0): slot-addressed pack entry — the worker derives
                # the byte range as slot * its actual per-key bytes.
                key_offsets.append((slot, 0))
                self._pending_disk_writes[key] = (pack_rel, slot, 0)
            else:
                rel = _key_relpath(key)
                relpaths.append(rel)
                key_offsets.append(None)
                self._pending_disk_writes[key] = (rel, None, None)

        if not stored:
            return None
        self._num_write_pending_blocks += len(stored)

        return PrepareStoreOutput(
            keys_to_store=stored,
            store_spec=DiskCPULoadStoreSpec(
                block_ids, row_counts[: len(stored)], relpaths[: len(stored)],
                group_idxs[: len(stored)], key_offsets[: len(stored)],
            ),
            evicted_keys=[],
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        super().complete_store(keys, req_context, success)
        for key in keys:
            pending = self._pending_disk_writes.pop(key, None)
            if pending is not None and not success and key in self._disk_lru:
                self._disk_bytes -= self._disk_lru.pop(key)
            if pending is not None and success and key not in self._disk_index:
                relpath, offset, length = pending
                record = (
                    None if offset is None else (relpath, offset, length)
                )
                self._disk_index[key] = record
                if record is not None:
                    self._pack_members.setdefault(relpath, []).append(key)
                self._append_index(key, record)

    def complete_load(self, keys, req_context) -> None:
        for key in keys:
            if self._disk_load_pins[key]:
                self._disk_load_pins[key] -= 1
                if not self._disk_load_pins[key]:
                    del self._disk_load_pins[key]
        if not self._stream_loads:
            return super().complete_load(keys, req_context)
        # Streaming: only unwind ref-count bumps this manager actually made
        # in prepare_load's CPU-cached branch. Streamed keys were never
        # inserted (and may coincidentally be in the policy via a later
        # re-store whose refs are not ours to touch).
        ours = []
        for key in keys:
            n = self._load_ref_bumped.get(key, 0)
            if n > 0:
                if n == 1:
                    del self._load_ref_bumped[key]
                else:
                    self._load_ref_bumped[key] = n - 1
                ours.append(key)
        if ours:
            super().complete_load(ours, req_context)

    def on_schedule_end(self, context) -> None:
        self._step_disk_hits.clear()
        self._step_committed_rows = 0
        self._step_granted_rows = 0
        self._apply_tombstones()
        return super().on_schedule_end(context)

    def reset_cache(self) -> None:
        super().reset_cache()
        self._step_committed_rows = 0
        self._step_granted_rows = 0
        self._load_ref_bumped.clear()
        self._disk_load_pins.clear()
        self._step_disk_hits.clear()

    def shutdown(self) -> None:
        try:
            self._index_fh.close()
        except Exception:
            pass
        return super().shutdown()


class _Wave:
    """One bounded slice of a streamed load job."""

    __slots__ = ("key_lo", "key_hi", "read_futures", "wave_id", "ring_used")

    def __init__(self, key_lo: int, key_hi: int):
        self.key_lo = key_lo
        self.key_hi = key_hi
        self.read_futures: "list[Future]" = []
        self.wave_id: "int | None" = None
        self.ring_used: list[int] = []


class _StreamJob:
    """A disk load pumped through the ring in waves.

    entries[i] = (rows, rel, gidx, off_len, dst_lo, streamed) per key, in
    spec order (parallel to the dst GPU block ids). Waves are contiguous
    key windows; each wave's streamed keys borrow ring rows for the disk
    read and release them when the wave's CPU->GPU copy completes.
    """

    __slots__ = (
        "job_id",
        "entries",
        "dst_block_ids",
        "orig_group_sizes",
        "orig_block_indices",
        "waves",
        "next_read",
        "next_submit",
        "waves_done",
        "consumed_per_group",
        "agg_bytes",
        "agg_time",
    )

    def __init__(self, job_id, entries, dst_spec, wave_cap_rows: int):
        self.job_id = job_id
        self.entries = entries
        self.dst_block_ids = dst_spec.block_ids
        self.orig_group_sizes = list(dst_spec.group_sizes)
        self.orig_block_indices = list(dst_spec.block_indices)
        self.waves: list[_Wave] = []
        lo = 0
        ring_need = 0
        for i, e in enumerate(entries):
            need = len(e[0]) if e[5] else 0
            if i > lo and ring_need + need > wave_cap_rows:
                self.waves.append(_Wave(lo, i))
                lo = i
                ring_need = 0
            ring_need += need
        self.waves.append(_Wave(lo, len(entries)))
        self.next_read = 0
        self.next_submit = 0
        self.waves_done = 0
        self.consumed_per_group = [0] * len(self.orig_group_sizes)
        self.agg_bytes = 0
        self.agg_time = 0.0


class NodeLocalDiskWorker(CPUOffloadingWorker):
    """CPU worker + per-rank disk IO (rows are GPU-block-sized, bpc=1)."""

    def __init__(
        self,
        *args,
        disk_root: str,
        disk_io_threads: int = 4,
        ring_first_row: int = -1,
        ring_rows: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._disk_root = disk_root
        os.makedirs(disk_root, exist_ok=True)
        self._pool = ThreadPoolExecutor(
            max_workers=disk_io_threads, thread_name_prefix="kv-disk"
        )
        self._lock = threading.Lock()
        # Streaming ring: tail rows of the CPU tensors reserved for pumping
        # disk loads in bounded waves (invisible to the manager's allocator).
        self._ring_free: list[int] = (
            list(range(ring_first_row, ring_first_row + ring_rows))
            if ring_rows > 0
            else []
        )
        # A wave may hold at most half the ring so two waves can overlap
        # (read wave N+1 from disk while wave N's H2D copy drains).
        self._wave_cap_rows = ring_rows // 2
        # real load job id -> _StreamJob
        self._stream_jobs: "dict[int, _StreamJob]" = {}
        # synthetic per-wave transfer id -> (real load job id, _Wave)
        self._wave_to_job: "dict[int, tuple[int, _Wave]]" = {}
        self._next_wave_id = -2  # negative ids never collide with real jobs
        # store job id -> [(row_ids, relpath, group_idx, off_len)] awaiting
        # GPU->CPU completion (off_len None = whole file, else pack range)
        self._store_disk_pairs: dict[int, list[tuple]] = {}
        # store job id -> disk-write futures
        self._store_disk_pending: dict[int, list[Future]] = {}
        # held results for stores awaiting disk writes
        self._store_results_held: dict[int, TransferResult] = {}
        # bytes snapshot-queued for disk writes (backpressure bound)
        self._pending_write_bytes = 0
        self._max_pending_write_bytes = 2 << 30  # 2 GiB
        # load job id -> (futures, src, dst) awaiting disk reads
        self._load_disk_pending: dict[
            int, tuple[list[Future], LoadStoreSpec, GPULoadStoreSpec]
        ] = {}
        self._cpu_tensors: list[torch.Tensor] = self._store_handler.dst_tensors

    # -- byte helpers --

    def _key_views(self, row_ids: list[int], group_idx: int):
        """Byte views of one key's data: per row, each group ref's page slice
        at the start of that ref tensor's CPU row (where the GPU->CPU copy
        placed it)."""
        refs = self._store_handler.layer_refs_per_group[group_idx]
        for row_id in row_ids:
            for ref in refs:
                row = self._cpu_tensors[ref.tensor_idx][row_id].numpy()
                yield memoryview(row)[: ref.page_size_bytes]

    def _group_key_bytes(self, row_ids: "list[int]", group_idx: int) -> int:
        """Actual on-disk payload bytes of one key: what _key_views yields.
        This is the single source of truth for pack slot sizing — both the
        writer and the reader derive it from the same tensor refs."""
        refs = self._store_handler.layer_refs_per_group[group_idx]
        return len(row_ids) * sum(ref.page_size_bytes for ref in refs)

    @staticmethod
    def _drop_page_cache(fh) -> None:
        """Flush THIS file's pages to disk and drop them from page cache.

        On unified-memory hosts (GB10) dirty page-cache accumulation from
        store writes competes with the model for the same physical memory —
        a large session's store burst can contribute to UMA exhaustion.
        fadvise(DONTNEED) is a no-op on dirty pages, so fsync first: the
        cost lands on IO pool threads only, never the engine thread."""
        fh.flush()
        if hasattr(os, "posix_fadvise"):
            os.fsync(fh.fileno())
            os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    def _write_key(
        self, row_ids: list[int], relpath: str, group_idx: int
    ) -> None:
        path = os.path.join(self._disk_root, relpath)
        if os.path.exists(path):
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            for view in self._key_views(row_ids, group_idx):
                fh.write(view)
            self._drop_page_cache(fh)
        os.replace(tmp, path)

    def _write_pack(
        self,
        relpath: str,
        entries: "list[tuple[list[int], int, tuple[int, int]]]",
    ) -> None:
        """Write one pack file: per-key payloads concatenated at the offsets
        the manager assigned (entries = [(row_ids, group_idx, (offset, len))]).

        Always overwrites: a stale same-named pack (from an earlier store
        whose index entries never landed) may have different membership, so
        the exists-skip of _write_key would leave offsets pointing into the
        wrong bytes. The asserts verify the manager's config-derived layout
        against the actual byte counts — layout drift fails the job loudly
        instead of corrupting restores.
        """
        path = os.path.join(self._disk_root, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            for row_ids, group_idx, (slot, _z) in sorted(
                entries, key=lambda e: e[2][0]
            ):
                key_bytes = self._group_key_bytes(row_ids, group_idx)
                pos = fh.tell()
                assert pos == slot * key_bytes, (
                    f"pack layout drift in {relpath}: writer at byte {pos}, "
                    f"slot {slot} expects {slot * key_bytes}"
                )
                written = 0
                for view in self._key_views(row_ids, group_idx):
                    fh.write(view)
                    written += view.nbytes
                assert written == key_bytes, (
                    f"pack entry bytes {written} != derived key bytes "
                    f"{key_bytes} in {relpath}"
                )
            self._drop_page_cache(fh)
        os.replace(tmp, path)

    def _snapshot_key(self, row_ids: list[int], group_idx: int) -> bytes:
        return b"".join(
            bytes(view) for view in self._key_views(row_ids, group_idx)
        )

    def _write_key_bytes(self, data: bytes, relpath: str) -> None:
        try:
            path = os.path.join(self._disk_root, relpath)
            if os.path.exists(path):
                return
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + f".tmp{os.getpid()}"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        finally:
            with self._lock:
                self._pending_write_bytes -= len(data)

    def _read_key(
        self,
        row_ids: list[int],
        relpath: str,
        group_idx: int,
        off_len: "tuple[int, int] | None" = None,
    ) -> None:
        """Read one key's payload into its CPU rows. off_len selects a byte
        range inside a pack file; None reads a whole-file key from offset 0.

        A bad file (missing/short/corrupt) is DELETED before the error
        propagates (fs-tier precedent) so it cannot poison the next boot's
        index scan; the failure itself still fails the load job for now
        (fail-soft is a separate track)."""
        path = os.path.join(self._disk_root, relpath)
        try:
            self._read_key_inner(path, row_ids, group_idx, off_len)
        except Exception:
            try:
                os.unlink(path)
                logger.warning("deleted bad KV file %s", relpath)
            except OSError:
                pass
            raise

    def _read_key_inner(
        self,
        path: str,
        row_ids: "list[int]",
        group_idx: int,
        off_len: "tuple[int, int] | None",
    ) -> None:
        with open(path, "rb") as fh:
            expected = None
            if off_len is not None:
                pos, length = off_len
                if length == 0:
                    # slot-addressed: byte range derived from actual refs
                    key_bytes = self._group_key_bytes(row_ids, group_idx)
                    fh.seek(pos * key_bytes)
                    expected = key_bytes
                else:
                    # byte-addressed (legacy pack entries)
                    fh.seek(pos)
                    expected = length
            total = 0
            for view in self._key_views(row_ids, group_idx):
                n = fh.readinto(view)
                assert n == view.nbytes, (
                    f"short read {n}/{view.nbytes} from {path}"
                )
                total += n
            if expected is not None:
                assert total == expected, (
                    f"packed key bytes {total} != expected {expected} "
                    f"in {path}"
                )

    @staticmethod
    def _key_row_runs(
        spec: DiskCPULoadStoreSpec,
    ) -> "list[tuple[list[int], str | None, int, tuple[int, int] | None]]":
        runs = []
        pos = 0
        ids = spec.block_ids
        for count, rel, gidx, off_len in zip(
            spec.key_row_counts,
            spec.key_relpaths,
            spec.key_group_idxs,
            spec.key_offsets,
        ):
            runs.append(
                ([int(b) for b in ids[pos : pos + count]], rel, gidx, off_len)
            )
            pos += count
        return runs

    # -- store: defer completion until disk writes land --

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        if isinstance(dst_spec, DiskCPULoadStoreSpec):
            for relpath in dst_spec.delete_relpaths:
                os.unlink(os.path.join(self._disk_root, relpath))
        ok = super().submit_store(job_id, src_spec, dst_spec)
        if ok and isinstance(dst_spec, DiskCPULoadStoreSpec):
            pairs = [
                (rows, rel, gidx, off_len)
                for rows, rel, gidx, off_len in self._key_row_runs(dst_spec)
                if rel
            ]
            if pairs:
                with self._lock:
                    self._store_disk_pairs[job_id] = pairs
        return ok

    # -- load: disk reads first, then chain CPU->GPU --

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        if isinstance(src_spec, DiskCPULoadStoreSpec):
            runs = self._key_row_runs(src_spec)
            streamed_any = any(
                rel is not None and rows and rows[0] == -1
                for rows, rel, _g, _o in runs
            )
            if streamed_any:
                entries = []
                dst_pos = 0
                for rows, rel, gidx, off_len in runs:
                    streamed = rel is not None and rows[0] == -1
                    entries.append(
                        (rows, rel, gidx, off_len, dst_pos, streamed)
                    )
                    dst_pos += len(rows)
                assert dst_pos == len(dst_spec.block_ids)
                job = _StreamJob(job_id, entries, dst_spec, self._wave_cap_rows)
                with self._lock:
                    self._stream_jobs[job_id] = job
                    self._start_wave_reads(job)
                    self._submit_ready_waves(job)
                return True
            pairs = [
                (rows, rel, gidx, off_len)
                for rows, rel, gidx, off_len in runs
                if rel
            ]
            if pairs:
                futures = [
                    self._pool.submit(self._read_key, rows, rel, gidx, off_len)
                    for rows, rel, gidx, off_len in pairs
                ]
                with self._lock:
                    self._load_disk_pending[job_id] = (futures, src_spec, dst_spec)
                return True
        return super().submit_load(job_id, src_spec, dst_spec)

    # -- streaming waves (called with self._lock held) --

    def _start_wave_reads(self, job: "_StreamJob") -> None:
        """Assign ring rows to the next waves and kick their disk reads."""
        while job.next_read < len(job.waves):
            wave = job.waves[job.next_read]
            window = job.entries[wave.key_lo : wave.key_hi]
            need = sum(len(e[0]) for e in window if e[5])
            if need > len(self._ring_free):
                break
            for idx in range(wave.key_lo, wave.key_hi):
                rows, rel, gidx, off_len, dst_lo, streamed = job.entries[idx]
                if not streamed:
                    continue
                n = len(rows)
                ring_rows = self._ring_free[:n]
                del self._ring_free[:n]
                wave.ring_used.extend(ring_rows)
                # rebind the entry to its assigned rows
                job.entries[idx] = (
                    ring_rows, rel, gidx, off_len, dst_lo, True
                )
                wave.read_futures.append(
                    self._pool.submit(
                        self._read_key, ring_rows, rel, gidx, off_len
                    )
                )
            job.next_read += 1

    def _submit_ready_waves(self, job: "_StreamJob") -> None:
        """Submit CPU->GPU transfers for read-complete waves, in order."""
        while job.next_submit < job.next_read:
            wave = job.waves[job.next_submit]
            if not all(f.done() for f in wave.read_futures):
                break
            for f in wave.read_futures:
                f.result()  # IO failures fail the job loudly (as today)
            window = job.entries[wave.key_lo : wave.key_hi]
            src_rows: list[int] = []
            n_groups = len(job.orig_group_sizes)
            group_sizes = [0] * n_groups
            for rows, _rel, gidx, _off, _dst, _st in window:
                src_rows.extend(int(r) for r in rows)
                group_sizes[gidx] += len(rows)
            block_indices = [
                job.orig_block_indices[g] + job.consumed_per_group[g]
                for g in range(n_groups)
            ]
            dst_lo = window[0][4]
            dst_hi = window[-1][4] + len(window[-1][0])
            wave_id = self._next_wave_id
            self._next_wave_id -= 1
            wave.wave_id = wave_id
            self._wave_to_job[wave_id] = (job.job_id, wave)
            ok = self._load_handler.transfer_async(
                wave_id,
                CPULoadStoreSpec(src_rows),
                GPULoadStoreSpec(
                    list(job.dst_block_ids[dst_lo:dst_hi]),
                    group_sizes,
                    block_indices,
                ),
            )
            assert ok
            for g in range(n_groups):
                job.consumed_per_group[g] += group_sizes[g]
            job.next_submit += 1

    # -- completion pump --

    def get_finished(self) -> list[TransferResult]:
        results = super().get_finished()
        out: list[TransferResult] = []
        ready_loads = []
        with self._lock:
            for res in results:
                wave_ref = self._wave_to_job.pop(res.job_id, None)
                if wave_ref is not None:
                    real_id, wave = wave_ref
                    job = self._stream_jobs.get(real_id)
                    if job is None:
                        continue  # job was flushed
                    self._ring_free.extend(wave.ring_used)
                    wave.ring_used = []
                    job.agg_bytes += res.transfer_size
                    job.agg_time += res.transfer_time
                    job.waves_done += 1
                    if job.waves_done == len(job.waves):
                        del self._stream_jobs[real_id]
                        out.append(
                            TransferResult(
                                job_id=real_id,
                                success=True,
                                transfer_size=job.agg_bytes,
                                transfer_time=job.agg_time,
                            )
                        )
                    continue
                pairs = self._store_disk_pairs.pop(res.job_id, None)
                if pairs is None:
                    out.append(res)
                    continue
                # All byte copying happens in pool threads; the job completion
                # is held until writes land so the CPU rows stay pinned.
                # (An engine-thread snapshot fast-path was tried and caused
                # multi-second generation stalls after each prefill.)
                # Packed keys sharing a relpath collapse into one pack write.
                packs: "dict[str, list]" = {}
                futures = []
                for rows, rel, gidx, off_len in pairs:
                    if off_len is None:
                        futures.append(
                            self._pool.submit(self._write_key, rows, rel, gidx)
                        )
                    else:
                        packs.setdefault(rel, []).append((rows, gidx, off_len))
                futures.extend(
                    self._pool.submit(self._write_pack, rel, entries)
                    for rel, entries in packs.items()
                )
                self._store_disk_pending[res.job_id] = futures
                self._store_results_held[res.job_id] = res

            done_ids = [
                job_id
                for job_id, futures in self._store_disk_pending.items()
                if all(f.done() for f in futures)
            ]
            for job_id in done_ids:
                for f in self._store_disk_pending.pop(job_id):
                    f.result()  # raise on IO failure
                out.append(self._store_results_held.pop(job_id))

            for job_id, (futures, src_spec, dst_spec) in list(
                self._load_disk_pending.items()
            ):
                if all(f.done() for f in futures):
                    for f in futures:
                        f.result()
                    ready_loads.append((job_id, src_spec, dst_spec))
                    self._load_disk_pending.pop(job_id)

            # advance streamed loads: start reads freed-up ring rows allow,
            # submit waves whose reads have landed
            for job in self._stream_jobs.values():
                self._start_wave_reads(job)
                self._submit_ready_waves(job)
        for job_id, src_spec, dst_spec in ready_loads:
            ok = super().submit_load(job_id, src_spec, dst_spec)
            assert ok
        return out

    def wait(self, job_ids: "set[int]") -> None:
        """Flush path (e.g. reset_cache): cancel pending disk-read waves and
        never submit their CPU->GPU copies — the destination GPU blocks may
        be reallocated after a flush. Also covers the legacy held-load path
        (pre-existing hazard: a flushed held load must not chain its copy
        later)."""
        extra: set[int] = set()
        pending: list[Future] = []
        flushed_ring: list[int] = []
        with self._lock:
            for jid in job_ids:
                job = self._stream_jobs.pop(jid, None)
                if job is not None:
                    for wave in job.waves:
                        if wave.wave_id is not None:
                            extra.add(wave.wave_id)
                            self._wave_to_job.pop(wave.wave_id, None)
                        for f in wave.read_futures:
                            f.cancel()
                        pending.extend(wave.read_futures)
                        flushed_ring.extend(wave.ring_used)
                        wave.ring_used = []
                legacy = self._load_disk_pending.pop(jid, None)
                if legacy is not None:
                    futures, _src, _dst = legacy
                    for f in futures:
                        f.cancel()
                    pending.extend(futures)
        if pending:
            futures_wait(pending)
        if flushed_ring:
            with self._lock:
                self._ring_free.extend(flushed_ring)
        super().wait(job_ids | extra)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        return super().shutdown()


class NodeLocalDiskOffloadingSpec(CPUOffloadingSpec):
    """CPU offloading + node-local per-rank disk tier (row-granular CPU)."""

    def _uses_shared_region(self):
        return False

    def __init__(self, config):
        super().__init__(config)
        # PR #54743 filters scratch groups but retains each original group ID.
        # Keys and worker transfer vectors still use those original IDs.
        num_groups = max(group.group_idx for group in config.groups) + 1
        self.blocks_per_chunk_per_group = (self.blocks_per_chunk,) * num_groups
        assert self.blocks_per_chunk == 1

        self.disk_root: str = str(
            self.extra_config.get("disk_root", "/root/.cache/vllm/kv-offload-l2")
        )
        self.disk_io_threads: int = int(self.extra_config.get("disk_io_threads", 4))

        # Row-granular CPU pool: rows are GPU-block-sized; capacity derives
        # from per-row bytes rather than per-chunk bytes.
        cpu_bytes_to_use = int(self.extra_config.get("cpu_bytes_to_use", 0))
        if config.worker_kv_bytes_per_block > 0:
            self.num_blocks = cpu_bytes_to_use // config.worker_kv_bytes_per_block

        # Streaming ring: tail rows of the CPU tensors reserved for pumping
        # disk loads in bounded waves. With a ring, the restore size is no
        # longer limited by the CPU pool (full-context restores). Set
        # load_ring_bytes: 0 to fall back to up-front staging with the
        # lookup budget cap.
        ring_bytes = int(self.extra_config.get("load_ring_bytes", 1 << 30))
        self.ring_rows = 0
        if ring_bytes > 0 and config.worker_kv_bytes_per_block > 0:
            self.ring_rows = min(
                ring_bytes // config.worker_kv_bytes_per_block,
                self.num_blocks // 4,
            )
            # A wave (= half the ring, for double-buffering) must fit the
            # largest single loadable key's row run. Derive it from the real
            # group geometry rather than guessing.
            load_skip = {
                int(g) for g in self.extra_config.get("load_skip_groups", ())
            }
            max_key_rows = max(
                (
                    bpc
                    for g, bpc in enumerate(self.blocks_per_chunk_per_group)
                    if g not in load_skip
                ),
                default=1,
            )
            min_ring = max(2 * max_key_rows, 8)
            if self.ring_rows < min_ring:
                logger.warning(
                    "NodeLocalDiskOffloadingSpec: ring of %d rows too small "
                    "(need >= %d); disabling load streaming",
                    self.ring_rows,
                    min_ring,
                )
                self.ring_rows = 0
        logger.warning(
            "NodeLocalDiskOffloadingSpec: %d CPU rows (%d B/row, %d ring), "
            "rows_per_key=%s, disk_root=%s",
            self.num_blocks,
            config.worker_kv_bytes_per_block,
            self.ring_rows,
            list(self.blocks_per_chunk_per_group),
            self.disk_root,
        )

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            os.makedirs(self.disk_root, exist_ok=True)
            # Per-key payload bytes per group (per rank): one CPU row per GPU
            # block, so key bytes = per-block group bytes * rows per key.
            # bytes_per_block is added to OffloadingGroupConfig by the divisor
            # patch; 0 (absent/unknown) disables packing for that group.
            key_bytes_per_group = [0] * len(self.blocks_per_chunk_per_group)
            for group in self.config.groups:
                key_bytes_per_group[group.group_idx] = (
                    int(getattr(group, "bytes_per_block", 0))
                    * self.blocks_per_chunk_per_group[group.group_idx]
                )
            key_bytes_per_group = tuple(key_bytes_per_group)
            self._manager = NodeLocalDiskManager(
                # ring rows live in the worker tensors but are invisible to
                # the manager's allocator
                num_blocks=self.num_blocks - self.ring_rows,
                cache_policy=self.eviction_policy,
                cache_policy_module_path=self.cache_policy_module_path,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=int(self.extra_config.get("store_threshold", 0)),
                max_tracker_size=int(
                    self.extra_config.get("max_tracker_size", 64_000)
                ),
                disk_root=self.disk_root,
                disk_max_bytes=int(self.extra_config["disk_max_bytes"]),
                reserved_bytes_per_key=self.config.worker_kv_bytes_per_block,
                rows_per_group=tuple(self.blocks_per_chunk_per_group),
                always_hit_groups=tuple(
                    int(g)
                    for g in self.extra_config.get("always_hit_groups", ())
                ),
                key_bytes_per_group=key_bytes_per_group,
                pack_groups=tuple(
                    int(g) for g in self.extra_config.get("pack_groups", ())
                ),
                stream_loads=self.ring_rows > 0,
                stat_on_lookup=bool(
                    self.extra_config.get("stat_on_lookup", True)
                ),
            )
        return self._manager

    def create_worker(self, kv_caches: CanonicalKVCaches) -> NodeLocalDiskWorker:
        # Row-granular worker: one CPU row per GPU block, private pinned
        # tensors only (the shared mmap region is single-host).
        return NodeLocalDiskWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=1,
            num_cpu_blocks=self.num_blocks,
            mmap_region=None,
            disk_root=self.disk_root,
            disk_io_threads=self.disk_io_threads,
            ring_first_row=self.num_blocks - self.ring_rows,
            ring_rows=self.ring_rows,
        )

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            self._worker = self.create_worker(kv_caches)
        assert self._worker is not None
        return self._worker
