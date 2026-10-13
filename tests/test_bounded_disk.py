# SPDX-License-Identifier: Apache-2.0
import tempfile,os
from pathlib import Path
from types import SimpleNamespace
import torch
from node_local_disk_offload import NodeLocalDiskManager,NodeLocalDiskWorker,_key_relpath
from vllm.v1.kv_offload.base import ReqContext,make_offload_key,LookupResult
root=tempfile.TemporaryDirectory()
m=NodeLocalDiskManager(num_blocks=8,disk_root=root.name,rows_per_group=(1,),pack_groups=(),stream_loads=False,disk_max_bytes=2048,reserved_bytes_per_key=1024)
ctx=ReqContext('test')
keys=[make_offload_key(bytes([i])*32,0) for i in (1,2,3,4)]
def store(k):
 r=m.prepare_store([k],ctx)
 for path in r.store_spec.delete_relpaths:Path(root.name,path).unlink()
 for path in r.store_spec.key_relpaths:
  if path:
   p=Path(root.name,path);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(bytes([k[0]])*1024)
 m.complete_store(r.keys_to_store,ctx)
 return r
store(keys[0]);store(keys[1])
# Touch key1; key2 must be the eviction victim.
m.lookup(keys[0],ctx);m.on_schedule_end(ctx)
store(keys[2])
assert keys[0] in m._disk_index and keys[1] not in m._disk_index and keys[2] in m._disk_index
assert m._disk_bytes==2048
# Pin both residents: admit CPU only, never delete a loading file.
m.prepare_load([keys[0],keys[2]],ctx)
r=store(keys[3]);assert r.store_spec.key_relpaths==[None]
assert not r.store_spec.delete_relpaths
m.complete_load([keys[0],keys[2]],ctx)
assert sum(p.stat().st_size for p in Path(root.name).rglob('*.kv'))==2048
# Actual byte I/O including padded row stride.
w=NodeLocalDiskWorker.__new__(NodeLocalDiskWorker)
w._disk_root=root.name
backing=torch.arange(4096,dtype=torch.int64).to(torch.int8)
w._cpu_tensors=[backing.as_strided((4,1024),(1024,1))]
w._store_handler=SimpleNamespace(layer_refs_per_group=[[SimpleNamespace(tensor_idx=0,page_size_bytes=584)]])
expected=bytes(w._key_views([1],0).__next__())
w._write_key([1],'io-test.kv',0)
w._cpu_tensors[0][2].zero_()
w._read_key([2],'io-test.kv',0)
assert bytes(w._key_views([2],0).__next__())==expected
assert torch.count_nonzero(w._cpu_tensors[0][2][584:])==0
print('PASS: bounded admission, LRU eviction, in-flight protection, exact padded-page disk roundtrip')
m.shutdown()
