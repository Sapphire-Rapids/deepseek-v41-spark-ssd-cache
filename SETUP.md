# Setup and restart replay

Start with an already working two-node V4.1 EXL3 service. Keep its model, network, TP=2 and launch settings. This recipe supplies the cache implementation and validation; it does not install weights or replace your distributed launcher.

## 1. Build from your working image

On each node, set `BASE_IMAGE` to the image used by your working service, then run from this bundle directory:

```sh
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t dsv41-ssd:local .
docker run --rm --network none --entrypoint python3 dsv41-ssd:local \
  /opt/dsv41-ssd/install.py verify
```

The build requires vLLM `0.1.dev20904+g179dd0fa9` and all eight original hashes in `overlay-manifest.json`. A mismatch stops the build before any overlay is written. Keep the old image available for rollback. Check both nodes; matching a tag alone does not establish matching source files.

The original deployment used read-only file mounts for the same Python changes. The derived-image build is a portable packaging path; its separate checks are in `results/package-checks.json`.

## 2. Add persistent storage and connector settings

Choose a new empty directory on each node's local SSD. Give it an identity tied to the weights/revision, runtime, TP=2 and cache layout. Keep both directories as a pair across restarts. Do not share one directory between model deployments or combine it with an older cache.

Recreate both serving containers with the derived image and a read/write bind mount from that directory to `/native-kv-cache`. Add these Docker arguments to the existing launch command:

```sh
--mount type=bind,src="$CACHE_DIR",dst=/native-kv-cache \
-e PYTHONHASHSEED=0 \
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

`CACHE_DIR` must be an absolute path. The serving user must be able to read and write it. The image includes the disk module in Python's normal package directory.

Pass this additional argument to **both vLLM processes**, after the Docker image and entrypoint rather than among Docker's own flags:

```sh
--kv-transfer-config "$(cat config/kv-transfer.json)"
```

The JSON is in this bundle on the host. If your launcher splits an `EXTRA_ARGS` string on whitespace, compact the JSON first and add it as one argument:

```sh
python3 -c 'import json; print(json.dumps(json.load(open("config/kv-transfer.json")), separators=(",", ":")))'
```

In the MiaAI launcher, use `IMAGE=dsv41-ssd:local` with `SKIP_PULL=1`, `SKIP_BUILD=1`, and `SKIP_SHIP=1` after building that tag on both nodes. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` in its `.env` too; the launcher otherwise overrides the image environment. Add the persistent bind mount to its head and worker `docker run` commands. Append the compact connector JSON to existing `EXTRA_ARGS` without deleting other required flags. This launcher integration step is manual.

The tested serving settings were:

| Setting | Value |
| --- | --- |
| TP / DSpark draft tokens | 2 / 3 |
| Maximum context / concurrent sequences | 262,144 / 4 |
| GPU KV bytes per node | 2,684,354,560 |
| Maximum batched tokens | 2,048 |
| Long-prefill token threshold | 1,792 |
| Prefix-cache retention interval | 4,096 |
| SSD payload budget per node | 137,438,953,472 bytes (128 GiB) |
| CPU pool / ring within that pool | 536,870,912 / 134,217,728 bytes |
| Disk I/O threads / packing | 2 / disabled |

The native connector rejects expandable CUDA segments in this configuration. Both nodes must use the allocator setting above. Changing weights, runtime, TP layout, or cache geometry requires a new empty pair of directories. The connector does not automatically check that a reused directory belongs to the current model.

## 3. Check CPU and GPU transfers

Run the CPU check on either node using the derived image:

```sh
docker run --rm --network none --cap-drop ALL --read-only \
  --memory 2g --memory-swap 2g --cpus 2 --tmpfs /tmp:rw,size=256m \
  -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 dsv41-ssd:local \
  /opt/dsv41-ssd/tests/test_offload.py
```

Run the GPU check on each node when the service is idle. Set `CONTAINER` to that node's serving container:

```sh
docker exec "$CONTAINER" python3 /opt/dsv41-ssd/tests/test_gpu_io.py
```

It uses a temporary directory and a small synthetic tensor. Expect `payload_bytes_equal: true`, `padding_preserved: true`, 40 disk-only keys and an eight-row ring. It creates an additional CUDA context, so allow memory for that process.

## 4. Fill, restart both processes, then replay

Use the direct server origin (without `/v1`) for `API_ORIGIN` and its served model ID for `MODEL`. The probe needs `/tokenize` and `/metrics` as well as the chat endpoint. It assumes a directly reachable endpoint without authentication.

```sh
python3 tests/cache_probe.py prepare --base-url "$API_ORIGIN" --model "$MODEL"
python3 tests/cache_probe.py run --label cold --base-url "$API_ORIGIN" --model "$MODEL"
```

Run with no unrelated traffic so metric deltas describe these requests. The probe saves random-identifier fixtures in `probe-results/fixtures.json`. **Keep this exact file for the replay.** To start another cold experiment, select a new `--work-dir` on all its probe commands.

Wait for the requests and disk writes to finish. Gracefully stop both serving processes and start both with the same SSD mounts. Wait for the model service to become ready. Then run:

```sh
python3 tests/cache_probe.py run --label restored --base-url "$API_ORIGIN" --model "$MODEL"
```

The script requires the exact identifier and a normal stop. In `probe-results/results/restored-*.json`, check cached prompt tokens, TTFT and external-cache/CPU-to-GPU metric deltas. Metric publication may lag the response; inspect `/metrics` again after a few seconds if needed. Confirm a disk lookup in the scheduler logs after restart. A warm request without a process restart cannot distinguish SSD reuse from an in-memory hit.

This tests clean process restart. It does not demonstrate host-reboot or power-loss recovery. Model initialization time is separate from replay TTFT.

## Rollback

Gracefully stop both nodes. Restore the previous image and launch settings, remove the connector argument, and restart both. Keep the SSD directories for inspection or a later compatible restart. They are not used when the connector is disabled.
