# SPDX-License-Identifier: Apache-2.0
"""Check or install the frozen Python overlay in a derived serving image."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=["check", "apply", "verify"])
args = parser.parse_args()
root = Path(__file__).resolve().parent
version = importlib.metadata.version("vllm")
expected_version = "0.1.dev20904+g179dd0fa9"
if version != expected_version:
    raise RuntimeError(f"Expected vLLM {expected_version}; found {version}. Rebase required.")
spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate the installed vLLM package")
target = Path(spec.origin).resolve().parent.parent
manifest = json.loads((root / "overlay-manifest.json").read_text())

# Check every input before writing any files. A mismatch is a different base,
# not an invitation to overwrite another release or another local patch.
for item in manifest:
    source = root / "overlay" / item["path"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != item["after_sha256"]:
        raise RuntimeError(f"Bundle checksum mismatch: {item['path']}")
    installed = target / item["path"]
    expected = item["after_sha256"] if args.action == "verify" else item["before_sha256"]
    if hashlib.sha256(installed.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Installed source mismatch: {item['path']}. Rebase required.")

module = "node_local_disk_offload.py"
if args.action == "apply":
    for item in manifest:
        shutil.copyfile(root / "overlay" / item["path"], target / item["path"])
    shutil.copyfile(root / module, target / module)
elif args.action == "verify":
    if (target / module).read_bytes() != (root / module).read_bytes():
        raise RuntimeError("Installed disk module differs from the bundle")
print(json.dumps({"action": args.action, "vllm": version, "files": len(manifest), "status": "passed"}))
