import os
from pathlib import Path

from rae_ablation.cache import compute_cache_statistics, read_manifest

root = Path(os.environ["CACHE"])
manifest = read_manifest(root / "manifest.json")

available = manifest.completed_shards
if not available:
    available = sorted(
        int(path.stem.split("-")[1])
        for path in root.glob("shard-*.safetensors")
    )

sample_count = min(32, len(available))
if sample_count == 1:
    selected = [available[0]]
else:
    selected = sorted({
        available[round(i * (len(available) - 1) / (sample_count - 1))]
        for i in range(sample_count)
    })

# This only changes the in-memory manifest; manifest.json remains complete.
manifest.completed_shards = selected
print(f"Computing approximate statistics from {len(selected)} shards: {selected}")
compute_cache_statistics(root, manifest)
print("Approximate statistics saved.")