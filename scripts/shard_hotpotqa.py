#!/usr/bin/env python3
"""Shard HotpotQA dev distractor dataset into fixed-size chunks."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    if not isinstance(data, list):
        raise ValueError(f"Unsupported dataset format in {path}")

    return data


def write_shard(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard HotpotQA dev dataset for multi-GPU evaluation")
    parser.add_argument("--input", default="hotpotQA/hotpot_dev_distractor_v1.json", help="Path to the HotpotQA dev distractor JSON file")
    parser.add_argument("--output_dir", default="data/hotpotQA/shards", help="Directory to write shard files")
    parser.add_argument("--prefix", default="dev", help="Prefix for shard filenames")
    parser.add_argument("--shard_size", type=int, default=1000, help="Number of examples per shard")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on total examples (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing shard files")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    dataset = load_dataset(input_path)
    if args.limit > 0:
        dataset = dataset[: args.limit]

    total = len(dataset)
    if total == 0:
        raise SystemExit("Dataset is empty; nothing to shard")

    shard_size = max(1, args.shard_size)
    num_shards = math.ceil(total / shard_size)

    print(f"Input dataset : {input_path}")
    print(f"Total examples: {total}")
    print(f"Shard size    : {shard_size}")
    print(f"Output dir    : {output_dir}")
    print(f"Shards        : {num_shards}")

    shards_meta: List[Dict[str, Any]] = []
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(total, start + shard_size)
        shard_rows = dataset[start:end]
        shard_name = f"{args.prefix}_{start}_{end}.json"
        shard_path = output_dir / shard_name

        if shard_path.exists() and not args.overwrite:
            print(f"[skip] {shard_name} (exists)")
        else:
            print(f"[write] {shard_name} ({len(shard_rows)} examples)")
            write_shard(shard_path, shard_rows)

        shards_meta.append({
            "file": shard_name,
            "start": start,
            "end": end,
            "num_examples": len(shard_rows),
        })

    metadata = {
        "source": str(input_path),
        "total_examples": total,
        "shard_size": shard_size,
        "num_shards": num_shards,
        "prefix": args.prefix,
        "shards": shards_meta,
    }

    meta_path = output_dir / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata written to {meta_path}")


if __name__ == "__main__":
    main()
