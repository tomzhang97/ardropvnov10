#!/usr/bin/env python3
"""
Shard HotpotQA dataset into 1000-example chunks for parallel GPU processing.

Usage:
    python shard_hotpot_dataset.py \
        --input hotpotQA/hotpot_dev_distractor_v1.json \
        --output_dir runs/_shards \
        --shard_size 1000
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any


def load_hotpot_data(input_path: str) -> List[Dict[str, Any]]:
    """Load HotpotQA dataset."""
    print(f"Loading data from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'data' in data:
        return data['data']
    else:
        raise ValueError(f"Unexpected data format in {input_path}")


def create_shards(
    data: List[Dict[str, Any]], 
    output_dir: str, 
    shard_size: int = 1000
) -> List[str]:
    """
    Create sharded dataset files.
    
    Returns:
        List of shard file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_examples = len(data)
    num_shards = (total_examples + shard_size - 1) // shard_size
    
    print(f"\nCreating {num_shards} shards:")
    print(f"  Total examples: {total_examples}")
    print(f"  Shard size: {shard_size}")
    print(f"  Output dir: {output_dir}")
    print()
    
    shard_paths = []
    
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, total_examples)
        
        shard_data = data[start:end]
        
        # Shard filename: dev_0_1000.json, dev_1000_2000.json, etc.
        shard_filename = f"dev_{start}_{end}.json"
        shard_path = os.path.join(output_dir, shard_filename)
        
        # Write shard
        with open(shard_path, 'w', encoding='utf-8') as f:
            json.dump(shard_data, f, ensure_ascii=False, indent=2)
        
        shard_paths.append(shard_path)
        
        print(f"  ✓ Shard {shard_idx + 1}/{num_shards}: {shard_filename} ({len(shard_data)} examples)")
    
    return shard_paths


def verify_shards(shard_paths: List[str], original_size: int):
    """Verify that shards contain all examples."""
    print(f"\nVerifying shards...")
    
    total_in_shards = 0
    shard_ids = set()
    
    for shard_path in shard_paths:
        with open(shard_path, 'r', encoding='utf-8') as f:
            shard_data = json.load(f)
        
        total_in_shards += len(shard_data)
        
        # Check for duplicate IDs
        for ex in shard_data:
            ex_id = ex.get('_id', ex.get('id'))
            if ex_id:
                if ex_id in shard_ids:
                    print(f"  ⚠️  WARNING: Duplicate ID {ex_id} in {os.path.basename(shard_path)}")
                shard_ids.add(ex_id)
    
    print(f"  Original dataset: {original_size} examples")
    print(f"  Total in shards:  {total_in_shards} examples")
    
    if total_in_shards == original_size:
        print(f"  ✓ All examples preserved")
    else:
        print(f"  ✗ Mismatch: {original_size - total_in_shards} examples missing/extra")
    
    print(f"  Unique IDs: {len(shard_ids)}")


def create_shard_manifest(shard_paths: List[str], output_dir: str):
    """Create manifest file listing all shards."""
    manifest = {
        "num_shards": len(shard_paths),
        "shards": [
            {
                "path": os.path.basename(path),
                "full_path": os.path.abspath(path)
            }
            for path in shard_paths
        ]
    }
    
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Manifest created: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Shard HotpotQA dataset for parallel processing"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to hotpot_dev_distractor_v1.json"
    )
    parser.add_argument(
        "--output_dir",
        default="runs/_shards",
        help="Directory for shard files"
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1000,
        help="Number of examples per shard"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing shards"
    )
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Check if output directory exists and has shards
    if os.path.exists(args.output_dir) and not args.force:
        existing_shards = list(Path(args.output_dir).glob("dev_*.json"))
        if existing_shards:
            print(f"Warning: {len(existing_shards)} existing shards found in {args.output_dir}")
            print("Use --force to overwrite")
            response = input("Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted")
                return 0
    
    print("=" * 70)
    print("HOTPOTQA DATASET SHARDING")
    print("=" * 70)
    
    # Load data
    data = load_hotpot_data(args.input)
    
    # Create shards
    shard_paths = create_shards(data, args.output_dir, args.shard_size)
    
    # Verify
    verify_shards(shard_paths, len(data))
    
    # Create manifest
    create_shard_manifest(shard_paths, args.output_dir)
    
    print("\n" + "=" * 70)
    print("SHARDING COMPLETE")
    print("=" * 70)
    print(f"\nCreated {len(shard_paths)} shards in {args.output_dir}")
    print("\nNext steps:")
    print("  1. Run evaluation on shards:")
    print(f"     python experiments/eval_complete_runnable.py \\")
    print(f"       --shard_glob '{args.output_dir}/dev_*.json' \\")
    print(f"       --gpus 0,1,2,3 \\")
    print(f"       --concurrency 4 \\")
    print(f"       --output_dir results/hotpotqa_full")
    print("\n  2. Aggregate results:")
    print(f"     python experiments/aggregate_eval_shards.py \\")
    print(f"       --root_dir results/hotpotqa_full \\")
    print(f"       --out_dir results/hotpotqa_full_agg")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())