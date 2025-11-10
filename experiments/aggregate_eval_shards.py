#!/usr/bin/env python3
"""
Aggregate evaluation results across shards.

Assumes each shard directory (under --root_dir) contains files like:
  vanilla_rag_results.jsonl
  agentragdrop_none_results.jsonl
  agentragdrop_lazy_greedy_results.jsonl
  agentragdrop_risk_controlled_results.jsonl
and a per-shard summary.json created by eval_complete_runnable.py.

Example usage:

  python experiments/aggregate_eval_shards.py \
      --root_dir results/hotpotqa_full \
      --out_dir results/hotpotqa_full_agg
"""

import os
import json
import time
import argparse
from typing import Dict, List, Any, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Bootstrap + metrics (same style as eval_complete_runnable.py)
# ---------------------------------------------------------------------------

def paired_bootstrap(
    metric1: List[float],
    metric2: List[float],
    n_bootstrap: int = 1000
) -> Dict[str, float]:
    """Paired bootstrap test."""
    assert len(metric1) == len(metric2)

    diffs = np.array(metric1) - np.array(metric2)
    observed_diff = np.mean(diffs)

    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(diffs), size=len(diffs), replace=True)
        boot_diffs.append(np.mean(diffs[idx]))

    boot_diffs = np.array(boot_diffs)

    # Two-tailed p-value
    p_value = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))

    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant_at_0.05": bool(p_value < 0.05),
        "significant_at_0.01": bool(p_value < 0.01),
    }


def compute_summary(all_results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """
    all_results: system_name -> question -> result_dict

    Returns: system_name -> summary_stats
    """
    summary: Dict[str, Dict[str, float]] = {}

    for sys_name, q2r in all_results.items():
        if not q2r:
            continue

        results = list(q2r.values())

        em_scores = [r.get("em", 0.0) for r in results]
        f1_scores = [r.get("f1", 0.0) for r in results]
        tokens = [r.get("tokens", 0.0) for r in results]
        latencies = [r.get("latency_ms", 0.0) for r in results]

        summary[sys_name] = {
            "n": len(results),
            "em_mean": float(np.mean(em_scores)),
            "em_std": float(np.std(em_scores)),
            "f1_mean": float(np.mean(f1_scores)),
            "f1_std": float(np.std(f1_scores)),
            "tokens_mean": float(np.mean(tokens)),
            "tokens_std": float(np.std(tokens)),
            "latency_mean": float(np.mean(latencies)),
            "latency_p50": float(np.percentile(latencies, 50)),
            "latency_p95": float(np.percentile(latencies, 95)),
        }

    return summary


def compute_significance_tests(all_results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """
    all_results: system_name -> question -> result_dict

    Returns: dict of test_name -> bootstrap_result
    """
    tests: Dict[str, Dict[str, float]] = {}

    baseline_name = "vanilla_rag"
    our_systems = ["agentragdrop_lazy_greedy", "agentragdrop_risk_controlled"]

    if baseline_name not in all_results:
        return tests

    baseline_q2r = all_results[baseline_name]

    for sys_name in our_systems:
        if sys_name not in all_results:
            continue

        sys_q2r = all_results[sys_name]

        # Align by question key intersection
        common_questions = sorted(set(sys_q2r.keys()) & set(baseline_q2r.keys()))
        if not common_questions:
            continue

        our_f1 = [sys_q2r[q]["f1"] for q in common_questions]
        base_f1 = [baseline_q2r[q]["f1"] for q in common_questions]

        our_tokens = [sys_q2r[q]["tokens"] for q in common_questions]
        base_tokens = [baseline_q2r[q]["tokens"] for q in common_questions]

        # F1 test
        f1_res = paired_bootstrap(our_f1, base_f1)
        tests[f"{sys_name}_vs_{baseline_name}_f1"] = f1_res

        # Token test
        tok_res = paired_bootstrap(our_tokens, base_tokens)
        tests[f"{sys_name}_vs_{baseline_name}_tokens"] = tok_res

    return tests


# ---------------------------------------------------------------------------
# Aggregation logic
# ---------------------------------------------------------------------------

def load_all_results(root_dir: str) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], List[str]]:
    """
    root_dir: directory containing shard subdirs (e.g. results/hotpotqa_full)

    Returns:
      - all_results: system_name -> question -> result_dict
      - shard_dirs: list of shard subdir names (relative to root_dir)
    """
    shard_dirs = [
        d for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for shard in shard_dirs:
        shard_path = os.path.join(root_dir, shard)
        # Any file ending with _results.jsonl is a system result file
        for fname in sorted(os.listdir(shard_path)):
            if not fname.endswith("_results.jsonl"):
                continue

            system_name = fname[:-len("_results.jsonl")]
            file_path = os.path.join(shard_path, fname)

            if system_name not in all_results:
                all_results[system_name] = {}

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    q = r.get("question")
                    if q is None:
                        # Skip if no question key
                        continue
                    # If duplicate question shows up, last one wins; this should
                    # not happen if shards are disjoint.
                    all_results[system_name][q] = r

    return all_results, shard_dirs


def print_summary_table(summary: Dict[str, Dict[str, float]], sig_tests: Dict[str, Dict[str, float]]):
    print("\n" + "=" * 90)
    print("AGGREGATED EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'System':<30} {'EM':<12} {'F1':<12} {'Tokens':<12} {'Lat(p95)'}")
    print("-" * 90)

    for sys_name in sorted(summary.keys()):
        stats = summary[sys_name]
        em_str = f"{stats['em_mean']*100:.1f}±{stats['em_std']*100:.1f}"
        f1_str = f"{stats['f1_mean']*100:.1f}±{stats['f1_std']*100:.1f}"
        tok_str = f"{stats['tokens_mean']:.0f}±{stats['tokens_std']:.0f}"
        lat_str = f"{stats['latency_p95']:.0f}ms"
        print(f"{sys_name:<30} {em_str:<12} {f1_str:<12} {tok_str:<12} {lat_str}")

    print("=" * 90)

    if sig_tests:
        print("\nAGGREGATED STATISTICAL SIGNIFICANCE (Paired Bootstrap, n=1000)")
        print("-" * 90)
        for test_name, result in sig_tests.items():
            sig = "***" if result["significant_at_0.01"] else ("**" if result["significant_at_0.05"] else "")
            metric = "F1" if "_f1" in test_name else "Tokens"
            print(
                f"{test_name}: {metric} Δ={result['observed_diff']:+.3f} "
                f"{sig} (p={result['p_value']:.4f})"
            )
        print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results across shards."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing shard subdirectories (e.g. results/hotpotqa_full)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Directory to write aggregate summary.json (default: <root_dir>/aggregate)",
    )

    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(root_dir, "aggregate")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("AGENTRAG-DROP: AGGREGATE SHARD EVALUATION")
    print("=" * 70)
    print(f"Root dir : {root_dir}")
    print(f"Out dir  : {out_dir}")
    print("=" * 70 + "\n")

    all_results, shard_dirs = load_all_results(root_dir)
    if not all_results:
        print(f"[ERROR] No *_results.jsonl files found under {root_dir}")
        return

    summary = compute_summary(all_results)
    sig_tests = compute_significance_tests(all_results)

    # Save aggregate summary
    out_path = os.path.join(out_dir, "summary.json")
    payload = {
        "aggregate_from": root_dir,
        "num_shards": len(shard_dirs),
        "shards": shard_dirs,
        "systems": sorted(all_results.keys()),
        "summary": summary,
        "significance_tests": sig_tests,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nAggregate summary written to: {out_path}")

    # Pretty-print table
    print_summary_table(summary, sig_tests)


if __name__ == "__main__":
    main()
