#!/usr/bin/env python3
"""
Deep diagnostics for evaluation pipeline bugs.

Usage:
    python diagnose_evaluation.py results/hotpotqa_full
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any
import hashlib


def check_file_identity(root_dir: str):
    """Check if different systems are using identical files (BUG #1)."""
    print("\n" + "="*70)
    print("FILE IDENTITY CHECK (Bug #1)")
    print("="*70)
    
    shard_dirs = [d for d in Path(root_dir).iterdir() if d.is_dir() and d.name.startswith("dev_")]
    
    systems = [
        "vanilla_rag",
        "agentragdrop_none",
        "agentragdrop_lazy_greedy",
        "agentragdrop_risk_controlled"
    ]
    
    for shard_dir in sorted(shard_dirs)[:2]:  # Check first 2 shards
        print(f"\n{shard_dir.name}:")
        
        file_hashes = {}
        
        for system in systems:
            results_file = shard_dir / f"{system}_results.jsonl"
            if not results_file.exists():
                print(f"  ✗ {system}: FILE MISSING")
                continue
            
            # Hash first 10 lines
            with open(results_file) as f:
                lines = [f.readline() for _ in range(10)]
                content = ''.join(lines)
                file_hash = hashlib.md5(content.encode()).hexdigest()
            
            file_hashes[system] = file_hash
            
            # Check actual predictions differ
            with open(results_file) as f:
                first_result = json.loads(f.readline())
                print(f"  {system}:")
                print(f"    - File hash: {file_hash[:8]}")
                print(f"    - First pred: {first_result['pred_answer'][:40]}...")
                print(f"    - Tokens: {first_result['tokens']}")
                print(f"    - Pruned: {first_result['agents_pruned']}")
        
        # Check for duplicates
        hash_counts = Counter(file_hashes.values())
        duplicates = [h for h, c in hash_counts.items() if c > 1]
        
        if duplicates:
            print(f"\n  ❌ FOUND IDENTICAL FILES:")
            for dup_hash in duplicates:
                dup_systems = [s for s, h in file_hashes.items() if h == dup_hash]
                print(f"    - {dup_systems} have IDENTICAL content!")
        else:
            print(f"\n  ✓ All systems have unique files")


def check_prediction_diversity(root_dir: str):
    """Check if predictions actually differ across systems."""
    print("\n" + "="*70)
    print("PREDICTION DIVERSITY CHECK")
    print("="*70)
    
    shard_dirs = [d for d in Path(root_dir).iterdir() if d.is_dir() and d.name.startswith("dev_")]
    
    if not shard_dirs:
        print("No shard directories found")
        return
    
    first_shard = sorted(shard_dirs)[0]
    
    systems = [
        "agentragdrop_none",
        "agentragdrop_lazy_greedy",
        "agentragdrop_risk_controlled"
    ]
    
    # Load all predictions for first 20 examples
    all_preds = {}
    
    for system in systems:
        results_file = first_shard / f"{system}_results.jsonl"
        if not results_file.exists():
            continue
        
        preds = []
        with open(results_file) as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                result = json.loads(line)
                preds.append({
                    'question': result['question'][:60],
                    'pred': result['pred_answer'],
                    'tokens': result['tokens'],
                    'pruned': result['agents_pruned']
                })
        
        all_preds[system] = preds
    
    # Compare predictions
    print(f"\nComparing first 20 examples from {first_shard.name}:")
    
    identical_preds = 0
    identical_tokens = 0
    
    for i in range(min(20, len(all_preds.get('agentragdrop_none', [])))):
        none_pred = all_preds['agentragdrop_none'][i]['pred']
        lazy_pred = all_preds.get('agentragdrop_lazy_greedy', [{}])[i].get('pred', '')
        risk_pred = all_preds.get('agentragdrop_risk_controlled', [{}])[i].get('pred', '')
        
        none_tokens = all_preds['agentragdrop_none'][i]['tokens']
        lazy_tokens = all_preds.get('agentragdrop_lazy_greedy', [{}])[i].get('tokens', 0)
        
        if none_pred == lazy_pred == risk_pred:
            identical_preds += 1
        
        if none_tokens == lazy_tokens:
            identical_tokens += 1
    
    print(f"  Identical predictions (none==lazy==risk): {identical_preds}/20")
    print(f"  Identical tokens (none==lazy): {identical_tokens}/20")
    
    if identical_preds > 18:
        print(f"\n  ❌ CRITICAL: {identical_preds}/20 predictions are identical!")
        print(f"     This means pruning is NOT affecting answers at all.")
    
    if identical_tokens > 18:
        print(f"\n  ❌ CRITICAL: {identical_tokens}/20 have identical token counts!")
        print(f"     This means pruning is NOT being executed.")
    
    # Show sample differences
    print(f"\nSample comparison (first 3 examples):")
    for i in range(min(3, len(all_preds.get('agentragdrop_none', [])))):
        print(f"\n  Example {i+1}:")
        print(f"    Question: {all_preds['agentragdrop_none'][i]['question']}")
        
        for system in systems:
            if system in all_preds and i < len(all_preds[system]):
                p = all_preds[system][i]
                print(f"    {system}:")
                print(f"      Pred: {p['pred'][:50]}")
                print(f"      Tokens: {p['tokens']}")
                print(f"      Pruned: {p['pruned']}")


def check_answer_formatting(root_dir: str):
    """Check answer formatting issues (Bug #2)."""
    print("\n" + "="*70)
    print("ANSWER FORMATTING CHECK (Bug #2)")
    print("="*70)
    
    shard_dirs = [d for d in Path(root_dir).iterdir() if d.is_dir() and d.name.startswith("dev_")]
    
    if not shard_dirs:
        print("No shard directories found")
        return
    
    first_shard = sorted(shard_dirs)[0]
    results_file = first_shard / "vanilla_rag_results.jsonl"
    
    if not results_file.exists():
        print(f"File not found: {results_file}")
        return
    
    print(f"\nAnalyzing: {results_file}")
    
    issues = {
        'explanation_prefix': 0,
        'too_long': 0,
        'has_punctuation': 0,
        'has_article': 0,
        'all_caps': 0,
        'empty': 0
    }
    
    samples = []
    
    with open(results_file) as f:
        for i, line in enumerate(f):
            if i >= 100:  # Check first 100
                break
            
            result = json.loads(line)
            pred = result['pred_answer']
            gold = result['gold_answer']
            
            if i < 10:
                samples.append((result['question'][:60], gold, pred, result['f1']))
            
            # Check for common issues
            if any(pred.lower().startswith(p) for p in ['the answer is', 'answer:', 'it is', 'according to']):
                issues['explanation_prefix'] += 1
            
            if len(pred.split()) > 10:
                issues['too_long'] += 1
            
            if pred.endswith(('.', ',', '!', '?')):
                issues['has_punctuation'] += 1
            
            if any(pred.lower().startswith(a) for a in ['a ', 'an ', 'the ']):
                issues['has_article'] += 1
            
            if pred.isupper():
                issues['all_caps'] += 1
            
            if not pred.strip():
                issues['empty'] += 1
    
    print(f"\nFormatting issues found (in first 100):")
    for issue, count in issues.items():
        if count > 0:
            print(f"  ❌ {issue}: {count}/100 examples")
    
    print(f"\nSample predictions (first 10):")
    for i, (q, gold, pred, f1) in enumerate(samples):
        print(f"\n  [{i+1}] Q: {q}...")
        print(f"      Gold: '{gold}'")
        print(f"      Pred: '{pred}'")
        print(f"      F1: {f1:.3f}")
        
        # Diagnose this example
        if f1 == 0 and gold.lower() in pred.lower():
            print(f"      ⚠️  Gold is IN pred but F1=0 (normalization issue)")


def check_retrieval_quality(root_dir: str):
    """Check retrieval quality (Bug #3)."""
    print("\n" + "="*70)
    print("RETRIEVAL QUALITY CHECK (Bug #3)")
    print("="*70)
    
    # This needs access to the original dataset to check support recall
    print("\nTo check retrieval quality, run:")
    print("  python check_retrieval_recall.py")


def check_budget_settings(root_dir: str):
    """Check if budget settings are causing no pruning."""
    print("\n" + "="*70)
    print("BUDGET SETTINGS CHECK")
    print("="*70)
    
    shard_dirs = [d for d in Path(root_dir).iterdir() if d.is_dir() and d.name.startswith("dev_")]
    
    if not shard_dirs:
        print("No shard directories found")
        return
    
    first_shard = sorted(shard_dirs)[0]
    summary_file = first_shard / "summary.json"
    
    if not summary_file.exists():
        print(f"Summary not found: {summary_file}")
        return
    
    with open(summary_file) as f:
        data = json.load(f)
        config = data.get('config', {})
    
    print(f"\nConfiguration used:")
    print(f"  budget_tokens: {config.get('budget_tokens', 'NOT SET')}")
    print(f"  budget_time_ms: {config.get('budget_time_ms', 'NOT SET')}")
    print(f"  lambda_redundancy: {config.get('lambda_redundancy', 'NOT SET')}")
    print(f"  risk_budget_alpha: {config.get('risk_budget_alpha', 'NOT SET')}")
    
    budget_tokens = config.get('budget_tokens', 0)
    
    if budget_tokens == 0:
        print(f"\n  ❌ CRITICAL: budget_tokens=0 (UNLIMITED)")
        print(f"     With unlimited budget, nothing gets pruned!")
        print(f"\n  SOLUTION: Re-run with --budget_tokens 600")
    elif budget_tokens > 900:
        print(f"\n  ⚠️  WARNING: budget_tokens={budget_tokens} is very high")
        print(f"     Average usage is ~537 tokens, so budget of {budget_tokens} allows all agents")
        print(f"\n  SUGGESTION: Try --budget_tokens 400 for aggressive pruning")
    else:
        print(f"\n  ✓ Budget looks reasonable: {budget_tokens} tokens")


def main():
    parser = argparse.ArgumentParser(
        description="Deep diagnostics for evaluation bugs"
    )
    parser.add_argument(
        "root_dir",
        help="Root directory containing shard results"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.root_dir):
        print(f"Error: Directory not found: {args.root_dir}")
        return 1
    
    print("="*70)
    print("DEEP EVALUATION DIAGNOSTICS")
    print("="*70)
    print(f"Directory: {args.root_dir}")
    
    # Run all checks
    check_file_identity(args.root_dir)
    check_prediction_diversity(args.root_dir)
    check_answer_formatting(args.root_dir)
    check_budget_settings(args.root_dir)
    check_retrieval_quality(args.root_dir)
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    
    print("\n📋 ACTION ITEMS:")
    print("1. If files are identical: Fix aggregator to read correct per-system files")
    print("2. If predictions identical: Check budget_tokens setting (should be 400-600)")
    print("3. If formatting issues: Implement answer cleaning in RAGComposerAgent")
    print("4. Run: python check_retrieval_recall.py to measure retrieval quality")


if __name__ == "__main__":
    exit(main())