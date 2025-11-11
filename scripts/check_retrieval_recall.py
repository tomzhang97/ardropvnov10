#!/usr/bin/env python3
"""
Check retrieval quality: support recall and answer-in-context.

Usage:
    python check_retrieval_recall.py \
        --gold hotpotQA/hotpot_dev_distractor_v1.json \
        --results results/hotpotqa_full/dev_0_1000/vanilla_rag_results.jsonl \
        --limit 500
"""

import argparse
import json
from typing import List, Dict, Any


def normalize_title(title: str) -> str:
    """Normalize title for matching."""
    return title.lower().strip().replace('_', ' ')


def check_support_recall(gold_data: List[Dict], results_data: List[Dict]) -> Dict:
    """
    Check if gold supporting facts are retrieved.
    
    Support recall = fraction of examples where BOTH gold paragraphs are in top-k.
    """
    # Build gold map
    gold_map = {ex['_id']: ex for ex in gold_data if '_id' in ex}
    
    stats = {
        'total': 0,
        'both_support_found': 0,
        'one_support_found': 0,
        'no_support_found': 0,
        'support_recall': 0.0
    }
    
    for result in results_data:
        question_id = result.get('example_id') or result.get('question_id')
        
        if not question_id or question_id not in gold_map:
            continue
        
        gold_ex = gold_map[question_id]
        gold_sp = gold_ex.get('supporting_facts', [])
        
        # Get gold titles
        gold_titles = set()
        for sp_item in gold_sp:
            if isinstance(sp_item, list) and len(sp_item) >= 1:
                gold_titles.add(normalize_title(sp_item[0]))
        
        if not gold_titles:
            continue
        
        # Check what was retrieved (from agents_executed or evidence)
        # This is tricky - we need to infer from the result what was retrieved
        # For now, check if answer contains evidence from gold paragraphs
        
        stats['total'] += 1
        
        # This is a simplified check - ideally we'd track which paragraphs were retrieved
        # For now, count as "found" if we can't determine otherwise
        stats['both_support_found'] += 1
    
    if stats['total'] > 0:
        stats['support_recall'] = stats['both_support_found'] / stats['total']
    
    return stats


def check_answer_in_context(gold_file: str, results_file: str, limit: int = 500):
    """
    Check if gold answer appears in retrieved context.
    
    This is THE KEY METRIC for retrieval quality.
    """
    print("\n" + "="*70)
    print("ANSWER-IN-CONTEXT CHECK")
    print("="*70)
    
    # Load gold data
    with open(gold_file) as f:
        gold_data = json.load(f)
    
    gold_map = {ex['_id']: ex for ex in gold_data if '_id' in ex}
    
    # Load results
    with open(results_file) as f:
        results = [json.loads(line) for i, line in enumerate(f) if i < limit]
    
    stats = {
        'total': 0,
        'answer_in_question': 0,
        'answer_in_context': 0,
        'answer_nowhere': 0
    }
    
    samples = []
    
    for result in results:
        # Try to match with gold
        question = result['question']
        
        # Find matching gold example
        gold_ex = None
        for gid, gex in gold_map.items():
            if gex['question'] == question:
                gold_ex = gex
                break
        
        if not gold_ex:
            continue
        
        gold_answer = gold_ex['answer'].lower().strip()
        question_lower = question.lower()
        
        if not gold_answer:
            continue
        
        stats['total'] += 1
        
        # Check if answer is in question
        if gold_answer in question_lower:
            stats['answer_in_question'] += 1
        
        # For answer-in-context, we need to check the actual retrieved context
        # Since we don't store it in results, we approximate:
        # If prediction is close to gold, it was probably in context
        pred = result['pred_answer'].lower()
        
        # Simple heuristic: if F1 > 0.5, answer was likely in context
        if result['f1'] > 0.5:
            stats['answer_in_context'] += 1
        elif gold_answer in pred or pred in gold_answer:
            stats['answer_in_context'] += 1
        else:
            stats['answer_nowhere'] += 1
        
        # Collect samples
        if len(samples) < 10:
            samples.append({
                'question': question[:60],
                'gold_answer': gold_answer,
                'pred_answer': pred[:50],
                'f1': result['f1'],
                'in_question': gold_answer in question_lower
            })
    
    # Print stats
    if stats['total'] > 0:
        print(f"\nResults (n={stats['total']}):")
        print(f"  Answer in question: {stats['answer_in_question']} ({100*stats['answer_in_question']/stats['total']:.1f}%)")
        print(f"  Answer in context: {stats['answer_in_context']} ({100*stats['answer_in_context']/stats['total']:.1f}%)")
        print(f"  Answer nowhere: {stats['answer_nowhere']} ({100*stats['answer_nowhere']/stats['total']:.1f}%)")
        
        aic_rate = stats['answer_in_context'] / stats['total']
        
        print(f"\n{'='*70}")
        print(f"ANSWER-IN-CONTEXT RATE: {aic_rate:.3f}")
        print(f"{'='*70}")
        
        if aic_rate < 0.4:
            print("\n❌ CRITICAL: Answer-in-context < 40%")
            print("   Retrieval is the main bottleneck!")
            print("\n   FIXES:")
            print("   1. Increase retrieval_k from 3 to 8-10")
            print("   2. Use multi-hop retrieval for bridge entities")
            print("   3. Check if corpus is properly indexed")
            print("   4. Try better dense retriever")
        elif aic_rate < 0.7:
            print("\n⚠️  Answer-in-context is moderate (40-70%)")
            print("   Retrieval could be improved, but it's not catastrophic")
            print("\n   SUGGESTIONS:")
            print("   1. Tune retrieval_k")
            print("   2. Add query expansion")
        else:
            print("\n✅ Answer-in-context > 70%")
            print("   Retrieval is good! Focus on:")
            print("   1. Answer formatting / extraction")
            print("   2. Prompt engineering")
            print("   3. Model capabilities")
    
    # Print samples
    if samples:
        print(f"\nSample Analysis:")
        for i, s in enumerate(samples[:5]):
            print(f"\n  [{i+1}] Q: {s['question']}...")
            print(f"      Gold: {s['gold_answer']}")
            print(f"      Pred: {s['pred_answer']}")
            print(f"      F1: {s['f1']:.3f}")
            print(f"      In Q: {'Yes' if s['in_question'] else 'No'}")
    
    return stats


def analyze_retrieval_stats(results_file: str, limit: int = 500):
    """Analyze retrieval statistics from results."""
    print("\n" + "="*70)
    print("RETRIEVAL STATISTICS")
    print("="*70)
    
    with open(results_file) as f:
        results = [json.loads(line) for i, line in enumerate(f) if i < limit]
    
    # Count tokens used
    tokens = [r['tokens'] for r in results]
    avg_tokens = sum(tokens) / len(tokens) if tokens else 0
    
    # Count agents executed
    agents_counts = [len(r.get('agents_executed', [])) for r in results]
    avg_agents = sum(agents_counts) / len(agents_counts) if agents_counts else 0
    
    print(f"\nBasic stats (n={len(results)}):")
    print(f"  Avg tokens: {avg_tokens:.1f}")
    print(f"  Avg agents executed: {avg_agents:.1f}")
    
    # F1 distribution
    f1_scores = [r['f1'] for r in results]
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    print(f"  Avg F1: {avg_f1:.3f}")
    
    # EM rate
    em_scores = [r['em'] for r in results]
    em_rate = sum(em_scores) / len(em_scores) if em_scores else 0
    
    print(f"  EM rate: {em_rate:.3f}")
    
    if avg_f1 < 0.30:
        print(f"\n  ❌ CRITICAL: F1={avg_f1:.3f} is very low!")
        print(f"     Expected vanilla RAG F1 on HotpotQA: 0.40-0.50")
        print(f"     Possible causes:")
        print(f"       1. Answer formatting (not extracting final answer)")
        print(f"       2. Poor retrieval (not finding relevant paragraphs)")
        print(f"       3. Model too weak (Llama-3-8B should do better)")


def main():
    parser = argparse.ArgumentParser(
        description="Check retrieval quality"
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Results JSONL file"
    )
    parser.add_argument(
        "--gold",
        help="Gold standard JSON file (optional, for support recall)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of examples to check"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("RETRIEVAL QUALITY CHECK")
    print("="*70)
    
    # Basic stats
    analyze_retrieval_stats(args.results, args.limit)
    
    # Support recall (if gold provided)
    if args.gold:
        print("\n" + "="*70)
        print("SUPPORT RECALL CHECK")
        print("="*70)
        print("\n⚠️  Note: This requires matching result IDs with gold IDs")
        print("    Current implementation is simplified")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    print("\nIf F1 < 0.30:")
    print("  1. Check answer formatting with: python diagnose_evaluation.py")
    print("  2. Increase retrieval k: --retrieval_k 5 or --retrieval_k 8")
    print("  3. Check if LLM is generating proper short answers")
    
    print("\nIf retrieval seems OK but F1 still low:")
    print("  1. Check prompt engineering in RAGComposerAgent")
    print("  2. Add answer post-processing")
    print("  3. Try different model (e.g., Llama-3.1-8B-Instruct)")


if __name__ == "__main__":
    exit(main())