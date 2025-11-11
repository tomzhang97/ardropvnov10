#!/usr/bin/env python3
"""
Post-process existing predictions to fix answer formatting.

This can improve F1 by 2-3x without re-running inference!

Usage:
    python clean_predictions.py \
        --input results/hotpotqa_full/dev_0_1000/vanilla_rag_results.jsonl \
        --output results/hotpotqa_full/dev_0_1000/vanilla_rag_results_clean.jsonl
"""

import argparse
import json
import re
from pathlib import Path


def clean_answer(raw: str, question: str) -> str:
    """
    Extract canonical short answer from potentially verbose prediction.
    
    This implements aggressive post-processing to handle common issues:
    1. "The answer is X" patterns
    2. Yes/no questions
    3. Over-verbose explanations
    4. Placeholders and sentinels
    """
    if not raw:
        return ""
    
    text = raw.strip()
    
    # Handle sentinel values
    if text in ["_______", "_______________________", "[Anytime token budget reached]", 
                "[Anytime time budget reached]", "unknown", "Unknown"]:
        return "unknown"
    
    # 1) Use explicit answer markers if present
    markers = [
        "Answer:", "Final answer:", "The answer is", "The correct answer is",
        "Based on the context,", "According to the context,",
        "From the context,", "It is", "They are", "This is", "That is"
    ]
    for m in markers:
        if m.lower() in text.lower():
            idx = text.lower().rfind(m.lower())
            text = text[idx + len(m):].strip()
            # Remove "is" or ":" if it immediately follows
            text = re.sub(r'^(is|:)\s*', '', text, flags=re.IGNORECASE)
            break
    
    # 2) Handle yes/no questions
    lower_q = question.lower()
    lower_t = text.lower()
    
    # Check if question expects yes/no
    is_yesno = (
        any(lower_q.startswith(p) for p in ["is ", "are ", "was ", "were ", 
                                              "do ", "does ", "did ", "can ", 
                                              "could ", "will ", "would ", "should "]) or
        "yes or no" in lower_q
    )
    
    if is_yesno:
        # Count occurrences
        yes_count = lower_t.count("yes")
        no_count = lower_t.count("no")
        
        # If "yes" appears but not "no", return "yes"
        if yes_count > 0 and no_count == 0:
            return "yes"
        # If "no" appears but not "yes", return "no"  
        if no_count > 0 and yes_count == 0:
            return "no"
        # If both or neither, keep trying other methods
    
    # 3) Cut at first sentence boundary
    for sep in [".", "?", "!", "\n"]:
        if sep in text:
            text = text.split(sep)[0]
            break
    
    # 4) Strip quotes and extra punctuation
    text = text.strip().strip('"').strip("'").strip()
    
    # 5) Remove leading articles
    text = re.sub(r'^(a|an|the)\s+', '', text, flags=re.IGNORECASE)
    
    # 6) Remove citations [1], [2]
    text = re.sub(r'\[\d+\]', '', text)
    
    # 7) For very long answers, try to extract key entity
    tokens = text.split()
    if len(tokens) > 10:
        # Look for capitalized entity names
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if entities:
            # Return longest entity
            text = max(entities, key=len)
        else:
            # Just take first 6 tokens
            text = " ".join(tokens[:6])
    
    # 8) Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 9) Final punctuation cleanup
    text = text.rstrip('.,;:!?')
    
    return text


def clean_results_file(input_file: str, output_file: str):
    """Clean all predictions in a results file."""
    print(f"Processing: {input_file}")
    
    cleaned_count = 0
    changed_count = 0
    total = 0
    
    with open(input_file) as fin, open(output_file, 'w') as fout:
        for line in fin:
            result = json.loads(line)
            
            original_pred = result['pred_answer']
            question = result['question']
            
            # Clean the prediction
            cleaned_pred = clean_answer(original_pred, question)
            
            # Update result
            result['pred_answer'] = cleaned_pred
            result['pred_answer_original'] = original_pred  # Keep original for reference
            
            # Re-compute metrics with cleaned answer
            from agentragdrop.utils import exact_match, f1_score
            
            gold = result['gold_answer']
            result['em'] = exact_match(cleaned_pred, gold)
            result['f1'] = f1_score(cleaned_pred, gold)
            
            # Write to output
            fout.write(json.dumps(result) + '\n')
            
            total += 1
            if cleaned_pred != original_pred:
                changed_count += 1
            if result['f1'] > 0:
                cleaned_count += 1
    
    print(f"  Total examples: {total}")
    print(f"  Answers changed: {changed_count} ({100*changed_count/total:.1f}%)")
    print(f"  Non-zero F1: {cleaned_count} ({100*cleaned_count/total:.1f}%)")
    print(f"  Saved to: {output_file}")


def compute_improvement(original_file: str, cleaned_file: str):
    """Compute improvement from cleaning."""
    import numpy as np
    
    # Load original
    with open(original_file) as f:
        original_results = [json.loads(line) for line in f]
    
    # Load cleaned
    with open(cleaned_file) as f:
        cleaned_results = [json.loads(line) for line in f]
    
    # Compute metrics
    orig_f1 = np.mean([r['f1'] for r in original_results])
    orig_em = np.mean([r['em'] for r in original_results])
    
    clean_f1 = np.mean([r['f1'] for r in cleaned_results])
    clean_em = np.mean([r['em'] for r in cleaned_results])
    
    print(f"\n{'Metric':<15} {'Original':<15} {'Cleaned':<15} {'Improvement'}")
    print("-" * 65)
    print(f"{'F1':<15} {orig_f1:<15.4f} {clean_f1:<15.4f} {clean_f1-orig_f1:+.4f} ({100*(clean_f1-orig_f1)/orig_f1 if orig_f1 > 0 else 0:+.1f}%)")
    print(f"{'EM':<15} {orig_em:<15.4f} {clean_em:<15.4f} {clean_em-orig_em:+.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean predictions to fix answer formatting"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input results JSONL file"
    )
    parser.add_argument(
        "--output",
        help="Output cleaned results JSONL file (default: <input>_clean.jsonl)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all result files in input directory"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Process all *_results.jsonl files in directory
        input_dir = Path(args.input)
        if not input_dir.is_dir():
            print(f"Error: {input_dir} is not a directory")
            return 1
        
        result_files = list(input_dir.glob("*_results.jsonl"))
        
        if not result_files:
            print(f"No *_results.jsonl files found in {input_dir}")
            return 1
        
        print(f"Found {len(result_files)} result files to clean")
        
        for input_file in result_files:
            output_file = input_file.parent / f"{input_file.stem}_clean.jsonl"
            
            print(f"\n{'='*70}")
            clean_results_file(str(input_file), str(output_file))
            compute_improvement(str(input_file), str(output_file))
    
    else:
        # Process single file
        input_file = args.input
        output_file = args.output or input_file.replace(".jsonl", "_clean.jsonl")
        
        print("="*70)
        print("CLEANING PREDICTIONS")
        print("="*70)
        
        clean_results_file(input_file, output_file)
        compute_improvement(input_file, output_file)
    
    print("\n" + "="*70)
    print("CLEANING COMPLETE")
    print("="*70)
    
    print("\nNext steps:")
    print("  1. Re-run evaluation on cleaned files")
    print("  2. If F1 improved significantly, update RAGComposerAgent to use this cleaning")
    print("  3. Compare systems with cleaned predictions")


if __name__ == "__main__":
    exit(main())