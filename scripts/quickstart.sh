#!/bin/bash
# scripts/quickstart.sh
# Quick start script for AgentRAG-Drop evaluation

set -e  # Exit on error

echo "========================================================================"
echo "AgentRAG-Drop: Quick Start Evaluation"
echo "========================================================================"

# Configuration
DATASET="hotpotqa"
LIMIT=100  # Use 100 examples for quick test (set to 0 for full evaluation)
DEVICE=0   # GPU device (use -1 for CPU)
OUTPUT_DIR="results/quickstart"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}Step 1/4:${NC} Checking dependencies..."
python -c "import torch; import transformers; import sentence_transformers; print('✓ All dependencies installed')"

echo ""
echo -e "${BLUE}Step 2/4:${NC} Preparing dataset (${DATASET}, limit=${LIMIT})..."
if [ ! -f "data/${DATASET}_eval.jsonl" ]; then
    python experiments/prepare_datasets.py \
        --dataset ${DATASET} \
        --split validation \
        --out_prefix data/
    echo -e "${GREEN}✓ Dataset prepared${NC}"
else
    echo "✓ Dataset already exists"
fi

echo ""
echo -e "${BLUE}Step 3/4:${NC} Running comprehensive evaluation..."
echo "This will evaluate:"
echo "  1. Vanilla RAG"
echo "  2. Self-RAG"
echo "  3. CRAG"
echo "  4. KET-RAG"
echo "  5. SAGE"
echo "  6. Plan-RAG"
echo "  7. AgentRAG-Drop (no pruning)"
echo "  8. AgentRAG-Drop (lazy greedy)"
echo "  9. AgentRAG-Drop (risk-controlled)"
echo ""
echo "Estimated time: ~10 minutes for ${LIMIT} examples"
echo ""

python experiments/eval_comprehensive.py \
    --dataset ${DATASET} \
    --data_path data/${DATASET}_eval.jsonl \
    --corpus_path data/${DATASET}_corpus.json \
    --limit ${LIMIT} \
    --device ${DEVICE} \
    --output_dir ${OUTPUT_DIR} \
    --seed 42

echo ""
echo -e "${BLUE}Step 4/4:${NC} Generating visualizations..."

# Create simple summary
python - <<EOF
import json
import os

# Load summary
with open('${OUTPUT_DIR}/summary.json') as f:
    data = json.load(f)

summary = data['summary']
tests = data['significance_tests']

print("\n" + "="*80)
print("QUICK START RESULTS SUMMARY")
print("="*80)
print(f"\nDataset: ${DATASET} (n={LIMIT} examples)")
print(f"Output directory: ${OUTPUT_DIR}")
print("\n" + "-"*80)
print(f"{'System':<30} {'F1':<10} {'Tokens':<10} {'P95 Latency'}")
print("-"*80)

# Sort by F1
systems_sorted = sorted(summary.items(), key=lambda x: x[1]['f1_mean'], reverse=True)

for sys_name, stats in systems_sorted:
    f1 = stats['f1_mean'] * 100
    tokens = stats['tokens_mean']
    lat_p95 = stats['latency_p95']
    
    print(f"{sys_name:<30} {f1:>6.2f}%   {tokens:>7.0f}    {lat_p95:>7.0f}ms")

print("="*80)

# Show key comparisons
print("\nKEY FINDINGS:")
print("-"*80)

# AgentRAG-Drop vs baselines
ours_lazy = summary.get('agentragdrop_lazy_greedy', {})
plan_rag = summary.get('plan_rag', {})
vanilla = summary.get('vanilla_rag', {})

if ours_lazy and plan_rag:
    f1_improvement = (ours_lazy['f1_mean'] - plan_rag['f1_mean']) * 100
    token_reduction = (1 - ours_lazy['tokens_mean'] / plan_rag['tokens_mean']) * 100
    
    print(f"\nAgentRAG-Drop (Lazy Greedy) vs Plan-RAG:")
    print(f"  F1: {f1_improvement:+.2f} points")
    print(f"  Tokens: {token_reduction:+.1f}% reduction")

if ours_lazy and vanilla:
    f1_improvement = (ours_lazy['f1_mean'] - vanilla['f1_mean']) * 100
    print(f"\nAgentRAG-Drop (Lazy Greedy) vs Vanilla RAG:")
    print(f"  F1: {f1_improvement:+.2f} points")

# Statistical significance
sig_count = sum(1 for t in tests.values() if t.get('significant_at_0.05', False))
print(f"\nStatistical Significance:")
print(f"  {sig_count}/{len(tests)} comparisons significant at p<0.05")

print("\n" + "="*80)
print("For detailed results, see:")
print(f"  ${OUTPUT_DIR}/summary.json")
print(f"  ${OUTPUT_DIR}/*_examples.jsonl")
print("="*80 + "\n")
EOF

echo -e "${GREEN}✓ Evaluation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. View detailed results: cat ${OUTPUT_DIR}/summary.json"
echo "  2. Run full evaluation: set LIMIT=0 in this script"
echo "  3. Try different budgets: --budget_tokens 400,600,800"
echo "  4. Generate plots: python scripts/plot_pareto.py ${OUTPUT_DIR}"
echo ""