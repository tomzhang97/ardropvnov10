#!/bin/bash
# scripts/quickstart_complete.sh
# Complete quick start script for AgentRAG-Drop

set -e  # Exit on error

echo "======================================================================"
echo "AgentRAG-Drop: Complete Quick Start"
echo "======================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DEVICE=0   # GPU device (use -1 for CPU)
LIMIT=5    # Use 5 examples for quick test
OUTPUT_DIR="results/quickstart"

echo -e "${BLUE}Configuration:${NC}"
echo "  Device: GPU $DEVICE"
echo "  Examples: $LIMIT"
echo "  Output: $OUTPUT_DIR"
echo ""

# Step 1: Check dependencies
echo -e "${BLUE}[1/6] Checking dependencies...${NC}"
python3 -c "
import sys
missing = []
try:
    import torch
except ImportError:
    missing.append('torch')
try:
    import transformers
except ImportError:
    missing.append('transformers')
try:
    import sentence_transformers
except ImportError:
    missing.append('sentence-transformers')
try:
    import numpy
except ImportError:
    missing.append('numpy')

if missing:
    print(f'Missing dependencies: {missing}')
    print('Install with: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('✓ All dependencies installed')
"

echo ""

# Step 2: Generate sample data
echo -e "${BLUE}[2/6] Generating sample data...${NC}"
if [ ! -f "data/hotpot_sample.json" ]; then
    python3 scripts/generate_sample_data.py --output data/hotpot_sample.json
    echo -e "${GREEN}✓ Sample data generated${NC}"
else
    echo "✓ Sample data already exists"
fi

echo ""

# Step 3: Test formal pruning module
echo -e "${BLUE}[3/6] Testing formal pruning module...${NC}"
python3 -c "
import sys
sys.path.insert(0, '.')
from agentragdrop.pruning_formal import test_submodularity, test_monotonicity

print('Running submodularity test...')
if test_submodularity():
    print('✓ Submodularity verified')
else:
    print('✗ Submodularity test failed')
    sys.exit(1)
"

echo ""

# Step 4: Run evaluation
echo -e "${BLUE}[4/6] Running evaluation...${NC}"
echo "This will evaluate:"
echo "  1. Vanilla RAG"
echo "  2. AgentRAG-Drop (no pruning)"
echo "  3. AgentRAG-Drop (lazy greedy)"
echo "  4. AgentRAG-Drop (risk-controlled)"
echo ""
echo "Estimated time: ~2 minutes for $LIMIT examples"
echo ""

python3 experiments/eval_complete_runnable.py \
    --data_path data/hotpot_sample.json \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE \
    --limit $LIMIT \
    --seed 42

echo ""

# Step 5: Display results
echo -e "${BLUE}[5/6] Processing results...${NC}"

python3 - <<EOF
import json
import os

# Load summary
summary_path = '${OUTPUT_DIR}/summary.json'
if not os.path.exists(summary_path):
    print('Error: Summary file not found')
    exit(1)

with open(summary_path) as f:
    data = json.load(f)

summary = data['summary']
tests = data.get('significance_tests', {})

print("\n" + "="*80)
print("QUICK START RESULTS")
print("="*80)
print(f"\nDataset: Sample HotpotQA (n=${LIMIT})")
print(f"Output: ${OUTPUT_DIR}")
print("\n" + "-"*80)
print(f"{'System':<35} {'F1 (%)':<12} {'Tokens':<10} {'Latency'}")
print("-"*80)

# Sort by F1
systems_sorted = sorted(summary.items(), key=lambda x: x[1]['f1_mean'], reverse=True)

for sys_name, stats in systems_sorted:
    f1 = stats['f1_mean'] * 100
    f1_std = stats['f1_std'] * 100
    tokens = stats['tokens_mean']
    lat_p95 = stats['latency_p95']
    
    print(f"{sys_name:<35} {f1:>5.1f}±{f1_std:<4.1f}  {tokens:>7.0f}    {lat_p95:>6.0f}ms")

print("="*80)

# Show key findings
if 'agentragdrop_lazy_greedy' in summary and 'vanilla_rag' in summary:
    lazy = summary['agentragdrop_lazy_greedy']
    vanilla = summary['vanilla_rag']
    
    f1_improvement = (lazy['f1_mean'] - vanilla['f1_mean']) * 100
    token_reduction = (1 - lazy['tokens_mean'] / vanilla['tokens_mean']) * 100 if vanilla['tokens_mean'] > 0 else 0
    
    print("\nKEY FINDINGS:")
    print("-"*80)
    print(f"AgentRAG-Drop (Lazy Greedy) vs Vanilla RAG:")
    print(f"  F1 improvement: {f1_improvement:+.1f} points")
    print(f"  Token reduction: {token_reduction:+.1f}%")

# Statistical significance
if tests:
    print(f"\nStatistical Tests: {len(tests)} comparisons")
    sig_count = sum(1 for t in tests.values() if t.get('significant_at_0.05', False))
    print(f"  Significant at p<0.05: {sig_count}/{len(tests)}")

print("\n" + "="*80)
print("Next Steps:")
print("  1. View detailed results: cat ${OUTPUT_DIR}/summary.json")
print("  2. Run full evaluation: use --limit 0")
print("  3. Try different budgets: --budget_tokens 400,600,800")
print("="*80 + "\n")
EOF

# Step 6: Verify outputs
echo -e "${BLUE}[6/6] Verifying outputs...${NC}"
if [ -f "${OUTPUT_DIR}/summary.json" ]; then
    echo -e "${GREEN}✓ Summary file created${NC}"
fi

if [ -f "${OUTPUT_DIR}/vanilla_rag_results.jsonl" ]; then
    echo -e "${GREEN}✓ Per-example results created${NC}"
fi

echo ""
echo -e "${GREEN}======================================================================"
echo "✓ Quick start completed successfully!"
echo "======================================================================${NC}"
echo ""
echo "Files created:"
echo "  - ${OUTPUT_DIR}/summary.json"
echo "  - ${OUTPUT_DIR}/vanilla_rag_results.jsonl"
echo "  - ${OUTPUT_DIR}/agentragdrop_*_results.jsonl"
echo ""
echo "To run full evaluation on real data:"
echo "  1. Download HotpotQA: python experiments/prepare_datasets.py --dataset hotpotqa"
echo "  2. Run evaluation: python experiments/eval_complete_runnable.py \\"
echo "       --data_path data/hotpotqa_eval.jsonl --limit 100"
echo ""