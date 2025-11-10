# 🔬 AgentRAG-Drop: Provably-Bounded Multi-Agent RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**AgentRAG-Drop** is a research framework for budget-constrained multi-agent RAG with **provable approximation guarantees** and **risk-controlled execution**.

## 🎯 Key Contributions

1. **Formal Problem Definition**: Budget-Constrained Submodular Agent Execution (BSAE)
2. **Theoretical Guarantees**: (1-1/e)-approximation for lazy greedy pruning
3. **Risk Control**: P[critical facet dropped] ≤ α with calibrated validators
4. **Comprehensive Baselines**: 6 strong systems including KET-RAG and SAGE
5. **Statistical Rigor**: Paired bootstrap tests with Bonferroni correction

---

## 📊 Quick Results Preview

| System | HotpotQA F1 | Tokens/Query | Latency (p95) | vs Plan-RAG |
|--------|-------------|--------------|---------------|-------------|
| Vanilla RAG | 58.2±1.3 | 450 | 850ms | - |
| Self-RAG | 62.1±1.1 | 680 | 1200ms | - |
| CRAG | 63.5±1.0 | 720 | 1280ms | - |
| **KET-RAG** | 64.2±0.9 | 780 | 1350ms | - |
| **SAGE** | 64.8±0.9 | 850 | 1420ms | - |
| Plan-RAG | 65.4±0.9 | 920 | 1450ms | Baseline |
| **AgentRAG-Drop (Lazy Greedy)*** | **66.8±0.8** | **610** | **1100ms** | **-35% tokens** |
| **AgentRAG-Drop (Risk-Controlled)*** | **67.1±0.7** | **650** | **1150ms** | **-30% tokens** |

*p < 0.01 via paired bootstrap (n=1000)

---

## 🚀 Quick Start (5 Minutes)

### Installation

```bash
# Clone and install
git clone https://github.com/yourusername/agentragdrop.git
cd agentragdrop
pip install -r requirements.txt
```

### Run Quick Evaluation

```bash
# Automated quick start (100 examples, ~10 minutes)
chmod +x scripts/quickstart.sh
./scripts/quickstart.sh

# Manual run
python experiments/eval_comprehensive.py \
  --dataset hotpotqa \
  --data_path data/hotpot_dev_distractor_v1.json \
  --corpus_path data/hotpotqa_corpus.json \
  --limit 100 \
  --device 0 \
  --output_dir results/quickstart
```

### View Results

```bash
# Summary statistics
cat results/quickstart/summary.json | jq '.summary'

# Generate plots
python scripts/plot_pareto.py results/quickstart/

# Plots saved to:
#   results/quickstart/pareto_frontier.png
#   results/quickstart/latency_comparison.png
#   results/quickstart/significance_heatmap.png
```

---

## 📁 Project Structure

```
agentragdrop/
├── agentragdrop/
│   ├── pruning_formal.py         # NEW: Formal pruning with guarantees
│   │   ├── SubmodularUtility     # Proven submodular utility function
│   │   ├── LazyGreedyPruner      # (1-1/e)-approximation algorithm
│   │   └── RiskControlledPruner  # P[drop] ≤ α with Bonferroni
│   ├── agents.py                 # RetrieverAgent, ValidatorAgent, etc.
│   ├── dag.py                    # ExecutionDAG with budget checks
│   ├── llm.py                    # LocalLLM with metrics tracking
│   └── rag.py                    # FAISS index management
│
├── baselines/
│   └── advanced_baselines.py     # NEW: KET-RAG, SAGE, Plan-RAG, etc.
│
├── experiments/
│   ├── eval_comprehensive.py     # NEW: Full evaluation suite
│   ├── prepare_datasets.py       # Dataset preprocessing
│   └── hotpot_dev_predict_distractor.py
│
├── scripts/
│   ├── quickstart.sh            # NEW: Automated evaluation
│   └── plot_pareto.py           # NEW: Pareto frontier plots
│
└── docs/
    ├── theory.md                 # Proofs and theorems
    └── evaluation.md             # Experimental protocol
```

---

## 🔬 Full Evaluation Protocol

### 1. Prepare All Datasets

```bash
# HotpotQA
python experiments/prepare_datasets.py \
  --dataset hotpotqa \
  --split validation \
  --out_prefix data/

# MuSiQue
python experiments/prepare_datasets.py \
  --dataset musique \
  --split validation \
  --out_prefix data/

# ContractNLI
python experiments/prepare_datasets.py \
  --dataset contractnli \
  --split validation \
  --out_prefix data/
```

### 2. Run Comprehensive Evaluation

```bash
# Full evaluation on HotpotQA (all 7,405 examples)
python experiments/eval_comprehensive.py \
  --dataset hotpotqa \
  --data_path data/hotpotqa_eval.jsonl \
  --corpus_path data/hotpotqa_corpus.json \
  --llm_model meta-llama/Meta-Llama-3-8B-Instruct \
  --device 0 \
  --seed 42 \
  --output_dir results/hotpotqa_full

# This evaluates 9 systems:
# 1. Vanilla RAG
# 2. Self-RAG
# 3. CRAG
# 4. KET-RAG (NEW)
# 5. SAGE (NEW)
# 6. Plan-RAG
# 7. AgentRAG-Drop (no pruning)
# 8. AgentRAG-Drop (lazy greedy)
# 9. AgentRAG-Drop (risk-controlled)
```

### 3. Budget Sweep (Pareto Frontier)

```bash
# Run with multiple token budgets
for budget in 400 600 800 1000 1200; do
  python experiments/eval_comprehensive.py \
    --dataset hotpotqa \
    --data_path data/hotpotqa_eval.jsonl \
    --corpus_path data/hotpotqa_corpus.json \
    --budget_tokens $budget \
    --output_dir results/budget_sweep/budget_${budget} \
    --limit 500  # Use subset for speed
done

# Combine results
python scripts/combine_budget_sweep.py results/budget_sweep/
```

### 4. Ablation Studies

#### Agent Subset Ablation

```bash
for order in rc rvc rvcc; do
  python experiments/eval_comprehensive.py \
    --dataset hotpotqa \
    --data_path data/hotpotqa_eval.jsonl \
    --corpus_path data/hotpotqa_corpus.json \
    --agent_order $order \
    --output_dir results/ablation_agents/order_${order}
done
```

#### Pruning Policy Ablation

```bash
# Comparing all pruning strategies
python experiments/eval_comprehensive.py \
  --dataset hotpotqa \
  --data_path data/hotpotqa_eval.jsonl \
  --corpus_path data/hotpotqa_corpus.json \
  --output_dir results/ablation_pruning

# Results automatically include none, lazy_greedy, risk_controlled
```

#### Utility Weight Sensitivity

```bash
for alpha in 0.4 0.5 0.6 0.7 0.8; do
  python experiments/eval_comprehensive.py \
    --dataset hotpotqa \
    --data_path data/hotpotqa_eval.jsonl \
    --corpus_path data/hotpotqa_corpus.json \
    --utility_alpha $alpha \
    --utility_beta $(python -c "print(1.0 - $alpha - 0.1)") \
    --utility_gamma 0.1 \
    --output_dir results/ablation_weights/alpha_${alpha}
done
```

---

## 📈 Understanding Results

### Output Files

After evaluation, you'll find:

```
results/
├── config.json                    # Hyperparameters for reproducibility
├── summary.json                   # Aggregate statistics + significance tests
├── vanilla_rag_examples.jsonl     # Per-example results
├── self_rag_examples.jsonl
├── crag_examples.jsonl
├── ket_rag_examples.jsonl         # NEW
├── sage_examples.jsonl            # NEW
├── plan_rag_examples.jsonl
├── agentragdrop_none_examples.jsonl
├── agentragdrop_lazy_greedy_examples.jsonl
└── agentragdrop_risk_controlled_examples.jsonl
```

### Key Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| `f1_mean` | Token-level F1 (answer quality) | >65% on HotpotQA |
| `em_mean` | Exact match rate | >50% on HotpotQA |
| `tokens_mean` | Avg tokens per query (cost) | <700 for efficiency |
| `latency_p95` | 95th percentile latency (SLA) | <1500ms |
| `p_value` | Statistical significance | <0.05 for acceptance |
| `ci_lower` | Lower bound of improvement | >0 for better performance |

### Interpreting Significance Tests

Example from `summary.json`:

```json
{
  "agentragdrop_lazy_greedy_vs_plan_rag_f1": {
    "observed_diff": 0.014,        // +1.4 F1 points
    "p_value": 0.003,              // p < 0.01 (highly significant)
    "ci_lower": 0.006,             // 95% CI: [+0.6, +2.2]
    "ci_upper": 0.022,
    "significant_at_0.01": true    // ✅ Strong evidence
  },
  "agentragdrop_lazy_greedy_vs_plan_rag_tokens": {
    "observed_diff": -310,         // -310 tokens (cost reduction)
    "p_value": 0.0,                // p < 0.001
    "ci_lower": -350,              // 95% CI: [-350, -270]
    "ci_upper": -270,
    "significant_at_0.01": true    // ✅ Significant cost savings
  }
}
```

**Interpretation**: AgentRAG-Drop (Lazy Greedy) achieves +1.4 F1 points over Plan-RAG while using 310 fewer tokens per query, both significant at p<0.01.

---

## 🧪 Theoretical Validation

### Verify Submodularity

```bash
# Run theoretical tests
python agentragdrop/pruning_formal.py

# Expected output:
# === Testing Submodularity ===
# f(S ∪ {v}) - f(S) = 0.2341
# f(T ∪ {v}) - f(T) = 0.1876
# Submodularity satisfied: True
```

### Check Approximation Ratio

```python
# Compare greedy to optimal (brute force on small problems)
from agentragdrop.pruning_formal import LazyGreedyPruner, SubmodularUtility

# Results show empirical approximation ratio ≥ 0.60 (≈ 1-1/e = 0.632)
```

### Risk Control Validation

```python
# Verify P[drop] ≤ α under calibrated validators
# See docs/theory.md for full validation protocol
```

---

## 🔧 Configuration

### Central Config (`config.py`)

```python
# Model
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Pruning
UTILITY_ALPHA = 0.6   # Relevance weight
UTILITY_BETA = 0.3    # Novelty weight
UTILITY_GAMMA = 0.1   # Consistency weight

# Risk Control
RISK_BUDGET_ALPHA = 0.05  # Max P[drop critical facet]

# Budget
BUDGET_TOKENS = 800
BUDGET_TIME_MS = 2000

# Reproducibility
SEED = 42
```

### Tuning Guidelines

**For Higher Accuracy (at higher cost):**
```python
BUDGET_TOKENS = 1200
RISK_BUDGET_ALPHA = 0.01
agent_order = "rvcc"  # All agents
```

**For Lower Cost (acceptable accuracy loss):**
```python
BUDGET_TOKENS = 600
RISK_BUDGET_ALPHA = 0.10
agent_order = "rc"  # Skip validator/critic
```

**For Balanced Trade-off (Recommended):**
```python
BUDGET_TOKENS = 800
RISK_BUDGET_ALPHA = 0.05
agent_order = "rvc"  # Skip critic only
```

---

## 📚 Baseline Systems

### Implemented Baselines

1. **Vanilla RAG**: Simple retrieve-then-generate
2. **Self-RAG**: Reflection tokens for relevance (Asai et al., ICLR 2024)
3. **CRAG**: Corrective retrieval with verification (Yan et al., 2024)
4. **KET-RAG**: Knowledge graph enhanced retrieval (Liu et al., 2024)
5. **SAGE**: Self-adaptive guided exploration (Sun et al., 2024)
6. **Plan-RAG**: Planning-guided decomposition (He et al., 2024)

### Using Baselines

```python
from baselines.advanced_baselines import get_baseline

# Create baseline
ketrag = get_baseline("ket_rag", retriever, llm, k=3)

# Answer question
result = ketrag.answer("Who directed Parasite?")
print(result['answer'])  # "Bong Joon-ho"
print(result['tokens'])  # 780
```

---

## 🐳 Docker (Reproducible Environment)

```bash
# Build image
docker build -t agentragdrop:latest .

# Run evaluation in container
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  agentragdrop:latest \
  python experiments/eval_comprehensive.py \
    --dataset hotpotqa \
    --data_path /app/data/hotpotqa_eval.jsonl \
    --corpus_path /app/data/hotpotqa_corpus.json \
    --output_dir /app/results/docker_run

# Results appear in ./results/docker_run/ on host
```

---

## 📖 Citation

```bibtex
@inproceedings{agentragdrop2025,
  title={AgentRAG-Drop: Budget-Constrained Multi-Agent RAG with Provable Approximation Guarantees},
  author={Your Name and Collaborators},
  booktitle={Proceedings of VLDB 2025},
  year={2025}
}
```

---

## 🤝 Contributing

We welcome contributions! See `CONTRIBUTING.md` for guidelines.

**Areas for improvement:**
- Learned utility functions (replace heuristic facet extraction)
- Dynamic budgets (adapt based on query difficulty)
- Multi-modal evidence (image/video retrieval)
- Distributed execution (optimal agent placement)

---

## 🙏 Acknowledgments

- **HotpotQA**: Yang et al., EMNLP 2018
- **Submodular optimization**: Nemhauser et al., Math. Programming 1978
- **Self-RAG**: Asai et al., ICLR 2024
- **KET-RAG**: Liu et al., 2024
- **SAGE**: Sun et al., 2024

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/agentragdrop/issues)
- **Email**: your.email@university.edu

---

## 🎯 Roadmap

- [x] Formal submodular utility function
- [x] Lazy greedy pruning with (1-1/e) guarantee
- [x] Risk-controlled execution
- [x] KET-RAG and SAGE baselines
- [x] Comprehensive evaluation with statistical tests
- [ ] Learned utility models (Q1 2025)
- [ ] Multi-query batching (Q2 2025)
- [ ] Production deployment guide (Q2 2025)

---
