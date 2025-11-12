# experiments/eval_comprehensive.py
"""
Comprehensive evaluation with all baselines including KET-RAG and SAGE.

This script runs:
1. Vanilla RAG
2. Self-RAG
3. CRAG
4. KET-RAG (NEW)
5. SAGE (NEW)
6. Plan-RAG
7. AgentRAG-Drop (no pruning)
8. AgentRAG-Drop (lazy greedy)
9. AgentRAG-Drop (risk-controlled)

With statistical significance testing, Pareto frontiers, and ablations.
"""

import argparse
import os
import sys
import json
import time
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

# Import AgentRAGDrop components
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agentragdrop import ExecutionDAG, Node, RetrieverAgent, get_llm, utils
from agentragdrop.agents import ValidatorAgent, CriticAgent, RAGComposerAgent
from agentragdrop.rag import make_retriever
from agentragdrop.pruning_formal import (
    LazyGreedyPruner, RiskControlledPruner, ExecutionCache
)

# Import baselines
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'baselines'))
from advanced_baselines import get_baseline


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Central configuration for reproducibility."""
    # Dataset
    dataset: str
    data_path: str
    corpus_path: str
    
    # Model
    llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: int = 0
    
    # Execution
    agent_order: str = "rvcc"
    retrieval_k: int = 6
    
    # Pruning
    utility_alpha: float = 0.6
    utility_beta: float = 0.3
    utility_gamma: float = 0.1
    risk_budget_alpha: float = 0.05
    
    # Budget
    budget_tokens: int = 0
    budget_time_ms: int = 0
    
    # Evaluation
    limit: int = 0  # 0 = all examples
    
    # Reproducibility
    seed: int = 42
    run_id: str = None
    
    def __post_init__(self):
        if self.run_id is None:
            self.run_id = hashlib.sha256(
                json.dumps(asdict(self), sort_keys=True).encode()
            ).hexdigest()[:12]


# ============================================================================
# METRICS & EVALUATION
# ============================================================================

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    import string
    import re
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em(pred: str, gold: str) -> float:
    """Exact match score."""
    return float(normalize_answer(pred) == normalize_answer(gold))


def compute_f1(pred: str, gold: str) -> float:
    """Token-level F1 score."""
    from collections import Counter
    
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


@dataclass
class ExampleResult:
    """Results for a single example."""
    example_id: str
    question: str
    gold_answer: str
    pred_answer: str
    
    em: float
    f1: float
    
    tokens_used: int
    latency_ms: float
    
    agents_executed: List[str]
    agents_pruned: List[str]
    
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def paired_bootstrap(
    metric1: List[float],
    metric2: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Paired bootstrap test for significance.
    
    Returns:
        Dictionary with p_value, confidence intervals, significance flags
    """
    assert len(metric1) == len(metric2), "Metrics must have same length"
    
    n = len(metric1)
    diffs = np.array(metric1) - np.array(metric2)
    observed_diff = np.mean(diffs)
    
    # Bootstrap resampling
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_diffs.append(np.mean(diffs[idx]))
    
    boot_diffs = np.array(boot_diffs)
    
    # Two-tailed p-value
    p_value = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))
    
    # Confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    
    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant_at_0.05": bool(p_value < 0.05),
        "significant_at_0.01": bool(p_value < 0.01)
    }


# ============================================================================
# MAIN EVALUATION ENGINE
# ============================================================================

class ComprehensiveEvaluator:
    """Main evaluation orchestrator."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._set_seed(config.seed)
        
        # Initialize models
        print(f"Loading LLM: {config.llm_model}...")
        self.llm = get_llm(model_name=config.llm_model, device=config.device)
        
        print(f"Loading retriever: {config.embed_model}...")
        self.retriever = make_retriever(
            config.corpus_path, 
            embed_model=config.embed_model,
            k=config.retrieval_k
        )
        
        # Load dataset
        print(f"Loading dataset: {config.data_path}...")
        self.dataset = utils.load_json_or_jsonl(config.data_path)
        if config.limit > 0:
            self.dataset = self.dataset[:config.limit]
        
        print(f"Evaluation ready: {len(self.dataset)} examples")
    
    def _set_seed(self, seed: int):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
    
    def build_agentragdrop(
        self, 
        pruning_policy: str
    ) -> Tuple[ExecutionDAG, Any]:
        """Build AgentRAG-Drop pipeline with specified pruning."""
        cache = ExecutionCache()
        dag = ExecutionDAG(cache=cache)
        
        # Agents
        A_r = RetrieverAgent(
            self.config.corpus_path,
            embed_model=self.config.embed_model,
            top_k=self.config.retrieval_k
        )
        A_v = ValidatorAgent(self.llm)
        A_c = CriticAgent(self.llm)
        A_p = RAGComposerAgent(self.retriever, self.llm)
        
        # DAG construction
        plan_map = {
            "retriever": lambda question, **_: A_r(question),
            "validator": lambda evidence=None, question=None, **_: A_v(question, evidence),
            "critic": lambda evidence=None, question=None, **_: A_c(question, evidence),
            "composer": lambda question, evidence=None, **_: A_p(question, evidence)
        }
        
        order_map = {
            "rvcc": ["retriever", "validator", "critic", "composer"],
            "rvc": ["retriever", "validator", "composer"],
            "rc": ["retriever", "composer"]
        }
        
        for name in order_map[self.config.agent_order]:
            dag.add(Node(name, plan_map[name]))
        
        # Pruner
        if pruning_policy == "none":
            pruner = None
        elif pruning_policy == "lazy_greedy":
            pruner = LazyGreedyPruner(
                lambda_redundancy=self.config.utility_beta
            )
        elif pruning_policy == "risk_controlled":
            pruner = RiskControlledPruner(
                risk_budget_alpha=self.config.risk_budget_alpha,
                lambda_redundancy=self.config.utility_beta
            )
        else:
            raise ValueError(f"Unknown pruning policy: {pruning_policy}")
        
        return dag, pruner
    
    def evaluate_system(
        self,
        system_name: str,
        system_fn: Any
    ) -> List[ExampleResult]:
        """Evaluate a single system."""
        results = []
        
        print(f"\nEvaluating {system_name} on {len(self.dataset)} examples...")
        
        for ex in tqdm(self.dataset, desc=system_name):
            question = ex.get("question", "")
            gold_answer = ex.get("answer", "")
            ex_id = ex.get("_id", ex.get("id", str(hash(question))))
            
            if not question or not gold_answer:
                continue
            
            try:
                # Run system
                output = system_fn(question)
                
                # Compute metrics
                pred_answer = output.get("answer", "")
                em = compute_em(pred_answer, gold_answer)
                f1 = compute_f1(pred_answer, gold_answer)
                
                result = ExampleResult(
                    example_id=ex_id,
                    question=question,
                    gold_answer=gold_answer,
                    pred_answer=pred_answer,
                    em=em,
                    f1=f1,
                    tokens_used=output.get("tokens", 0),
                    latency_ms=output.get("latency_ms", 0),
                    agents_executed=output.get("agents_executed", []),
                    agents_pruned=output.get("agents_pruned", []),
                    cache_hit=output.get("cache_hit", False)
                )
                
                results.append(result)
            except Exception as e:
                print(f"Error on example {ex_id}: {e}")
                continue
        
        return results
    
    def run_full_evaluation(
        self,
        output_dir: str = "results/comprehensive"
    ) -> Dict[str, Any]:
        """Run complete evaluation suite."""
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        
        # Save config
        config_file = os.path.join(output_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        
        print("\n" + "="*70)
        print("EVALUATING ALL SYSTEMS")
        print("="*70)
        print(f"Config saved: {config_file}")
        print(f"Run ID: {self.config.run_id}")
        print("="*70)
        
        # 1. Vanilla RAG
        print("\n[1/9] Vanilla RAG")
        vanilla = get_baseline("vanilla_rag", self.retriever, self.llm, k=self.config.retrieval_k)
        all_results["vanilla_rag"] = self.evaluate_system(
            "Vanilla RAG",
            lambda q: vanilla.answer(q)
        )
        
        # 2. Self-RAG
        print("\n[2/9] Self-RAG")
        selfrag = get_baseline("self_rag", self.retriever, self.llm, k=self.config.retrieval_k)
        all_results["self_rag"] = self.evaluate_system(
            "Self-RAG",
            lambda q: selfrag.answer(q)
        )
        
        # 3. CRAG
        print("\n[3/9] CRAG")
        crag = get_baseline("crag", self.retriever, self.llm, k=self.config.retrieval_k)
        all_results["crag"] = self.evaluate_system(
            "CRAG",
            lambda q: crag.answer(q)
        )
        
        # 4. KET-RAG (NEW)
        print("\n[4/9] KET-RAG")
        ketrag = get_baseline("ket_rag", self.retriever, self.llm, k=self.config.retrieval_k)
        all_results["ket_rag"] = self.evaluate_system(
            "KET-RAG",
            lambda q: ketrag.answer(q)
        )
        
        # 5. SAGE (NEW)
        print("\n[5/9] SAGE")
        sage = get_baseline("sage", self.retriever, self.llm, k=self.config.retrieval_k, max_hops=2)
        all_results["sage"] = self.evaluate_system(
            "SAGE",
            lambda q: sage.answer(q)
        )
        
        # 6. Plan-RAG
        print("\n[6/9] Plan-RAG")
        planrag = get_baseline("plan_rag", self.retriever, self.llm, k=self.config.retrieval_k)
        all_results["plan_rag"] = self.evaluate_system(
            "Plan-RAG",
            lambda q: planrag.answer(q)
        )
        
        # 7-9. AgentRAG-Drop variants
        for idx, pruning in enumerate(["none", "lazy_greedy", "risk_controlled"], start=7):
            print(f"\n[{idx}/9] AgentRAG-Drop ({pruning})")
            dag, pruner = self.build_agentragdrop(pruning)
            
            def system_fn(q):
                t_start = time.perf_counter()
                outputs = dag.run(
                    {"question": q},
                    pruner=pruner,
                    budget_tokens=self.config.budget_tokens or None,
                    budget_time_ms=self.config.budget_time_ms or None
                )
                latency_ms = (time.perf_counter() - t_start) * 1000
                
                answer = outputs.get("composer", {}).get("answer", "")
                tokens = sum(
                    o.get("tokens_est", 0) 
                    for o in outputs.values() 
                    if isinstance(o, dict)
                )
                
                executed = list(outputs.keys())
                pruned = []
                if pruner:
                    logs = pruner.export_logs()
                    pruned = [
                        log["agent"] 
                        for log in logs 
                        if log.get("decision") == "pruned"
                    ]
                    pruner.reset_logs()
                
                return {
                    "answer": answer,
                    "tokens": tokens,
                    "latency_ms": latency_ms,
                    "agents_executed": executed,
                    "agents_pruned": pruned,
                    "cache_hit": False
                }
            
            all_results[f"agentragdrop_{pruning}"] = self.evaluate_system(
                f"AgentRAG-Drop ({pruning})",
                system_fn
            )
        
        # Compute statistics
        print("\n" + "="*70)
        print("COMPUTING STATISTICS")
        print("="*70)
        
        summary = self._compute_summary_statistics(all_results)
        
        # Statistical significance tests
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*70)
        
        sig_tests = self._run_significance_tests(all_results)
        
        # Save results
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)
        
        # Per-example results
        for system_name, results in all_results.items():
            output_file = os.path.join(output_dir, f"{system_name}_examples.jsonl")
            with open(output_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r.to_dict()) + "\n")
            print(f"Saved: {output_file}")
        
        # Summary
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, "w") as f:
            json.dump({
                "config": asdict(self.config),
                "summary": summary,
                "significance_tests": sig_tests,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        print(f"Saved: {summary_file}")
        
        # Print summary table
        self._print_summary_table(summary)
        self._print_significance_tests(sig_tests)
        
        return {
            "summary": summary,
            "significance_tests": sig_tests,
            "per_example_results": all_results
        }
    
    def _compute_summary_statistics(
        self,
        all_results: Dict[str, List[ExampleResult]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute mean/std for all metrics."""
        summary = {}
        
        for system_name, results in all_results.items():
            em_scores = [r.em for r in results]
            f1_scores = [r.f1 for r in results]
            tokens = [r.tokens_used for r in results]
            latencies = [r.latency_ms for r in results]
            
            summary[system_name] = {
                "n_examples": len(results),
                "em_mean": float(np.mean(em_scores)),
                "em_std": float(np.std(em_scores)),
                "f1_mean": float(np.mean(f1_scores)),
                "f1_std": float(np.std(f1_scores)),
                "tokens_mean": float(np.mean(tokens)),
                "tokens_std": float(np.std(tokens)),
                "latency_mean": float(np.mean(latencies)),
                "latency_std": float(np.std(latencies)),
                "latency_p50": float(np.percentile(latencies, 50)),
                "latency_p95": float(np.percentile(latencies, 95)),
                "latency_p99": float(np.percentile(latencies, 99)),
                "throughput": float(len(results) / (sum(latencies) / 1000)) if sum(latencies) > 0 else 0
            }
        
        return summary
    
    def _run_significance_tests(
        self,
        all_results: Dict[str, List[ExampleResult]]
    ) -> Dict[str, Any]:
        """Run paired bootstrap tests."""
        tests = {}
        
        # Our system names
        our_systems = ["agentragdrop_lazy_greedy", "agentragdrop_risk_controlled"]
        
        # Baseline systems
        baseline_systems = ["vanilla_rag", "self_rag", "crag", "ket_rag", "sage", "plan_rag"]
        
        for our_system in our_systems:
            if our_system not in all_results:
                continue
            
            ours = all_results[our_system]
            our_f1 = [r.f1 for r in ours]
            our_tokens = [r.tokens_used for r in ours]
            
            for baseline_name in baseline_systems:
                if baseline_name not in all_results:
                    continue
                
                baseline = all_results[baseline_name]
                baseline_f1 = [r.f1 for r in baseline]
                baseline_tokens = [r.tokens_used for r in baseline]
                
                # Match lengths (should be same)
                if len(our_f1) == len(baseline_f1):
                    # F1 test
                    f1_test = paired_bootstrap(our_f1, baseline_f1, n_bootstrap=1000)
                    tests[f"{our_system}_vs_{baseline_name}_f1"] = f1_test
                    
                    # Token cost test
                    token_test = paired_bootstrap(our_tokens, baseline_tokens, n_bootstrap=1000)
                    tests[f"{our_system}_vs_{baseline_name}_tokens"] = token_test
        
        return tests
    
    def _print_summary_table(self, summary: Dict[str, Dict[str, float]]):
        """Print nice summary table."""
        print("\n" + "="*110)
        print("EVALUATION SUMMARY")
        print("="*110)
        print(f"{'System':<30} {'EM':<12} {'F1':<12} {'Tokens':<15} {'Lat(p50/p95)':<20} {'Throughput':<12}")
        print("-"*110)
        
        for system_name, stats in sorted(summary.items()):
            em_str = f"{stats['em_mean']*100:.2f}±{stats['em_std']*100:.2f}"
            f1_str = f"{stats['f1_mean']*100:.2f}±{stats['f1_std']*100:.2f}"
            tokens_str = f"{stats['tokens_mean']:.0f}±{stats['tokens_std']:.0f}"
            latency_str = f"{stats['latency_p50']:.0f}/{stats['latency_p95']:.0f}ms"
            throughput_str = f"{stats['throughput']:.2f} ex/s"
            
            print(f"{system_name:<30} {em_str:<12} {f1_str:<12} {tokens_str:<15} {latency_str:<20} {throughput_str:<12}")
        
        print("="*110)
    
    def _print_significance_tests(self, tests: Dict[str, Any]):
        """Print significance test results."""
        print("\n" + "="*110)
        print("STATISTICAL SIGNIFICANCE TESTS (Paired Bootstrap, n=1000)")
        print("="*110)
        
        for test_name, result in sorted(tests.items()):
            parts = test_name.split("_vs_")
            if len(parts) == 2:
                our_system, rest = parts
                baseline_metric = rest.rsplit("_", 1)
                if len(baseline_metric) == 2:
                    baseline, metric = baseline_metric
                    
                    sig_marker = ""
                    if result["significant_at_0.01"]:
                        sig_marker = "***"
                    elif result["significant_at_0.05"]:
                        sig_marker = "**"
                    
                    print(f"\n{our_system} vs {baseline} ({metric.upper()}):")
                    print(f"  Difference: {result['observed_diff']:+.4f} {sig_marker}")
                    print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
                    print(f"  P-value: {result['p_value']:.4f}")
        
        print("\n" + "="*110)
        print("Significance: *** p<0.01, ** p<0.05")
        print("="*110)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation with KET-RAG and SAGE"
    )
    
    # Dataset
    parser.add_argument("--dataset", required=True, 
                       choices=["hotpotqa", "musique", "contractnli"])
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    
    # Model
    parser.add_argument("--llm_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=int, default=0)
    
    # Execution
    parser.add_argument("--agent_order", default="rvcc")
    parser.add_argument("--retrieval_k", type=int, default=6)
    
    # Pruning
    parser.add_argument("--utility_alpha", type=float, default=0.6)
    parser.add_argument("--utility_beta", type=float, default=0.3)
    parser.add_argument("--utility_gamma", type=float, default=0.1)
    parser.add_argument("--risk_budget_alpha", type=float, default=0.05)
    
    # Budget
    parser.add_argument("--budget_tokens", type=int, default=0)
    parser.add_argument("--budget_time_ms", type=int, default=0)
    
    # Evaluation
    parser.add_argument("--limit", type=int, default=0, help="Limit examples (0=all)")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/comprehensive")
    
    args = parser.parse_args()
    
    # Build config
    config = ExperimentConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        corpus_path=args.corpus_path,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        device=args.device,
        agent_order=args.agent_order,
        retrieval_k=args.retrieval_k,
        utility_alpha=args.utility_alpha,
        utility_beta=args.utility_beta,
        utility_gamma=args.utility_gamma,
        risk_budget_alpha=args.risk_budget_alpha,
        budget_tokens=args.budget_tokens,
        budget_time_ms=args.budget_time_ms,
        limit=args.limit,
        seed=args.seed
    )
    
    print("="*70)
    print("AGENTRAG-DROP: COMPREHENSIVE EVALUATION")
    print("="*70)
    print(f"Dataset: {config.dataset}")
    print(f"LLM: {config.llm_model}")
    print(f"Seed: {config.seed}")
    print(f"Run ID: {config.run_id}")
    print("="*70)
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(config)
    results = evaluator.run_full_evaluation(output_dir=args.output_dir)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"Results saved to: {args.output_dir}")
    print(f"Run ID: {config.run_id}")


if __name__ == "__main__":
    main()