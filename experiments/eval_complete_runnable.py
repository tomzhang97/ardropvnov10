#!/usr/bin/env python3
# experiments/eval_complete_runnable.py (SHARDED + FIXED)
"""
Complete runnable evaluation script for AgentRAG-Drop.

Modes:
- Single file (default):
    python eval_complete_runnable.py \
        --data_path hotpotQA/hotpot_dev_distractor_v1.json \
        --output_dir results/hotpotqa_full \
        --device 0

- Sharded multi-GPU master:
    python eval_complete_runnable.py \
        --shard_glob "runs/_shards/dev_*_*.json" \
        --output_dir results/hotpotqa_full \
        --gpus 0,1,2,3 \
        --concurrency 4

Internally, master launches worker processes with:
    --worker --data_path <shard.json> --device 0
and sets CUDA_VISIBLE_DEVICES to a single physical GPU for each worker.
"""

import argparse
import os
import sys
import json
import time
import random
import hashlib
import subprocess
import glob
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AgentRAGDrop components
try:
    from agentragdrop import ExecutionDAG, Node, get_llm, utils
    from agentragdrop.agents import RetrieverAgent, ValidatorAgent, CriticAgent, RAGComposerAgent
    from agentragdrop.rag import make_retriever
    from agentragdrop.pruning_formal import (
        LazyGreedyPruner,
        RiskControlledPruner,
        ExecutionCache
    )
    from agentragdrop.pruning import StaticPruner
except ImportError as e:
    print(f"Error importing agentragdrop: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Try importing baselines
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'baselines'))
    from advanced_baselines import get_baseline
    BASELINES_AVAILABLE = True
except ImportError:
    print("Warning: Advanced baselines not available, will use simple baselines only")
    BASELINES_AVAILABLE = False


# ============================================================================
# SIMPLE BASELINES (BUILT-IN)
# ============================================================================

class SimpleVanillaRAG:
    """Simple baseline if advanced baselines not available."""
    def __init__(self, retriever, llm, k=3):
        self.retriever = retriever
        self.llm = llm
        self.k = k
        # Reuse the Hotpot-specific composer for consistent prompts
        self.composer = RAGComposerAgent(retriever=None, llm=llm)

    def answer(self, question: str) -> Dict[str, Any]:
        t_start = time.perf_counter()

        if hasattr(self.retriever, "invoke"):
            docs = self.retriever.invoke(question)
        elif hasattr(self.retriever, "get_relevant_documents"):
            docs = self.retriever.get_relevant_documents(question)
        else:
            docs = self.retriever._get_relevant_documents(question)

        evidence = [getattr(d, "page_content", str(d)) for d in docs[: self.k]]

        composer_result = self.composer(question, evidence=evidence)
        answer = composer_result.get("answer", "unknown")
        raw_answer = composer_result.get("raw_answer", answer)

        latency_ms = (time.perf_counter() - t_start) * 1000

        return {
            "answer": answer,
            "raw_answer": raw_answer.strip(),
            "tokens": composer_result.get("tokens_est", 0),
            "latency_ms": latency_ms,
            "agents_executed": ["retriever", "composer"],
            "agents_pruned": [],
            "retrieved_context": evidence,
            "evidence": evidence
        }


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # Data
    data_path: str
    output_dir: str = "results/eval"

    # Model
    llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: int = 0  # single GPU id, -1 for CPU

    # Evaluation
    limit: int = 0  # 0 = all examples
    retrieval_k: int = 4

    # Pruning
    lambda_redundancy: float = 0.3
    risk_budget_alpha: float = 0.05
    budget_tokens: int = 0  # 0 = unlimited

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        self.run_id = hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:12]


# ============================================================================
# METRICS
# ============================================================================

def normalize_answer(s: str) -> str:
    """Normalize answer for fair comparison."""
    import string
    import re

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em(pred: str, gold: str) -> float:
    """Exact match."""
    return float(normalize_answer(pred) == normalize_answer(gold))


def compute_f1(pred: str, gold: str) -> float:
    """Token-level F1."""
    from collections import Counter

    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


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
        "significant_at_0.01": bool(p_value < 0.01)
    }


# ============================================================================
# EVALUATOR
# ============================================================================

class Evaluator:
    """Main evaluation orchestrator."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self._set_seed(config.seed)

        print("Initializing evaluation...")
        print(f"Run ID: {config.run_id}")

        # Load data
        print(f"Loading data from: {config.data_path}")
        self.dataset = self._load_data(config.data_path)

        if config.limit > 0:
            self.dataset = self.dataset[:config.limit]

        print(f"Dataset size: {len(self.dataset)} examples")

        # Build corpus from dataset for retrieval
        print("Building retrieval corpus...")
        self.corpus_path = self._build_corpus()

        # Initialize models
        print(f"Loading LLM: {config.llm_model}")
        self.llm = get_llm(model_name=config.llm_model, device=config.device)

        print(f"Loading retriever: {config.embed_model}")
        self.retriever = make_retriever(
            self.corpus_path,
            embed_model=config.embed_model,
            k=config.retrieval_k
        )

        print("Initialization complete!\n")

    def _set_seed(self, seed: int):
        """Set all random seeds."""
        random.seed(seed)
        np.random.seed(seed)

    def _load_data(self, path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON (.json) or JSONL (.jsonl/.ndjson)."""

        # JSONL / NDJSON: one JSON object per line
        if path.endswith(".jsonl") or path.endswith(".ndjson"):
            rows: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Failed to parse line {line_no} in {path} as JSONL: {e}"
                        ) from e
                    rows.append(obj)
            return rows

        # Regular JSON file (Hotpot dev, shards, etc.)
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse {path} as JSON. "
                    f"If this is JSONL, rename it to .jsonl. Original error: {e}"
                ) from e

        # Handle Hotpot-style {"data": [...]} or plain list
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "data" in data:
            return data["data"]
        else:
            raise ValueError(
                f"Unknown data format in {path}: expected list or dict with 'data' key."
            )

    def _build_corpus(self) -> str:
        """
        Build corpus for retrieval from dataset.

        Handles multiple formats:
        - HotpotQA: {context: [[title, [sent1, sent2, ...]], ...]}
        - Standard: {context: "text"} or {context: ["text1", "text2"]}
        """
        corpus_dir = os.path.join(self.config.output_dir, "corpus")
        os.makedirs(corpus_dir, exist_ok=True)

        corpus_path = os.path.join(corpus_dir, "corpus.json")

        if os.path.exists(corpus_path):
            # Verify corpus is not empty
            with open(corpus_path, 'r', encoding='utf-8') as f:
                existing_corpus = json.load(f)
            if existing_corpus and len(existing_corpus) > 0:
                print(f"Using existing corpus: {corpus_path} ({len(existing_corpus)} docs)")
                return corpus_path
            else:
                print("Existing corpus is empty, rebuilding...")

        # Extract context from dataset
        corpus_texts = set()  # Use set for deduplication

        for ex in tqdm(self.dataset, desc="Extracting corpus"):
            # Try different context field names
            context = ex.get("context", ex.get("contexts", ex.get("passages", [])))

            if isinstance(context, list):
                for ctx_item in context:
                    # HotpotQA format: [[title, [sent1, sent2, ...]], ...]
                    if isinstance(ctx_item, (list, tuple)) and len(ctx_item) >= 2:
                        title = str(ctx_item[0])
                        sentences = ctx_item[1]

                        if isinstance(sentences, list):
                            for sent in sentences:
                                if sent and isinstance(sent, str) and sent.strip():
                                    full_text = f"{title}. {sent.strip()}"
                                    corpus_texts.add(full_text)
                        elif isinstance(sentences, str) and sentences.strip():
                            full_text = f"{title}. {sentences.strip()}"
                            corpus_texts.add(full_text)

                    # Standard list format
                    elif isinstance(ctx_item, str) and ctx_item.strip():
                        corpus_texts.add(ctx_item.strip())

                    # Dict format
                    elif isinstance(ctx_item, dict):
                        text = ctx_item.get("text", ctx_item.get("content", ""))
                        if text and text.strip():
                            corpus_texts.add(text.strip())

            # String context
            elif isinstance(context, str) and context.strip():
                corpus_texts.add(context.strip())

        corpus_texts = list(corpus_texts)

        if not corpus_texts:
            raise ValueError(
                f"No corpus texts extracted from dataset!\n"
                f"Please check the format of your data file: {self.config.data_path}\n"
                f"Expected 'context' field with text or [[title, [sentences]], ...] format"
            )

        # Save corpus
        corpus_data = [{"text": t} for t in corpus_texts]
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)

        print(f"Built corpus: {len(corpus_data)} unique documents")
        return corpus_path

    def build_agentragdrop(self, pruning_policy: str):
        """Build AgentRAG-Drop pipeline."""
        cache = ExecutionCache()
        dag = ExecutionDAG(cache=cache)

        # Agents
        A_r = RetrieverAgent(
            self.corpus_path,
            embed_model=self.config.embed_model,
            top_k=self.config.retrieval_k
        )
        A_v = ValidatorAgent(self.llm)
        A_c = CriticAgent(self.llm)
        A_p = RAGComposerAgent(self.retriever, self.llm)

        # DAG
        dag.add(Node("retriever", lambda question, **_: A_r(question)))
        dag.add(Node("validator", lambda evidence=None, question=None, **_: A_v(question, evidence)))
        dag.add(Node("critic", lambda evidence=None, question=None, **_: A_c(question, evidence)))
        dag.add(Node("composer", lambda question, evidence=None, **_: A_p(question, evidence)))

        # Pruner
        if pruning_policy == "none":
            pruner = StaticPruner(keep_set=("retriever", "composer"))
        elif pruning_policy == "lazy_greedy":
            pruner = LazyGreedyPruner(lambda_redundancy=self.config.lambda_redundancy)
        elif pruning_policy == "risk_controlled":
            pruner = RiskControlledPruner(
                risk_budget_alpha=self.config.risk_budget_alpha,
                lambda_redundancy=self.config.lambda_redundancy
            )
        else:
            raise ValueError(f"Unknown pruning policy: {pruning_policy}")

        return dag, pruner

    def evaluate_system(self, system_name: str, system_fn) -> List[Dict[str, Any]]:
        """Evaluate a single system."""
        results = []

        print(f"\nEvaluating: {system_name}")

        for ex in tqdm(self.dataset, desc=system_name):
            question = ex.get("question", "")
            gold_answer = ex.get("answer", "")
            example_id = ex.get("_id") or ex.get("id")

            if not question or not gold_answer:
                continue

            try:
                output = system_fn(question)

                pred_answer = output.get("answer", "")
                raw_pred = output.get("raw_answer", pred_answer)
                em = compute_em(pred_answer, gold_answer)
                f1 = compute_f1(pred_answer, gold_answer)

                results.append({
                    "example_id": example_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "raw_pred_answer": raw_pred,
                    "em": em,
                    "f1": f1,
                    "tokens": output.get("tokens", 0),
                    "latency_ms": output.get("latency_ms", 0),
                    "agents_executed": output.get("agents_executed", []),
                    "agents_pruned": output.get("agents_pruned", []),
                    "retrieved_context": output.get("retrieved_context", [])
                })
            except Exception as e:
                print(f"\nError on question: {question[:50]}... - {e}")
                import traceback
                traceback.print_exc()
                continue

        return results

    def run_evaluation(self, only_vanilla: bool = False) -> Dict[str, Any]:
        """Run complete evaluation."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        all_results: Dict[str, List[Dict[str, Any]]] = {}

        print("\n" + "="*70)
        print("RUNNING EVALUATION")
        print("="*70)

        # 1. Vanilla RAG
        print("\n[1/4] Vanilla RAG")
        vanilla = SimpleVanillaRAG(self.retriever, self.llm, k=self.config.retrieval_k)
        vanilla_results = self.evaluate_system(
            "Vanilla RAG",
            lambda q: vanilla.answer(q)
        )
        if not vanilla_results:
            raise RuntimeError("Vanilla RAG produced no results. Check logs for errors.")
        all_results["vanilla_rag"] = vanilla_results

        # If we only want Vanilla, stop here
        if only_vanilla:
            print("\n[only_vanilla] Skipping AgentRAG-Drop variants.")
            print("\n" + "="*70)
            print("COMPUTING STATISTICS")
            print("="*70)

            summary = self._compute_statistics(all_results)
            sig_tests = {}  # no significance tests without comparators

            self._save_results(all_results, summary, sig_tests)
            self._print_summary(summary, sig_tests)

            return {
                "summary": summary,
                "significance_tests": sig_tests,
                "results": all_results,
            }

        # 2-4. AgentRAG-Drop variants
        for idx, pruning in enumerate(["none", "lazy_greedy", "risk_controlled"], start=2):
            print(f"\n[{idx}/4] AgentRAG-Drop ({pruning})")

            dag, pruner = self.build_agentragdrop(pruning)

            def system_fn(q):
                t_start = time.perf_counter()
                outputs = dag.run(
                    {"question": q},
                    pruner=pruner,
                    budget_tokens=self.config.budget_tokens or None
                )
                latency_ms = (time.perf_counter() - t_start) * 1000

                answer = outputs.get("composer", {}).get("answer", "")
                raw_answer = outputs.get("composer", {}).get("raw_answer", answer)
                tokens = sum(
                    o.get("tokens_est", 0)
                    for o in outputs.values()
                    if isinstance(o, dict)
                )

                executed = list(outputs.keys())
                pruned = []
                if pruner:
                    logs = pruner.export_logs()
                    for log in logs:
                        if log.get("decision") == "pruned":
                            node_name = log.get("agent") or log.get("node")
                            if node_name:
                                pruned.append(node_name)
                    pruner.reset_logs()

                return {
                    "answer": answer,
                    "raw_answer": raw_answer,
                    "tokens": tokens,
                    "latency_ms": latency_ms,
                    "agents_executed": executed,
                    "agents_pruned": pruned,
                    "retrieved_context": outputs.get("retriever", {}).get("evidence", [])
                }

            system_results = self.evaluate_system(
                f"AgentRAG-Drop ({pruning})",
                system_fn
            )
            if not system_results:
                raise RuntimeError(f"AgentRAG-Drop ({pruning}) produced no results. Check logs for errors.")
            all_results[f"agentragdrop_{pruning}"] = system_results

        # Compute statistics
        print("\n" + "="*70)
        print("COMPUTING STATISTICS")
        print("="*70)

        summary = self._compute_statistics(all_results)
        sig_tests = self._run_significance_tests(all_results)

        # Save results
        self._save_results(all_results, summary, sig_tests)

        # Print summary
        self._print_summary(summary, sig_tests)

        return {
            "summary": summary,
            "significance_tests": sig_tests,
            "results": all_results
        }

    def _compute_statistics(self, all_results: Dict) -> Dict:
        """Compute aggregate statistics."""
        summary: Dict[str, Dict[str, float]] = {}

        for system_name, results in all_results.items():
            if not results:
                continue

            em_scores = [r["em"] for r in results]
            f1_scores = [r["f1"] for r in results]
            tokens = [r["tokens"] for r in results]
            latencies = [r["latency_ms"] for r in results]

            summary[system_name] = {
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

    def _run_significance_tests(self, all_results: Dict) -> Dict:
        """Run paired bootstrap tests."""
        tests: Dict[str, Dict[str, float]] = {}

        # Compare our systems vs vanilla
        our_systems = ["agentragdrop_lazy_greedy", "agentragdrop_risk_controlled"]

        for our_sys in our_systems:
            if our_sys not in all_results:
                continue

            ours = all_results[our_sys]
            our_f1 = [r["f1"] for r in ours]
            our_tokens = [r["tokens"] for r in ours]

            if "vanilla_rag" in all_results:
                baseline = all_results["vanilla_rag"]
                baseline_f1 = [r["f1"] for r in baseline]
                baseline_tokens = [r["tokens"] for r in baseline]

                if len(our_f1) == len(baseline_f1):
                    # F1 test
                    f1_test = paired_bootstrap(our_f1, baseline_f1)
                    tests[f"{our_sys}_vs_vanilla_rag_f1"] = f1_test

                    # Token test
                    token_test = paired_bootstrap(our_tokens, baseline_tokens)
                    tests[f"{our_sys}_vs_vanilla_rag_tokens"] = token_test

        return tests

    def _save_results(self, all_results, summary, sig_tests):
        """Save all results to disk."""
        # Per-system results
        for system_name, results in all_results.items():
            output_file = os.path.join(
                self.config.output_dir,
                f"{system_name}_results.jsonl"
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

        # Summary
        summary_file = os.path.join(self.config.output_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "config": asdict(self.config),
                "summary": summary,
                "significance_tests": sig_tests,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

        print(f"\nResults saved to: {self.config.output_dir}")

    def _print_summary(self, summary, sig_tests):
        """Print summary table."""
        print("\n" + "="*90)
        print("EVALUATION SUMMARY")
        print("="*90)
        print(f"{'System':<30} {'EM':<12} {'F1':<12} {'Tokens':<12} {'Lat(p95)'}")
        print("-"*90)

        for sys_name in sorted(summary.keys()):
            stats = summary[sys_name]
            em_str = f"{stats['em_mean']*100:.1f}±{stats['em_std']*100:.1f}"
            f1_str = f"{stats['f1_mean']*100:.1f}±{stats['f1_std']*100:.1f}"
            tok_str = f"{stats['tokens_mean']:.0f}±{stats['tokens_std']:.0f}"
            lat_str = f"{stats['latency_p95']:.0f}ms"

            print(f"{sys_name:<30} {em_str:<12} {f1_str:<12} {tok_str:<12} {lat_str}")

        print("="*90)

        if sig_tests:
            print("\nSTATISTICAL SIGNIFICANCE (Paired Bootstrap, n=1000)")
            print("-"*90)
            for test_name, result in sig_tests.items():
                sig = "***" if result["significant_at_0.01"] else ("**" if result["significant_at_0.05"] else "")
                metric = "F1" if "_f1" in test_name else "Tokens"
                print(f"{test_name}: {metric} Δ={result['observed_diff']:+.3f} {sig} (p={result['p_value']:.4f})")
            print("="*90)


# ============================================================================
# SHARDED MASTER
# ============================================================================

def run_sharded_master(args: argparse.Namespace):
    """Launch one worker per shard, distributed across GPUs."""
    shards = sorted(glob.glob(args.shard_glob))
    if not shards:
        print(f"[ERROR] No shard files matched pattern: {args.shard_glob}")
        sys.exit(1)

    if args.gpus:
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    else:
        gpu_ids = ["0"]

    concurrency = max(1, min(args.concurrency, len(gpu_ids), len(shards)))

    print(f"Found {len(shards)} shard files.")
    print(f"GPUs: {gpu_ids} | concurrency: {concurrency}")
    print(f"Root output_dir: {args.output_dir}")

    indexed_shards = list(enumerate(shards))  # (idx, path)
    next_idx = 0
    active: List[Tuple[subprocess.Popen, int, str, str, str]] = []

    while next_idx < len(indexed_shards) or active:
        # Launch new workers while we have capacity
        while next_idx < len(indexed_shards) and len(active) < concurrency:
            shard_idx, shard_path = indexed_shards[next_idx]
            next_idx += 1

            gpu = gpu_ids[shard_idx % len(gpu_ids)]
            shard_name = os.path.basename(shard_path).rsplit(".", 1)[0]
            out_dir = os.path.join(args.output_dir, shard_name)
            os.makedirs(out_dir, exist_ok=True)

            env = os.environ.copy()
            # Inside worker we use --device 0; here we map that to a physical GPU id.
            env["CUDA_VISIBLE_DEVICES"] = gpu

            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--worker",
                "--data_path", shard_path,
                "--output_dir", out_dir,
                "--llm_model", args.llm_model,
                "--embed_model", args.embed_model,
                "--device", "0",
                "--retrieval_k", str(args.retrieval_k),
                "--lambda_redundancy", str(args.lambda_redundancy),
                "--risk_budget_alpha", str(args.risk_budget_alpha),
                "--budget_tokens", str(args.budget_tokens),
                "--seed", str(args.seed),
            ]

            if args.only_vanilla:
                cmd.append("--only_vanilla")

            print(f"[launch] shard {shard_idx+1}/{len(shards)} {shard_name} on GPU {gpu}")
            p = subprocess.Popen(cmd, env=env)
            active.append((p, shard_idx, shard_path, out_dir, gpu))

        if not active:
            break

        time.sleep(1.0)
        new_active: List[Tuple[subprocess.Popen, int, str, str, str]] = []
        for p, shard_idx, shard_path, out_dir, gpu in active:
            ret = p.poll()
            if ret is None:
                new_active.append((p, shard_idx, shard_path, out_dir, gpu))
            else:
                status = "OK" if ret == 0 else f"EXIT {ret}"
                shard_name = os.path.basename(shard_path)
                print(f"[done] shard {shard_idx+1}/{len(shards)} {shard_name} on GPU {gpu} -> {status}")
        active = new_active

    print("All shard workers completed.")
    print("Per-shard summaries are in:")
    print(f"  {args.output_dir}/<shard_name>/summary.json")
    print("If you want a global aggregate across shards, we can add an aggregator script later.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete runnable evaluation for AgentRAG-Drop (single or sharded)."
    )

    parser.add_argument(
        "--only_vanilla",
        action="store_true",
        help="Only run Vanilla RAG baseline (skip AgentRAG-Drop variants)",
    )

    # Single / worker mode
    parser.add_argument("--data_path", type=str,
                        help="Path to evaluation dataset (JSON/JSONL) or shard file")

    parser.add_argument("--output_dir", default="results/eval", help="Output directory")
    parser.add_argument("--llm_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=int, default=0, help="GPU device (-1 for CPU)")
    parser.add_argument("--limit", type=int, default=0, help="Limit examples (0=all)")
    parser.add_argument("--retrieval_k", type=int, default=4)
    parser.add_argument("--lambda_redundancy", type=float, default=0.3)
    parser.add_argument("--risk_budget_alpha", type=float, default=0.05)
    parser.add_argument("--budget_tokens", type=int, default=0, help="Token budget (0=unlimited)")
    parser.add_argument("--seed", type=int, default=42)

    # Sharded master mode
    parser.add_argument("--shard_glob", type=str, default="",
                        help="Glob for sharded dataset files, e.g. 'runs/_shards/dev_*_*.json'")
    parser.add_argument("--gpus", type=str, default="",
                        help="Comma-separated GPU ids for sharded master, e.g. '0,1,2,3'")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Max concurrent workers in sharded master mode")

    # Worker flag (internal)
    parser.add_argument("--worker", action="store_true",
                        help="Internal flag: run in worker mode on a single shard")

    args = parser.parse_args()

    # Worker mode: run on a single data_path (shard or full file)
    if args.worker:
        if not args.data_path:
            parser.error("--worker requires --data_path pointing to a single shard file")
        config = EvalConfig(
            data_path=args.data_path,
            output_dir=args.output_dir,
            llm_model=args.llm_model,
            embed_model=args.embed_model,
            device=args.device,
            limit=args.limit,
            retrieval_k=args.retrieval_k,
            lambda_redundancy=args.lambda_redundancy,
            risk_budget_alpha=args.risk_budget_alpha,
            budget_tokens=args.budget_tokens,
            seed=args.seed
        )

        print("="*70)
        print("AGENTRAG-DROP: EVALUATION (WORKER MODE)")
        print("="*70)
        print(f"Data (shard): {config.data_path}")
        print(f"Output dir  : {config.output_dir}")
        print(f"LLM         : {config.llm_model}")
        print(f"Device      : {'GPU ' + str(config.device) if config.device >= 0 else 'CPU'}")
        print(f"Seed        : {config.seed}")
        print("="*70 + "\n")

        evaluator = Evaluator(config)
        _ = evaluator.run_evaluation(only_vanilla=args.only_vanilla)
        return

    # Sharded master mode
    if args.shard_glob:
        # In master mode we don't need data_path; we just coordinate shards.
        print("="*70)
        print("AGENTRAG-DROP: SHARDED EVALUATION (MASTER MODE)")
        print("="*70)
        print(f"Shard glob  : {args.shard_glob}")
        print(f"Output root : {args.output_dir}")
        print(f"LLM         : {args.llm_model}")
        print(f"GPUs        : {args.gpus or '0'}")
        print(f"Concurrency : {args.concurrency}")
        print("="*70 + "\n")

        run_sharded_master(args)
        return

    # Single-file mode (original behavior)
    if not args.data_path:
        parser.error("Either --data_path (single/worker mode) or --shard_glob (sharded master mode) must be provided")

    config = EvalConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        device=args.device,
        limit=args.limit,
        retrieval_k=args.retrieval_k,
        lambda_redundancy=args.lambda_redundancy,
        risk_budget_alpha=args.risk_budget_alpha,
        budget_tokens=args.budget_tokens,
        seed=args.seed
    )

    print("="*70)
    print("AGENTRAG-DROP: COMPLETE EVALUATION (SINGLE FILE)")
    print("="*70)
    print(f"Data   : {config.data_path}")
    print(f"Output : {config.output_dir}")
    print(f"LLM    : {config.llm_model}")
    print(f"Device : {'GPU ' + str(config.device) if config.device >= 0 else 'CPU'}")
    print(f"Seed   : {config.seed}")
    print("="*70 + "\n")

    evaluator = Evaluator(config)
    _ = evaluator.run_evaluation(only_vanilla=args.only_vanilla)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"Results: {config.output_dir}")
    print(f"Run ID : {config.run_id}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
