# scripts/plot_pareto.py
"""
Generate Pareto frontier plots for cost-quality tradeoffs.

Usage:
    python scripts/plot_pareto.py results/comprehensive/
"""

import argparse
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_summary(results_dir):
    """Load summary.json from results directory."""
    summary_path = Path(results_dir) / "summary.json"
    with open(summary_path) as f:
        data = json.load(f)
    return data['summary'], data.get('significance_tests', {})


def plot_pareto_frontier(summary, output_path, show_ci=True):
    """
    Plot Pareto frontier: F1 vs Tokens.
    
    Args:
        summary: Summary dictionary
        output_path: Where to save plot
        show_ci: Whether to show confidence intervals
    """
    # System categories
    baselines = ["vanilla_rag", "self_rag", "crag", "ket_rag", "sage", "plan_rag"]
    ours = ["agentragdrop_none", "agentragdrop_lazy_greedy", "agentragdrop_risk_controlled"]
    
    # Colors
    color_map = {
        "vanilla_rag": "#888888",
        "self_rag": "#888888",
        "crag": "#888888",
        "ket_rag": "#888888",
        "sage": "#888888",
        "plan_rag": "#d62728",  # Red for Plan-RAG (main competitor)
        "agentragdrop_none": "#9467bd",  # Purple
        "agentragdrop_lazy_greedy": "#2ca02c",  # Green (ours)
        "agentragdrop_risk_controlled": "#1f77b4",  # Blue (ours)
    }
    
    # Marker styles
    marker_map = {
        "vanilla_rag": "o",
        "self_rag": "s",
        "crag": "^",
        "ket_rag": "D",
        "sage": "v",
        "plan_rag": "X",
        "agentragdrop_none": "p",
        "agentragdrop_lazy_greedy": "*",
        "agentragdrop_risk_controlled": "P",
    }
    
    # Labels
    label_map = {
        "vanilla_rag": "Vanilla RAG",
        "self_rag": "Self-RAG",
        "crag": "CRAG",
        "ket_rag": "KET-RAG",
        "sage": "SAGE",
        "plan_rag": "Plan-RAG",
        "agentragdrop_none": "AgentRAG-Drop (no pruning)",
        "agentragdrop_lazy_greedy": "AgentRAG-Drop (Lazy Greedy)",
        "agentragdrop_risk_controlled": "AgentRAG-Drop (Risk-Controlled)",
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot systems
    for sys_name, stats in summary.items():
        if sys_name not in color_map:
            continue
        
        tokens = stats['tokens_mean']
        f1 = stats['f1_mean'] * 100
        
        color = color_map[sys_name]
        marker = marker_map[sys_name]
        label = label_map[sys_name]
        
        # Marker size
        if "agentragdrop" in sys_name:
            markersize = 200 if "lazy_greedy" in sys_name else 150
        else:
            markersize = 100
        
        # Plot point
        ax.scatter(tokens, f1, s=markersize, color=color, marker=marker, 
                  label=label, alpha=0.8, edgecolors='black', linewidth=1.5)
        
        # Error bars (confidence intervals)
        if show_ci:
            f1_std = stats['f1_std'] * 100
            tokens_std = stats['tokens_std']
            ax.errorbar(tokens, f1, yerr=f1_std, xerr=tokens_std, 
                       fmt='none', color=color, alpha=0.3, capsize=3)
    
    # Styling
    ax.set_xlabel('Average Tokens per Query', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Cost-Quality Pareto Frontier\n(Higher F1, Lower Tokens is Better)', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # Add annotations for our best systems
    for sys_name in ["agentragdrop_lazy_greedy", "agentragdrop_risk_controlled"]:
        if sys_name in summary:
            stats = summary[sys_name]
            tokens = stats['tokens_mean']
            f1 = stats['f1_mean'] * 100
            
            ax.annotate(
                f"F1={f1:.1f}%\nTokens={tokens:.0f}",
                xy=(tokens, f1),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5)
            )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Pareto plot: {output_path}")
    
    return fig, ax


def plot_latency_comparison(summary, output_path):
    """
    Bar plot comparing latency distributions.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    systems = ["vanilla_rag", "self_rag", "crag", "ket_rag", "sage", "plan_rag",
               "agentragdrop_none", "agentragdrop_lazy_greedy", "agentragdrop_risk_controlled"]
    
    labels = [s.replace("_", " ").replace("agentragdrop", "AgentRAG-Drop").title() for s in systems]
    
    # Extract latencies
    p50 = [summary[s]['latency_p50'] for s in systems if s in summary]
    p95 = [summary[s]['latency_p95'] for s in systems if s in summary]
    p99 = [summary[s]['latency_p99'] for s in systems if s in summary]
    
    x = np.arange(len(labels))
    width = 0.25
    
    ax.bar(x - width, p50, width, label='P50', color='#2ca02c', alpha=0.8)
    ax.bar(x, p95, width, label='P95', color='#ff7f0e', alpha=0.8)
    ax.bar(x + width, p99, width, label='P99', color='#d62728', alpha=0.8)
    
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Distribution Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved latency plot: {output_path}")
    
    return fig, ax


def plot_significance_heatmap(tests, output_path):
    """
    Heatmap showing statistical significance of comparisons.
    """
    # Parse test names
    our_systems = set()
    baselines = set()
    
    for test_name in tests.keys():
        if "_vs_" in test_name and "_f1" in test_name:
            parts = test_name.split("_vs_")
            if len(parts) == 2:
                our_sys = parts[0]
                baseline = parts[1].replace("_f1", "")
                our_systems.add(our_sys)
                baselines.add(baseline)
    
    our_systems = sorted(our_systems)
    baselines = sorted(baselines)
    
    if not our_systems or not baselines:
        print("No significance tests found")
        return None, None
    
    # Build matrix
    matrix = np.zeros((len(our_systems), len(baselines)))
    
    for i, our_sys in enumerate(our_systems):
        for j, baseline in enumerate(baselines):
            test_key = f"{our_sys}_vs_{baseline}_f1"
            if test_key in tests:
                p_value = tests[test_key]['p_value']
                # Map p-value to significance level
                if p_value < 0.01:
                    matrix[i, j] = 3  # ***
                elif p_value < 0.05:
                    matrix[i, j] = 2  # **
                elif p_value < 0.1:
                    matrix[i, j] = 1  # *
                else:
                    matrix[i, j] = 0  # n.s.
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=3, aspect='auto')
    
    # Labels
    ax.set_xticks(np.arange(len(baselines)))
    ax.set_yticks(np.arange(len(our_systems)))
    ax.set_xticklabels([b.replace("_", " ").title() for b in baselines], rotation=45, ha='right')
    ax.set_yticklabels([o.replace("_", " ").replace("agentragdrop", "AgentRAG-Drop").title() for o in our_systems])
    
    # Annotate cells with significance markers
    for i in range(len(our_systems)):
        for j in range(len(baselines)):
            val = matrix[i, j]
            if val == 3:
                text = "***"
            elif val == 2:
                text = "**"
            elif val == 1:
                text = "*"
            else:
                text = "n.s."
            
            ax.text(j, i, text, ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Statistical Significance of F1 Improvements\n(Paired Bootstrap, n=1000)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.set_label('Significance Level', rotation=270, labelpad=20, fontweight='bold')
    cbar.ax.set_yticklabels(['n.s.', '* (p<0.1)', '** (p<0.05)', '*** (p<0.01)'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved significance heatmap: {output_path}")
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description="Generate Pareto frontier plots")
    parser.add_argument("results_dir", help="Directory containing summary.json")
    parser.add_argument("--output_dir", default=None, help="Output directory for plots (default: same as results_dir)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Output format")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading results from: {args.results_dir}")
    summary, tests = load_summary(args.results_dir)
    
    # Output directory
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # 1. Pareto frontier
    pareto_path = os.path.join(output_dir, f"pareto_frontier.{args.format}")
    plot_pareto_frontier(summary, pareto_path, show_ci=True)
    
    # 2. Latency comparison
    latency_path = os.path.join(output_dir, f"latency_comparison.{args.format}")
    plot_latency_comparison(summary, latency_path)
    
    # 3. Significance heatmap
    if tests:
        significance_path = os.path.join(output_dir, f"significance_heatmap.{args.format}")
        plot_significance_heatmap(tests, significance_path)
    
    print("\n" + "="*70)
    print("All plots generated successfully!")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()