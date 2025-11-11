import sys
sys.path.insert(0, '.')

from agentragdrop.pruning_formal import LazyGreedyPruner, RiskControlledPruner

# Test pruning with mock data
context = {
    "question": "Who directed Parasite?",
    "evidence": ["Parasite is a film.", "It won Best Picture."]
}

candidate_agents = ["retriever", "validator", "critic", "composer"]

# Test Lazy Greedy
print("="*60)
print("Testing Lazy Greedy Pruner")
print("="*60)

pruner = LazyGreedyPruner(lambda_redundancy=0.3)
pruned = pruner.decide(candidate_agents, context, budget_tokens=600)

print(f"Candidates: {candidate_agents}")
print(f"Pruned: {pruned}")
print(f"Should execute: {[a for a in candidate_agents if a not in pruned]}")
print()

# Test Risk Controlled
print("="*60)
print("Testing Risk Controlled Pruner")
print("="*60)

pruner = RiskControlledPruner(risk_budget_alpha=0.05)
pruned = pruner.decide(candidate_agents, context, budget_tokens=600)

print(f"Candidates: {candidate_agents}")
print(f"Pruned: {pruned}")
print(f"Should execute: {[a for a in candidate_agents if a not in pruned]}")
print()

# Check logs
if pruner.logs:
    print("Pruner logs:")
    for log in pruner.logs[-5:]:
        print(f"  {log}")
