# baselines/advanced_baselines.py
"""
Advanced baseline implementations for comparison.

Includes:
1. VanillaRAG: Simple retrieve-then-generate
2. SelfRAG: Reflection tokens for relevance
3. CRAG: Corrective RAG with web search fallback
4. KET-RAG: Knowledge graph enhanced retrieval
5. SAGE: Self-adaptive guided exploration
6. PlanRAG: Planning-guided decomposition

All baselines use the same LLM and retrieval index for fair comparison.
"""

import time
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict

from agentragdrop.answer_cleaning import clean_answer


# ============================================================================
# VANILLA RAG (Simple Baseline)
# ============================================================================

class VanillaRAG:
    """
    Simple retrieve-then-generate baseline.
    
    Pipeline:
    1. Retrieve top-k documents
    2. Generate answer from retrieved context
    
    No reflection, no verification, no planning.
    """
    
    def __init__(self, retriever, llm, k: int = 3):
        """
        Initialize Vanilla RAG.
        
        Args:
            retriever: LangChain retriever (FAISS-based)
            llm: LocalLLM instance
            k: Number of documents to retrieve
        """
        self.retriever = retriever
        self.llm = llm
        self.k = k
    
    def answer(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using simple RAG.
        
        Returns:
            Dictionary with answer, tokens, latency, etc.
        """
        t_start = time.perf_counter()
        
        # Retrieve
        docs = self.retriever.get_relevant_documents(question)
        evidence = [d.page_content for d in docs[:self.k]]
        
        # Generate
        prompt = (
            f"Answer the question using only the provided context.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n" + "\n".join(f"{i+1}. {e[:300]}" for i, e in enumerate(evidence)) + "\n\n"
            f"Answer:"
        )
        
        raw_answer = self.llm.generate(prompt, max_new_tokens=64)
        cleaned_answer = clean_answer(raw_answer, question)

        latency_ms = (time.perf_counter() - t_start) * 1000
        tokens = len(prompt.split()) + len(raw_answer.split())

        return {
            "answer": cleaned_answer,
            "raw_answer": raw_answer.strip(),
            "tokens": tokens,
            "latency_ms": latency_ms,
            "agents_executed": ["retriever", "composer"],
            "agents_pruned": [],
            "retrieved_context": evidence,
            "evidence": evidence
        }


# ============================================================================
# SELF-RAG (Reflection Baseline)
# ============================================================================

class SelfRAG:
    """
    Self-RAG with reflection tokens.
    
    Based on: Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", ICLR 2024
    
    Pipeline:
    1. Retrieve documents
    2. For each document, assess relevance (reflection)
    3. Filter irrelevant documents
    4. Generate answer with filtered context
    """
    
    def __init__(self, retriever, llm, k: int = 3):
        """
        Initialize Self-RAG.
        
        Args:
            retriever: LangChain retriever
            llm: LocalLLM instance
            k: Number of documents to retrieve
        """
        self.retriever = retriever
        self.llm = llm
        self.k = k
    
    def answer(self, question: str) -> Dict[str, Any]:
        """Answer with self-reflection."""
        t_start = time.perf_counter()
        
        # Retrieve
        docs = self.retriever.get_relevant_documents(question)
        evidence = [d.page_content for d in docs[:self.k]]
        
        # Assess relevance (reflection token)
        filtered_evidence = []
        for doc in evidence[:3]:  # Check top 3
            relevance_prompt = (
                f"Is this context relevant to the question? Answer YES or NO.\n"
                f"Question: {question}\n"
                f"Context: {doc[:200]}\n"
                f"Relevant:"
            )
            relevance = self.llm.generate(relevance_prompt, max_new_tokens=8)
            
            if "yes" in relevance.lower():
                filtered_evidence.append(doc)
        
        # Use filtered evidence (fallback to top 1 if all filtered)
        if not filtered_evidence:
            filtered_evidence = evidence[:1]
        
        # Generate
        prompt = (
            f"Answer the question using the context.\n"
            f"Question: {question}\n"
            f"Context:\n" + "\n".join(f"{i+1}. {e[:300]}" for i, e in enumerate(filtered_evidence)) + "\n"
            f"Answer:"
        )
        answer = self.llm.generate(prompt, max_new_tokens=64)
        
        latency_ms = (time.perf_counter() - t_start) * 1000
        tokens = (len(relevance_prompt.split()) * len(evidence[:3]) + 
                 len(prompt.split()) + len(answer.split()))
        
        return {
            "answer": answer.strip(),
            "tokens": tokens,
            "latency_ms": latency_ms,
            "agents_executed": ["retriever", "validator", "composer"],
            "agents_pruned": [],
            "evidence": filtered_evidence
        }


# ============================================================================
# CRAG (Corrective RAG)
# ============================================================================

class CRAG:
    """
    Corrective RAG with verification and fallback.
    
    Based on: Yan et al., "Corrective Retrieval Augmented Generation", 2024
    
    Pipeline:
    1. Retrieve documents
    2. Verify relevance with confidence score
    3. If low confidence, retrieve again with query reformulation
    4. Generate with corrected context
    """
    
    def __init__(self, retriever, llm, k: int = 3, confidence_threshold: float = 0.6):
        """
        Initialize CRAG.
        
        Args:
            retriever: LangChain retriever
            llm: LocalLLM instance
            k: Number of documents
            confidence_threshold: Minimum confidence for accepting retrieval
        """
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.confidence_threshold = confidence_threshold
    
    def answer(self, question: str) -> Dict[str, Any]:
        """Answer with corrective retrieval."""
        t_start = time.perf_counter()
        
        # Initial retrieval
        docs = self.retriever.get_relevant_documents(question)
        evidence = [d.page_content for d in docs[:self.k]]
        
        # Verify relevance
        verify_prompt = (
            f"Rate the relevance of this context to the question on a scale of 0-10.\n"
            f"Question: {question}\n"
            f"Context: {evidence[0][:200] if evidence else ''}\n"
            f"Relevance score (0-10):"
        )
        score_text = self.llm.generate(verify_prompt, max_new_tokens=8)
        
        # Extract numeric score
        score_match = re.search(r'\d+', score_text)
        confidence = float(score_match.group()) / 10 if score_match else 0.5
        
        # Corrective retrieval if low confidence
        if confidence < self.confidence_threshold:
            # Reformulate query
            reformulate_prompt = (
                f"Rephrase this question to make it clearer:\n"
                f"Original: {question}\n"
                f"Rephrased:"
            )
            reformulated = self.llm.generate(reformulate_prompt, max_new_tokens=32)
            
            # Re-retrieve
            docs = self.retriever.get_relevant_documents(reformulated.strip())
            evidence = [d.page_content for d in docs[:self.k]]
        
        # Generate
        prompt = (
            f"Answer the question using the context.\n"
            f"Question: {question}\n"
            f"Context:\n" + "\n".join(f"{i+1}. {e[:300]}" for i, e in enumerate(evidence)) + "\n"
            f"Answer:"
        )
        answer = self.llm.generate(prompt, max_new_tokens=64)
        
        latency_ms = (time.perf_counter() - t_start) * 1000
        tokens = len(verify_prompt.split()) + len(prompt.split()) + len(answer.split())
        
        return {
            "answer": answer.strip(),
            "tokens": tokens,
            "latency_ms": latency_ms,
            "agents_executed": ["retriever", "validator", "composer"],
            "agents_pruned": [],
            "evidence": evidence,
            "corrective_retrieval": confidence < self.confidence_threshold
        }


# ============================================================================
# KET-RAG (Knowledge Graph Enhanced)
# ============================================================================

class KETRAG:
    """
    KET-RAG: Knowledge graph enhanced retrieval.
    
    Based on: Liu et al., "KET-RAG: Knowledge Graph Enhanced RAG", 2024
    
    Pipeline:
    1. Extract entities from question
    2. Retrieve documents mentioning entities
    3. Build entity-relation graph from context
    4. Use graph structure to guide answer generation
    
    Simplified implementation without full KG construction.
    """
    
    def __init__(self, retriever, llm, k: int = 3):
        """
        Initialize KET-RAG.
        
        Args:
            retriever: LangChain retriever
            llm: LocalLLM instance
            k: Number of documents
        """
        self.retriever = retriever
        self.llm = llm
        self.k = k
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simple heuristic: capitalized words)."""
        return re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    
    def answer(self, question: str) -> Dict[str, Any]:
        """Answer with KG-enhanced retrieval."""
        t_start = time.perf_counter()
        
        # Extract entities from question
        entities = self._extract_entities(question)
        
        # Retrieve documents mentioning entities
        if entities:
            # Build enhanced query with entities
            enhanced_query = question + " " + " ".join(entities)
            docs = self.retriever.get_relevant_documents(enhanced_query)
        else:
            docs = self.retriever.get_relevant_documents(question)
        
        evidence = [d.page_content for d in docs[:self.k]]
        
        # Extract entity relations (simplified: co-occurrence)
        entity_context = defaultdict(list)
        for ent in entities:
            for doc in evidence:
                if ent in doc:
                    entity_context[ent].append(doc[:200])
        
        # Build KG-informed prompt
        kg_info = "\n".join([
            f"- {ent}: {'; '.join(ctx[:2])}" 
            for ent, ctx in entity_context.items()
        ])
        
        prompt = (
            f"Answer using the context and entity information.\n"
            f"Question: {question}\n\n"
            f"Entity Information:\n{kg_info}\n\n"
            f"Context:\n" + "\n".join(f"{i+1}. {e[:300]}" for i, e in enumerate(evidence)) + "\n\n"
            f"Answer:"
        )
        answer = self.llm.generate(prompt, max_new_tokens=64)
        
        latency_ms = (time.perf_counter() - t_start) * 1000
        tokens = len(prompt.split()) + len(answer.split())
        
        return {
            "answer": answer.strip(),
            "tokens": tokens,
            "latency_ms": latency_ms,
            "agents_executed": ["retriever", "kg_builder", "composer"],
            "agents_pruned": [],
            "evidence": evidence,
            "entities_used": entities
        }


# ============================================================================
# SAGE (Self-Adaptive Guided Exploration)
# ============================================================================

class SAGE:
    """
    SAGE: Self-adaptive guided exploration for multi-hop QA.
    
    Based on: Sun et al., "SAGE: Self-Adaptive Guidance for Multi-Hop Reasoning", 2024
    
    Pipeline:
    1. Retrieve initial documents
    2. Assess answer completeness
    3. If incomplete, extract missing information needs
    4. Retrieve additional documents targeting missing info
    5. Iteratively refine until complete or budget exhausted
    """
    
    def __init__(self, retriever, llm, k: int = 3, max_hops: int = 2):
        """
        Initialize SAGE.
        
        Args:
            retriever: LangChain retriever
            llm: LocalLLM instance
            k: Documents per hop
            max_hops: Maximum retrieval iterations
        """
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.max_hops = max_hops
    
    def answer(self, question: str) -> Dict[str, Any]:
        """Answer with adaptive exploration."""
        t_start = time.perf_counter()
        
        all_evidence = []
        
        # Hop 1: Initial retrieval
        docs = self.retriever.get_relevant_documents(question)
        evidence = [d.page_content for d in docs[:self.k]]
        all_evidence.extend(evidence)
        
        # Iterative refinement
        for hop in range(1, self.max_hops):
            # Check completeness
            check_prompt = (
                f"Can this context fully answer the question? Answer YES or NO.\n"
                f"Question: {question}\n"
                f"Context: {' '.join(all_evidence[:2])[:400]}\n"
                f"Complete:"
            )
            completeness = self.llm.generate(check_prompt, max_new_tokens=8)
            
            if "yes" in completeness.lower():
                break  # Answer is complete
            
            # Extract missing information
            missing_prompt = (
                f"What information is missing to answer this question?\n"
                f"Question: {question}\n"
                f"Current context: {' '.join(all_evidence[:2])[:300]}\n"
                f"Missing:"
            )
            missing_info = self.llm.generate(missing_prompt, max_new_tokens=32)
            
            # Retrieve targeting missing info
            follow_up_query = question + " " + missing_info.strip()
            docs = self.retriever.get_relevant_documents(follow_up_query)
            new_evidence = [d.page_content for d in docs[:self.k]]
            all_evidence.extend(new_evidence)
        
        # Generate final answer
        prompt = (
            f"Answer the question using all the context.\n"
            f"Question: {question}\n"
            f"Context:\n" + "\n".join(f"{i+1}. {e[:300]}" for i, e in enumerate(all_evidence[:6])) + "\n"
            f"Answer:"
        )
        answer = self.llm.generate(prompt, max_new_tokens=64)
        
        latency_ms = (time.perf_counter() - t_start) * 1000
        tokens = len(prompt.split()) + len(answer.split()) + (hop * 50)  # Approximate
        
        return {
            "answer": answer.strip(),
            "tokens": tokens,
            "latency_ms": latency_ms,
            "agents_executed": ["retriever", "validator", "composer"],
            "agents_pruned": [],
            "evidence": all_evidence,
            "hops_used": hop + 1
        }


# ============================================================================
# PLAN-RAG (Planning-Guided)
# ============================================================================

class PlanRAG:
    """
    Plan-RAG: Planning-guided decomposition.
    
    Based on: He et al., "Plan-RAG: Planning-Guided Retrieval", 2024
    
    Pipeline:
    1. Decompose question into sub-questions
    2. For each sub-question, retrieve and answer
    3. Synthesize final answer from sub-answers
    """
    
    def __init__(self, retriever, llm, k: int = 3):
        """
        Initialize Plan-RAG.
        
        Args:
            retriever: LangChain retriever
            llm: LocalLLM instance
            k: Documents per sub-question
        """
        self.retriever = retriever
        self.llm = llm
        self.k = k
    
    def answer(self, question: str) -> Dict[str, Any]:
        """Answer with planning decomposition."""
        t_start = time.perf_counter()
        
        # Decompose into sub-questions
        plan_prompt = (
            f"Break this question into 2-3 simpler sub-questions.\n"
            f"Question: {question}\n"
            f"Sub-questions (numbered):\n"
        )
        plan = self.llm.generate(plan_prompt, max_new_tokens=128)
        
        # Extract sub-questions (numbered lines)
        sub_questions = []
        for line in plan.split('\n'):
            if re.match(r'^\d+[.)]', line.strip()):
                sub_q = re.sub(r'^\d+[.)]', '', line).strip()
                if sub_q:
                    sub_questions.append(sub_q)
        
        # Limit to 3 sub-questions
        sub_questions = sub_questions[:3]
        
        # Answer each sub-question
        sub_answers = []
        all_evidence = []
        for sub_q in sub_questions:
            docs = self.retriever.get_relevant_documents(sub_q)
            evidence = [d.page_content for d in docs[:self.k]]
            all_evidence.extend(evidence)
            
            sub_prompt = (
                f"Answer: {sub_q}\n"
                f"Context: {evidence[0][:200] if evidence else ''}\n"
                f"Answer:"
            )
            sub_ans = self.llm.generate(sub_prompt, max_new_tokens=32)
            sub_answers.append(sub_ans.strip())
        
        # Synthesize final answer
        synthesis_prompt = (
            f"Combine these sub-answers to answer the main question.\n"
            f"Main question: {question}\n"
            f"Sub-answers:\n" + "\n".join(f"- {sa}" for sa in sub_answers) + "\n"
            f"Final answer:"
        )
        answer = self.llm.generate(synthesis_prompt, max_new_tokens=64)
        
        latency_ms = (time.perf_counter() - t_start) * 1000
        tokens = len(plan_prompt.split()) + len(synthesis_prompt.split()) + len(answer.split())
        
        return {
            "answer": answer.strip(),
            "tokens": tokens,
            "latency_ms": latency_ms,
            "agents_executed": ["planner", "retriever", "composer"],
            "agents_pruned": [],
            "evidence": all_evidence,
            "sub_questions": sub_questions,
            "sub_answers": sub_answers
        }


# ============================================================================
# BASELINE FACTORY
# ============================================================================

def get_baseline(
    name: str, 
    retriever, 
    llm, 
    **kwargs
) -> Any:
    """
    Factory function to create baseline systems.
    
    Args:
        name: Baseline name (vanilla_rag, self_rag, crag, ket_rag, sage, plan_rag)
        retriever: LangChain retriever
        llm: LocalLLM instance
        **kwargs: Additional arguments (k, confidence_threshold, etc.)
        
    Returns:
        Baseline instance
    """
    baselines = {
        "vanilla_rag": VanillaRAG,
        "self_rag": SelfRAG,
        "crag": CRAG,
        "ket_rag": KETRAG,
        "sage": SAGE,
        "plan_rag": PlanRAG
    }
    
    if name not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(baselines.keys())}")
    
    return baselines[name](retriever, llm, **kwargs)