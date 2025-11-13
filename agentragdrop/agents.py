from .utils import token_estimate
from .rag import make_retriever
from .answer_cleaning import clean_answer
from typing import Optional, List, Dict, Any
import time

MAX_CONTEXT_ITEMS = 4
MAX_SNIPPET_CHARS = 180
MAX_NEW_TOKENS = 16

_FEW_SHOT_EXAMPLES = (
    "Examples of good answers:\n\n"
    "Q: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\n"
    "Context: 1. Shirley Temple held the position of U.S. Ambassador... 2. Kiss and Tell starred Shirley Temple as Corliss Archer...\n"
    "A: U.S. Ambassador\n\n"
    "Q: Were Scott Derrickson and Ed Wood of the same nationality?\n"
    "Context: 1. Scott Derrickson is an American filmmaker. 2. Ed Wood was an American filmmaker...\n"
    "A: yes\n\n"
    "Q: What year was the creator of Vampire: The Masquerade born?\n"
    "Context: 1. Mark Rein-Hagen created Vampire: The Masquerade. 2. Mark Rein-Hagen was born in 1964...\n"
    "A: 1964\n\n"
)


def _format_evidence_snippets(
    evidence: Optional[List[str]],
    *,
    limit: int = MAX_CONTEXT_ITEMS,
    snippet_chars: int = MAX_SNIPPET_CHARS,
) -> str:
    """Format evidence snippets for the composer prompts."""

    if not evidence:
        return "1. No supporting context available."

    formatted: List[str] = []
    for idx, snippet in enumerate(evidence[:limit]):
        if not isinstance(snippet, str):
            snippet = str(snippet)
        snippet = snippet.strip()
        if not snippet:
            continue
        formatted.append(f"{idx + 1}. {snippet[:snippet_chars]}")

    if not formatted:
        return "1. No supporting context available."

    return "\n".join(formatted)


def _build_direct_answer_prompt(question: str, evidence: Optional[List[str]]) -> str:
    ctx_block = _format_evidence_snippets(evidence)
    return (
        "You are answering questions using provided context. "
        "Give ONLY the direct answer – a name, number, yes/no, or short phrase. "
        "Do NOT write explanations.\n"
        "If the answer cannot be determined from the context, output exactly: unknown.\n\n"
        f"{_FEW_SHOT_EXAMPLES}"
        "Now answer this question:\n\n"
        f"Q: {question}\n"
        f"Context:\n{ctx_block}\n"
        "A:"
    )

class AgentResult(dict):
    pass

class RetrieverAgent:
    def __init__(self, data_path, embed_model="sentence-transformers/all-MiniLM-L6-v2", top_k=4):
        self.retriever = make_retriever(data_path, embed_model=embed_model, k=top_k)
        self.vs = self.retriever.vectorstore
        self.top_k = top_k
        
        # Metrics
        self.total_calls = 0
        self.total_latency = 0.0

    def __call__(self, question, k: Optional[int] = None) -> Dict[str, Any]:
        t_start = time.perf_counter()
        
        k = k or self.top_k
        fetch_k = max(k * 2, k)
        try:
            results = self.vs.max_marginal_relevance_search_with_score(
                question, k=k, fetch_k=fetch_k
            )
        except AttributeError:
            results = self.vs.similarity_search_with_score(question, k=k)

        hits, evidence = [], []
        for doc, score in results:
            text = doc.page_content
            raw_score = float(score)
            similarity = 1.0 / (1.0 + abs(raw_score))
            hits.append(({"text": text}, similarity))
            evidence.append(text)
        
        t_elapsed = time.perf_counter() - t_start
        self.total_calls += 1
        self.total_latency += t_elapsed

        return {
            "hits": hits, 
            "evidence": evidence, 
            "tokens_est": 0,
            "latency_ms": t_elapsed * 1000
        }
    
    def get_metrics(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_latency_s": round(self.total_latency, 2),
            "avg_latency_ms": round((self.total_latency / self.total_calls) * 1000, 2) if self.total_calls > 0 else 0
        }

class ValidatorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.total_calls = 0
        self.total_latency = 0.0

    def __call__(self, question, evidence: List[str]):
        t_start = time.perf_counter()
        
        prompt = (
            "Answer only YES or NO.\n"
            f"Question: {question}\n"
            "Evidence:\n" + "\n".join(f"- {e[:300]}" for e in (evidence or [])[:3]) + "\n"
            "Is the evidence relevant to the question? Answer YES or NO:"
        )
        out = self.llm.generate(prompt, max_new_tokens=16)
        raw = out.strip()
        verdict_lower = raw.lower()
        if verdict_lower.startswith("yes"):
            verdict = "yes"
        elif verdict_lower.startswith("no"):
            verdict = "no"
        else:
            verdict = "unknown"

        t_elapsed = time.perf_counter() - t_start
        self.total_calls += 1
        self.total_latency += t_elapsed

        return AgentResult({
            "verdict": verdict,
            "verdict_raw": raw,
            "tokens_est": token_estimate(prompt + out),
            "latency_ms": t_elapsed * 1000
        })
    
    def get_metrics(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_latency_s": round(self.total_latency, 2),
            "avg_latency_ms": round((self.total_latency / self.total_calls) * 1000, 2) if self.total_calls > 0 else 0
        }

class CriticAgent:
    def __init__(self, llm, threshold=0.2):
        self.llm = llm
        self.threshold = threshold
        self.total_calls = 0
        self.total_latency = 0.0

    def __call__(self, question, evidence: List[str]):
        t_start = time.perf_counter()
        
        notes = ""
        if evidence and len(evidence) >= 2:
            set1, set2 = set(evidence[0].split()), set(evidence[1].split())
            overlap = len(set1.intersection(set2)) / max(1, len(set1.union(set2)))
            if overlap < self.threshold:
                notes = f"Inconsistent evidence (overlap={overlap:.2f})"

        if not notes:
            prompt = (
                "Critique if the provided evidence items conflict with each other. Be brief.\n"
                f"Q: {question}\nEvidence:\n" + "\n".join(f"- {e[:300]}" for e in (evidence or [])[:3])
            )
            notes = self.llm.generate(prompt, max_new_tokens=64)
            tokens = token_estimate(prompt + notes)
        else:
            tokens = token_estimate(notes)

        raw_notes = notes.strip()
        notes_lower = raw_notes.lower()
        if not raw_notes:
            label = "ok"
        elif any(term in notes_lower for term in ("inconsistent", "conflict", "contradict", "disagree", "mismatch")):
            label = "conflict"
        else:
            label = "ok"

        t_elapsed = time.perf_counter() - t_start
        self.total_calls += 1
        self.total_latency += t_elapsed

        return AgentResult({
            "notes": label,
            "notes_raw": raw_notes,
            "tokens_est": tokens,
            "latency_ms": t_elapsed * 1000
        })
    
    def get_metrics(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_latency_s": round(self.total_latency, 2),
            "avg_latency_ms": round((self.total_latency / self.total_calls) * 1000, 2) if self.total_calls > 0 else 0
        }

class ComposerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.total_calls = 0
        self.total_latency = 0.0

    def __call__(self, question, evidence: List[str], validator=None, critic=None):
        t_start = time.perf_counter()

        prompt = _build_direct_answer_prompt(question, evidence)
        ans = self.llm.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
        cleaned = clean_answer(ans, question)
        if not cleaned:
            cleaned = "unknown"

        t_elapsed = time.perf_counter() - t_start
        self.total_calls += 1
        self.total_latency += t_elapsed

        return AgentResult({
            "answer": cleaned,
            "raw_answer": ans.strip(),
            "tokens_est": token_estimate(prompt + ans),
            "latency_ms": t_elapsed * 1000
        })
    
    def get_metrics(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_latency_s": round(self.total_latency, 2),
            "avg_latency_ms": round((self.total_latency / self.total_calls) * 1000, 2) if self.total_calls > 0 else 0
        }

class RAGComposerAgent:
    """Composer with few-shot examples for HotpotQA."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.total_calls = 0
        self.total_latency = 0.0

    def __call__(self, question, evidence=None, **kwargs):
        t_start = time.perf_counter()

        evidence_list: List[str] = []
        if evidence:
            evidence_list = list(evidence)
        elif self.retriever:
            docs = []
            if hasattr(self.retriever, "invoke"):
                docs = self.retriever.invoke(question)
            elif hasattr(self.retriever, "get_relevant_documents"):
                docs = self.retriever.get_relevant_documents(question)
            else:
                vecstore = getattr(self.retriever, "vectorstore", None)
                if vecstore is not None:
                    try:
                        docs = vecstore.similarity_search(question, k=MAX_CONTEXT_ITEMS + 2)
                    except Exception:
                        docs = []
            evidence_list = [getattr(d, "page_content", str(d)) for d in (docs or [])]

        prompt = _build_direct_answer_prompt(question, evidence_list)

        raw_ans = self.llm.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
        ans = self._clean_answer(raw_ans, question)
        if not ans:
            ans = "unknown"

        t_elapsed = time.perf_counter() - t_start
        self.total_calls += 1
        self.total_latency += t_elapsed

        return AgentResult({
            "answer": ans,
            "raw_answer": raw_ans.strip(),
            "tokens_est": token_estimate(prompt + raw_ans),
            "latency_ms": t_elapsed * 1000
        })

    def _clean_answer(self, ans: str, question: str) -> str:
        return clean_answer(ans, question)
    
    def get_metrics(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_latency_s": round(self.total_latency, 2),
            "avg_latency_ms": round((self.total_latency / self.total_calls) * 1000, 2) if self.total_calls > 0 else 0
        }