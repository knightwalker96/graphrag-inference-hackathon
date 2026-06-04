"""Pipeline 2: Basic RAG — embed question → retrieve chunks → LLM. Primary baseline."""

from __future__ import annotations
import os
import time

from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

from metrics import CallMetrics, MetricsTimer
from llm_client import get_llm_response
import config

# MMR retrieval + cross-encoder reranking (toggle off with P2_RERANK=0)
P2_RERANK        = os.getenv("P2_RERANK", "1") != "0"
RERANK_POOL      = int(os.getenv("P2_RERANK_POOL", "20"))   # MMR candidates before rerank
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_INPUT_CHARS = 600                                    # model has a 512-token limit

_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print(f"[p2] Loading cross-encoder ({RERANKER_MODEL})…")
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context_block = "\n\n---\n\n".join(
        f"[Passage {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    return (
        f"{config.SYSTEM_PROMPT}\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\nAnswer:"
    )


def run_single(question: str, vectorstore: Chroma) -> CallMetrics:
    t_start = time.perf_counter()

    if P2_RERANK:
        # MMR pulls a diverse candidate pool (good for 2-hop questions that need
        # chunks from two different articles), then a cross-encoder reranks them
        # and we keep the top TOP_K for the prompt.
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RERANK_POOL, "fetch_k": RERANK_POOL * 3, "lambda_mult": 0.7},
        )
        candidates = retriever.invoke(question)
        if candidates:
            ce = _get_cross_encoder()
            scores = ce.predict([(question, d.page_content[:RERANK_INPUT_CHARS]) for d in candidates])
            ranked = [d for _, d in sorted(zip(scores, candidates), key=lambda x: -x[0])]
        else:
            ranked = candidates
        context_chunks = [d.page_content for d in ranked[:config.TOP_K]]
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": config.TOP_K},
        )
        context_chunks = [doc.page_content for doc in retriever.invoke(question)]

    prompt = build_prompt(question, context_chunks)
    with MetricsTimer("Pipeline 2: Basic RAG", question) as timer:
        answer, _ = get_llm_response(prompt)

    metrics = timer.record_manual(prompt, answer, retrieved_chunks=len(context_chunks))
    metrics.latency_seconds = round(time.perf_counter() - t_start, 3)
    return metrics


def run_batch(questions: list[str], vectorstore: Chroma, verbose: bool = False) -> list[CallMetrics]:
    from tqdm import tqdm
    results = []
    for q in tqdm(questions, desc="Pipeline 2 (Basic RAG)"):
        m = run_single(q, vectorstore)
        results.append(m)
        if verbose:
            print(f"  Q: {q[:60]}...")
            print(f"  A: {m.answer[:80]}...")
            print(f"  Chunks: {m.retrieved_chunks} | Tokens: {m.total_tokens} | Latency: {m.latency_seconds}s")
    return results
