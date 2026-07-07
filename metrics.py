"""Token counting, latency, and cost tracking per pipeline call."""

from __future__ import annotations
import re
import time
from dataclasses import dataclass, asdict, field

import tiktoken

import config


_PREFIX_RE = re.compile(
    r'^(the\s+answer\s+is[:,]?\s*'
    r'|answer[:,]?\s*'
    r'|based\s+on\s+(the\s+)?(context|passages|information)[:,]?\s*'
    r'|according\s+to\s+(the\s+)?(context|passage|information)[:,]?\s*)',
    re.IGNORECASE,
)


def normalize_answer(s: str) -> str:
    """Strip preambles, trailing punctuation, quotes - keeps short factoid answers clean."""
    if not s:
        return s
    s = s.strip()
    s = s.strip('"""\'`')
    s = _PREFIX_RE.sub('', s)
    # Take only the first non-empty line - guards against models appending
    # explanation after "Answer: <factoid>\nExplanation: ..."
    first_line = s.split('\n')[0].strip()
    if first_line:
        s = first_line
    s = s.rstrip('.,;:! \t\n')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


@dataclass
class CallMetrics:
    pipeline: str = ""
    question: str = ""
    answer: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    retrieved_chunks: int = 0

    # --- Pipeline-3 graph provenance (reviewer fix #1c) ---
    # Proof, per question, that the LIVE cloud graph was queried and what it returned.
    # Defaults keep P1/P2 rows clean (no graph => tg_live False, empty doc list).
    tg_live: bool = False               # True only if the live TigerGraph BFS ran for this question
    tg_host: str = ""                   # which Savanna instance answered
    graph_doc_ids: list = field(default_factory=list)  # doc IDs the graph BFS returned
    graph_entity_count: int = 0         # entities the graph expansion surfaced

    def to_dict(self) -> dict:
        return asdict(self)


def count_tokens_tiktoken(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    if config.LLM_PROVIDER == "openai":
        rates = config.OPENAI_PRICING.get(config.OPENAI_MODEL)
        if not rates:
            return 0.0
        in_per_1k, out_per_1k = rates
        return (prompt_tokens / 1000.0) * in_per_1k + (completion_tokens / 1000.0) * out_per_1k
    return 0.0


class MetricsTimer:
    def __init__(self, pipeline_name: str, question: str):
        self.pipeline = pipeline_name
        self.question = question
        self._start: float = 0.0
        self.metrics = CallMetrics(pipeline=pipeline_name, question=question)

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.metrics.latency_seconds = round(time.perf_counter() - self._start, 3)

    def record_manual(self, prompt: str, answer: str, retrieved_chunks: int = 0) -> CallMetrics:
        cleaned = normalize_answer(answer)
        self.metrics.answer = cleaned
        self.metrics.prompt_tokens = count_tokens_tiktoken(prompt)
        self.metrics.completion_tokens = count_tokens_tiktoken(cleaned)
        self.metrics.total_tokens = self.metrics.prompt_tokens + self.metrics.completion_tokens
        self.metrics.cost_usd = calculate_cost(self.metrics.prompt_tokens, self.metrics.completion_tokens)
        self.metrics.retrieved_chunks = retrieved_chunks
        return self.metrics


def aggregate_metrics(all_metrics: list[CallMetrics]) -> dict:
    if not all_metrics:
        return {}
    n = len(all_metrics)
    return {
        "pipeline": all_metrics[0].pipeline,
        "num_queries": n,
        "avg_prompt_tokens": round(sum(m.prompt_tokens for m in all_metrics) / n, 1),
        "avg_completion_tokens": round(sum(m.completion_tokens for m in all_metrics) / n, 1),
        "avg_total_tokens": round(sum(m.total_tokens for m in all_metrics) / n, 1),
        "avg_latency_seconds": round(sum(m.latency_seconds for m in all_metrics) / n, 3),
        "avg_cost_usd": round(sum(m.cost_usd for m in all_metrics) / n, 6),
        "total_cost_usd": round(sum(m.cost_usd for m in all_metrics), 4),
        "avg_retrieved_chunks": round(sum(m.retrieved_chunks for m in all_metrics) / n, 1),
        # --- Graph provenance rollup (reviewer fix #1c) ---
        # tg_live_rate = fraction of questions the live cloud graph actually answered.
        # For a valid GraphRAG run this must be 1.0; avg_graph_docs shows the graph's reach.
        "tg_live_rate": round(sum(1 for m in all_metrics if m.tg_live) / n, 3),
        "avg_graph_docs": round(sum(len(m.graph_doc_ids) for m in all_metrics) / n, 1),
    }
