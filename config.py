"""Central configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OPENAI_PRICING = {
    "gpt-4o-mini":     (0.00015,    0.00060),
    "gpt-4o":          (0.0025,     0.01),
    "gpt-4-turbo":     (0.01,       0.03),
    "gpt-4":           (0.03,       0.06),
    "gpt-3.5-turbo":   (0.0005,     0.0015),
}

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

TOP_K = 10

# --- Fair-comparison context budget (reviewer fix #2) ---
# Basic RAG (P2) and GraphRAG (P3) use the SAME number of context items AND the
# SAME per-item character cap, so neither pipeline is fed more text than the other.
# 2000 chars covers the p90 of both P2 chroma chunks (~1817) and P3 docs (~2066),
# so this is a generous "level-up" cap that leaves both essentially un-starved.
CONTEXT_NUM_CHUNKS  = int(os.getenv("CONTEXT_NUM_CHUNKS", "10"))
CONTEXT_CHUNK_CHARS = int(os.getenv("CONTEXT_CHUNK_CHARS", "2000"))

HOTPOTQA_SPLIT = "train"
MAX_DOCS_TO_INDEX = int(os.getenv("MAX_DOCS_TO_INDEX", "10000"))
EVAL_SAMPLE_SIZE = 50

# ---- Dataset selection (configurable by parameter) ----
# DATASET picks the corpus + QA source and tunes the answer style. Add new
# datasets here + in data/datasets.py.
DATASET = os.getenv("DATASET", "bioasq")
# Max corpus tokens to load (high default = use the full available corpus).
TARGET_CORPUS_TOKENS = int(os.getenv("TARGET_CORPUS_TOKENS", "100000000"))

# Biomedical datasets use the biomedical answer prompt, biomedical entity/relation
# extraction, and a PubMed-tuned embedder. Add new biomedical datasets here.
BIOMED_DATASETS = {"bioasq"}


def ds_paths(ds: str | None = None) -> dict:
    """Standard per-dataset artifact paths (non-destructive: each dataset isolated)."""
    ds = ds or DATASET
    if ds == "hotpotqa":
        return {  # keep original hotpotqa layout for backward compatibility
            "chroma": "./chroma_db/hotpotqa", "collection": "hotpotqa_docs",
            "ground_truth": "./results/ground_truth.json",
            "index_dir": "pipeline3_networkx/index",
            "graph": os.getenv("TIGERGRAPH_GRAPH", "hotpotqa"),
            "benchmark": "./results/benchmark_results.json",
        }
    return {
        "chroma": f"./chroma_db_{ds}/main", "collection": f"{ds}_docs",
        "ground_truth": f"./results/ground_truth_{ds}.json",
        "index_dir": f"pipeline3_{ds}/index",
        "graph": ds,
        "benchmark": f"./results/benchmark_{ds}.json",
    }


# Chroma paths are dataset-aware — always derived from DATASET so all consumers
# (chroma_store.py, app.py) automatically point at the right index.
_ds = ds_paths()
CHROMA_PERSIST_DIR     = _ds["chroma"]
CHROMA_COLLECTION_NAME = _ds["collection"]

HF_TOKEN = os.getenv("HF_TOKEN", "")
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "hf")

BENCHMARK_RESULTS_PATH = "./results/benchmark_results.json"
GROUND_TRUTH_PATH      = "./results/ground_truth.json"

TEMPERATURE = 0.0

# Factoid datasets want terse answers; biomedical datasets want complete answers.
_FACTOID_PROMPT = (
    "Answer the factoid question with only the exact answer — a name, date, place, "
    "number, or short noun phrase. Use the context passages if provided; otherwise "
    "use your general knowledge. No sentences. No trailing period. No 'The answer is'. "
    "No explanations."
)
_BIOMED_PROMPT = (
    "You are a biomedical expert. Answer the question accurately and completely using "
    "the provided context passages when available; otherwise use your knowledge. "
    "Give a focused answer of 1-3 sentences that states the key finding and names the "
    "relevant biomedical entities (genes, proteins, diseases, drugs, pathways). "
    "Do not add citations, disclaimers, or 'The answer is'."
)

# Biomedical datasets get a focused 1-3 sentence prompt + room to answer; factoid stays terse.
if DATASET in BIOMED_DATASETS:
    SYSTEM_PROMPT, _MAX_TOKENS_DEFAULT = _BIOMED_PROMPT, 256
else:
    SYSTEM_PROMPT, _MAX_TOKENS_DEFAULT = _FACTOID_PROMPT, 32
MAX_TOKENS = int(os.getenv("MAX_TOKENS", str(_MAX_TOKENS_DEFAULT)))
