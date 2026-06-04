"""Three-pipeline benchmark runner.

Usage:
  DATASET=bioasq python run_benchmark.py            # all 3 pipelines
  DATASET=bioasq P3_ONLY=1 python run_benchmark.py  # only Pipeline 3 (fast iteration)

Env overrides: N_QUESTIONS, GT_PATH, BENCH_PATH, EMBEDDING_MODEL, JUDGE_PROVIDER.
Biomedical datasets (config.BIOMED_DATASETS) auto-use the PubMedBERT embedder.
Do NOT set HF_HUB_OFFLINE — it blocks the HF judge API.
"""
import os

DATASET = os.environ.setdefault("DATASET", "bioasq")

import config
p = config.ds_paths(DATASET)

# Biomedical datasets -> PubMedBERT embedder (index + query must match)
if DATASET in config.BIOMED_DATASETS:
    config.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")

# Pipeline-3 (Savanna) wiring + tuned GraphRAG defaults
os.environ["TIGERGRAPH_GRAPH"] = p["graph"]
os.environ.setdefault("P3_INDEX_DIR", p["index_dir"])
os.environ.setdefault("TG_MIN_CO_COUNT", "1")   # Gemini edges have rel_count ~1
os.environ.setdefault("NUM_HOPS", "2")
os.environ.setdefault("P3_GRAPH_TRIPLES", "1")
os.environ.setdefault("GRAPH_BRIDGE_RESERVE", "2")

config.CHROMA_PERSIST_DIR     = p["chroma"]
config.CHROMA_COLLECTION_NAME = p["collection"]
config.GROUND_TRUTH_PATH      = os.getenv("GT_PATH", p["ground_truth"])
config.BENCHMARK_RESULTS_PATH = os.getenv("BENCH_PATH", "./results/benchmark_results.json")
config.LLM_PROVIDER           = "openai"
config.OPENAI_MODEL           = "gpt-4o-mini"
config.JUDGE_PROVIDER         = os.getenv("JUDGE_PROVIDER", "hf")

import main

if __name__ == "__main__":
    n = int(os.getenv("N_QUESTIONS", str(config.EVAL_SAMPLE_SIZE)))
    print(f"[benchmark] dataset={DATASET} | graph={p['graph']} | embedder={config.EMBEDDING_MODEL} | "
          f"judge={config.JUDGE_PROVIDER} | answer={config.OPENAI_MODEL} | MAX_TOKENS={config.MAX_TOKENS}")
    if os.getenv("P3_ONLY"):
        main.mode_p3only(n_questions=n)
    else:
        main.mode_benchmark(n_questions=n, include_p3=True)
