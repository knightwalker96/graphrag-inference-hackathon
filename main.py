"""
Entry point: builds the index and runs the three-pipeline benchmark.

Usage:
  python main.py --mode index       # Build the ChromaDB index (run once)
  python main.py --mode benchmark   # Run all 3 pipelines + evaluate + save results
  python main.py --mode quick       # Quick test with 5 questions (sanity check)
  python main.py --mode p3only      # Run pipeline 3 only, append to existing results
"""

from __future__ import annotations
import argparse
import json
import os
import random

import config

os.makedirs("./results", exist_ok=True)

def _import_p3():
    import pipeline3_tigergraph_cloud.pipeline as p3
    return p3


def mode_index():
    from data.loader import load_hotpotqa, save_ground_truth
    from chroma_store import build_vectorstore

    documents, qa_pairs = load_hotpotqa()
    save_ground_truth(qa_pairs[:config.EVAL_SAMPLE_SIZE * 10])
    print(f"[main] Saved {min(len(qa_pairs), config.EVAL_SAMPLE_SIZE * 10)} QA pairs as ground truth.")

    build_vectorstore(documents)
    print("\n[main] ✅ Indexing complete. Run `python main.py --mode benchmark` next.")


def mode_benchmark(n_questions: int = config.EVAL_SAMPLE_SIZE, include_p3: bool = True):
    from data.loader import load_ground_truth
    from chroma_store import load_vectorstore
    import pipeline1_llm_only as p1
    import pipeline2_basic_rag as p2
    from metrics import aggregate_metrics
    from accuracy import AccuracyEvaluator

    all_qa = load_ground_truth()
    random.seed(42)
    eval_qa = random.sample(all_qa, min(n_questions, len(all_qa)))
    questions = [qa["question"] for qa in eval_qa]
    correct_answers = [qa["answer"] for qa in eval_qa]
    print(f"[main] Running benchmark on {len(eval_qa)} questions...")

    vectorstore = load_vectorstore()

    print("\n[main] === Pipeline 1: LLM-Only ===")
    p1_metrics = p1.run_batch(questions)
    p1_answers = [m.answer for m in p1_metrics]

    print("\n[main] === Pipeline 2: Basic RAG ===")
    p2_metrics = p2.run_batch(questions, vectorstore)
    p2_answers = [m.answer for m in p2_metrics]

    p3_metrics, p3_answers, p3_agg, p3_accuracy = [], [], None, None
    if include_p3:
        p3 = _import_p3()
        print("\n[main] === Pipeline 3: GraphRAG (TigerGraph Savanna) ===")
        try:
            p3_metrics = p3.run_batch(questions, vectorstore=vectorstore)
            p3_answers = [m.answer for m in p3_metrics]
        except Exception as e:
            # Reviewer fix #1a: the live graph is mandatory BY DEFAULT — do NOT
            # downgrade to a P1/P2-only result; abort the whole benchmark. Only a
            # deliberate opt-out (P3_REQUIRE_TG=0) or the graph-OFF ablation
            # (P3_DISABLE_GRAPH=1) allows the run to continue without the graph.
            require_tg = (os.getenv("P3_REQUIRE_TG", "1") != "0"
                          and os.getenv("P3_DISABLE_GRAPH", "0") == "0")
            if require_tg:
                raise
            print(f"[main] ⚠️  Pipeline 3 failed: {e}")
            include_p3 = False

    p1_agg = aggregate_metrics(p1_metrics)
    p2_agg = aggregate_metrics(p2_metrics)
    if include_p3 and p3_metrics:
        p3_agg = aggregate_metrics(p3_metrics)

    print("\n[main] === Performance Summary ===")
    aggs = [p1_agg, p2_agg] + ([p3_agg] if p3_agg else [])
    for agg in aggs:
        print(f"\n  {agg['pipeline']}")
        print(f"    Avg total tokens:   {agg['avg_total_tokens']}")
        print(f"    Avg latency:        {agg['avg_latency_seconds']}s")
        print(f"    Avg cost per query: ${agg['avg_cost_usd']:.6f}")

    if p1_agg["avg_total_tokens"] > 0:
        token_delta = (p1_agg["avg_total_tokens"] - p2_agg["avg_total_tokens"]) / p1_agg["avg_total_tokens"] * 100
        print(f"\n  Basic RAG vs LLM-Only token change: {token_delta:+.1f}%")
    if p3_agg and p2_agg["avg_total_tokens"] > 0:
        token_delta_p3 = (p2_agg["avg_total_tokens"] - p3_agg["avg_total_tokens"]) / p2_agg["avg_total_tokens"] * 100
        print(f"  GraphRAG vs Basic RAG token change:  {token_delta_p3:+.1f}%")

    print("\n[main] === Accuracy Evaluation ===")
    evaluator = AccuracyEvaluator()
    skip_judge = config.JUDGE_PROVIDER == "hf" and not config.HF_TOKEN
    if skip_judge:
        print("[main] ⚠️  HF_TOKEN not set. Skipping LLM-as-a-Judge — only BERTScore will run.")

    p1_accuracy = evaluator.evaluate_pipeline("Pipeline 1: LLM-Only", questions, correct_answers, p1_answers, skip_judge=skip_judge)
    p2_accuracy = evaluator.evaluate_pipeline("Pipeline 2: Basic RAG", questions, correct_answers, p2_answers, skip_judge=skip_judge)
    if include_p3 and p3_answers:
        p3_accuracy = evaluator.evaluate_pipeline("Pipeline 3: GraphRAG", questions, correct_answers, p3_answers, skip_judge=skip_judge)

    full_results = {
        "config": {
            "llm_provider": config.LLM_PROVIDER,
            "llm_model": config.OPENAI_MODEL if config.LLM_PROVIDER == "openai" else config.OLLAMA_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
            "top_k": config.TOP_K,
            "chunk_size": config.CHUNK_SIZE,
            "n_questions": len(eval_qa),
            "ground_truth_path": config.GROUND_TRUTH_PATH,
            # Reviewer-relevant knobs, recorded so each results file is self-documenting.
            "context_num_chunks": config.CONTEXT_NUM_CHUNKS,
            "context_chunk_chars": config.CONTEXT_CHUNK_CHARS,
            "graph_boost": os.getenv("GRAPH_BOOST", "4.0"),
            "graph_disabled": os.getenv("P3_DISABLE_GRAPH", "0") != "0",
            "require_live_tg": (os.getenv("P3_REQUIRE_TG", "1") != "0"
                                and os.getenv("P3_DISABLE_GRAPH", "0") == "0"),
            "tigergraph_host": os.getenv("TIGERGRAPH_HOST", ""),
            "tigergraph_graph": os.getenv("TIGERGRAPH_GRAPH", ""),
        },
        "performance": {
            "pipeline1": p1_agg,
            "pipeline2": p2_agg,
            **( {"pipeline3": p3_agg} if p3_agg else {} ),
        },
        "accuracy": {
            "pipeline1": {k: v for k, v in p1_accuracy.items() if k != "per_item"},
            "pipeline2": {k: v for k, v in p2_accuracy.items() if k != "per_item"},
            **( {"pipeline3": {k: v for k, v in p3_accuracy.items() if k != "per_item"}} if p3_accuracy else {} ),
        },
        "per_question": {
            "pipeline1": [m.to_dict() for m in p1_metrics],
            "pipeline2": [m.to_dict() for m in p2_metrics],
            **( {"pipeline3": [m.to_dict() for m in p3_metrics]} if p3_metrics else {} ),
        },
        "accuracy_detail": {
            "pipeline1": p1_accuracy["per_item"],
            "pipeline2": p2_accuracy["per_item"],
            **( {"pipeline3": p3_accuracy["per_item"]} if p3_accuracy else {} ),
        },
    }

    with open(config.BENCHMARK_RESULTS_PATH, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n[main] ✅ Results saved to {config.BENCHMARK_RESULTS_PATH}")
    print("[main] Launch the dashboard: streamlit run app.py")


def mode_p3only(n_questions: int = config.EVAL_SAMPLE_SIZE, skip_judge: bool = False):
    """Run pipeline 3 only and merge results into existing benchmark_results.json."""
    p3 = _import_p3()

    from data.loader import load_ground_truth
    from chroma_store import load_vectorstore
    from metrics import aggregate_metrics
    from accuracy import AccuracyEvaluator

    all_qa = load_ground_truth()
    random.seed(42)
    eval_qa = random.sample(all_qa, min(n_questions, len(all_qa)))
    questions = [qa["question"] for qa in eval_qa]
    correct_answers = [qa["answer"] for qa in eval_qa]

    vectorstore = load_vectorstore()

    print(f"\n[main] === Pipeline 3: GraphRAG ({len(eval_qa)} questions) ===")
    p3_metrics = p3.run_batch(questions, vectorstore=vectorstore, verbose=True)
    p3_answers = [m.answer for m in p3_metrics]
    p3_agg = aggregate_metrics(p3_metrics)

    evaluator = AccuracyEvaluator()
    skip_judge = skip_judge or (config.JUDGE_PROVIDER == "hf" and not config.HF_TOKEN)
    p3_accuracy = evaluator.evaluate_pipeline("Pipeline 3: GraphRAG", questions, correct_answers, p3_answers, skip_judge=skip_judge)

    print(f"\n[main] Pipeline 3 results:")
    print(f"  Avg tokens:    {p3_agg['avg_total_tokens']}")
    print(f"  Avg latency:   {p3_agg['avg_latency_seconds']}s")
    print(f"  Judge pass:    {p3_accuracy.get('llm_judge_pass_rate', 'N/A')}")
    print(f"  BERTScore F1:  {p3_accuracy.get('bertscore_f1_rescaled', 'N/A')}")

    existing = {}
    if os.path.exists(config.BENCHMARK_RESULTS_PATH):
        with open(config.BENCHMARK_RESULTS_PATH) as f:
            existing = json.load(f)

    existing.setdefault("performance", {})["pipeline3"] = p3_agg
    existing.setdefault("accuracy", {})["pipeline3"] = {k: v for k, v in p3_accuracy.items() if k != "per_item"}
    existing.setdefault("per_question", {})["pipeline3"] = [m.to_dict() for m in p3_metrics]
    existing.setdefault("accuracy_detail", {})["pipeline3"] = p3_accuracy["per_item"]

    with open(config.BENCHMARK_RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n[main] ✅ Pipeline 3 results merged into {config.BENCHMARK_RESULTS_PATH}")


def mode_quick():
    from chroma_store import load_vectorstore
    import pipeline1_llm_only as p1
    import pipeline2_basic_rag as p2

    questions = [
        "What is the capital of France?",
        "Who wrote the novel 1984?",
        "What year did World War II end?",
        "Who is the founder of Microsoft?",
        "What is the speed of light?",
    ]

    print("[main] === Quick test: Pipeline 1 ===")
    for q in questions:
        m = p1.run_single(q)
        print(f"  Q: {q}\n  A: {m.answer}\n  Tokens: {m.total_tokens} | Latency: {m.latency_seconds}s\n")

    print("[main] === Quick test: Pipeline 2 ===")
    vs = load_vectorstore()
    for q in questions:
        m = p2.run_single(q, vs)
        print(f"  Q: {q}\n  A: {m.answer}\n  Tokens: {m.total_tokens} | Chunks: {m.retrieved_chunks} | Latency: {m.latency_seconds}s\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG Hackathon - HotpotQA (All 3 Pipelines)")
    parser.add_argument("--mode", choices=["index", "benchmark", "quick", "p3only"], required=True)
    parser.add_argument("--n", type=int, default=config.EVAL_SAMPLE_SIZE,
                        help=f"Questions for benchmark (default: {config.EVAL_SAMPLE_SIZE})")
    parser.add_argument("--no-p3", action="store_true", help="Skip pipeline 3 in benchmark mode")
    parser.add_argument("--llm", choices=["ollama", "openai"], default=None,
                        help="Answering LLM provider (overrides LLM_PROVIDER from .env).")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="Model name within the chosen --llm provider.")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM-as-a-Judge; use BERTScore only (avoids HF rate limits).")
    args = parser.parse_args()

    if args.llm:
        config.LLM_PROVIDER = args.llm
    if args.llm_model:
        if config.LLM_PROVIDER == "openai":
            config.OPENAI_MODEL = args.llm_model
        else:
            config.OLLAMA_MODEL = args.llm_model
    _active_model = config.OPENAI_MODEL if config.LLM_PROVIDER == "openai" else config.OLLAMA_MODEL
    print(f"[main] Answering LLM: provider={config.LLM_PROVIDER}, model={_active_model}")

    if args.mode == "index":
        mode_index()
    elif args.mode == "benchmark":
        mode_benchmark(n_questions=args.n, include_p3=not args.no_p3)
    elif args.mode == "quick":
        mode_quick()
    elif args.mode == "p3only":
        mode_p3only(n_questions=args.n, skip_judge=args.skip_judge)
