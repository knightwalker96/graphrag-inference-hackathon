# GraphRAG vs Basic RAG — Token-Efficient LLM Inference on TigerGraph

**TigerGraph GraphRAG Inference Hackathon.** Three pipelines answer the same biomedical
questions; the numbers tell the story: **GraphRAG cuts tokens ~32% vs Basic RAG while
*improving* accuracy** — and every GraphRAG answer is produced against a **live TigerGraph
Savanna** graph (verified per question; the run aborts rather than silently fall back).

## 🔄 What's new in this submission

Changes incorporated since the previous submission, addressing reviewer feedback on
proving the graph is real and making the pipeline comparison fair:

- **Live TigerGraph is now enforced by default.** Pipeline 3 performs a health-checked
  round-trip to the cloud graph and **aborts loudly if it is unreachable**, instead of
  silently falling back to a local copy. A "GraphRAG" result can no longer be produced
  without the live graph (opt out only via `P3_REQUIRE_TG=0`).
- **Per-question proof the graph ran.** Every result now records `tg_live`, `tg_host`, and
  the exact `graph_doc_ids` the graph returned, plus an aggregate `tg_live_rate` (1.0 = live
  on every question) and `avg_graph_docs`.
- **The graph now has real retrieval weight.** The graph-document boost was a decorative
  hard-coded `0.5` (vs vector search `5.0`); it is now a configurable `GRAPH_BOOST`
  (default `4.0`) that materially changes which documents are retrieved.
- **Fair, equal context budget.** Basic RAG and GraphRAG now share the same retrieval budget
  (`CONTEXT_NUM_CHUNKS` × `CONTEXT_CHUNK_CHARS`), so neither pipeline is fed more text than
  the other — the accuracy difference reflects retrieval quality, not context size.
- **Accurate chunk reporting.** The retrieved-chunk count now reflects the chunks actually
  placed in the prompt, not the larger reranker candidate pool.
- **Graph ON/OFF ablation support** (`P3_DISABLE_GRAPH`) for transparently measuring the
  graph's contribution.

## 🏆 Headline — BioASQ (biomedical multi-hop QA)

| Pipeline | LLM-as-a-Judge | BERTScore | Avg tokens |
|---|---|---|---|
| P1 — LLM-Only | 90% | 0.385 | 183 |
| P2 — Basic RAG | 92% | 0.358 | 3,375 |
| **P3 — GraphRAG (TigerGraph)** | **96%** | **0.402** | **2,288 (−32%)** |

All three run the **same** answering model (`gpt-4o-mini`) and the **same** LLM-judge.
**GraphRAG leads on both accuracy metrics (96% judge, 0.402 BERTScore) while using 32% fewer
tokens than Basic RAG** — its multi-hop graph traversal + entity-seeded retrieval surface the
answer-bearing passages that pure vector search dilutes. The graph is queried live on every
question (`tg_live_rate = 1.0` in the results file), and Pipeline 3 refuses to produce a
result if TigerGraph is unreachable.

## The three pipelines

| | Pipeline | How it works |
|---|---|---|
| **P1** | LLM-Only | Prompt in, answer out. No retrieval. Worst-case baseline. |
| **P2** | Basic RAG | ChromaDB vector retrieval + cross-encoder rerank → LLM. The baseline to beat. |
| **P3** | **GraphRAG** | Gemini-extracted knowledge graph in **TigerGraph Savanna** → multi-hop BFS + entity-seeded retrieval + rerank → LLM synthesis. |

## Architecture

![Architecture](architecture.png)

Corpus → **ChromaDB** (P2 index) and → **Gemini KG extraction** → **TigerGraph Savanna** (P3 graph).
Full source for the diagram: [`architecture.html`](architecture.html).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env          # then fill in the keys
```

**Keys in `.env`:** `OPENAI_API_KEY` (answering LLM, gpt-4o-mini) · `HF_TOKEN` (LLM-judge) ·
`GOOGLE_API_KEY` (Gemini KG extraction) · `TIGERGRAPH_HOST` / `TIGERGRAPH_SECRET` (Savanna).
Get a free TigerGraph Savanna instance at [tgcloud.io](https://tgcloud.io).

## Run the full pipeline

```bash
# 1. Build the corpus + ChromaDB vector index (Pipeline 2)
DATASET=bioasq python build_corpus.py

# 2. Extract the knowledge graph with Gemini (Pipeline 3) — resumable + checkpointed
DATASET=bioasq GEMINI_INDEX_DIR=pipeline3_bioasq/index python build_entity_index_gemini.py

# 3. Ingest the KG into TigerGraph Savanna
TG_INDEX_DIR=pipeline3_bioasq/index TIGERGRAPH_GRAPH=bioasq \
  python tigergraph_cloud_ingest.py --fresh

# 4. Run the 3-pipeline benchmark (LLM-judge + BERTScore + tokens/latency)
DATASET=bioasq python run_benchmark.py

# 5. Launch the side-by-side comparison dashboard
streamlit run app.py
```

## Dashboard

**Live:** https://knightwalker96-graphrag-inference-hackathon-app-k9q1vq.streamlit.app
(or run locally with `streamlit run app.py`).

The interactive comparison dashboard has two tabs:

- **Live Query** — type any question; all three pipelines answer it live and show their
  answers + per-query tokens, latency, cost, and chunks retrieved, side by side.
- **Benchmark Results** — loads `results/benchmark_results.json` (written by every
  `run_benchmark.py` run) and renders the full comparison: accuracy (LLM-judge + BERTScore),
  token/latency charts, the BERTScore distribution, and per-question breakdowns.

## Evaluation (per the hackathon spec)

- **LLM-as-a-Judge** — `meta-llama/Llama-3.1-8B-Instruct` returns PASS/FAIL per answer.
- **BERTScore** — semantic similarity to the reference answer (`rescale_with_baseline=True`).
- **Cost** — prompt + completion tokens, latency, and $/query per pipeline.

Implemented in [`accuracy.py`](accuracy.py) and [`metrics.py`](metrics.py).

## Repository layout

```
config.py                      central config + per-dataset paths
main.py                        3-pipeline benchmark orchestration
run_benchmark.py               dataset-configurable entry point
app.py                         Streamlit comparison dashboard
accuracy.py / metrics.py       evaluation + cost tracking
llm_client.py                  LLM provider wrapper
chroma_store.py                ChromaDB build/query (Pipeline 2)
build_corpus.py                corpus + ChromaDB builder
build_entity_index_gemini.py   Gemini KG extraction (Pipeline 3)
tigergraph_cloud_ingest.py     TigerGraph Savanna schema + ingest
pipeline1_llm_only.py          Pipeline 1
pipeline2_basic_rag.py         Pipeline 2
pipeline3_tigergraph_cloud/    Pipeline 3 (GraphRAG retrieval + synthesis)
data/  embeddings/             dataset loaders + embedder
```

> Large generated artifacts (`chroma_db_bioasq/`, `pipeline3_bioasq/index/`) are gitignored —
> regenerate them with steps 1–2 above.

## Credits

Pipeline 3 is built on the [TigerGraph GraphRAG](https://github.com/tigergraph/graphrag) approach,
running on **TigerGraph Savanna**. Dataset: [`rag-mini-bioasq`](https://huggingface.co/datasets/enelpol/rag-mini-bioasq).
