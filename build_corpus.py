"""Dataset-configurable corpus builder (run before extraction/benchmark).

Builds, for DATASET (env), at the standard per-dataset paths:
  pipeline3_<ds>/index/corpus.json   — {doc_id: text}  (shared by P2 chroma + P3 extraction)
  results/ground_truth_<ds>.json     — [{question, answer}]  eval set
  chroma_db_<ds>/main                — ChromaDB for Pipeline 2

Usage:
  DATASET=bioasq TARGET_CORPUS_TOKENS=50000 python build_corpus.py
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import config
from data.datasets import load_corpus_qa


def main() -> None:
    ds = config.DATASET
    p = config.ds_paths(ds)
    target = config.TARGET_CORPUS_TOKENS
    # Biomedical datasets -> PubMedBERT embedder (must match the benchmark query embedder)
    if ds in config.BIOMED_DATASETS and not os.getenv("EMBEDDING_MODEL"):
        config.EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
    print(f"[build_corpus] dataset={ds} target_tokens={target:,} embedder={config.EMBEDDING_MODEL}")

    docs, qa = load_corpus_qa(ds, target)

    # corpus.json (shared input for chroma + Gemini extraction)
    idx_dir = Path(p["index_dir"]); idx_dir.mkdir(parents=True, exist_ok=True)
    corpus = {d.metadata["title"]: d.page_content for d in docs}
    json.dump(corpus, open(idx_dir / "corpus.json", "w"))
    print(f"[build_corpus] wrote {len(corpus):,} docs -> {idx_dir/'corpus.json'}")

    # ground truth (eval set)
    os.makedirs(os.path.dirname(p["ground_truth"]), exist_ok=True)
    json.dump(qa, open(p["ground_truth"], "w"), indent=2)
    alens = [len(x["answer"].split()) for x in qa]
    stats = (f"avg={sum(alens)//len(alens)}, max={max(alens)}" if alens
             else "none yet — eval generated post-extraction")
    print(f"[build_corpus] wrote {len(qa):,} qa -> {p['ground_truth']} (answer words: {stats})")

    # ChromaDB for P2
    config.CHROMA_PERSIST_DIR = p["chroma"]
    config.CHROMA_COLLECTION_NAME = p["collection"]
    import shutil
    if os.path.exists(p["chroma"]):
        shutil.rmtree(p["chroma"])
    from chroma_store import build_vectorstore
    build_vectorstore(docs)
    print(f"[build_corpus] ChromaDB -> {p['chroma']} (collection {p['collection']})")
    print(f"[build_corpus] ✅ done. Next: Gemini extraction with GEMINI_INDEX_DIR={p['index_dir']}")


if __name__ == "__main__":
    main()
