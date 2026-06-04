"""Dataset-configurable corpus + QA loader.

Returns (documents, qa_pairs) where:
  documents = list[langchain Document]  (page_content = passage, metadata title = doc_id)
  qa_pairs  = list[{"question","answer"}]

Datasets:
  hotpotqa  — multi-hop Wikipedia (short factoid answers)
  bioasq    — biomedical PubMed QA (rag-mini-bioasq), longer answers. The mini
              corpus isn't ID-aligned to the QA, so we build a focused, answerable
              corpus by TF-IDF aligning each question to its top passages.
"""
from __future__ import annotations
import warnings
from pathlib import Path

from langchain_core.documents import Document

warnings.filterwarnings("ignore")


def _est_tokens(s: str) -> int:
    return len(s) // 4


def load_bioasq(target_tokens: int, max_questions: int = 120,
                passages_per_q: int = 5) -> tuple[list[Document], list[dict]]:
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    print("[bioasq] loading rag-mini-bioasq…")
    corp = load_dataset("enelpol/rag-mini-bioasq", "text-corpus", split="test")
    qa = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages", split="test")
    passages = list(corp["passage"])
    pids = [str(i) for i in corp["id"]]

    # Take passages corpus-wide until the budget is hit (TF-IDF alignment below is
    # only for tiny demo corpora).
    if target_tokens > 2_000_000:
        docs, toks = [], 0
        for p, pid in zip(passages, pids):
            if not p.strip():
                continue
            docs.append(Document(page_content=p, metadata={"title": pid, "source": "bioasq"}))
            toks += _est_tokens(p)
            if toks >= target_tokens:
                break
        qa_pairs = [{"question": q, "answer": a}
                    for q, a in zip(qa["question"][:300], qa["answer"][:300])]
        print(f"[bioasq] loaded {len(docs):,} passages")
        return docs, qa_pairs

    print(f"[bioasq] TF-IDF over {len(passages):,} passages…")
    vec = TfidfVectorizer(stop_words="english", max_features=50000)
    P = vec.fit_transform(passages)

    corpus_order: list[int] = []
    in_corpus: set[int] = set()
    qa_pairs: list[dict] = []
    toks = 0
    for i in range(min(max_questions, len(qa))):
        q, a = qa["question"][i], qa["answer"][i]
        sims = linear_kernel(vec.transform([q + " " + a]), P).ravel()
        top = sims.argsort()[::-1][:passages_per_q]
        added = False
        for idx in top:
            if sims[idx] <= 0:
                continue
            idx = int(idx)
            if idx not in in_corpus:
                in_corpus.add(idx); corpus_order.append(idx); toks += _est_tokens(passages[idx])
            added = True
        if added:
            qa_pairs.append({"question": q, "answer": a})
        if toks >= target_tokens:
            break

    documents = [Document(page_content=passages[idx],
                          metadata={"title": pids[idx], "source": "bioasq"})
                 for idx in corpus_order]
    print(f"[bioasq] corpus={len(documents):,} passages (~{toks:,} tokens), qa={len(qa_pairs):,}")
    return documents, qa_pairs


def load_jsonl_dataset(dataset: str, target_tokens: int) -> tuple[list[Document], list[dict]]:
    """File-based dataset at datasets/<dataset>/{corpus,eval_set}.jsonl.

    corpus.jsonl: {"title": doc_id, "text": passage}   (or "id"/"text"/"content"/"passage")
    eval_set.jsonl: {"question": ..., "answer": ...}    (extra fields like id/type/context_docs ignored)
    """
    import json
    base = Path("datasets") / dataset
    cpath, epath = base / "corpus.jsonl", base / "eval_set.jsonl"
    if not cpath.exists():
        raise FileNotFoundError(f"Expected {cpath} (put corpus.jsonl + eval_set.jsonl in datasets/{dataset}/)")

    def field(r, *names, default=""):
        for n in names:
            if n in r and r[n]:
                return r[n]
        return default

    docs, toks, seen = [], 0, set()
    for line in open(cpath):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        doc_id = str(field(r, "title", "id", "doc_id", "_id"))
        text = str(field(r, "text", "content", "passage", "body"))
        if not text.strip() or doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append(Document(page_content=text, metadata={"title": doc_id, "source": dataset}))
        toks += _est_tokens(text)
        if target_tokens and toks >= target_tokens:
            break

    qa_pairs = []
    if epath.exists():
        for line in open(epath):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q, a = field(r, "question", "query"), field(r, "answer", "ideal_answer", "answers")
            if isinstance(a, list):
                a = " ".join(map(str, a))
            if q and a:
                qa_pairs.append({"question": str(q), "answer": str(a)})
    print(f"[{dataset}] corpus={len(docs):,} docs (~{toks:,} tokens), qa={len(qa_pairs):,}")
    return docs, qa_pairs


def load_corpus_qa(dataset: str, target_tokens: int) -> tuple[list[Document], list[dict]]:
    if dataset == "bioasq":
        return load_bioasq(target_tokens)
    if dataset == "hotpotqa":
        from data.loader import load_hotpotqa
        return load_hotpotqa()
    # File-based dataset: datasets/<dataset>/{corpus,eval_set}.jsonl
    if (Path("datasets") / dataset / "corpus.jsonl").exists():
        return load_jsonl_dataset(dataset, target_tokens)
    raise ValueError(f"Unknown dataset {dataset!r} (no datasets/{dataset}/corpus.jsonl)")
