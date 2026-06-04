"""HotpotQA dataset loader and ground-truth IO."""

from __future__ import annotations
import json
import os

from datasets import load_dataset as hf_load
from langchain_core.documents import Document
from tqdm import tqdm

import config


def load_hotpotqa() -> tuple[list[Document], list[dict[str, str]]]:
    split = config.HOTPOTQA_SPLIT
    max_docs = config.MAX_DOCS_TO_INDEX

    print(f"[loader] Loading HF hotpot_qa ({split} split, first {max_docs} examples)...")
    ds = hf_load("hotpot_qa", "distractor", split=split)

    documents: list[Document] = []
    qa_pairs: list[dict[str, str]] = []
    seen_titles: set[str] = set()

    for i, ex in enumerate(tqdm(ds, total=min(max_docs, len(ds)), desc="hotpotqa")):
        if i >= max_docs:
            break
        qa_pairs.append({"question": ex["question"], "answer": ex["answer"]})
        for title, sentences in zip(ex["context"]["title"], ex["context"]["sentences"]):
            if title in seen_titles:
                continue
            seen_titles.add(title)
            documents.append(Document(
                page_content=" ".join(sentences),
                metadata={"title": title, "source": "hotpotqa", "example_index": i},
            ))

    total_chars = sum(len(d.page_content) for d in documents)
    print(f"[loader] Loaded {len(documents)} unique passages, {len(qa_pairs)} QA pairs.")
    print(f"[loader] Estimated corpus: ~{total_chars // 4:,} tokens ({total_chars:,} chars)")
    return documents, qa_pairs


def save_ground_truth(qa_pairs: list[dict], path: str | None = None) -> None:
    path = path or config.GROUND_TRUTH_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"[loader] Ground truth saved to {path}")


def load_ground_truth(path: str | None = None) -> list[dict]:
    path = path or config.GROUND_TRUTH_PATH
    with open(path) as f:
        return json.load(f)
