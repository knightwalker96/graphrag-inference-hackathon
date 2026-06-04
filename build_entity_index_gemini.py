"""Build the Pipeline-3 index using GEMINI for entity + relationship extraction.

Drop-in replacement for build_entity_index.py's spaCy NER. For each HotpotQA
passage, calls Gemini (flash-lite) with structured-JSON output to extract:
  - typed entities  (name, type)
  - relationships   (source, relation, target)

Outputs (pipeline3_gemini/index/), same contract as the spaCy index + one new file:
  entity_to_docs.json  — {entity_name: [doc_id, ...]}
  doc_content.json     — {doc_id: full_text}
  entity_types.json    — {entity_name: TYPE}
  relationships.json    — [{"source","relation","target","count"}]   (NEW: real edges)

Concurrent + checkpointed (resumable): re-running skips docs already in doc_content.json.

Env:
  NUM_DOCS         number of passages to process (default 300)
  GEMINI_MODEL     default gemini-2.5-flash-lite
  EXTRACT_WORKERS  thread pool size (default 8)
"""
from __future__ import annotations
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, ".")
import config

OUT_DIR     = Path(os.getenv("GEMINI_INDEX_DIR", "pipeline3_gemini/index"))
NUM_DOCS    = int(os.getenv("NUM_DOCS", "0"))   # 0 = all docs in the corpus
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
WORKERS     = int(os.getenv("EXTRACT_WORKERS", "8"))
CHECKPOINT_EVERY = 200
MAX_DOC_CHARS = 8000          # cap very long passages
PER_CALL_RETRIES = 4
BATCH_SIZE = int(os.getenv("GEMINI_BATCH_SIZE", "1"))   # docs per Gemini call (>1 = batched: far fewer API calls)

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
                "required": ["name", "type"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "relation": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["source", "relation", "target"],
            },
        },
    },
    "required": ["entities", "relationships"],
}

# --- General type normalizer: canonical spaCy-aligned set ---
def _normalize_general(raw: str) -> str:
    t = str(raw).lower().replace("_", " ").strip()
    def has(*words): return any(w in t for w in words)
    if has("person", "people", "individual", "human", "author", "musician", "politician", "athlete"): return "PERSON"
    if has("nation", "ethnic", "religio", "politic group", "demonym"): return "NORP"
    if has("countr", "city", "state", "province", "capital", "town", "village", "geopolit"): return "GPE"
    if has("organ", "institut", "compan", "band", "team", "club", "univers", "college", "school", "agency", "label", "group", "government", "party", "publisher", "studio"): return "ORG"
    if has("build", "road", "bridge", "airport", "stadium", "facilit", "courthouse", "museum", "station", "venue", "arena", "highway", "street"): return "FAC"
    if has("location", "place", "region", "area", "territory", "continent", "mountain", "river", "lake", "sea", "ocean", "island", "park", "geograph"): return "LOC"
    if has("work", "art", "book", "song", "film", "movie", "album", "magazine", "novel", "paint", "show", "series", "game", "publication", "single", "composition", "poem", "play", "tv"): return "WORK_OF_ART"
    if has("event", "championship", "war", "battle", "tournament", "festival", "competition", "election", "season", "olympic", "ceremony", "conflict"): return "EVENT"
    if has("product", "vehicle", "chemical", "compound", "device", "software", "weapon", "aircraft", "car", "drug", "material", "brand", "model"): return "PRODUCT"
    if has("date", "year", "time", "month", "century", "decade", "period", "era"): return "DATE"
    if has("award", "prize", "medal", "honor"): return "WORK_OF_ART"
    return "MISC"


# --- Biomedical (BioASQ) type normalizer: canonical biomedical set ---
def _normalize_biomed(raw: str) -> str:
    t = str(raw).lower().replace("_", " ").strip()
    def has(*words): return any(w in t for w in words)
    if has("gene", "allele", "locus", "mrna", "transcript", "mutation", "snp", "variant"): return "GENE"
    if has("protein", "enzyme", "receptor", "antibody", "kinase", "antigen", "transcription factor", "peptide", "hormone"): return "PROTEIN"
    if has("disease", "disorder", "syndrome", "cancer", "tumor", "tumour", "infection", "carcinoma", "condition", "pathology", "symptom"): return "DISEASE"
    if has("drug", "inhibitor", "agonist", "antagonist", "therapy", "treatment", "medication", "vaccine", "compound", "agent", "antibiotic"): return "DRUG"
    if has("chemical", "molecule", "metabolite", "acid", "ion", "sugar", "lipid", "substrate", "ligand"): return "CHEMICAL"
    if has("pathway", "signaling", "cascade", "process", "mechanism", "regulation", "metabolism", "cycle", "response"): return "PATHWAY"
    if has("cell", "tissue", "organ", "neuron", "lymphocyte", "anatom", "membrane", "nucleus"): return "CELL_ANATOMY"
    if has("organism", "bacteri", "virus", "species", "patient", "human", "mouse", "yeast", "strain"): return "ORGANISM"
    if has("assay", "scale", "score", "test", "measure", "biomarker", "index", "technique", "method", "sequenc"): return "MEASURE"
    return "BIOENTITY"


def normalize_type(raw: str) -> str:
    return _normalize_biomed(raw) if config.DATASET in config.BIOMED_DATASETS else _normalize_general(raw)


_PROMPT_GENERAL = (
    "Extract a knowledge graph from the passage. Identify concrete real-world "
    "entities (people, organizations, locations, works of art, products, events, "
    "dates) and the meaningful relationships between them. Ignore generic words. "
    "Use the canonical entity name as it appears. Relations should be short verb "
    "phrases (e.g. founded, located_in, member_of, released, directed_by, "
    "married_to, part_of). Only include relationships where BOTH endpoints are "
    "entities you listed.\n\nPassage:\n"
)

_PROMPT_BIOMED = (
    "Extract a BIOMEDICAL knowledge graph from the passage. Identify biomedical "
    "entities with a 'type' from: GENE, PROTEIN, DISEASE, DRUG, CHEMICAL, PATHWAY, "
    "CELL_ANATOMY, ORGANISM, MEASURE. Use the canonical biomedical name (e.g. gene "
    "symbols like PCSK9, drug names like inclisiran, diseases like glioblastoma). "
    "Extract directed relationships using precise biomedical verbs: inhibits, "
    "activates, treats, causes, associated_with, binds, targets, mutated_in, "
    "expressed_in, regulates, interacts_with, metabolized_by, biomarker_for, "
    "encodes, part_of. Only include relationships where BOTH endpoints are entities "
    "you listed. Ignore citations and generic words.\n\nPassage:\n"
)


def _select_prompt() -> str:
    return _PROMPT_BIOMED if config.DATASET in config.BIOMED_DATASETS else _PROMPT_GENERAL


# Optional external prompt override (keeps dataset-specific prompts out of shared code):
#   EXTRACT_PROMPT_FILE=datasets/<ds>/extract_prompt.txt
_PROMPT_FILE = os.getenv("EXTRACT_PROMPT_FILE")
PROMPT = Path(_PROMPT_FILE).read_text(encoding="utf-8") if _PROMPT_FILE else _select_prompt()


def _get_key() -> str:
    k = os.getenv("GOOGLE_API_KEY", "")
    if not k:
        raise EnvironmentError("GOOGLE_API_KEY not set. Add it to .env (get one at "
                               "https://aistudio.google.com/app/apikey).")
    return k


KEY = _get_key()


def extract_one(text: str) -> dict | None:
    """Call Gemini for one passage; return {'entities':[...], 'relationships':[...]} or None."""
    body = {
        "contents": [{"parts": [{"text": PROMPT + text[:MAX_DOC_CHARS]}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }
    for attempt in range(PER_CALL_RETRIES):
        try:
            r = requests.post(f"{API_URL}?key={KEY}", json=body, timeout=60)
            if r.status_code == 200:
                txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(txt)
            if r.status_code in (429, 500, 503):
                time.sleep(2 ** attempt + 0.5)
                continue
            # other errors: don't retry
            return None
        except Exception:
            time.sleep(2 ** attempt + 0.5)
    return None


# ---- Doc-batching: pack multiple docs per Gemini call (cuts API calls ~BATCH_SIZE×) ----
_BATCH_INSTRUCTIONS = (
    "\n\nYou will be given MULTIPLE documents, each starting with a line '### DOC <i>'. "
    "For EACH document independently, extract its entities and relationships using the "
    "format described above. Return a JSON array; each element is "
    '{"doc_index": <i>, "entities": [...], "relationships": [...]}.\n\n'
)

BATCH_RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "doc_index": {"type": "integer"},
            "entities": RESPONSE_SCHEMA["properties"]["entities"],
            "relationships": RESPONSE_SCHEMA["properties"]["relationships"],
        },
        "required": ["doc_index", "entities", "relationships"],
    },
}


def _batch_header() -> str:
    """Reuse the active prompt's instructions, dropping its trailing single-doc marker."""
    base = PROMPT
    for marker in ("Passage:", "Abstract:", "Text:"):
        idx = base.rfind(marker)
        if idx != -1:
            base = base[:idx]
            break
    return base.rstrip() + _BATCH_INSTRUCTIONS


def extract_batch(texts: list[str]) -> list[dict | None]:
    """Extract entities/rels for a batch of docs in ONE call. Returns list aligned to `texts`."""
    parts = [_batch_header()]
    for i, t in enumerate(texts):
        parts.append(f"### DOC {i}\n{t[:MAX_DOC_CHARS]}\n\n")
    body = {
        "contents": [{"parts": [{"text": "".join(parts)}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": BATCH_RESPONSE_SCHEMA,
        },
    }
    out: list[dict | None] = [None] * len(texts)
    for attempt in range(PER_CALL_RETRIES):
        try:
            r = requests.post(f"{API_URL}?key={KEY}", json=body, timeout=180)
            if r.status_code == 200:
                arr = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
                for el in arr:
                    idx = el.get("doc_index")
                    if isinstance(idx, int) and 0 <= idx < len(texts):
                        out[idx] = {"entities": el.get("entities", []),
                                    "relationships": el.get("relationships", [])}
                return out
            if r.status_code in (429, 500, 503):
                time.sleep(2 ** attempt + 0.5)
                continue
            return out
        except Exception:
            time.sleep(2 ** attempt + 0.5)
    return out


def _accumulate(doc_id, body, result, doc_content, entity_to_docs, entity_types, rel_counts) -> int:
    """Fold one doc's extraction result into the index structures. Returns 1 if the call failed."""
    doc_content[doc_id] = body
    if result is None:
        return 1
    for e in result.get("entities", []):
        name = str(e.get("name", "")).lower().strip()
        if len(name) < 2:
            continue
        entity_to_docs[name].append(doc_id)
        entity_types.setdefault(name, normalize_type(e.get("type", "")))
    for rel in result.get("relationships", []):
        s = str(rel.get("source", "")).lower().strip()
        t = str(rel.get("target", "")).lower().strip()
        rname = str(rel.get("relation", "")).lower().strip()
        if len(s) >= 2 and len(t) >= 2 and rname:
            rel_counts[(s, rname, t)] += 1
    return 0


def _load_corpus() -> dict[str, str]:
    """Corpus = {doc_id: text}. Prefer a pre-built corpus.json (so P2/P3 share the
    exact same corpus); else build it from the configured dataset."""
    cj = os.getenv("GEMINI_CORPUS", "")
    if cj and Path(cj).exists():
        return json.load(open(cj))
    if (OUT_DIR / "corpus.json").exists():
        return json.load(open(OUT_DIR / "corpus.json"))
    if NUM_DOCS and config.DATASET == "hotpotqa":
        config.MAX_DOCS_TO_INDEX = max(50, NUM_DOCS // 7 + 20)
    from data.datasets import load_corpus_qa
    docs, _ = load_corpus_qa(config.DATASET, config.TARGET_CORPUS_TOKENS)
    return {d.metadata["title"]: d.page_content for d in docs}


def main() -> None:
    corpus = _load_corpus()
    items = list(corpus.items())
    if NUM_DOCS > 0:
        items = items[:NUM_DOCS]
    print(f"[gemini-index] model={GEMINI_MODEL} workers={WORKERS} dir={OUT_DIR} docs={len(items)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc_content: dict[str, str] = {}
    entity_to_docs: dict[str, list[str]] = defaultdict(list)
    entity_types: dict[str, str] = {}
    rel_counts: dict[tuple, int] = defaultdict(int)

    # Resume: load any existing checkpoint
    cp = OUT_DIR / "doc_content.json"
    done_ids: set[str] = set()
    if cp.exists():
        doc_content = json.load(open(cp))
        done_ids = set(doc_content)
        if (OUT_DIR / "entity_to_docs.json").exists():
            for n, ds in json.load(open(OUT_DIR / "entity_to_docs.json")).items():
                entity_to_docs[n].extend(ds)
        if (OUT_DIR / "entity_types.json").exists():
            entity_types.update(json.load(open(OUT_DIR / "entity_types.json")))
        if (OUT_DIR / "relationships.json").exists():
            for r in json.load(open(OUT_DIR / "relationships.json")):
                rel_counts[(r["source"], r["relation"], r["target"])] = r["count"]
        print(f"[gemini-index] resuming — {len(done_ids):,} docs already done")

    todo = [(doc_id, text) for doc_id, text in items
            if doc_id not in done_ids and text.strip()]
    print(f"[gemini-index] {len(todo):,} docs to extract")

    lock = threading.Lock()
    n_done = 0
    n_fail = 0

    def save_checkpoint():
        json.dump(doc_content, open(OUT_DIR / "doc_content.json", "w"))
        json.dump({k: v for k, v in entity_to_docs.items()}, open(OUT_DIR / "entity_to_docs.json", "w"))
        json.dump(entity_types, open(OUT_DIR / "entity_types.json", "w"))
        json.dump([{"source": s, "relation": rel, "target": t, "count": c}
                   for (s, rel, t), c in rel_counts.items()],
                  open(OUT_DIR / "relationships.json", "w"))

    if BATCH_SIZE > 1:
        batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
        print(f"[gemini-index] batched: {len(batches):,} calls of up to {BATCH_SIZE} docs/call")
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(extract_batch, [t for _, t in b]): b for b in batches}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Gemini batch"):
                batch = futs[fut]
                results = fut.result()
                with lock:
                    for (doc_id, text), res in zip(batch, results):
                        n_fail += _accumulate(doc_id, text, res, doc_content,
                                              entity_to_docs, entity_types, rel_counts)
                        n_done += 1
                    if n_done % CHECKPOINT_EVERY < BATCH_SIZE:
                        save_checkpoint()
    else:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(extract_one, body): (doc_id, body) for doc_id, body in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Gemini extract"):
                doc_id, body = futs[fut]
                result = fut.result()
                with lock:
                    n_fail += _accumulate(doc_id, body, result, doc_content,
                                          entity_to_docs, entity_types, rel_counts)
                    n_done += 1
                    if n_done % CHECKPOINT_EVERY == 0:
                        save_checkpoint()

    # de-dup doc lists
    for n in entity_to_docs:
        entity_to_docs[n] = sorted(set(entity_to_docs[n]))
    save_checkpoint()

    print(f"\n[gemini-index] ✅ done. docs={len(doc_content):,} failed={n_fail:,} "
          f"entities={len(entity_to_docs):,} relationships={len(rel_counts):,}")
    print(f"[gemini-index] output in {OUT_DIR}/")
    print(f"[gemini-index] next: TG_INDEX_DIR={OUT_DIR} TIGERGRAPH_GRAPH=<name> python tigergraph_cloud_ingest.py --fresh")


if __name__ == "__main__":
    main()
