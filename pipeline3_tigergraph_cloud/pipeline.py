"""Pipeline 3: GraphRAG on TigerGraph Savanna.

Multi-hop BFS over the knowledge graph runs on TigerGraph Savanna; document content
lookup uses a local entity/doc cache; ChromaDB adds a semantic retrieval signal.

Pre-requisites:
  1. python tigergraph_cloud_ingest.py
  2. TIGERGRAPH_HOST, TIGERGRAPH_GRAPH, TIGERGRAPH_USERNAME,
     TIGERGRAPH_PASSWORD, TIGERGRAPH_SECRET set in .env
"""

from __future__ import annotations
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import spacy
import pyTigerGraph as tg
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

import config
from metrics import CallMetrics, MetricsTimer
from llm_client import get_llm_response

load_dotenv()

TG_HOST     = os.getenv("TIGERGRAPH_HOST", "")
TG_GRAPH    = os.getenv("TIGERGRAPH_GRAPH", "HotpotQA")
TG_USERNAME = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
TG_PASSWORD = os.getenv("TIGERGRAPH_PASSWORD", "")
TG_SECRET   = os.getenv("TIGERGRAPH_SECRET", "")

INDEX_DIR      = os.getenv("P3_INDEX_DIR", config.ds_paths()["index_dir"])
ENTITY_LABELS  = {"PERSON", "ORG", "GPE", "NORP", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LOC"}
HUB_MENTION_THRESHOLD = 700
TG_HUB_MAX = 100
# Gemini relationship edges mostly have count=1, unlike spaCy co-occurrence (often >=5).
# Set TG_MIN_CO_COUNT=1 when querying a Gemini-built graph.
TG_MIN_CO_COUNT = int(os.getenv("TG_MIN_CO_COUNT", "5"))
MAX_CHUNK_CHARS = 800
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "7"))
NUM_HOPS = int(os.getenv("NUM_HOPS", "1"))   # BFS depth for multi_hop_expand (set 2 for multi-hop datasets)
GRAPH_BRIDGE_RESERVE = int(os.getenv("GRAPH_BRIDGE_RESERVE", "0"))  # ctx slots reserved for graph-traversed docs (bypass reranker demotion)
CHROMA_ENTITY_SEED_LIMIT = 16

RERANKER_MODEL       = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_N       = 20   # candidates retrieved before reranking
RERANKER_INPUT_CHARS = 600  # passage chars fed to reranker (model has 512-token limit)

# Wh-word → expected answer entity types. Docs containing entities of these types
# get a small scoring boost when the question asks "who/where/when/which".
WH_TYPE_HINTS = {
    "who":   {"PERSON", "ORG", "NORP"},
    "whom":  {"PERSON", "ORG"},
    "whose": {"PERSON", "ORG"},
    "where": {"GPE", "LOC", "FAC"},
    "when":  {"DATE", "TIME", "CARDINAL"},
    "which": {"PERSON", "ORG", "GPE", "WORK_OF_ART", "PRODUCT"},
}
WH_TYPE_BOOST = 0.5

# Subset reliable enough for hard server-side BFS filtering. Excludes "when"
# (spaCy is inconsistent on dates) and "which" (answer-type space too broad).
TRUSTED_WH_TYPE_HINTS = {
    "who":   {"PERSON", "ORG", "NORP"},
    "whom":  {"PERSON", "ORG"},
    "whose": {"PERSON", "ORG"},
    "where": {"GPE", "LOC", "FAC"},
}

_conn: tg.TigerGraphConnection | None = None
_nlp = None
_cross_encoder: CrossEncoder | None = None
_entity_to_docs: dict | None = None
_doc_to_entities: dict | None = None
_doc_content:    dict | None = None
_entity_names:   set  | None = None
_entity_mentions: dict | None = None
_entity_types:    dict | None = None
_alias_siblings:  dict | None = None
_entity_rels:     dict | None = None   # entity name -> [(source, relation, target), ...]
_tg_available:   bool = True
_tg_cache:       dict = {}   # frozenset(seeds) + params → (entities, doc_ids)

ALIAS_SEED_CAP = 25


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print(f"[p3-tg-cloud] Loading cross-encoder ({RERANKER_MODEL})…")
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


def _rerank(question: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return candidates
    ce = _get_cross_encoder()
    pairs = [(question, r["content"][:RERANKER_INPUT_CHARS]) for r in candidates]
    scores = ce.predict(pairs)
    return [r for _, r in sorted(zip(scores, candidates), key=lambda x: -x[0])]


def _graph_bridge_boost(reranked: list[dict], graph_doc_ids: set[str],
                        reserve: int, ctx: int) -> list[dict]:
    """Guarantee up to `reserve` graph-traversed docs land inside the top-`ctx` context
    window, even if the cross-encoder demoted them. The hop-2 'bridge' paper that answers
    a multi-hop question is rarely similar to the question text, so the reranker drops it;
    this pins the best graph bridges back in while keeping the top query-relevant docs."""
    if reserve <= 0 or not graph_doc_ids:
        return reranked
    head = reranked[: max(0, ctx - reserve)]                 # top reranked (query-similar)
    head_ids = {r["doc_id"] for r in head}
    bridges = [r for r in reranked
               if r["doc_id"] in graph_doc_ids and r["doc_id"] not in head_ids][:reserve]
    if not bridges:
        return reranked
    bridge_ids = {r["doc_id"] for r in bridges}
    rest = [r for r in reranked if r["doc_id"] not in head_ids and r["doc_id"] not in bridge_ids]
    return head + bridges + rest


def _load_local_index() -> None:
    global _nlp, _entity_to_docs, _doc_to_entities, _doc_content
    global _entity_names, _entity_mentions, _entity_types, _alias_siblings, _entity_rels
    if _doc_content is not None:
        return

    idx = Path(INDEX_DIR)
    if not (idx / "doc_content.json").exists():
        raise FileNotFoundError(
            f"Local index not found at {INDEX_DIR}/. "
            "Expected doc_content.json, entity_to_docs.json (and optionally "
            "entity_types.json, wikidata_aliases.json) in that folder."
        )

    print("[p3-tg-cloud] Loading local index (once)…")
    _nlp = spacy.load("en_core_web_sm")
    with open(idx / "entity_to_docs.json") as f:
        _entity_to_docs = json.load(f)
    with open(idx / "doc_content.json") as f:
        _doc_content = json.load(f)
    _entity_names = set(_entity_to_docs.keys())
    _entity_mentions = {name: len(set(docs)) for name, docs in _entity_to_docs.items()}

    type_cache = idx / "entity_types.json"
    if type_cache.exists():
        with open(type_cache) as f:
            _entity_types = json.load(f)
    else:
        _entity_types = {}

    # Wikidata alias siblings: entity name → list of OTHER names with the same
    # QID. Used to broaden BFS reach across alias-equivalent entities.
    _alias_siblings = {}
    alias_cache = idx / "wikidata_aliases.json"
    if alias_cache.exists():
        with open(alias_cache) as f:
            raw = json.load(f)
        by_qid: dict[str, list[str]] = {}
        for name, info in raw.items():
            qid = info.get("qid", "")
            if qid:
                by_qid.setdefault(qid, []).append(name)
        for qid, group in by_qid.items():
            if len(group) <= 1:
                continue
            for n in group:
                _alias_siblings[n] = [s for s in group if s != n]

    _doc_to_entities = {}
    for name, docs in _entity_to_docs.items():
        for doc_id in set(docs):
            _doc_to_entities.setdefault(doc_id, []).append(name)

    # Relationship triples (for feeding the LLM the graph reasoning chains)
    _entity_rels = {}
    rel_path = idx / "relationships.json"
    if rel_path.exists():
        with open(rel_path) as f:
            for r in json.load(f):
                s, rel, t = r.get("source"), r.get("relation"), r.get("target")
                if s and t:
                    trip = (s, rel or "related_to", t)
                    _entity_rels.setdefault(s, []).append(trip)
                    _entity_rels.setdefault(t, []).append(trip)

    print(
        f"[p3-tg-cloud] {len(_entity_to_docs):,} entities, "
        f"{len(_doc_content):,} docs, "
        f"{len(_doc_to_entities):,} doc→entity mappings, "
        f"{sum(1 for v in _entity_types.values() if v):,} typed, "
        f"{len(_alias_siblings):,} alias-linked"
    )


def _get_connection() -> tg.TigerGraphConnection | None:
    global _conn, _tg_available
    if not _tg_available:
        return None
    if _conn is not None:
        return _conn
    if not TG_HOST:
        print("[p3-tg-cloud] TIGERGRAPH_HOST not set — running without cloud graph (local fallback)")
        _tg_available = False
        return None
    try:
        print(f"[p3-tg-cloud] Connecting to TigerGraph at {TG_HOST}…")
        conn = tg.TigerGraphConnection(
            host=TG_HOST,
            graphname=TG_GRAPH,
            gsqlSecret=TG_SECRET or None,
            username=TG_USERNAME,
            password=TG_PASSWORD,
        )
        _conn = conn
        print("[p3-tg-cloud] Connected.")
        return _conn
    except Exception as e:
        print(f"[p3-tg-cloud] Connection failed ({e}) — falling back to local-only retrieval")
        _tg_available = False
        return None


def _extract_entities(question: str) -> list[str]:
    doc = _nlp(question)
    found: set[str] = set()
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            k = ent.text.strip().lower()
            if len(k) > 2 and not k.isdigit():
                found.add(k)
    for chunk in doc.noun_chunks:
        k = chunk.text.strip().lower()
        if len(k) > 3 and k in _entity_names:
            found.add(k)
    # Domain seeds: match question n-grams (1-3 words) directly against the
    # entity index. spaCy's general NER misses biomedical entities (genes, drugs);
    # this surfaces graph entities like 'pcsk9', 'inclisiran', 'glioblastoma' so the
    # graph BFS gets real seeds.
    words = [w.strip(".,?!;:()[]\"'").lower() for w in question.split()]
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i + n]).strip()
            if len(gram) > 2 and gram in _entity_names:
                found.add(gram)
    return list(found)


def _keyword_fallback(question: str, max_seeds: int = 5) -> list[str]:
    words = [w.lower() for w in question.split() if len(w) > 4]
    seeds: list[str] = []
    for name in _entity_names:
        if any(w in name for w in words):
            seeds.append(name)
            if len(seeds) >= max_seeds:
                break
    return seeds


def _title_match_docs(question: str, max_docs: int = 8) -> set[str]:
    """Direct keyword match of question words against document IDs.

    HotpotQA doc IDs are entity names (e.g. 'Deal_Timeball', 'Wollongong_Wolves').
    Two-word matches are high-precision; a single 7+ char distinctive word also qualifies.
    """
    stopwords = {"what", "when", "where", "who", "which", "does", "did", "was", "were",
                 "have", "been", "that", "this", "from", "with", "into", "both", "also",
                 "than", "more", "most", "some", "such", "each", "many", "much", "their",
                 "about", "these", "those", "other", "after", "before", "between", "through"}
    words = [
        w.strip(".,?!\"'()").lower()
        for w in question.split()
        if len(w) > 3 and w.lower().strip(".,?!\"'()") not in stopwords
    ]
    # Long distinctive words (7+ chars) count double — single-word match is enough for them
    strong_words = {w for w in words if len(w) >= 7}
    matched: dict[str, float] = {}
    for doc_id in _doc_content:
        doc_lower = doc_id.lower().replace("_", " ")
        cnt = sum(1 for w in words if w in doc_lower)
        strong_cnt = sum(1 for w in strong_words if w in doc_lower)
        # Accept: 2+ words, OR 1 strong word (7+ chars) that matches
        score = cnt + strong_cnt * 0.5
        if cnt >= 2 or strong_cnt >= 1:
            matched[doc_id] = score
    top = sorted(matched.items(), key=lambda x: -x[1])[:max_docs]
    return {doc_id for doc_id, _ in top}


def _tg_expand(
    seed_names: list[str],
    num_hops: int = 1,
    top_k: int = 20,
    expand_types: set[str] | None = None,
) -> tuple[list[str], set[str]]:
    """Run multi_hop_expand on TigerGraph Cloud.

    Returns (expanded_entities, graph_recommended_doc_ids). Results are cached
    by (frozenset(seeds), num_hops, top_k, expand_types) to avoid duplicate
    TG roundtrips within a batch run.
    """
    conn = _get_connection()
    if conn is None or not seed_names:
        return seed_names, set()

    cache_key = (frozenset(seed_names), num_hops, top_k, frozenset(expand_types or []))
    if cache_key in _tg_cache:
        return _tg_cache[cache_key]

    use_filter = bool(expand_types)
    types_list = sorted(expand_types) if use_filter else ["_"]

    try:
        result = conn.runInstalledQuery("multi_hop_expand", {
            "seed_names":      seed_names,
            "expand_types":    types_list,
            "use_type_filter": use_filter,
            "num_hops":        num_hops,
            "top_k":           top_k,
            "hub_max":         TG_HUB_MAX,
            "doc_limit":       8,
            "min_co_count":    TG_MIN_CO_COUNT,
        })
        if result and isinstance(result, list):
            entities = list(result[0].get("entities", []))
            # Cap to 20 — GSQL SetAccum ignores LIMIT so all matched docs accumulate
            raw_docs = list(result[1].get("docs", [])) if len(result) > 1 else []
            docs = set(raw_docs[:20])
            _tg_cache[cache_key] = (entities, docs)
            return entities, docs
    except Exception as e:
        print(f"[p3-tg-cloud] Query error ({e}), using seed entities only")
    return seed_names, set()


def _resolve_doc_id(title: str) -> str | None:
    """Map a chroma hit title to a doc_content key. Gemini index uses RAW titles;
    older spaCy index used underscore form — try both."""
    if title in _doc_content:
        return title
    alt = title.replace(" ", "_").replace("/", "_")[:120]
    return alt if alt in _doc_content else None


def _chroma_doc_ids(question: str, vectorstore, k: int = 15) -> set[str]:
    if vectorstore is None:
        return set()
    try:
        # MMR gives diverse docs — critical for 2-hop questions needing bridge + answer doc
        hits = vectorstore.max_marginal_relevance_search(question, k=k, fetch_k=50)
        ids: set[str] = set()
        for hit in hits:
            did = _resolve_doc_id(hit.metadata.get("title", ""))
            if did:
                ids.add(did)
        return ids
    except Exception:
        try:
            hits = vectorstore.similarity_search(question, k=k)
            ids = set()
            for hit in hits:
                did = _resolve_doc_id(hit.metadata.get("title", ""))
                if did:
                    ids.add(did)
            return ids
        except Exception:
            return set()


def _chroma_entity_seeds(chroma_doc_ids: set[str]) -> list[str]:
    if not chroma_doc_ids or _doc_to_entities is None:
        return []
    seeds: set[str] = set()
    for doc_id in chroma_doc_ids:
        for name in _doc_to_entities.get(doc_id, [])[:CHROMA_ENTITY_SEED_LIMIT]:
            if _entity_mentions.get(name, 0) <= HUB_MENTION_THRESHOLD:
                seeds.add(name)
    return list(seeds)


def _expand_with_aliases(seeds: set[str], cap: int = ALIAS_SEED_CAP) -> set[str]:
    if not _alias_siblings or not seeds:
        return seeds
    expanded = set(seeds)
    for s in list(seeds):
        for sibling in _alias_siblings.get(s, []):
            if sibling in _entity_names and sibling not in expanded:
                expanded.add(sibling)
                if len(expanded) >= cap:
                    return expanded
    return expanded



def _expected_answer_types(question: str) -> set[str]:
    q_lower = question.lower()
    types: set[str] = set()
    for wh, tset in WH_TYPE_HINTS.items():
        if re.search(rf"\b{wh}\b", q_lower):
            types |= tset
    return types


def _trusted_expand_types(question: str) -> set[str]:
    q_lower = question.lower()
    types: set[str] = set()
    for wh, tset in TRUSTED_WH_TYPE_HINTS.items():
        if re.search(rf"\b{wh}\b", q_lower):
            types |= tset
    return types


def _score_and_retrieve(
    seed_entities: set[str],
    expanded_entities: list[str],
    chroma_ids: set[str],
    graph_doc_ids: set[str],
    title_ids: set[str],
    expected_types: set[str],
    top_k: int = 8,
) -> list[dict]:
    """Score: seed_hits + chroma_boost + title_boost + graph_boost + wh_type_boost."""
    candidate_docs: set[str] = set(chroma_ids) | set(graph_doc_ids) | set(title_ids)
    for name in expanded_entities:
        for doc_id in set(_entity_to_docs.get(name, [])):
            candidate_docs.add(doc_id)

    relevant_set: set[str] = set(seed_entities) | set(expanded_entities)

    doc_scores: dict[str, float] = {}
    for doc_id in candidate_docs:
        doc_ents = _doc_to_entities.get(doc_id, [])
        seed_hits = sum(1.0 for e in doc_ents if e in seed_entities)
        chroma_boost = 5.0 if doc_id in chroma_ids else 0.0
        title_boost = 3.5 if doc_id in title_ids else 0.0
        graph_boost = 0.5 if doc_id in graph_doc_ids else 0.0

        wh_boost = 0.0
        if expected_types and _entity_types:
            type_hits = sum(
                1 for e in doc_ents
                if e in relevant_set
                and _entity_types.get(e, "") in expected_types
            )
            wh_boost = min(1.5, type_hits * WH_TYPE_BOOST)

        score = seed_hits + chroma_boost + title_boost + graph_boost + wh_boost
        if score > 0:
            doc_scores[doc_id] = score

    sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]
    return [
        {"doc_id": d, "content": _doc_content.get(d, ""), "score": s}
        for d, s in sorted_docs
        if _doc_content.get(d)
    ]


MAX_TRIPLES = 15


def _relevant_triples(entities: list[str], max_n: int = MAX_TRIPLES) -> list[str]:
    """Graph reasoning chains involving the (seed+expanded) entities — fed to the
    LLM so it can reason over relationships, not just passages (#2: real GraphRAG)."""
    if not _entity_rels:
        return []
    seen: set = set()
    out: list[str] = []
    for e in entities:
        for (s, rel, t) in _entity_rels.get(e, []):
            if (s, rel, t) in seen:
                continue
            seen.add((s, rel, t))
            out.append(f"{s} —{rel}→ {t}")
            if len(out) >= max_n:
                return out
    return out


def _build_prompt(question: str, retrieved: list[dict], triples: list[str] | None = None) -> str:
    passages = "\n\n---\n\n".join(
        f"[Passage {i+1} | {r['doc_id']}]\n{r['content'][:MAX_CHUNK_CHARS]}"
        for i, r in enumerate(retrieved[:MAX_CONTEXT_CHUNKS])
    )
    rel_block = ""
    if triples:
        rel_block = ("Knowledge-graph relationships (entity —relation→ entity):\n"
                     + "\n".join(f"- {t}" for t in triples) + "\n\n")
    return (
        f"{config.SYSTEM_PROMPT}\n"
        f"{rel_block}"
        f"Context:\n{passages}\n\n"
        f"Question: {question}\nAnswer:"
    )


def run_single(
    question: str,
    vectorstore=None,
    num_hops: int = 2,
) -> CallMetrics:
    _load_local_index()
    t_start = time.perf_counter()  # start BEFORE TigerGraph BFS

    # NER, ChromaDB retrieval and title-matching are independent — run them concurrently
    # so only the TigerGraph BFS (which needs the seeds) stays on the critical path.
    def _ner_seeds():
        seeds = _extract_entities(question) or _keyword_fallback(question)
        return [e for e in seeds if _entity_mentions.get(e, 0) <= HUB_MENTION_THRESHOLD]

    def _chroma():
        cids = _chroma_doc_ids(question, vectorstore)
        return cids, _chroma_entity_seeds(cids)

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_ner, f_chroma = ex.submit(_ner_seeds), ex.submit(_chroma)
        f_title = ex.submit(_title_match_docs, question)
        ner_seeds = f_ner.result()
        chroma_ids, chroma_seeds = f_chroma.result()
        title_ids = f_title.result()

    seed_entities = {*ner_seeds, *chroma_seeds}
    # Type filter for "where"/"who" questions reduces BFS noise
    expand_types = _trusted_expand_types(question)
    expanded, graph_doc_ids = _tg_expand(
        list(seed_entities),
        num_hops=NUM_HOPS,
        top_k=15,
        expand_types=expand_types,
    )

    expected_types = _expected_answer_types(question)
    candidates = _score_and_retrieve(
        seed_entities, expanded, chroma_ids, graph_doc_ids, title_ids,
        expected_types=expected_types, top_k=RERANKER_TOP_N,
    )
    retrieved = _rerank(question, candidates)
    retrieved = _graph_bridge_boost(retrieved, set(graph_doc_ids), GRAPH_BRIDGE_RESERVE, MAX_CONTEXT_CHUNKS)

    # Optionally surface graph relationship chains for the LLM (set P3_GRAPH_TRIPLES=1).
    triples = _relevant_triples(list(seed_entities) + list(expanded)) if os.getenv("P3_GRAPH_TRIPLES") else None
    prompt = _build_prompt(question, retrieved, triples=triples)
    with MetricsTimer("Pipeline 3: GraphRAG (TigerGraph Cloud)", question) as timer:
        answer, _ = get_llm_response(prompt)
    metrics = timer.record_manual(prompt, answer, retrieved_chunks=len(retrieved))
    metrics.latency_seconds = round(time.perf_counter() - t_start, 3)  # true end-to-end
    return metrics


def run_batch(
    questions: list[str],
    vectorstore=None,
    num_hops: int = 2,
    verbose: bool = False,
) -> list[CallMetrics]:
    from tqdm import tqdm
    _load_local_index()
    _get_connection()

    results: list[CallMetrics] = []
    for q in tqdm(questions, desc="Pipeline 3 (TigerGraph Cloud GraphRAG)"):
        m = run_single(q, vectorstore=vectorstore, num_hops=num_hops)
        results.append(m)
        if verbose:
            print(f"  Q: {q[:70]}…")
            print(f"  A: {m.answer[:100]}…")
            print(f"  Chunks: {m.retrieved_chunks} | Tokens: {m.total_tokens} | Latency: {m.latency_seconds:.2f}s")
    return results
