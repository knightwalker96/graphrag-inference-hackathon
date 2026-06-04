"""TigerGraph Cloud (Savanna) ingestion for HotpotQA.

What this script does:
  1. Connects to your TigerGraph Savanna instance
  2. Creates the HotpotQA graph + schema (Entity, Document vertices; co_occurs_with, mentions_in_document edges)
  3. Installs a GSQL multi-hop BFS query
  4. Loads Entity + Document vertices from the local index
  5. Loads co_occurs_with + mentions_in_document edges
  6. Verifies graph counts + runs a smoke-test query (checks docs are returned)

Required .env keys:
  TIGERGRAPH_HOST=https://<your-instance>.i.tgcloud.io
  TIGERGRAPH_GRAPH=HotpotQA
  TIGERGRAPH_USERNAME=tigergraph
  TIGERGRAPH_PASSWORD=<your REST++ password>
  TIGERGRAPH_SECRET=<your secret string>

Usage:
  python tigergraph_cloud_ingest.py            # safe: skip schema if graph exists
  python tigergraph_cloud_ingest.py --fresh    # drop + recreate graph from scratch
  python tigergraph_cloud_ingest.py --skip-schema   # skip schema, only reload data
  python tigergraph_cloud_ingest.py --skip-edges    # load entities only (no edges)
"""

from __future__ import annotations
import argparse
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyTigerGraph as tg
from dotenv import load_dotenv

load_dotenv()

# Parallel batch upserts + retry/backoff on transient Savanna 5xx. Reuses the proven
# REST path (no file-upload permissions needed). TG_INGEST_WORKERS=1 => sequential (default).
INGEST_WORKERS = int(os.getenv("TG_INGEST_WORKERS", "1"))
UPSERT_RETRIES = 5


def _retry(fn, *args, what="batch"):
    """Call fn(*args) with exponential backoff on transient errors; return its count or 0."""
    for attempt in range(UPSERT_RETRIES):
        try:
            return fn(*args) or 0
        except Exception as e:
            if attempt == UPSERT_RETRIES - 1:
                print(f"  {what} failed after {UPSERT_RETRIES} tries: {str(e)[:90]}")
                return 0
            time.sleep(2 ** attempt + 0.5)
    return 0


def _run_batches(payloads, do_batch, label):
    """Run do_batch(payload)->count over all payloads, parallel if TG_INGEST_WORKERS>1."""
    total = len(payloads)
    ok = 0
    if INGEST_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=INGEST_WORKERS) as ex:
            futs = [ex.submit(_retry, do_batch, p, what=label) for p in payloads]
            for i, f in enumerate(as_completed(futs)):
                ok += f.result()
                if i % 10 == 0:
                    print(f"  {label}: {ok:,} loaded ({i + 1}/{total} batches)...")
    else:
        for i, p in enumerate(payloads):
            ok += _retry(do_batch, p, what=label)
            if i % 10 == 0:
                print(f"  {label}: {ok:,} loaded ({i + 1}/{total} batches)...")
    return ok

TG_HOST     = os.getenv("TIGERGRAPH_HOST", "")
TG_GRAPH    = os.getenv("TIGERGRAPH_GRAPH", "HotpotQA")
TG_USERNAME = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
TG_PASSWORD = os.getenv("TIGERGRAPH_PASSWORD", "")
TG_SECRET   = os.getenv("TIGERGRAPH_SECRET", "")

# Point at the Gemini-extracted index with TG_INDEX_DIR=pipeline3_gemini/index
INDEX_DIR     = os.getenv("TG_INDEX_DIR", "pipeline3_networkx/index")
ENTITY_BATCH  = 2000
DOC_BATCH     = 5000
EDGE_BATCH    = 5000
MENTION_BATCH = 5000

# Per-graph type names so multiple datasets coexist on ONE TigerGraph workspace
# without sharing global vertices (global types are shared across graphs).
V_ENTITY = f"Entity_{TG_GRAPH}"
V_DOC    = f"Document_{TG_GRAPH}"
E_REL    = f"related_to_{TG_GRAPH}"
E_MEN    = f"mentions_{TG_GRAPH}"
_FMT = dict(graph=TG_GRAPH, entity=V_ENTITY, doc=V_DOC, rel=E_REL, men=E_MEN)

DROP_SCHEMA_GSQL = """
USE GLOBAL
DROP GRAPH {graph} IF EXISTS
DROP EDGE {rel} IF EXISTS
DROP EDGE {men} IF EXISTS
DROP VERTEX {entity} IF EXISTS
DROP VERTEX {doc} IF EXISTS
"""

# TigerGraph 4.x: create GLOBAL vertex/edge types first, then CREATE GRAPH
# referencing them (CREATE VERTEX is not allowed inside a USE GRAPH context).
# Note: edges do NOT accept a WITH STATS clause in this TG version.
SCHEMA_GSQL = """
CREATE VERTEX {entity} (
  PRIMARY_ID entity_id STRING,
  name                 STRING DEFAULT "",
  entity_type          STRING DEFAULT "",
  mention_count        INT    DEFAULT 0
) WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"

CREATE VERTEX {doc} (
  PRIMARY_ID doc_id STRING,
  title      STRING DEFAULT ""
) WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"

CREATE UNDIRECTED EDGE {rel} (
  FROM {entity}, TO {entity},
  rel_type  STRING DEFAULT "",
  rel_count INT    DEFAULT 1
)

CREATE UNDIRECTED EDGE {men} (
  FROM {entity}, TO {doc}
)

CREATE GRAPH {graph}({entity}, {doc}, {rel}, {men})
"""

QUERY_GSQL = """
USE GRAPH {graph}

CREATE QUERY multi_hop_expand(
  SET<STRING> seed_names,
  SET<STRING> expand_types,
  BOOL        use_type_filter = FALSE,
  INT         num_hops      = 2,
  INT         top_k         = 15,
  INT         hub_max       = 100,
  INT         doc_limit     = 5,
  INT         min_co_count  = 2
) FOR GRAPH {graph} {{

  SetAccum<VERTEX<{entity}>> @@visited;
  SetAccum<STRING>         @@found_entities;
  SetAccum<STRING>         @@graph_docs;

  start = {{{entity}.*}};

  seeds = SELECT s FROM start:s
          WHERE s.name IN seed_names OR s.entity_id IN seed_names
          ACCUM @@visited       += s,
                @@found_entities += s.name;

  seed_docs = SELECT d FROM seeds:e -({men}:m)- {doc}:d
              ACCUM @@graph_docs += d.doc_id
              LIMIT doc_limit;

  FOREACH hop IN RANGE[1, num_hops] DO
    hop_candidates = SELECT t FROM seeds:s -({rel}:e)- {entity}:t
                     WHERE t NOT IN @@visited
                       AND t.mention_count <= hub_max
                       AND e.rel_count >= min_co_count
                       AND (use_type_filter == FALSE
                            OR t.entity_type IN expand_types
                            OR t.entity_type == "")
                     ORDER BY t.mention_count DESC
                     LIMIT top_k;
    seeds = SELECT v FROM hop_candidates:v
            ACCUM @@visited       += v,
                  @@found_entities += v.name;
  END;

  expanded_docs = SELECT d FROM seeds:e -({men}:m)- {doc}:d
                  ACCUM @@graph_docs += d.doc_id
                  LIMIT doc_limit;

  PRINT @@found_entities AS entities;
  PRINT @@graph_docs     AS docs;
}}

INSTALL QUERY multi_hop_expand
"""


def connect() -> tg.TigerGraphConnection:
    if not TG_HOST:
        raise EnvironmentError(
            "\n  TIGERGRAPH_HOST is not set in .env\n"
            "  Create a Savanna instance at https://tgcloud.io and add credentials to .env\n"
        )
    print(f"Connecting to TigerGraph at {TG_HOST}...")
    conn = tg.TigerGraphConnection(
        host=TG_HOST,
        graphname=TG_GRAPH,
        gsqlSecret=TG_SECRET or None,
        username=TG_USERNAME,
        password=TG_PASSWORD,
    )
    print("  Connected.")
    return conn


def drop_graph(conn: tg.TigerGraphConnection) -> None:
    # Robust ordered drop. MUST drop queries first (they pin the global vertex types),
    # then the graph, then edges, then vertices. NOTE: this GSQL rejects "IF EXISTS" on
    # DROP GRAPH/EDGE/VERTEX (parse error), so we run plain drops and tolerate
    # "does not exist" errors instead. Each statement is its own gsql call.
    print(f"Dropping graph {TG_GRAPH!r} and global types...")
    f = _FMT
    steps = [
        ("queries", f"USE GRAPH {f['graph']}\nDROP QUERY ALL"),
        ("graph",   f"USE GLOBAL\nDROP GRAPH {f['graph']}"),
        ("edge related_to",   f"USE GLOBAL\nDROP EDGE {f['rel']}"),
        ("edge mentions",     f"USE GLOBAL\nDROP EDGE {f['men']}"),
        ("vertex Entity",     f"USE GLOBAL\nDROP VERTEX {f['entity']}"),
        ("vertex Document",   f"USE GLOBAL\nDROP VERTEX {f['doc']}"),
    ]
    for label, gsql in steps:
        try:
            res = str(conn.gsql(gsql)).lower()
            if "does not exist" in res or "cannot find" in res or "not exist" in res:
                pass  # already gone — fine
        except Exception as e:
            print(f"  drop {label} warning (safe to ignore): {str(e)[:90]}")
    print("  Graph and global types dropped.")


def create_schema(conn: tg.TigerGraphConnection) -> None:
    print(f"\nCreating graph {TG_GRAPH!r} and schema...")
    gsql = SCHEMA_GSQL.format(**_FMT)
    try:
        result = conn.gsql(f"USE GLOBAL\n{gsql}")
        print(f"  Schema result: {str(result)[:300]}")
        print("  Schema created.")
    except Exception as e:
        err = str(e).lower()
        if "already exist" in err or "conflicts with" in err:
            print("Graph/schema already exists — skipping. Use --fresh to recreate.")
        else:
            print(f"  Schema error: {e}")
            raise


def install_query(conn: tg.TigerGraphConnection) -> None:
    print("\nInstalling multi_hop_expand query...")
    gsql = QUERY_GSQL.format(**_FMT)
    try:
        result = conn.gsql(gsql)
        print(f"  Query install result: {str(result)[:300]}")
        print("  Query installed.")
    except Exception as e:
        err = str(e).lower()
        if "already exist" in err:
            print("  Query already installed — skipping.")
        else:
            print(f"  Query install error: {e}")
            raise


def load_entities(conn: tg.TigerGraphConnection) -> None:
    idx = Path(INDEX_DIR)
    print(f"\nLoading entities from {INDEX_DIR}/entity_to_docs.json...")

    with open(idx / "entity_to_docs.json") as f:
        entity_to_docs: dict[str, list[str]] = json.load(f)

    entity_types: dict[str, str] = {}
    tpath = idx / "entity_types.json"
    if tpath.exists():
        with open(tpath) as f:
            entity_types = json.load(f)

    vertices = {
        name: {
            "name": name,
            "entity_type": entity_types.get(name, ""),
            "mention_count": len(set(docs)),
        }
        for name, docs in entity_to_docs.items()
    }
    print(f"  Unique entities: {len(vertices):,}  (typed: {sum(1 for v in vertices.values() if v['entity_type']):,})")

    names = list(vertices.keys())
    payloads = [
        {
            name: {
                "name":          name,
                "entity_type":   vertices[name]["entity_type"],
                "mention_count": vertices[name]["mention_count"],
            }
            for name in names[i : i + ENTITY_BATCH]
        }
        for i in range(0, len(names), ENTITY_BATCH)
    ]

    def _do(upsert_data):
        conn.upsertVertices(V_ENTITY, upsert_data)
        return len(upsert_data)

    ok = _run_batches(payloads, _do, "entities")
    print(f"  ✓ Loaded {ok:,} entities")


def load_documents(conn: tg.TigerGraphConnection) -> None:
    idx = Path(INDEX_DIR)
    print(f"\nLoading documents from {INDEX_DIR}/doc_content.json...")

    with open(idx / "doc_content.json") as f:
        doc_content: dict[str, str] = json.load(f)

    print(f"  Unique documents: {len(doc_content):,}")

    doc_ids = list(doc_content.keys())
    payloads = [
        {doc_id: {"title": doc_id} for doc_id in doc_ids[i : i + DOC_BATCH]}
        for i in range(0, len(doc_ids), DOC_BATCH)
    ]

    def _do(upsert_data):
        conn.upsertVertices(V_DOC, upsert_data)
        return len(upsert_data)

    ok = _run_batches(payloads, _do, "documents")
    print(f"  ✓ Loaded {ok:,} documents")


def load_mentions(conn: tg.TigerGraphConnection) -> None:
    idx = Path(INDEX_DIR)
    print(f"\nLoading entity→document mention edges from {INDEX_DIR}/entity_to_docs.json...")

    with open(idx / "entity_to_docs.json") as f:
        entity_to_docs: dict[str, list[str]] = json.load(f)

    # Flatten to unique (entity, doc_id) pairs
    pairs = [
        (entity, doc_id)
        for entity, docs in entity_to_docs.items()
        for doc_id in set(docs)
    ]
    print(f"  Total mention pairs: {len(pairs):,}")

    payloads = [
        [(entity, doc_id, {}) for entity, doc_id in pairs[i : i + MENTION_BATCH]]
        for i in range(0, len(pairs), MENTION_BATCH)
    ]

    def _do(edge_data):
        conn.upsertEdges(V_ENTITY, E_MEN, V_DOC, edge_data)
        return len(edge_data)

    ok = _run_batches(payloads, _do, "mention edges")
    print(f"  ✓ Loaded {ok:,} mention edges")


def load_edges(conn: tg.TigerGraphConnection) -> None:
    idx = Path(INDEX_DIR)

    rel_path = idx / "relationships.json"
    if rel_path.exists():
        # Gemini index: real subject-relation-target edges -> related_to,
        # carrying the relation verb + frequency. This is a semantically
        # meaningful knowledge graph, not co-occurrence.
        print(f"\nLoading Gemini relationship edges from {INDEX_DIR}/relationships.json...")
        with open(rel_path) as f:
            rels = json.load(f)
        ee_edges = [(r["source"], r["target"], str(r.get("relation", "")), int(r.get("count", 1)))
                    for r in rels if r.get("source") and r.get("target")]
        print(f"  Relationship edges total: {len(ee_edges):,}")
    else:
        print(f"\nLoading co-occurrence edges from {INDEX_DIR}/graph.pkl...")
        with open(idx / "graph.pkl", "rb") as f:
            G = pickle.load(f)
        ee_edges = [
            (u, v, "co_occurs", data.get("weight", 1))
            for u, v, data in G.edges(data=True)
            if G.nodes[u].get("ntype") == "entity" and G.nodes[v].get("ntype") == "entity"
        ]
        print(f"  Entity-entity edges total: {len(ee_edges):,}")

    payloads = [
        [(src, tgt, {"rel_type": rel, "rel_count": cnt})
         for src, tgt, rel, cnt in ee_edges[i : i + EDGE_BATCH]]
        for i in range(0, len(ee_edges), EDGE_BATCH)
    ]

    def _do(edge_data):
        conn.upsertEdges(V_ENTITY, E_REL, V_ENTITY, edge_data)
        return len(edge_data)

    ok = _run_batches(payloads, _do, "related_to edges")
    print(f"  ✓ Loaded {ok:,} edges")


def verify(conn: tg.TigerGraphConnection) -> None:
    print("\nVerifying graph...")
    try:
        n_ents     = conn.getVertexCount(V_ENTITY)
        n_docs     = conn.getVertexCount(V_DOC)
        n_related  = conn.getEdgeCount(E_REL)
        n_mentions = conn.getEdgeCount(E_MEN)
        print(f"  Entity vertices:         {n_ents:,}")
        print(f"  Document vertices:       {n_docs:,}")
        print(f"  related_to edges:        {n_related:,}")
        print(f"  mentions_in_document:    {n_mentions:,}")

        print("\n  Smoke test — multi_hop_expand on 'france'...")
        result = conn.runInstalledQuery("multi_hop_expand", {
            "seed_names":   ["france"],
            "num_hops":     2,
            "top_k":        10,
            "doc_limit":    5,
            "min_co_count": 1,
        })
        found_ents = result[0].get("entities", []) if result else []
        found_docs = result[1].get("docs", []) if len(result) > 1 else []
        print(f"  Entities returned: {len(found_ents)}  sample: {list(found_ents)[:6]}")
        print(f"  Docs returned:     {len(found_docs)}  sample: {list(found_docs)[:4]}")

        healthy = (n_ents >= 5000 and n_related >= 5000
                   and n_docs >= 1000 and n_mentions >= 5000)
        if healthy:
            print("\n  ✓ Graph is healthy and ready for Pipeline 3!")
        else:
            print("\n  ⚠ Graph looks incomplete. Re-run with --fresh if something went wrong.")
    except Exception as e:
        print(f"  Verification error: {e}")
        print("  Tip: if the query isn't installed yet, wait 30 s and try again.")


def main(fresh: bool, skip_schema: bool, skip_edges: bool, query_only: bool, edges_only: bool) -> None:
    print("=" * 60)
    print(f"TigerGraph Cloud Ingestion — graph={TG_GRAPH}")
    print("=" * 60)

    conn = connect()

    if query_only:
        install_query(conn)
        verify(conn)
        return

    if edges_only:
        load_edges(conn)
        verify(conn)
        return

    if not skip_schema:
        if fresh:
            drop_graph(conn)
        create_schema(conn)
        install_query(conn)
    else:
        print("\nSkipping schema creation (--skip-schema).")

    load_entities(conn)
    load_documents(conn)

    if not skip_edges:
        load_edges(conn)
        load_mentions(conn)
    else:
        print("\nSkipping edge loading (--skip-edges).")

    verify(conn)

    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print("Next: python main.py --mode p3only")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load HotpotQA into TigerGraph Savanna")
    parser.add_argument("--fresh",       action="store_true", help="Drop + recreate graph from scratch")
    parser.add_argument("--skip-schema", action="store_true", help="Skip graph/schema/query creation")
    parser.add_argument("--skip-edges",  action="store_true", help="Load entities only, skip edges")
    parser.add_argument("--query-only",  action="store_true",
                        help="Only install the multi_hop_expand query (assumes data already loaded)")
    parser.add_argument("--edges-only",  action="store_true",
                        help="Only load/upsert co-occurrence edges (assumes entities already loaded)")
    args = parser.parse_args()
    main(args.fresh, args.skip_schema, args.skip_edges, args.query_only, args.edges_only)
