"""app.py — Streamlit comparison dashboard.

Shows Pipeline 1 vs Pipeline 2 vs Pipeline 3 side-by-side:
- Enter any query → runs all three pipelines live
- Displays responses + metrics (tokens, latency, cost, accuracy)
- Loads and displays benchmark results from results/benchmark_results.json
  (written by every `run_benchmark.py` run)

Run with: streamlit run app.py  (from the project root)
"""

import os
import json
import base64
import re
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# On Streamlit Cloud, surface secrets as env vars BEFORE importing config so config + the
# pipelines (which read os.getenv) pick up the API keys, LLM provider, embedder, and
# TigerGraph creds. Set these in the app's "Secrets" settings:
#   OPENAI_API_KEY, HF_TOKEN, TIGERGRAPH_HOST, TIGERGRAPH_SECRET, TIGERGRAPH_GRAPH=bioasq,
#   TIGERGRAPH_USERNAME, TIGERGRAPH_PASSWORD, EMBEDDING_MODEL=all-MiniLM-L6-v2
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass
os.environ.setdefault("LLM_PROVIDER", "openai")            # hosted container has no local Ollama
os.environ.setdefault("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # lighter than PubMedBERT (RAM)
# Benchmarks fail loud by default (P3_REQUIRE_TG=1). The interactive demo instead degrades
# gracefully if the free Savanna instance is asleep — but the UI always shows the live-graph
# status per query (tg_live), so a fallback is never misrepresented as a real GraphRAG result.
os.environ.setdefault("P3_REQUIRE_TG", "0")

import config

st.set_page_config(
    page_title="GraphRAG Inference Benchmark",
    page_icon="🔬",
    layout="wide",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _asset_to_base64(filename: str) -> str:
    path = Path(__file__).parent / filename
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


bg_base64 = _asset_to_base64("background.avif")
bg_layer = (
    f'url("data:image/avif;base64,{bg_base64}")'
    if bg_base64
    else "linear-gradient(140deg, #030303 0%, #16120d 45%, #492300 100%)"
)

st.markdown(
    """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Anton&display=swap');

            :root {
                --card: rgba(17, 17, 17, 0.82);
                --ink: #f8fafc;
                --muted: #cbd5e1;
                --line: rgba(226, 232, 240, 0.28);
                --brand: #f8fafc;
                --brand-soft: #111827;
                --accent: #e2e8f0;
                --danger: #ef4444;
                --hero-progress: 0;
            }

            .stApp {
                background-image:
                    linear-gradient(
                        122deg,
                        rgba(3, 7, 18, 0.88) 0%,
                        rgba(15, 23, 42, 0.84) 36%,
                        rgba(30, 41, 59, 0.72) 76%,
                        rgba(51, 65, 85, 0.58) 100%
                    ),
                    __BG_LAYER__;
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }

            .block-container {
                max-width: 1120px;
                padding-top: 0;
                padding-bottom: 2.4rem;
                overflow: visible;
            }

            .story-root { position: relative; overflow: visible; }
            .hero-stage { height: 100vh; position: relative; }

            .hero-wrap {
                position: sticky;
                top: 0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                background: transparent;
                text-align: center;
                transform: scale(calc(1 - (var(--hero-progress) * 0.18))) translateY(calc(var(--hero-progress) * -70px));
                opacity: calc(1 - (var(--hero-progress) * 0.9));
                transition: transform 80ms linear, opacity 80ms linear;
                will-change: transform, opacity;
                overflow: visible;
            }

            .hero-wrap::before {
                content: "";
                position: absolute;
                inset: -8%;
                background: radial-gradient(circle at 55% 42%, rgba(148, 163, 184, 0.28) 0%, transparent 54%);
                transform: translateY(calc(var(--hero-progress) * 45px));
                transition: transform 80ms linear;
                pointer-events: none;
            }

            .hero-title {
                position: relative;
                color: #f8fafc;
                font-size: clamp(6rem, 14vw, 13rem) !important;
                font-weight: 400;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                margin: 0 auto;
                line-height: 0.82;
                white-space: nowrap;
                background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 42%, #94a3b8 100%);
                -webkit-background-clip: text;
                background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 24px 55px rgba(15, 23, 42, 0.7);
                font-family: "Anton", "Impact", "Arial Black", sans-serif;
            }

            .hero-sub {
                position: relative;
                color: rgba(248, 250, 252, 0.78);
                margin-top: 1.4rem;
                font-size: 0.78rem;
                letter-spacing: 0.32em;
                text-transform: uppercase;
                font-weight: 600;
            }

            h2, h3, h4 { color: #f8fafc !important; }

            [data-testid="stMarkdownContainer"] h2,
            [data-testid="stMarkdownContainer"] h3,
            [data-testid="stMarkdownContainer"] h4,
            [data-testid="stCaptionContainer"] { color: #f8fafc !important; }

            [data-testid="stCaptionContainer"] code,
            [data-testid="stMarkdownContainer"] code {
                color: #f59e0b !important;
                background: rgba(245, 158, 11, 0.12) !important;
                padding: 0.05rem 0.35rem;
                border-radius: 6px;
                font-weight: 700;
            }

            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li,
            [data-testid="stMarkdownContainer"] strong,
            [data-testid="stMarkdownContainer"] em,
            [data-testid="stMarkdownContainer"] span { color: #f8fafc !important; }

            .stTabs [data-baseweb="tab-list"] {
                position: sticky;
                top: 0.35rem;
                z-index: 40;
                background: rgba(2, 6, 23, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.26);
                border-radius: 14px;
                padding: 0.25rem;
                backdrop-filter: blur(8px);
            }

            .stTabs [data-baseweb="tab-list"] button {
                height: auto !important;
                padding: 0.85rem 1.6rem !important;
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: 12px;
                color: #cbd5e1;
                font-weight: 800;
                font-size: 1.15rem !important;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }

            .stTabs [data-baseweb="tab"] p,
            .stTabs [data-baseweb="tab"] div {
                font-size: 1.15rem !important;
                font-weight: 800 !important;
                letter-spacing: 0.04em;
            }

            .stTabs [aria-selected="true"] {
                background: rgba(148, 163, 184, 0.24) !important;
                color: #f8fafc !important;
            }

            div[data-testid="stMetric"] {
                background: transparent;
                border: 1.5px solid rgba(255, 255, 255, 0.85);
                border-radius: 18px;
                padding: 0.75rem 0.9rem;
                box-shadow: 0 12px 30px rgba(2, 6, 23, 0.35);
                transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease, background 220ms ease;
            }

            div[data-testid="stMetric"]:hover {
                transform: translateY(-2px);
                border-color: #ffffff;
                background: rgba(255, 255, 255, 0.06);
                box-shadow: 0 18px 40px rgba(2, 6, 23, 0.5);
            }

            div[data-testid="stMetricLabel"],
            div[data-testid="stMetricLabel"] *,
            div[data-testid="stMetricValue"],
            div[data-testid="stMetricValue"] * {
                color: #ffffff !important;
                font-weight: 800;
            }

            div[data-testid="stMetricLabel"] {
                text-transform: uppercase;
                letter-spacing: 0.06em;
                font-size: 0.72rem;
                font-weight: 700;
            }

            div[data-testid="stDataFrame"] {
                border-radius: 18px;
                overflow: hidden;
                border: 1.5px solid rgba(255, 255, 255, 0.85);
                background: transparent;
                box-shadow: 0 12px 30px rgba(2, 6, 23, 0.35);
                --gdg-bg-cell: rgba(0,0,0,0);
                --gdg-text-dark: #ffffff;
                --gdg-text-medium: #e2e8f0;
                --gdg-text-header: #ffffff;
                --gdg-border-color: rgba(255,255,255,0.18);
                --gdg-accent-color: #f59e0b;
            }

            div[data-testid="stPlotlyChart"] {
                border-radius: 18px;
                overflow: hidden;
                padding: 6px;
                background: transparent;
                border: 1.5px solid rgba(255, 255, 255, 0.85);
                box-shadow: 0 12px 30px rgba(2, 6, 23, 0.35);
                transition: transform 220ms ease, box-shadow 220ms ease;
            }

            div[data-testid="stPlotlyChart"]:hover {
                transform: translateY(-2px);
                border-color: #ffffff;
                box-shadow: 0 18px 40px rgba(2, 6, 23, 0.5);
            }

            .styled-table-wrap {
                border-radius: 18px;
                overflow: hidden;
                background: transparent;
                border: 1.5px solid rgba(255, 255, 255, 0.85);
                box-shadow: 0 12px 30px rgba(2, 6, 23, 0.35);
                transition: transform 220ms ease, box-shadow 220ms ease;
                margin: 0.4rem 0 0.9rem 0;
            }

            .styled-table-wrap:hover {
                transform: translateY(-2px);
                border-color: #ffffff;
                box-shadow: 0 18px 40px rgba(2, 6, 23, 0.5);
            }

            .styled-table-wrap .scroll { overflow-x: auto; }

            table.styled-table {
                width: 100%;
                border-collapse: collapse;
                background: transparent;
                color: #ffffff;
                font-family: "Inter", system-ui, sans-serif;
                font-size: 0.92rem;
            }

            table.styled-table thead th {
                background: rgba(255, 255, 255, 0.08);
                color: #ffffff;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-size: 0.74rem;
                text-align: left;
                padding: 0.7rem 0.9rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.18);
                white-space: nowrap;
            }

            table.styled-table tbody td {
                color: #f8fafc;
                padding: 0.6rem 0.9rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                vertical-align: top;
            }

            table.styled-table tbody tr:last-child td { border-bottom: none; }
            table.styled-table tbody tr:nth-child(even) td { background: rgba(255,255,255,0.03); }
            table.styled-table tbody tr:hover td { background: rgba(255,255,255,0.07); }

            .pipeline-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 1.1rem;
                margin: 1.4rem 0 0.85rem 0;
            }

            .pipeline-card {
                background: transparent;
                border: 1.5px solid rgba(255, 255, 255, 0.85);
                border-radius: 18px;
                padding: 1.1rem 1.15rem;
                box-shadow: 0 12px 30px rgba(2, 6, 23, 0.35);
                transition: transform 320ms cubic-bezier(0.22, 1, 0.36, 1),
                            box-shadow 320ms cubic-bezier(0.22, 1, 0.36, 1),
                            border-color 320ms ease, background 320ms ease;
            }

            .pipeline-card:hover {
                transform: translateY(-4px);
                border-color: #ffffff;
                background: rgba(255,255,255,0.06);
                box-shadow: 0 18px 40px rgba(2, 6, 23, 0.5);
            }

            .pipeline-name  { font-size: 1rem; font-weight: 800; color: #ffffff !important; margin: 0; }
            .pipeline-desc  { margin: 0.35rem 0 0.4rem 0; color: rgba(255,255,255,0.82) !important; font-size: 0.86rem; min-height: 2.2rem; }
            .pipeline-badge { display: inline-block; padding: 0.2rem 0.45rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700; }
            .badge-live  { background: #22c55e; color: #ffffff; }
            .badge-ready { background: #3b82f6; color: #ffffff; }

            .benchmark-shell {
                margin-top: 1.4rem;
                background: transparent;
                border: 1.5px solid rgba(255, 255, 255, 0.85);
                border-radius: 18px;
                padding: 1rem 1rem 0.4rem 1rem;
                box-shadow: 0 12px 30px rgba(2, 6, 23, 0.35);
            }

            .benchmark-title { color: #f8fafc; font-size: 1.15rem; font-weight: 800; letter-spacing: 0.03em; margin: 0 0 0.85rem 0; }

            div[data-testid="stAlert"] {
                background: rgba(15, 23, 42, 0.78) !important;
                border: 1px solid rgba(148, 163, 184, 0.35) !important;
                color: #f8fafc !important;
            }

            div[data-testid="stAlert"] * { color: #f8fafc !important; }

            .stTextInput input {
                background: rgba(248, 250, 252, 0.96);
                border: 1px solid rgba(148, 163, 184, 0.65);
                border-radius: 999px;
                color: #0f172a;
                height: 3.2rem;
                padding: 0 1.1rem;
                font-weight: 600;
                box-shadow: 0 10px 22px rgba(2, 6, 23, 0.2);
                transition: box-shadow 260ms ease, transform 260ms ease, border-color 260ms ease;
            }

            .stTextInput input:focus {
                border-color: #38bdf8;
                box-shadow: 0 0 0 3px rgba(56,189,248,0.24), 0 14px 26px rgba(15,23,42,0.28);
            }

            .stButton button {
                border-radius: 999px;
                border: 1px solid rgba(148, 163, 184, 0.75);
                background: linear-gradient(120deg, #0f172a 0%, #334155 100%);
                color: #f8fafc;
                font-weight: 800;
                transition: transform 220ms ease, box-shadow 220ms ease;
            }

            .stButton button:hover {
                transform: translateY(-1px);
                box-shadow: 0 9px 20px rgba(2, 6, 23, 0.28);
            }

            .intro-stage { position: relative; height: 130vh; }

            .intro-sticky {
                position: sticky;
                top: 0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                overflow: visible;
            }

            .intro-heading {
                font-family: "Anton", "Impact", "Arial Black", sans-serif;
                font-weight: 400;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                line-height: 0.85;
                margin: 0 auto;
                color: #f8fafc;
                background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 45%, #94a3b8 100%);
                -webkit-background-clip: text;
                background-clip: text;
                -webkit-text-fill-color: #f8fafc;
                text-shadow: 0 24px 55px rgba(15, 23, 42, 0.7);
                font-size: clamp(2.6rem, 10vw, 8rem) !important;
            }

            @supports ((-webkit-background-clip: text) or (background-clip: text)) {
                .intro-heading, .hero-title { -webkit-text-fill-color: transparent; }
            }

            .intro-sub {
                color: rgba(248, 250, 252, 0.78);
                margin-top: 1.4rem;
                font-size: 0.78rem;
                letter-spacing: 0.32em;
                text-transform: uppercase;
                font-weight: 600;
            }

            .search-row { max-width: 700px; margin: 0 auto; }
            .center-shell { max-width: 1040px; margin: 0 auto; }
            .question-shell { max-width: 820px; margin: 0 auto; }

            @media (max-width: 900px) {
                .pipeline-grid { grid-template-columns: 1fr; }
                .stButton button { width: 100%; }
            }
        </style>
        """.replace("__BG_LAYER__", bg_layer),
    unsafe_allow_html=True,
)

# ── Animated graph-pulse background ──────────────────────────────────────────
st.html(
    """
    <script>
    (() => {
      const parentDoc = window.parent.document;
      if (!parentDoc) return;

      const root = parentDoc.documentElement;
      const heroStage = parentDoc.getElementById('hero-stage');

      const clamp = (v, mn, mx) => Math.min(mx, Math.max(mn, v));
      const ease  = (t) => t < 0.5 ? 2*t*t : -1+(4-2*t)*t;

      const updateHero = () => {
        if (!heroStage) return;
        const rect = heroStage.getBoundingClientRect();
        const total = Math.max(rect.height - window.parent.innerHeight, 1);
        root.style.setProperty('--hero-progress', ease(clamp(-rect.top/total,0,1)).toFixed(4));
      };

      window.parent.addEventListener('scroll', updateHero, { passive: true });
      window.parent.addEventListener('resize', updateHero);
      updateHero();

      const stApp = parentDoc.querySelector('.stApp');
      if (stApp && !parentDoc.getElementById('graph-pulse-bg')) {
        const canvas = parentDoc.createElement('canvas');
        canvas.id = 'graph-pulse-bg';
        Object.assign(canvas.style, {
          position:'fixed', inset:'0', width:'100%', height:'100%',
          zIndex:'0', pointerEvents:'none', opacity:'0.55', mixBlendMode:'screen',
        });
        stApp.prepend(canvas);

        const blockContainer = parentDoc.querySelector('.block-container');
        if (blockContainer) { blockContainer.style.position='relative'; blockContainer.style.zIndex='1'; }

        const ctx = canvas.getContext('2d');
        let W=0, H=0, dpr = window.parent.devicePixelRatio || 1;
        const nodes=[], pulses=[];

        const resize = () => {
          W=canvas.clientWidth; H=canvas.clientHeight;
          canvas.width=W*dpr; canvas.height=H*dpr;
          ctx.setTransform(dpr,0,0,dpr,0,0);
        };
        resize();
        window.parent.addEventListener('resize', resize);

        for (let i=0;i<64;i++) nodes.push({ x:Math.random()*W, y:Math.random()*H,
          vx:(Math.random()-0.5)*0.25, vy:(Math.random()-0.5)*0.25, phase:Math.random()*Math.PI*2 });

        let lastPulse=0;
        const draw = (t) => {
          if (t-lastPulse>1400) { const n=nodes[Math.floor(Math.random()*nodes.length)]; pulses.push({x:n.x,y:n.y,r:0,life:1}); lastPulse=t; }
          ctx.clearRect(0,0,W,H);
          for (let i=0;i<nodes.length;i++) for (let j=i+1;j<nodes.length;j++) {
            const a=nodes[i],b=nodes[j]; const dx=a.x-b.x,dy=a.y-b.y,d=Math.sqrt(dx*dx+dy*dy);
            if (d<170) { ctx.strokeStyle=`rgba(148,197,255,${(1-d/170)*0.35})`; ctx.lineWidth=0.7;
              ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke(); }
          }
          for (const n of nodes) {
            n.x+=n.vx; n.y+=n.vy;
            if (n.x<0||n.x>W) n.vx*=-1; if (n.y<0||n.y>H) n.vy*=-1;
            n.phase+=0.018; const r=1.6*(1+Math.sin(n.phase)*0.35);
            const g=ctx.createRadialGradient(n.x,n.y,0,n.x,n.y,r*6);
            g.addColorStop(0,'rgba(186,220,255,0.95)'); g.addColorStop(1,'rgba(186,220,255,0)');
            ctx.fillStyle=g; ctx.beginPath(); ctx.arc(n.x,n.y,r*6,0,Math.PI*2); ctx.fill();
            ctx.fillStyle='rgba(226,240,255,0.95)'; ctx.beginPath(); ctx.arc(n.x,n.y,r,0,Math.PI*2); ctx.fill();
          }
          for (let i=pulses.length-1;i>=0;i--) {
            const p=pulses[i]; p.r+=1.6; p.life-=0.008;
            if (p.life<=0){pulses.splice(i,1);continue;}
            ctx.strokeStyle=`rgba(125,211,252,${p.life*0.55})`; ctx.lineWidth=1.2;
            ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2); ctx.stroke();
          }
          window.parent.requestAnimationFrame(draw);
        };
        window.parent.requestAnimationFrame(draw);
      }
    })();
    </script>
    """
)

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def load_vectorstore():
    """Load the ChromaDB index; on the hosted/first run, build it from the shipped KG docs."""
    from chroma_store import load_vectorstore as _load, build_vectorstore
    if not Path(config.CHROMA_PERSIST_DIR).exists():
        idx = Path(os.getenv("P3_INDEX_DIR", config.ds_paths()["index_dir"])) / "doc_content.json"
        from langchain_core.documents import Document
        with st.spinner("First run — building the vector index (one-time, ~1–2 min)…"):
            dc = json.load(open(idx))
            cap = int(os.getenv("HOSTED_MAX_DOCS", "8000"))   # keep the hosted build fast + light
            docs = [Document(page_content=t, metadata={"title": d})
                    for d, t in list(dc.items())[:cap]]
            build_vectorstore(docs)
    return _load()


BENCHMARK_RESULTS_FILE = Path("results/benchmark_results.json")


@st.cache_data
def load_benchmark_results():
    """Read the canonical results/benchmark_results.json that every benchmark run writes."""
    if BENCHMARK_RESULTS_FILE.exists():
        with open(BENCHMARK_RESULTS_FILE) as f:
            return json.load(f)
    return None


def render_styled_table(df: pd.DataFrame) -> None:
    html = df.to_html(index=False, escape=True, classes="styled-table", border=0)
    st.markdown(
        f'<div class="styled-table-wrap"><div class="scroll">{html}</div></div>',
        unsafe_allow_html=True,
    )


# ── Hero + intro ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="story-root">
        <section id="hero-stage" class="hero-stage">
            <div class="hero-wrap">
                <div>
                    <p class="hero-title">TIGERGRAPH</p>
                    <p class="hero-sub">Scroll to Enter the Search Experience</p>
                </div>
            </div>
        </section>
        <section class="search-stage">
          <div id="intro-stage" class="intro-stage">
            <div class="intro-sticky">
              <div>
                <h1 class="intro-heading">ASK ANYTHING</h1>
                <p class="intro-sub">Live comparison across pipelines</p>
              </div>
            </div>
          </div>
        </section>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "live_results" not in st.session_state:
    st.session_state.live_results = {}
if "live_error" not in st.session_state:
    st.session_state.live_error = ""
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_benchmark = st.tabs(["Live Query", "Benchmark Results"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Query
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown('<div class="question-shell">', unsafe_allow_html=True)
    st.markdown('<div class="search-row">', unsafe_allow_html=True)

    query = st.text_input(
        "Search anything",
        key="live_query_input",
        label_visibility="collapsed",
        placeholder="Search anything...",
    )
    run_query_clicked = st.button("Run Query", key="run_query_btn", use_container_width=True)

    if run_query_clicked and not query.strip():
        st.warning("Please enter a question before running the query.")

    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="center-shell" style="margin-top:1rem;">
            <div class="pipeline-grid">
                <div class="pipeline-card">
                    <p class="pipeline-name">Pipeline 1: LLM-Only</p>
                    <p class="pipeline-desc">Direct LLM answers without retrieval for baseline comparison.</p>
                    <span class="pipeline-badge badge-live">Live</span>
                </div>
                <div class="pipeline-card">
                    <p class="pipeline-name">Pipeline 2: Basic RAG</p>
                    <p class="pipeline-desc">Vector retrieval with contextual grounding before generation.</p>
                    <span class="pipeline-badge badge-live">Live</span>
                </div>
                <div class="pipeline-card">
                    <p class="pipeline-name">Pipeline 3: GraphRAG</p>
                    <p class="pipeline-desc">Knowledge-graph augmented retrieval via TigerGraph BFS + rerank.</p>
                    <span class="pipeline-badge badge-ready">Live</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if run_query_clicked and query.strip():
        import pipeline1_llm_only as p1
        import pipeline2_basic_rag as p2
        import pipeline3_tigergraph_cloud.pipeline as p3

        results = {}
        st.session_state.live_error = ""
        st.session_state.last_query = query

        try:
            with st.spinner("Running Pipeline 1 (LLM-Only)..."):
                results["p1"] = p1.run_single(query)
        except Exception as e:
            st.session_state.live_error = f"Pipeline 1 failed: {e}"

        vs = None
        try:
            with st.spinner("Loading vector store & running Pipeline 2 (Basic RAG)..."):
                vs = load_vectorstore()
                results["p2"] = p2.run_single(query, vs)
        except Exception as e:
            st.session_state.live_error += f" | Pipeline 2 failed: {e}"

        try:
            with st.spinner("Running Pipeline 3 (GraphRAG — TigerGraph)..."):
                if vs is None:
                    vs = load_vectorstore()
                results["p3"] = p3.run_single(query, vectorstore=vs)
        except Exception as e:
            st.session_state.live_error += f" | Pipeline 3 failed: {e}"

        st.session_state.live_results = results

    if st.session_state.live_error:
        st.error(st.session_state.live_error.strip(" |"))

    if st.session_state.live_results:
        if st.session_state.last_query:
            st.caption(f"Showing results for: **{st.session_state.last_query}**")

        cols = st.columns(len(st.session_state.live_results))
        for i, (key, metrics) in enumerate(st.session_state.live_results.items()):
            with cols[i]:
                st.markdown(f"### {metrics.pipeline}")
                st.markdown("**Answer:**")
                st.info(metrics.answer or "_No answer_")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Total Tokens", f"{metrics.total_tokens:,}")
                    st.metric("Latency", f"{metrics.latency_seconds:.2f}s")
                with c2:
                    st.metric("Cost", f"${metrics.cost_usd:.6f}")
                    st.metric("Chunks Retrieved", metrics.retrieved_chunks)
                with st.expander("Token breakdown"):
                    st.write(f"Prompt tokens: {metrics.prompt_tokens:,}")
                    st.write(f"Completion tokens: {metrics.completion_tokens:,}")

                # Live-graph provenance (reviewer #1c): prove the cloud graph was queried
                # for this question — never let a fallback look like a real GraphRAG result.
                if key == "p3":
                    if getattr(metrics, "tg_live", False):
                        st.success(
                            f"🟢 Live TigerGraph queried — "
                            f"{len(getattr(metrics, 'graph_doc_ids', []))} graph docs, "
                            f"{getattr(metrics, 'graph_entity_count', 0)} entities"
                        )
                        if metrics.tg_host:
                            st.caption(f"Graph host: `{metrics.tg_host}`")
                        if metrics.graph_doc_ids:
                            with st.expander("Documents returned by the graph"):
                                st.write(", ".join(str(d) for d in metrics.graph_doc_ids[:30]))
                    else:
                        st.warning(
                            "⚠️ Graph offline — vector-only fallback "
                            "(not a true GraphRAG result)"
                        )

        if len(st.session_state.live_results) > 1:
            st.divider()
            st.subheader("Side-by-side comparison")
            render_styled_table(pd.DataFrame([
                {
                    "Pipeline": m.pipeline,
                    "Total Tokens": f"{m.total_tokens:,}",
                    "Latency (s)": f"{m.latency_seconds:.2f}",
                    "Cost (USD)": f"${m.cost_usd:.6f}",
                    "Chunks": m.retrieved_chunks,
                    "Live Graph": (f"🟢 {len(m.graph_doc_ids)} docs"
                                   if getattr(m, "tg_live", False)
                                   else "—"),
                }
                for m in st.session_state.live_results.values()
            ]))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Benchmark Results
# ══════════════════════════════════════════════════════════════════════════════
with tab_benchmark:
    st.markdown('<div class="benchmark-shell"><p class="benchmark-title">Benchmark Results</p>', unsafe_allow_html=True)
    results_data = load_benchmark_results()

    if results_data is None:
        st.warning(
            "No benchmark results found. "
            "Run `python run_benchmark.py` (or `main.py --mode benchmark`) first, then refresh."
        )
    else:
        cfg  = results_data["config"]
        perf = results_data["performance"]
        acc  = results_data["accuracy"]

        st.caption(
            f"Model: `{cfg['llm_model']}` | "
            f"Top-K: {cfg['top_k']} | "
            f"Chunk size: {cfg['chunk_size']} | "
            f"Questions evaluated: {cfg['n_questions']}"
        )

        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1: st.metric("Questions Evaluated", cfg["n_questions"])
        with kc2: st.metric("Top-K", cfg["top_k"])
        # Live-graph proof (reviewer #1a/#1c): tg_live_rate == 1.0 means every GraphRAG
        # answer in this benchmark was produced against the live TigerGraph cloud graph.
        p3_perf = perf.get("pipeline3", {})
        if "tg_live_rate" in p3_perf:
            with kc3:
                st.metric("P3 Live-Graph Rate", f"{p3_perf['tg_live_rate']:.0%}")
            with kc4:
                st.metric("P3 Avg Graph Docs", f"{p3_perf.get('avg_graph_docs', 0):.0f}")

        # ── Performance ────────────────────────────────────────────────────
        st.subheader("📈 Performance Metrics")

        pipelines_perf = [perf["pipeline1"], perf["pipeline2"]]
        if "pipeline3" in perf:
            pipelines_perf.append(perf["pipeline3"])

        render_styled_table(pd.DataFrame([
            {
                "Pipeline": p["pipeline"],
                "Avg Total Tokens": f"{p['avg_total_tokens']:.0f}",
                "Avg Prompt Tokens": f"{p['avg_prompt_tokens']:.0f}",
                "Avg Completion Tokens": f"{p['avg_completion_tokens']:.0f}",
                "Avg Latency (s)": f"{p['avg_latency_seconds']:.2f}",
                "Avg Cost (USD)": f"${p['avg_cost_usd']:.6f}",
                "Total Cost (USD)": f"${p['total_cost_usd']:.4f}",
            }
            for p in pipelines_perf
        ]))

        # ── Charts ─────────────────────────────────────────────────────────
        chart_layout = dict(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, system-ui, sans-serif", color="#f8fafc", size=13),
            title_font=dict(color="#f8fafc", size=16),
            title_x=0.02, title_xanchor="left",
            margin=dict(l=50, r=20, t=60, b=50),
            xaxis=dict(showgrid=False, zeroline=False, color="#f8fafc",
                       linecolor="rgba(255,255,255,0.25)", tickfont=dict(color="#e2e8f0")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False,
                       color="#f8fafc", linecolor="rgba(255,255,255,0.25)", tickfont=dict(color="#e2e8f0")),
            legend=dict(font=dict(color="#f8fafc"), bgcolor="rgba(15,23,42,0.5)",
                        bordercolor="rgba(255,255,255,0.2)", borderwidth=1),
        )
        bar_colors = ["#f59e0b", "#3b82f6", "#22c55e"]
        px = [p["pipeline"].replace("Pipeline ", "P") for p in pipelines_perf]

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Bar(
                x=px, y=[p["avg_total_tokens"] for p in pipelines_perf],
                marker=dict(color=bar_colors[:len(pipelines_perf)], line=dict(color="rgba(255,255,255,0.6)", width=1.2)),
                text=[f"{p['avg_total_tokens']:.0f}" for p in pipelines_perf],
                textposition="outside", textfont=dict(color="#f8fafc", size=13),
                hovertemplate="<b>%{x}</b><br>Tokens: %{y:,.0f}<extra></extra>",
            ))
            fig.update_layout(title_text="Average Total Tokens per Query", yaxis_title="Tokens", showlegend=False, **chart_layout)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(go.Bar(
                x=px, y=[p["avg_latency_seconds"] for p in pipelines_perf],
                marker=dict(color=bar_colors[:len(pipelines_perf)], line=dict(color="rgba(255,255,255,0.6)", width=1.2)),
                text=[f"{p['avg_latency_seconds']:.2f}s" for p in pipelines_perf],
                textposition="outside", textfont=dict(color="#f8fafc", size=13),
                hovertemplate="<b>%{x}</b><br>Latency: %{y:.2f}s<extra></extra>",
            ))
            fig.update_layout(title_text="Average Latency per Query", yaxis_title="Seconds", showlegend=False, **chart_layout)
            st.plotly_chart(fig, use_container_width=True)

        # ── Accuracy ────────────────────────────────────────────────────────
        st.subheader("🎯 Accuracy Metrics")

        render_styled_table(pd.DataFrame([
            {
                "Pipeline": a["pipeline"],
                "LLM-Judge Pass Rate": (
                    f"{a['llm_judge_pass_rate']:.1%}" if a.get("llm_judge_pass_rate") is not None else "N/A"
                ),
                "BERTScore F1 (rescaled)": f"{a['bertscore_f1_rescaled']:.4f}",
                "Questions Scored": a.get("num_judge_scored", a.get("num_evaluated", "N/A")),
                "Judge Errors": a.get("num_judge_errored", 0),
                "Judge Bonus (≥90%)":   "✅" if a.get("judge_bonus_achieved") else "❌",
                "BERTScore Bonus (≥0.55)": "✅" if a.get("bert_bonus_achieved") else "❌",
                "Max Bonus": "🏆" if a.get("max_bonus_achieved") else "❌",
            }
            for a in acc.values()
        ]))

        # ── BERTScore distribution ─────────────────────────────────────────
        if "accuracy_detail" in results_data:
            st.subheader("BERTScore F1 Distribution")
            fig = go.Figure()
            colors = {"pipeline1": "#f59e0b", "pipeline2": "#3b82f6", "pipeline3": "#22c55e"}
            for key, items in results_data["accuracy_detail"].items():
                scores = [item["bertscore_f1"] for item in items]
                label = items[0].get("pipeline", key) if items else key
                fig.add_trace(go.Histogram(
                    x=scores, name=label, opacity=0.75,
                    marker=dict(color=colors.get(key, "#94a3b8"), line=dict(color="rgba(255,255,255,0.4)", width=1)),
                    hovertemplate="<b>%{fullData.name}</b><br>F1: %{x:.3f}<br>Count: %{y}<extra></extra>",
                ))
            fig.update_layout(barmode="overlay", xaxis_title="BERTScore F1 (rescaled)",
                              yaxis_title="Count", title_text="BERTScore F1 Distribution per Pipeline", **chart_layout)
            st.plotly_chart(fig, use_container_width=True)

        # ── Per-question table ─────────────────────────────────────────────
        pipeline_keys = [k for k in ["pipeline3", "pipeline2", "pipeline1"] if k in results_data.get("per_question", {})]
        for pk in pipeline_keys:
            label = {"pipeline1": "P1: LLM-Only", "pipeline2": "P2: Basic RAG", "pipeline3": "P3: GraphRAG"}[pk]
            with st.expander(f"📋 Per-question results — {label}"):
                pq_data   = results_data["per_question"][pk]
                acc_detail = results_data.get("accuracy_detail", {}).get(pk, [])
                rows = []
                for i, m in enumerate(pq_data):
                    row = {
                        "Question":    (m["question"][:80] + "...") if len(m["question"]) > 80 else m["question"],
                        "Answer":      (m["answer"][:80] + "...")   if len(m.get("answer","")) > 80 else m.get("answer",""),
                        "Tokens":      m["total_tokens"],
                        "Latency (s)": m["latency_seconds"],
                        "Cost (USD)":  f"${m['cost_usd']:.6f}",
                    }
                    if i < len(acc_detail):
                        row["Judge"]      = "✅" if acc_detail[i].get("judge_pass") else "❌"
                        row["BERTScore"]  = f"{acc_detail[i].get('bertscore_f1', 0):.3f}"
                    rows.append(row)
                render_styled_table(pd.DataFrame(rows))

    st.markdown('</div>', unsafe_allow_html=True)
