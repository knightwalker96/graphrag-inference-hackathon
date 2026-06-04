"""Build and query the ChromaDB vector store."""

from __future__ import annotations
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from tqdm import tqdm

import config
from embeddings.embedder import get_embedder


def _get_splitter() -> RecursiveCharacterTextSplitter:
    # chunk_size is in characters; ×4 ≈ token count
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 4,
        chunk_overlap=config.CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def build_vectorstore(documents: list[Document]) -> Chroma:
    print(f"[chroma] Splitting {len(documents)} documents into chunks...")
    splitter = _get_splitter()

    all_chunks: list[Document] = []
    for doc in tqdm(documents, desc="Splitting"):
        all_chunks.extend(splitter.split_documents([doc]))

    print(f"[chroma] Total chunks: {len(all_chunks)}")
    print(f"[chroma] Building index at '{config.CHROMA_PERSIST_DIR}'...")

    BATCH_SIZE = 5000
    vectorstore = None
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        n_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"[chroma] Batch {i // BATCH_SIZE + 1}/{n_batches} ({len(batch)} chunks)...")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=get_embedder(),
                persist_directory=config.CHROMA_PERSIST_DIR,
                collection_name=config.CHROMA_COLLECTION_NAME,
            )
        else:
            vectorstore.add_documents(batch)

    print(f"[chroma] Done. {len(all_chunks)} chunks stored.")
    return vectorstore


def load_vectorstore() -> Chroma:
    if not os.path.exists(config.CHROMA_PERSIST_DIR):
        raise FileNotFoundError(
            f"No index at '{config.CHROMA_PERSIST_DIR}'. Run `python main.py --mode index` first."
        )
    print(f"[chroma] Loading index from '{config.CHROMA_PERSIST_DIR}'...")
    vectorstore = Chroma(
        persist_directory=config.CHROMA_PERSIST_DIR,
        embedding_function=get_embedder(),
        collection_name=config.CHROMA_COLLECTION_NAME,
    )
    print(f"[chroma] {vectorstore._collection.count():,} chunks loaded.")
    return vectorstore


def get_retriever(vectorstore: Chroma):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.TOP_K, "fetch_k": config.TOP_K * 3, "lambda_mult": 0.7},
    )
