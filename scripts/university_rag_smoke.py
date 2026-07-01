#!/usr/bin/env python3
"""Deterministic university RAG smoke check without external services.

Indexes synthetic university-like chunks in memory, runs vector/hybrid retrieval over
an exact phrase, prints top chunks, and saves debug output under
ktem_app_data/evaluations/<timestamp>_smoke/.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kotaemon.base import Document, DocumentWithEmbedding
from kotaemon.embeddings.base import BaseEmbeddings
from kotaemon.indices.splitters import UniversityPDFChunker
from kotaemon.indices.vectorindex import VectorIndexing, VectorRetrieval
from kotaemon.storages.docstores.in_memory import InMemoryDocumentStore


class KeywordEmbedding(BaseEmbeddings):
    terms = ["mündliche", "prüfung", "akteneinsicht", "frist"]

    def invoke(self, docs, *args, **kwargs):
        if not isinstance(docs, list):
            docs = [docs]
        out = []
        for doc in docs:
            text = (doc.text if isinstance(doc, Document) else str(doc)).lower()
            out.append(
                DocumentWithEmbedding(
                    embedding=[float(term in text) for term in self.terms]
                )
            )
        return out


class MemoryVectorStore:
    def __init__(self):
        self.vectors = {}

    def add(self, embeddings, ids):
        for embedding, doc_id in zip(embeddings, ids):
            self.vectors[doc_id] = list(embedding.embedding)

    def query(self, embedding, top_k=1, doc_ids=None, **kwargs):
        allowed = set(doc_ids or self.vectors)
        scored = []
        for doc_id, vector in self.vectors.items():
            if doc_id not in allowed:
                continue
            score = sum(a * b for a, b in zip(embedding, vector))
            scored.append((score, doc_id))
        scored.sort(reverse=True)
        scored = scored[:top_k]
        return [], [score for score, _ in scored], [doc_id for _, doc_id in scored]

    def delete(self, ids, **kwargs):
        for doc_id in ids:
            self.vectors.pop(doc_id, None)

    def drop(self):
        self.vectors = {}


def main() -> int:
    app_data = ROOT / "ktem_app_data"
    run_dir = app_data / "evaluations" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)

    source_docs = [
        Document(
            text="§ 13 Bewertung der Prüfungsleistungen",
            metadata={
                "file_name": "smoke_APO.pdf",
                "page_label": 8,
                "order": 1,
                "element_type": "heading",
            },
        ),
        Document(
            text="(1) Prüfungen können schriftlich oder als mündliche Prüfung durchgeführt werden.",
            metadata={
                "file_name": "smoke_APO.pdf",
                "page_label": 8,
                "order": 2,
                "element_type": "paragraph",
            },
        ),
    ]
    chunks = UniversityPDFChunker().run(source_docs)
    docstore = InMemoryDocumentStore()
    vectorstore = MemoryVectorStore()
    embedding = KeywordEmbedding()
    VectorIndexing(
        vector_store=vectorstore, doc_store=docstore, embedding=embedding
    ).run(chunks)

    retrieval = VectorRetrieval(
        vector_store=vectorstore,
        doc_store=docstore,
        embedding=embedding,
        retrieval_mode="hybrid",
        first_round_top_k_mult=5,
    )
    docs = retrieval.run("oral exam", top_k=3, do_extend=True, expand_parent="siblings")
    debug = retrieval.last_debug
    (run_dir / "retrieval_debug.jsonl").write_text(
        json.dumps(debug, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_dir / "settings.json").write_text(
        json.dumps({"smoke": True}, indent=2), encoding="utf-8"
    )
    print(f"Saved smoke debug to {run_dir}")
    for rank, doc in enumerate(docs, start=1):
        print(
            f"#{rank} {doc.doc_id} score={doc.score} {doc.text[:180].replace(chr(10), ' ')}"
        )
    if not docs:
        raise SystemExit("No documents retrieved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
