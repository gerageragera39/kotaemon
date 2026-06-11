from __future__ import annotations

import json
import sys
import types

import pytest

pytest.importorskip("theflow")

from kotaemon.base import Document, DocumentWithEmbedding
from kotaemon.embeddings.base import BaseEmbeddings
from kotaemon.indices.rankings.base import BaseReranking
from kotaemon.indices.vectorindex import VectorIndexing, VectorRetrieval
from kotaemon.storages.docstores.base import BaseDocumentStore
from kotaemon.storages.vectorstores.base import BaseVectorStore


class DummyEmbedding(BaseEmbeddings):
    embedded_doc_ids: list[str] = []

    def invoke(self, docs, *args, **kwargs):
        if not isinstance(docs, list):
            docs = [docs]
        self.embedded_doc_ids.extend(
            doc.doc_id for doc in docs if isinstance(doc, Document)
        )
        return [DocumentWithEmbedding(embedding=[1.0, 0.0]) for _ in docs]


class RecordingVectorStore(BaseVectorStore):
    def __init__(self, ids: list[str] | None = None, scores: list[float] | None = None):
        self.ids = ids or []
        self.scores = scores or []
        self.query_calls: list[dict] = []
        self.added_ids: list[str] = []

    def add(self, embeddings, ids):
        self.added_ids.extend(ids)

    def query(self, embedding, top_k=1, doc_ids=None, **kwargs):
        self.query_calls.append(
            {"embedding": embedding, "top_k": top_k, "doc_ids": doc_ids}
        )
        return [], self.scores[:top_k], self.ids[:top_k]

    def delete(self, ids, **kwargs):
        return None

    def drop(self):
        return None


class RecordingDocStore(BaseDocumentStore):
    def __init__(
        self,
        docs: list[Document] | None = None,
        query_docs: list[Document] | None = None,
    ):
        self.docs = {doc.doc_id: doc for doc in docs or []}
        self.query_docs = query_docs or []
        self.query_calls: list[dict] = []

    def add(self, docs, ids=None, **kwargs):
        if not isinstance(docs, list):
            docs = [docs]
        for doc in docs:
            self.docs[doc.doc_id] = doc

    def get(self, ids):
        if not isinstance(ids, list):
            ids = [ids]
        return [self.docs[doc_id] for doc_id in ids]

    def get_all(self):
        return list(self.docs.values())

    def count(self):
        return len(self.docs)

    def query(self, query, top_k=10, doc_ids=None):
        self.query_calls.append({"query": query, "top_k": top_k, "doc_ids": doc_ids})
        if doc_ids is None:
            return self.query_docs[:top_k]
        allowed = set(doc_ids)
        return [doc for doc in self.query_docs if doc.doc_id in allowed][:top_k]

    def delete(self, ids):
        if not isinstance(ids, list):
            ids = [ids]
        for doc_id in ids:
            self.docs.pop(doc_id, None)

    def drop(self):
        self.docs = {}


class RecordingReranker(BaseReranking):
    seen_doc_ids: list[str] = []

    def run(self, documents, query):
        self.seen_doc_ids = [doc.doc_id for doc in documents]
        return documents


def doc(doc_id: str, text: str | None = None, **metadata) -> Document:
    return Document(text=text or doc_id, id_=doc_id, metadata=metadata)


def test_text_mode_queries_docstore_when_scope_is_none():
    docstore = RecordingDocStore(query_docs=[doc("text-hit")])
    retrieval = VectorRetrieval(
        vector_store=RecordingVectorStore(),
        doc_store=docstore,
        embedding=DummyEmbedding(),
        retrieval_mode="text",
    )

    results = retrieval.run("hello", top_k=3, expand_parent=False)

    assert [result.doc_id for result in results] == ["text-hit"]
    assert docstore.query_calls == [{"query": "hello", "top_k": 3, "doc_ids": None}]


def test_hybrid_mode_queries_vectorstore_and_docstore_when_scope_is_none():
    vector_doc = doc("vector-hit")
    text_doc = doc("text-hit")
    docstore = RecordingDocStore(docs=[vector_doc, text_doc], query_docs=[text_doc])
    vectorstore = RecordingVectorStore(ids=["vector-hit"], scores=[0.42])
    retrieval = VectorRetrieval(
        vector_store=vectorstore,
        doc_store=docstore,
        embedding=DummyEmbedding(),
        retrieval_mode="hybrid",
    )

    results = retrieval.run("hello", top_k=5, expand_parent=False)

    assert vectorstore.query_calls[0]["doc_ids"] is None
    assert docstore.query_calls == [{"query": "hello", "top_k": 5, "doc_ids": None}]
    assert {result.doc_id for result in results} == {"vector-hit", "text-hit"}


def test_rrf_fuse_deduplicates_by_doc_id_and_rewards_overlap():
    retrieval = VectorRetrieval(
        vector_store=RecordingVectorStore(),
        doc_store=RecordingDocStore(),
        embedding=DummyEmbedding(),
    )
    shared = doc("shared")
    dense_only = doc("dense-only")
    sparse_only = doc("sparse-only")

    fused = retrieval._rrf_fuse(
        vector_docs=[shared, dense_only, shared],
        vector_scores=[0.9, 0.8, 0.1],
        text_docs=[shared, sparse_only],
    )

    assert [result.doc_id for result in fused].count("shared") == 1
    assert fused[0].doc_id == "shared"
    assert fused[0].metadata["_retrieval_sources"] == ["vector", "text"]
    assert fused[0].metadata["_vector_score"] == 0.9
    assert fused[0].metadata["_dense_rank"] == 1
    assert fused[0].metadata["_sparse_rank"] == 1
    assert fused[0].metadata["_fusion_score"] == fused[0].score


def test_parent_docs_are_skipped_in_vector_embedding():
    parent = doc("parent", index_role="parent")
    child = doc("child", index_role="child", parent_id="parent")
    embedding = DummyEmbedding()
    vectorstore = RecordingVectorStore()
    indexing = VectorIndexing(
        vector_store=vectorstore,
        doc_store=RecordingDocStore(),
        embedding=embedding,
    )

    indexing.add_to_vectorstore([parent, child])

    assert embedding.embedded_doc_ids == ["child"]
    assert vectorstore.added_ids == ["child"]


def test_parent_docs_are_filtered_before_reranking_and_raw_results():
    parent = doc("parent", index_role="parent")
    child = doc("child", index_role="child", parent_id="parent")
    reranker = RecordingReranker()
    docstore = RecordingDocStore(query_docs=[parent, child])
    retrieval = VectorRetrieval(
        vector_store=RecordingVectorStore(),
        doc_store=docstore,
        embedding=DummyEmbedding(),
        retrieval_mode="text",
        rerankers=[reranker],
    )

    results = retrieval.run("hello", top_k=5, expand_parent=False)

    assert reranker.seen_doc_ids == ["child"]
    assert [result.doc_id for result in results] == ["child"]
    assert all(result.metadata.get("index_role") != "parent" for result in results)


def test_lancedb_fts_prefilters_parent_docs_before_limit(monkeypatch, tmp_path):
    events: list[tuple] = []

    class FakeSearch:
        def where(self, query_filter, prefilter=False):
            events.append(("where", query_filter, prefilter))
            return self

        def limit(self, top_k):
            events.append(("limit", top_k))
            return self

        def to_list(self):
            events.append(("to_list",))
            return [
                {
                    "id": "child",
                    "text": "child hit",
                    "attributes": json.dumps({"index_role": "child"}),
                }
            ]

    class FakeTable:
        def search(self, query=None, query_type=None):
            events.append(("search", query, query_type))
            return FakeSearch()

    class FakeDB:
        def table_names(self):
            return ["docstore"]

        def open_table(self, name):
            return FakeTable()

    fake_lancedb = types.SimpleNamespace(connect=lambda _path: FakeDB())
    monkeypatch.setitem(sys.modules, "lancedb", fake_lancedb)

    from kotaemon.storages.docstores.lancedb import LanceDBDocumentStore

    store = LanceDBDocumentStore(path=str(tmp_path), collection_name="docstore")
    results = store.query("Prüfende", top_k=1, doc_ids=["parent", "child"])

    assert [result.doc_id for result in results] == ["child"]
    where_event = next(event for event in events if event[0] == "where")
    limit_event = next(event for event in events if event[0] == "limit")
    assert events.index(where_event) < events.index(limit_event)
    assert "index_role != 'parent'" in where_event[1]
    assert "id in ('parent', 'child')" in where_event[1]
    assert where_event[2] is True
