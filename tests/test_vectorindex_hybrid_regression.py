from __future__ import annotations

import json
import sys
import types

import pytest

pytest.importorskip("theflow")

from kotaemon.base import Document, DocumentWithEmbedding, RetrievedDocument
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


class MetadataCopyEmbedding(BaseEmbeddings):
    def invoke(self, docs, *args, **kwargs):
        if not isinstance(docs, list):
            docs = [docs]
        return [
            DocumentWithEmbedding(
                embedding=[1.0, 0.0],
                metadata=dict(doc.metadata or {}) if isinstance(doc, Document) else {},
            )
            for doc in docs
        ]


class RecordingVectorStore(BaseVectorStore):
    def __init__(self, ids: list[str] | None = None, scores: list[float] | None = None):
        self.ids = ids or []
        self.scores = scores or []
        self.query_calls: list[dict] = []
        self.added_ids: list[str] = []
        self.added_embeddings = []

    def add(self, embeddings, ids):
        self.added_ids.extend(ids)
        self.added_embeddings.extend(embeddings)

    def query(self, embedding, top_k=1, doc_ids=None, **kwargs):
        self.query_calls.append(
            {"embedding": embedding, "top_k": top_k, "doc_ids": doc_ids}
        )
        return [], self.scores[:top_k], self.ids[:top_k]

    def delete(self, ids, **kwargs):
        return None

    def drop(self):
        return None


class IdsRecordingVectorStore(RecordingVectorStore):
    def query(self, embedding, top_k=1, ids=None, **kwargs):
        self.query_calls.append(
            {"embedding": embedding, "top_k": top_k, "ids": ids, "kwargs": kwargs}
        )
        return [], self.scores[:top_k], self.ids[:top_k]


class FilterSensitiveVectorStore(RecordingVectorStore):
    def query(self, embedding, top_k=1, doc_ids=None, **kwargs):
        self.query_calls.append(
            {
                "embedding": embedding,
                "top_k": top_k,
                "doc_ids": doc_ids,
                "kwargs": kwargs,
            }
        )
        if "filters" in kwargs:
            return [], [], []
        return [], self.scores[:top_k], self.ids[:top_k]


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
    assert docstore.query_calls == [{"query": "hello", "top_k": 30, "doc_ids": None}]


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
    assert docstore.query_calls == [{"query": "hello", "top_k": 50, "doc_ids": None}]
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
    assert fused[0].metadata["_fusion_score"] == fused[0].metadata["_ranking_score"]
    assert fused[0].metadata["retrieval_score"] == fused[0].score == 0.9


def test_vector_scope_uses_canonical_ids_parameter_for_vectorstore_contract():
    scoped_doc = doc("scoped-hit")
    docstore = RecordingDocStore(docs=[scoped_doc])
    vectorstore = IdsRecordingVectorStore(ids=["scoped-hit"], scores=[0.87])
    retrieval = VectorRetrieval(
        vector_store=vectorstore,
        doc_store=docstore,
        embedding=DummyEmbedding(),
        retrieval_mode="vector",
    )

    results = retrieval.run(
        "hello", top_k=3, scope=["scoped-hit"], expand_parent=False
    )

    assert vectorstore.query_calls == [
        {"embedding": [1.0, 0.0], "top_k": 30, "ids": ["scoped-hit"], "kwargs": {}}
    ]
    assert [result.doc_id for result in results] == ["scoped-hit"]
    assert results[0].score == 0.87
    assert results[0].metadata["_ranking_score"] != results[0].score


def test_vector_query_retries_without_metadata_filters_when_scoped_ids_are_enough():
    scoped_doc = doc("scoped-hit")
    docstore = RecordingDocStore(docs=[scoped_doc])
    vectorstore = FilterSensitiveVectorStore(ids=["scoped-hit"], scores=[0.91])
    retrieval = VectorRetrieval(
        vector_store=vectorstore,
        doc_store=docstore,
        embedding=DummyEmbedding(),
        retrieval_mode="vector",
    )

    results = retrieval.run(
        "hello",
        top_k=3,
        scope=["scoped-hit"],
        filters=object(),
        expand_parent=False,
    )

    assert [result.doc_id for result in results] == ["scoped-hit"]
    assert len(vectorstore.query_calls) == 2
    assert "filters" in vectorstore.query_calls[0]["kwargs"]
    assert "filters" not in vectorstore.query_calls[1]["kwargs"]
    assert retrieval.last_debug["vector_filter_fallbacks"] == 1


def test_text_only_display_score_is_not_reported_as_perfect_relevance():
    retrieval = VectorRetrieval(
        vector_store=RecordingVectorStore(),
        doc_store=RecordingDocStore(),
        embedding=DummyEmbedding(),
    )

    fused = retrieval._rrf_fuse([], [], [doc("text-hit")])

    assert fused[0].metadata["retrieval_source"] == "text"
    assert 0 < fused[0].score <= 0.35


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


def test_vector_metadata_is_flattened_before_vectorstore_insert():
    child = doc(
        "child",
        index_role="child",
        section_path=["2. Aufbau des Studiengangs", "2.6. Studienprofile"],
        nearest_heading="2.6. Studienprofile",
        is_probably_latest=True,
        nested={"origin": "docling"},
    )
    vectorstore = RecordingVectorStore()
    indexing = VectorIndexing(
        vector_store=vectorstore,
        doc_store=RecordingDocStore(),
        embedding=MetadataCopyEmbedding(),
    )

    indexing.add_to_vectorstore([child])

    metadata = vectorstore.added_embeddings[0].metadata
    assert metadata["section_path"] == (
        "2. Aufbau des Studiengangs > 2.6. Studienprofile"
    )
    assert metadata["is_probably_latest"] == 1
    assert metadata["nested"] == '{"origin": "docling"}'
    assert all(
        value is None or type(value) in {str, int, float}
        for value in metadata.values()
    )


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


def test_candidate_multiplier_controls_first_round_pool_when_extended():
    vector_doc = doc("vector-hit")
    docstore = RecordingDocStore(docs=[vector_doc])
    vectorstore = RecordingVectorStore(ids=["vector-hit"], scores=[0.42])
    retrieval = VectorRetrieval(
        vector_store=vectorstore,
        doc_store=docstore,
        embedding=DummyEmbedding(),
        retrieval_mode="vector",
        first_round_top_k_mult=20,
    )

    retrieval.run("hello", top_k=15, do_extend=True, expand_parent=False)

    assert vectorstore.query_calls[0]["top_k"] == 300


def test_candidate_multiplier_widens_first_round_pool_by_default():
    vector_doc = doc("vector-hit")
    docstore = RecordingDocStore(docs=[vector_doc])
    vectorstore = RecordingVectorStore(ids=["vector-hit"], scores=[0.42])
    retrieval = VectorRetrieval(
        vector_store=vectorstore,
        doc_store=docstore,
        embedding=DummyEmbedding(),
        retrieval_mode="vector",
        first_round_top_k_mult=4,
    )

    retrieval.run("hello", top_k=5, expand_parent=False)

    assert vectorstore.query_calls[0]["top_k"] == 20


def test_hybrid_branch_errors_are_propagated_to_caller():
    class FailingVectorStore(RecordingVectorStore):
        def query(self, *args, **kwargs):
            raise ValueError("vector boom")

    docstore = RecordingDocStore(query_docs=[doc("text-hit")])
    retrieval = VectorRetrieval(
        vector_store=FailingVectorStore(),
        doc_store=docstore,
        embedding=DummyEmbedding(),
        retrieval_mode="hybrid",
    )

    with pytest.raises(RuntimeError) as exc_info:
        retrieval.run("hello", top_k=5, expand_parent=False)

    assert "vectorstore" in str(exc_info.value)
    assert "vector boom" in str(exc_info.value)


def test_file_retrieval_defaults_include_hybrid_candidate_multiplier_and_context_window():
    from ktem.index.file.pipelines import DocumentRetrievalPipeline
    from ktem.reasoning.simple import FullQAPipeline
    from kotaemon.indices.qa.format_context import PrepareEvidencePipeline

    retrieval_settings = DocumentRetrievalPipeline.get_user_settings()
    reasoning_settings = FullQAPipeline.get_user_settings()

    assert retrieval_settings["retrieval_mode"]["value"] == "hybrid"
    assert retrieval_settings["num_retrieval"]["value"] == 15
    assert retrieval_settings["candidate_multiplier"]["value"] == 20
    assert retrieval_settings["context_expansion_mode"]["value"] == "none"
    assert reasoning_settings["max_context_length"]["value"] == 32000
    assert PrepareEvidencePipeline().max_context_length == 32000


def test_reranking_sees_extended_candidates_before_final_top_k_cut():
    docs = [doc(f"text-hit-{idx}") for idx in range(6)]
    docstore = RecordingDocStore(query_docs=docs)
    reranker = RecordingReranker()
    retrieval = VectorRetrieval(
        vector_store=RecordingVectorStore(),
        doc_store=docstore,
        embedding=DummyEmbedding(),
        retrieval_mode="text",
        first_round_top_k_mult=3,
        rerankers=[reranker],
    )

    results = retrieval.run("hello", top_k=2, do_extend=True, expand_parent=False)

    assert reranker.seen_doc_ids == [doc.doc_id for doc in docs]
    assert [result.doc_id for result in results] == ["text-hit-0", "text-hit-1"]


def test_query_expansion_adds_german_oral_exam_variant():
    retrieval = VectorRetrieval(
        vector_store=RecordingVectorStore(),
        doc_store=RecordingDocStore(),
        embedding=DummyEmbedding(),
        enable_query_expansion=True,
    )

    variants = retrieval.query_variants("When is an oral exam allowed?")

    assert any("mündliche Prüfung" in variant for variant in variants)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many ECTS are in the compulsory area?", "Pflichtbereich"),
        ("How many ECTS are in the elective compulsory area?", "Wahlpflichtbereich"),
        ("Who is the bachelor's program aimed at?", "Zielgruppe"),
        ("What teaching and learning methods are used?", "didaktische Konzepte"),
        ("What foreign-language competence is expected?", "Fremdsprachenkompetenz"),
        ("When is the foundations and orientation exam passed?", "40 ECTS-Punkten"),
        ("Which main study areas does the program combine?", "Informationsverarbeitende Systeme"),
        ("Which semester is suitable for study abroad?", "fünfte Studiensemester"),
        ("Which semester is suitable for studying abroad?", "Internationalisierung"),
        ("What must an unsupervised written assignment include?", "ohne Aufsicht"),
        ("How may I cite aids or sources?", "Verzeichnis der benutzten Hilfsmittel"),
        ("When does artificial intelligence count as cheating?", "Ghostwriter"),
        ("What algorithm topics are taught?", "Eigenschaften von Algorithmen"),
        (
            "What does the bachelor's thesis module require students to produce?",
            "Formulierung einer Forschungsfrage",
        ),
    ],
)
def test_query_expansion_adds_university_structure_terms(question, expected):
    retrieval = VectorRetrieval(
        vector_store=RecordingVectorStore(),
        doc_store=RecordingDocStore(),
        embedding=DummyEmbedding(),
        enable_query_expansion=True,
    )

    assert any(expected in variant for variant in retrieval.query_variants(question))


def test_sibling_expansion_includes_matched_child_before_neighbors():
    parent = "parent-1"
    docs = [
        doc("child-1", index_role="child", parent_id=parent, child_index=1),
        doc("child-2", index_role="child", parent_id=parent, child_index=2),
        doc("child-3", index_role="child", parent_id=parent, child_index=3),
    ]
    docstore = RecordingDocStore(docs=docs)
    retrieval = VectorRetrieval(
        vector_store=RecordingVectorStore(),
        doc_store=docstore,
        embedding=DummyEmbedding(),
    )
    matched = RetrievedDocument(
        text="child-2",
        id_="child-2",
        score=0.8,
        metadata={"index_role": "child", "parent_id": parent, "child_index": 2},
    )

    expanded = retrieval._expand_sibling_context([matched], window=1)

    assert [item.doc_id for item in expanded] == ["child-2", "child-1", "child-3"]
    assert expanded[0].metadata["context_role"] == "matched_child"
    assert expanded[1].metadata["context_role"] == "sibling_context"


def test_context_formatter_keeps_ranked_metadata_and_non_empty_context():
    from kotaemon.base import RetrievedDocument
    from kotaemon.indices.qa.format_context import PrepareEvidencePipeline

    formatter = PrepareEvidencePipeline(max_context_length=200)
    doc_ = RetrievedDocument(
        text="Mündliche Prüfungen sind in § 13 geregelt.",
        id_="doc-1",
        score=0.9,
        metadata={
            "source_file": "a_APO_ab_WS_14_15.pdf",
            "page_label_start": 8,
            "section_id": "§ 13",
            "paragraph_id": "Abs. 3",
            "index_role": "child",
            "retrieval_source": "both",
        },
    )

    _, evidence, _ = formatter.run([doc_]).content

    assert "[Context 1]" in evidence
    assert "source=a_APO_ab_WS_14_15.pdf" in evidence
    assert "§ 13" in evidence
    assert formatter.last_debug["context_tokens"] > 0
    assert formatter.last_debug["used_docs"] == 1
    assert formatter.last_debug["dropped_docs"] == []


def test_context_formatter_exposes_module_section_and_final_ranking():
    from kotaemon.indices.qa.format_context import PrepareEvidencePipeline

    formatter = PrepareEvidencePipeline(max_context_length=300)
    content = RetrievedDocument(
        text="Selbstständige Bearbeitung eines Themas.",
        id_="thesis-content",
        score=0.7,
        metadata={
            "source_file": "Modulkatalog.pdf",
            "module_title": "Bachelor Thesis",
            "section_title": "Inhalte und Themen",
            "section_path": ["Bachelor Thesis", "Inhalte und Themen"],
            "module_section": "contents",
            "_ranking_score": 0.42,
        },
    )

    _, evidence, _ = formatter.run([content]).content

    assert "module_title=Bachelor Thesis" in evidence
    assert "section_title=Inhalte und Themen" in evidence
    assert "module_section=contents" in evidence
    assert "ranking_score=0.42" in evidence


def test_context_formatter_reports_budget_dropped_docs():
    from kotaemon.base import RetrievedDocument
    from kotaemon.indices.qa.format_context import PrepareEvidencePipeline

    formatter = PrepareEvidencePipeline(max_context_length=80)
    docs = [
        RetrievedDocument(
            text=" ".join(["wichtiger Kontext"] * 10),
            id_=f"doc-{idx}",
            score=0.9,
            metadata={"source_file": "po.pdf", "section_id": f"§ {idx}"},
        )
        for idx in range(1, 4)
    ]

    formatter.run(docs)

    assert formatter.last_debug["input_docs"] == 3
    assert formatter.last_debug["used_docs"] < 3
    assert formatter.last_debug["dropped_docs"]


def test_rrf_lexical_signal_promotes_exact_university_candidate():
    retrieval = VectorRetrieval(
        vector_store=RecordingVectorStore(),
        doc_store=RecordingDocStore(),
        embedding=DummyEmbedding(),
    )
    generic = doc("generic", text="Allgemeine Hinweise zur Prüfung.")
    exact = doc(
        "exact",
        text="Mündliche Prüfungen dauern mindestens 15 und höchstens 60 Minuten.",
        section_title="§ 13 Bewertung der Prüfungsleistungen",
    )

    fused = retrieval._rrf_fuse(
        vector_docs=[generic, exact],
        vector_scores=[0.91, 0.89],
        text_docs=[],
        query_variants=["mündliche Prüfung 15 60 Minuten"],
    )

    assert fused[0].doc_id == "exact"
    assert fused[0].metadata["_lexical_score"] > fused[1].metadata["_lexical_score"]


def test_rrf_metadata_signal_prefers_named_module_assessment_chunk():
    retrieval = VectorRetrieval(
        vector_store=RecordingVectorStore(),
        doc_store=RecordingDocStore(),
        embedding=DummyEmbedding(),
    )
    wrong_module = doc(
        "digital-project",
        text="Modulnote: Schriftliche Ausarbeitung 50 Prozent.",
        module_title="Digital Project",
        module_section="assessment",
        section_title="Modulnote",
    )
    requested_module = doc(
        "digital-seminar",
        text="Softwareimplementierung 50 Prozent, Hausarbeit 30 Prozent, Präsentation 20 Prozent.",
        module_title="Digital Seminar in Data Science & Quantitative Applications",
        module_section="assessment",
        section_title="Modulnote",
    )

    fused = retrieval._rrf_fuse(
        vector_docs=[wrong_module, requested_module],
        vector_scores=[0.91, 0.89],
        text_docs=[],
        query_variants=[
            "How is Digital Seminar in Data Science & Quantitative Applications assessed?"
        ],
    )

    assert fused[0].doc_id == "digital-seminar"
    assert fused[0].metadata["_metadata_score"] == 1.0
    assert fused[1].metadata["_metadata_score"] == 0.4


def test_rrf_metadata_signal_prefers_study_description_prose_for_program_question():
    retrieval = VectorRetrieval(
        vector_store=RecordingVectorStore(),
        doc_store=RecordingDocStore(),
        embedding=DummyEmbedding(),
    )
    module = doc(
        "module-teaching",
        text="Lehr- und Lernformen: Vorlesung und Übung.",
        doc_type="module_catalog",
        chunk_type="module",
    )
    program = doc(
        "program-teaching",
        text="Die Lehre kombiniert Vorlesungen, Übungen und Projektarbeit.",
        doc_type="study_description",
        chunk_type="heading",
        section_path=["Didaktisches Konzept"],
    )

    fused = retrieval._rrf_fuse(
        vector_docs=[module, program],
        vector_scores=[0.91, 0.89],
        text_docs=[],
        query_variants=["What teaching and learning methods are used in the program?"],
    )

    assert fused[0].doc_id == "program-teaching"
    assert fused[0].metadata["_metadata_score"] == 0.5


@pytest.mark.parametrize(
    ("question", "direct_metadata", "distractor_metadata", "direct_text"),
    [
        (
            "What does the Bachelor Thesis module require students to produce?",
            {
                "doc_type": "module_catalog",
                "module_title": "Bachelor Thesis",
                "module_section": "contents",
                "section_title": "Inhalte und Themen",
            },
            {
                "doc_type": "module_catalog",
                "module_title": "Bachelor Thesis",
                "module_section": "assessment",
                "section_title": "Modulnote",
            },
            "Selbstständige Bearbeitung und Formulierung einer Forschungsfrage.",
        ),
        (
            "What algorithm topics are taught in Algorithms and Data Structures?",
            {
                "doc_type": "module_catalog",
                "module_title": "Algorithms and Data Structures",
                "module_section": "contents",
                "section_title": "Inhalte und Themen",
            },
            {
                "doc_type": "module_catalog",
                "module_title": "Algorithms and Data Structures",
                "module_section": "assessment",
                "section_title": "Modulnote",
            },
            "Eigenschaften von Algorithmen; Effizienz, Komplexität und Rekursion.",
        ),
    ],
)
def test_rrf_content_intent_prefers_contents_over_assessment(
    question, direct_metadata, distractor_metadata, direct_text
):
    retrieval = VectorRetrieval(
        vector_store=RecordingVectorStore(),
        doc_store=RecordingDocStore(),
        embedding=DummyEmbedding(),
    )
    distractor = doc("assessment", text="Klausur (100%).", **distractor_metadata)
    direct = doc("contents", text=direct_text, **direct_metadata)

    fused = retrieval._rrf_fuse(
        vector_docs=[distractor, direct],
        vector_scores=[0.91, 0.89],
        text_docs=[],
        query_variants=retrieval.query_variants(question),
    )

    assert fused[0].doc_id == "contents"
    assert fused[0].metadata["_metadata_score"] == 1.0
    assert fused[1].metadata["_metadata_score"] == 0.6


def test_rrf_legal_intent_prefers_exact_apo_paragraph():
    retrieval = VectorRetrieval(
        vector_store=RecordingVectorStore(),
        doc_store=RecordingDocStore(),
        embedding=DummyEmbedding(),
    )
    generic = doc(
        "generic-rule",
        text="Allgemeine Regelungen zu Prüfungsleistungen.",
        doc_type="amendment",
        section_title="§ 17 Prüfungsformen",
    )
    direct = doc(
        "ai-rule",
        text=(
            "Täuschung durch Ghostwriter oder Einsatz einer künstlichen Intelligenz, "
            "wenn diese nicht als Hilfsmittel zugelassen ist."
        ),
        doc_type="amendment",
        section_title="§ 27 Täuschung, Ordnungsverstoß",
    )

    fused = retrieval._rrf_fuse(
        vector_docs=[generic, direct],
        vector_scores=[0.91, 0.89],
        text_docs=[],
        query_variants=retrieval.query_variants(
            "When does artificial intelligence count as cheating under the exam rules?"
        ),
    )

    assert fused[0].doc_id == "ai-rule"
    assert fused[0].metadata["_metadata_score"] > fused[1].metadata["_metadata_score"]


@pytest.mark.parametrize(
    "question",
    [
        "Which main study areas does the program combine?",
        "Which semester is suitable for study abroad?",
    ],
)
def test_rrf_study_description_intent_prefers_prose_over_profile_tables(question):
    retrieval = VectorRetrieval(
        vector_store=RecordingVectorStore(),
        doc_store=RecordingDocStore(),
        embedding=DummyEmbedding(),
    )
    table = doc(
        "profile-table",
        text="Exemplarisches Studienprofil",
        doc_type="study_description",
        chunk_type="table",
    )
    prose = doc(
        "program-prose",
        text="Grundsätzliche Studienbereiche und Ausgestaltung der Internationalisierung.",
        doc_type="study_description",
        chunk_type="heading",
    )
    overview = doc(
        "unrelated-module-overview",
        text="Turnus des Angebots: fifth semester.",
        doc_type="module_catalog",
        module_title="Unrelated Module",
        module_section="overview",
        section_title="Module overview",
    )

    fused = retrieval._rrf_fuse(
        vector_docs=[overview, table, prose],
        vector_scores=[0.92, 0.91, 0.89],
        text_docs=[],
        query_variants=retrieval.query_variants(question),
    )

    assert fused[0].doc_id == "program-prose"
    assert fused[0].metadata["_metadata_score"] == 0.5


def test_context_formatter_uses_final_ranking_before_display_score():
    from kotaemon.base import RetrievedDocument
    from kotaemon.indices.qa.format_context import PrepareEvidencePipeline

    formatter = PrepareEvidencePipeline(max_context_length=500)
    docs = [
        RetrievedDocument(
            text="high display score but weak fused rank",
            id_="display-high",
            score=0.99,
            metadata={"_ranking_score": 0.01},
        ),
        RetrievedDocument(
            text="lower display score but best final rank",
            id_="rank-high",
            score=0.2,
            metadata={"_ranking_score": 0.5},
        ),
    ]

    _, evidence, _ = formatter.run(docs).content

    assert evidence.index("rank-high") < evidence.index("display-high")


def test_lancedb_get_all_supports_sibling_expansion(monkeypatch, tmp_path):
    events: list[tuple] = []

    class FakeSearch:
        def limit(self, top_k):
            events.append(("limit", top_k))
            return self

        def to_list(self):
            return [
                {
                    "id": "child-1",
                    "text": "first sibling",
                    "attributes": json.dumps(
                        {
                            "index_role": "child",
                            "parent_id": "parent",
                            "child_index": 1,
                        }
                    ),
                },
                {
                    "id": "child-2",
                    "text": "matched sibling",
                    "attributes": json.dumps(
                        {
                            "index_role": "child",
                            "parent_id": "parent",
                            "child_index": 2,
                        }
                    ),
                },
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
    docs = store.get_all()

    assert [doc.doc_id for doc in docs] == ["child-1", "child-2"]
    assert events == [("search", None, None), ("limit", 10**4)]
