from __future__ import annotations

from kotaemon.base import Document, DocumentWithEmbedding
from kotaemon.indices.splitters import UniversityPDFChunker
from kotaemon.embeddings.base import BaseEmbeddings
from kotaemon.indices.vectorindex import VectorIndexing
from kotaemon.storages.docstores.in_memory import InMemoryDocumentStore


class DummyEmbedding(BaseEmbeddings):
    def invoke(self, docs, *args, **kwargs):
        if not isinstance(docs, list):
            docs = [docs]
        return [DocumentWithEmbedding(embedding=[1.0, 0.0], metadata=doc.metadata) for doc in docs]


class DummyVectorStore:
    def __init__(self):
        self.ids = []

    def add(self, embeddings, ids):
        self.ids.extend(ids)


def structured_doc(text, order, element_type="paragraph", page=1, file_name="PO_BSc_Test.pdf"):
    return Document(
        text=text,
        metadata={
            "file_name": file_name,
            "file_path": f"/tmp/{file_name}",
            "order": order,
            "element_type": element_type,
            "page_label": page,
        },
    )


def test_regulation_chunks_preserve_section_title_and_required_metadata():
    docs = [
        structured_doc("Prüfungsordnung", 0, "heading"),
        structured_doc("§ 1 Zweck\n(1) Diese Ordnung regelt das Studium.\n(2) Sie gilt für alle Studierenden.", 1),
    ]
    chunks = UniversityPDFChunker().run(docs)
    children = [d for d in chunks if d.metadata["index_role"] == "child"]
    parents = [d for d in chunks if d.metadata["index_role"] == "parent"]

    assert parents
    assert children
    child = children[0]
    assert child.metadata["doc_type"] == "exam_regulation"
    assert child.metadata["parent_id"]
    assert child.metadata["chunk_id"]
    assert child.metadata["source_file"] == "PO_BSc_Test.pdf"
    assert child.metadata["section_title"].startswith("§ 1")
    assert child.metadata["section_title"] in child.text
    assert "Dokument: PO_BSc_Test.pdf" in child.text


def test_table_chunks_keep_markdown_syntax():
    docs = [
        structured_doc("Studienverlaufsplan", 0, "heading", file_name="Studienverlaufsplan_BA_D3B.pdf"),
        structured_doc(
            "Zusammenfassung\n| Semester | ECTS |\n| --- | --- |\n| 1 | 30 |",
            1,
            "table",
            file_name="Studienverlaufsplan_BA_D3B.pdf",
        ),
    ]
    chunks = UniversityPDFChunker().run(docs)
    table_children = [d for d in chunks if d.metadata.get("chunk_type") == "table" and d.metadata.get("index_role") == "child"]
    assert table_children
    assert "|" in table_children[0].text
    assert "---" in table_children[0].text


def test_module_chunks_include_module_title():
    docs = [
        structured_doc("Modul: Data Analytics", 0, "heading", file_name="Module_DataCompetence.pdf"),
        structured_doc("Inhalte\nDatenanalyse und Visualisierung", 1, file_name="Module_DataCompetence.pdf"),
    ]
    chunks = UniversityPDFChunker().run(docs)
    children = [d for d in chunks if d.metadata.get("index_role") == "child"]
    assert children
    assert children[0].metadata["doc_type"] == "module_catalog"
    assert children[0].metadata["module_title"] == "Data Analytics"


def test_vector_indexing_does_not_embed_parent_docs():
    parent = Document(text="parent", id_="parent-1", metadata={"index_role": "parent"})
    child = Document(text="child", id_="child-1", metadata={"index_role": "child", "parent_id": "parent-1"})
    vector_store = DummyVectorStore()
    indexing = VectorIndexing(vector_store=vector_store, doc_store=InMemoryDocumentStore(), embedding=DummyEmbedding())

    indexing.run([parent, child])

    assert vector_store.ids == ["child-1"]
    assert indexing.doc_store.count() == 2


def test_vector_retrieval_expands_child_to_parent_context():
    from kotaemon.base import RetrievedDocument
    from kotaemon.indices.vectorindex import VectorRetrieval

    parent = Document(
        text="full parent section",
        id_="parent-1",
        metadata={"index_role": "parent", "parent_id": "parent-1"},
    )
    child = RetrievedDocument(
        text="child section",
        id_="child-1",
        score=0.75,
        metadata={"index_role": "child", "parent_id": "parent-1"},
    )
    doc_store = InMemoryDocumentStore()
    doc_store.add(parent)
    retrieval = VectorRetrieval(vector_store=DummyVectorStore(), doc_store=doc_store, embedding=DummyEmbedding())

    expanded = retrieval._expand_parent_context([child])

    assert len(expanded) == 1
    assert expanded[0].doc_id == "parent-1"
    assert expanded[0].text == "full parent section"
    assert expanded[0].score == 0.75
    assert expanded[0].metadata["expanded_from_child_ids"] == ["child-1"]
