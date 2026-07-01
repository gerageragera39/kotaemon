from __future__ import annotations

from kotaemon.base import Document, DocumentWithEmbedding
from kotaemon.embeddings.base import BaseEmbeddings
from kotaemon.indices.splitters import UniversityPDFChunker
from kotaemon.indices.vectorindex import VectorIndexing
from kotaemon.storages.docstores.in_memory import InMemoryDocumentStore


class DummyEmbedding(BaseEmbeddings):
    def invoke(self, docs, *args, **kwargs):
        if not isinstance(docs, list):
            docs = [docs]
        return [
            DocumentWithEmbedding(embedding=[1.0, 0.0], metadata=doc.metadata)
            for doc in docs
        ]


class DummyVectorStore:
    def __init__(self):
        self.ids = []

    def add(self, embeddings, ids):
        self.ids.extend(ids)


def structured_doc(
    text, order, element_type="paragraph", page=1, file_name="PO_BSc_Test.pdf"
):
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
        structured_doc(
            "§ 1 Zweck\n(1) Diese Ordnung regelt das Studium.\n(2) Sie gilt für alle Studierenden.",
            1,
        ),
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
    assert "Abschnitt: § 1 Zweck" in child.text
    assert "Studiengang:" not in child.text
    assert "Section:" not in child.text
    assert child.metadata["doc_family"] == "exam_regulation"
    assert child.metadata["page_label_start"] == 1
    assert child.metadata["page_label_end"] == 1
    assert child.metadata["token_count"] > 0


def test_regulation_two_line_title_and_major_heading_are_preserved():
    docs = [
        structured_doc("Prüfungsordnung", 0, "heading"),
        structured_doc("III. PRÜFUNGSORGANE", 1, "heading"),
        structured_doc("§ 8", 2),
        structured_doc("Prüfende, Beisitzende, Aufsichtsführende", 3),
        structured_doc("(1) Der Prüfungsausschuss bestellt Prüfende.", 4),
        structured_doc(
            "§ 9 Prüfungsausschuss\n(1) Der Prüfungsausschuss entscheidet.", 5
        ),
    ]

    chunks = UniversityPDFChunker(max_child_size=80).run(docs)
    children = [d for d in chunks if d.metadata["index_role"] == "child"]
    section_8 = [d for d in children if d.metadata.get("section_id") == "§ 8"]
    assert section_8
    assert all(
        d.metadata["section_title"] == "§ 8 Prüfende, Beisitzende, Aufsichtsführende"
        for d in section_8
    )
    assert all(d.metadata["major_heading"] == "III. PRÜFUNGSORGANE" for d in section_8)
    assert "Hauptüberschrift: III. PRÜFUNGSORGANE" in section_8[0].text
    assert not any("§ 9 Prüfungsausschuss" in d.text for d in section_8)


def test_regulation_child_never_contains_two_real_section_starts():
    docs = [
        structured_doc(
            "Prüfungsordnung\n§ 8\nPrüfende, Beisitzende, Aufsichtsführende\n"
            "(1) Inhalt zu § 8.\n§ 9 Prüfungsausschuss\n(1) Inhalt zu § 9.",
            0,
        )
    ]

    chunks = UniversityPDFChunker(max_child_size=500).run(docs)
    children = [d for d in chunks if d.metadata["index_role"] == "child"]

    assert {d.metadata["section_id"] for d in children} == {"§ 8", "§ 9"}
    for child in children:
        assert len(__import__("re").findall(r"(?m)^§\s*\d+", child.text)) <= 1
    assert not any("§ 8" in d.text and "§ 9" in d.text for d in children)


def test_table_chunks_keep_markdown_syntax():
    docs = [
        structured_doc(
            "Studienverlaufsplan",
            0,
            "heading",
            file_name="Studienverlaufsplan_BA_D3B.pdf",
        ),
        structured_doc(
            "Zusammenfassung\n| Semester | ECTS |\n| --- | --- |\n| 1 | 30 |",
            1,
            "table",
            file_name="Studienverlaufsplan_BA_D3B.pdf",
        ),
    ]
    chunks = UniversityPDFChunker().run(docs)
    table_children = [
        d
        for d in chunks
        if d.metadata.get("chunk_type") == "table"
        and d.metadata.get("index_role") == "child"
    ]
    assert table_children
    assert "|" in table_children[0].text
    assert "---" in table_children[0].text
    assert "Section path:" in table_children[0].text
    assert table_children[0].metadata.get("nearest_heading") == "Studienverlaufsplan"


def test_study_description_filename_wins_over_toc_studienverlaufsplan():
    chunker = UniversityPDFChunker()

    doc_type = chunker.detect_doc_type(
        "Studiengangsbeschreibung_BA_D3B.pdf",
        "Inhaltsverzeichnis\nStudienverlaufsplan ................................ 12",
    )

    assert doc_type == "study_description"


def test_study_description_uses_section_path_table_chunks():
    docs = [
        structured_doc(
            "Studiengangsbeschreibung",
            0,
            "heading",
            file_name="Studiengangsbeschreibung_BA_D3B.pdf",
        ),
        structured_doc(
            "2.6. Studienprofile",
            1,
            "heading",
            file_name="Studiengangsbeschreibung_BA_D3B.pdf",
        ),
        structured_doc(
            "Supply Chain Management & Logistics",
            2,
            "heading",
            file_name="Studiengangsbeschreibung_BA_D3B.pdf",
        ),
        structured_doc(
            "| Modul | ECTS |\n| --- | --- |\n| SCM Projektstudium | 5 |",
            3,
            "table",
            file_name="Studiengangsbeschreibung_BA_D3B.pdf",
        ),
        structured_doc(
            "federführende Fakultät: Wirtschaftswissenschaftliche Fakultät; "
            "verantwortlich beteiligt: Mathematisch-Geographische Fakultät",
            4,
            file_name="Studiengangsbeschreibung_BA_D3B.pdf",
        ),
    ]

    chunks = UniversityPDFChunker().run(docs)
    children = [d for d in chunks if d.metadata.get("index_role") == "child"]
    table = next(d for d in children if d.metadata.get("chunk_type") == "table")

    assert table.metadata["doc_type"] == "study_description"
    assert table.metadata["nearest_heading"] == "Supply Chain Management & Logistics"
    assert "2.6. Studienprofile > Supply Chain Management & Logistics" in table.text
    assert (
        table.metadata["semantic_title"]
        == "Supply Chain Management & Logistics profile table"
    )
    assert any("Wirtschaftswissenschaftliche Fakultät" in d.text for d in children)


def test_study_description_prose_keeps_its_own_section_path():
    file_name = "Studiengangsbeschreibung_BA_D3B.pdf"
    docs = [
        structured_doc("1. Studiengang", 0, "heading", page=2, file_name=file_name),
        structured_doc("1.2. Zielgruppe", 1, "heading", page=3, file_name=file_name),
        structured_doc(
            "Der Studiengang richtet sich an analytisch interessierte Studierende.",
            2,
            page=3,
            file_name=file_name,
        ),
        structured_doc(
            "2.6. Studienprofile", 3, "heading", page=8, file_name=file_name
        ),
        structured_doc(
            "| Profil | ECTS |\n| --- | --- |\n| Finance & Economics | 30 |",
            4,
            "table",
            page=8,
            file_name=file_name,
        ),
    ]

    chunks = UniversityPDFChunker().run(docs)
    prose = next(
        doc
        for doc in chunks
        if doc.metadata.get("index_role") == "child"
        and "analytisch interessierte" in doc.text
    )

    assert prose.metadata["section_path"] == ["1. Studiengang", "1.2. Zielgruppe"]
    assert prose.metadata["nearest_heading"] == "1.2. Zielgruppe"
    assert prose.metadata["chunk_type"] == "heading"
    assert "Finance & Economics" not in prose.text


def test_study_profile_heading_list_gets_summary_chunk():
    docs = [
        structured_doc(
            "2.6. Studienprofile\nEs werden in der Regel die folgenden Studienprofile angeboten:",
            0,
            "heading",
            file_name="Studiengangsbeschreibung_BA_D3B.pdf",
        ),
        structured_doc(
            "Accounting, Taxation & Controlling",
            1,
            "list_item",
            file_name="Studiengangsbeschreibung_BA_D3B.pdf",
        ),
        structured_doc(
            "Finance & Economics",
            2,
            "list_item",
            file_name="Studiengangsbeschreibung_BA_D3B.pdf",
        ),
        structured_doc(
            "Marketing, Organization, Innovation",
            3,
            "list_item",
            file_name="Studiengangsbeschreibung_BA_D3B.pdf",
        ),
        structured_doc(
            "Supply Chain Management & Logistics",
            4,
            "list_item",
            file_name="Studiengangsbeschreibung_BA_D3B.pdf",
        ),
    ]

    chunks = UniversityPDFChunker().run(docs)
    summaries = [
        d
        for d in chunks
        if d.metadata.get("index_role") == "child"
        and d.metadata.get("chunk_type") == "section_summary"
    ]

    assert summaries
    summary = summaries[0]
    assert summary.metadata["nearest_heading"] == "2.6. Studienprofile"
    for expected in [
        "Accounting, Taxation & Controlling",
        "Finance & Economics",
        "Marketing, Organization, Innovation",
        "Supply Chain Management & Logistics",
    ]:
        assert expected in summary.text


def test_module_chunks_include_module_title():
    docs = [
        structured_doc(
            "Modul: Data Analytics", 0, "heading", file_name="Module_DataCompetence.pdf"
        ),
        structured_doc(
            "Inhalte\nDatenanalyse und Visualisierung",
            1,
            file_name="Module_DataCompetence.pdf",
        ),
    ]
    chunks = UniversityPDFChunker().run(docs)
    children = [d for d in chunks if d.metadata.get("index_role") == "child"]
    assert children
    assert children[0].metadata["doc_type"] == "module_catalog"
    assert children[0].metadata["module_title"] == "Data Analytics"


def test_module_catalog_bare_headings_define_module_boundaries_and_metadata():
    docs = [
        structured_doc(
            "Bachelorarbeit",
            0,
            "heading",
            page=49,
            file_name="Modulkatalog_Bachelor_D3B_DE.pdf",
        ),
        structured_doc(
            "| Modultitel | Bachelorarbeit |\n| --- | --- |\n"
            "| Modultitel Englisch | Bachelor Thesis |\n"
            "| Modulnummer | 82-021-H-BA-0507 |\n"
            "| Leistungspunkte ECTS-Punkte | 10 ECTS |",
            1,
            "table",
            page=49,
            file_name="Modulkatalog_Bachelor_D3B_DE.pdf",
        ),
        structured_doc(
            "Modulnote :",
            2,
            "heading",
            page=50,
            file_name="Modulkatalog_Bachelor_D3B_DE.pdf",
        ),
        structured_doc(
            "Schriftliche Arbeit (100%)",
            3,
            "list_item",
            page=50,
            file_name="Modulkatalog_Bachelor_D3B_DE.pdf",
        ),
        structured_doc(
            "Digitales Projekt",
            4,
            "heading",
            page=51,
            file_name="Modulkatalog_Bachelor_D3B_DE.pdf",
        ),
        structured_doc(
            "| Modultitel | Digitales Projekt |\n| --- | --- |\n"
            "| Modultitel Englisch | Digital Project |\n"
            "| Modulnummer | 82-021-D3B03-H-0721 |\n"
            "| Leistungspunkte ECTS-Punkte | 10 |",
            5,
            "table",
            page=51,
            file_name="Modulkatalog_Bachelor_D3B_DE.pdf",
        ),
        structured_doc(
            "Modulnote :",
            6,
            "heading",
            page=52,
            file_name="Modulkatalog_Bachelor_D3B_DE.pdf",
        ),
        structured_doc(
            "Schriftliche Ausarbeitung (50 %)",
            7,
            "list_item",
            page=52,
            file_name="Modulkatalog_Bachelor_D3B_DE.pdf",
        ),
        structured_doc(
            "Endpräsentation (25 %)",
            8,
            "list_item",
            page=52,
            file_name="Modulkatalog_Bachelor_D3B_DE.pdf",
        ),
        structured_doc(
            "Digital Seminar in Data Science & Quantitative Applications",
            9,
            "heading",
            page=53,
            file_name="Modulkatalog_Bachelor_D3B_DE.pdf",
        ),
        structured_doc(
            "| Modultitel | Digital Seminar in Data Science & Quantitative Applications |\n| --- | --- |\n"
            "| Modultitel Englisch | Digital Seminar in Data Science & Quantitative Applications |\n"
            "| Modulnummer | 82-021-D3B09-H-0124 |",
            10,
            "table",
            page=53,
            file_name="Modulkatalog_Bachelor_D3B_DE.pdf",
        ),
        structured_doc(
            "Modulnote :",
            11,
            "heading",
            page=55,
            file_name="Modulkatalog_Bachelor_D3B_DE.pdf",
        ),
        structured_doc(
            "Softwareimplementierung (50 %)",
            12,
            "list_item",
            page=55,
            file_name="Modulkatalog_Bachelor_D3B_DE.pdf",
        ),
    ]

    chunks = UniversityPDFChunker(max_child_size=120, target_child_size=90).run(docs)
    children = [d for d in chunks if d.metadata.get("index_role") == "child"]
    parents = [d for d in chunks if d.metadata.get("index_role") == "parent"]

    assert all(d.metadata.get("module_title") for d in children)
    assert {
        "Bachelor Thesis",
        "Digital Project",
        "Digital Seminar in Data Science & Quantitative Applications",
    } <= {d.metadata.get("module_title") for d in children}

    digital_project = [
        d for d in children if d.metadata.get("module_title") == "Digital Project"
    ]
    assert digital_project
    assert all("Schriftliche Arbeit (100%)" not in d.text for d in digital_project)
    assert all(
        "Digital Seminar in Data Science & Quantitative Applications" not in d.text
        for d in digital_project
    )
    assessment = next(
        d for d in digital_project if "Schriftliche Ausarbeitung (50 %)" in d.text
    )
    assert assessment.metadata["section_title"] == "Modulnote"
    assert assessment.metadata["module_code"] == "82-021-D3B03-H-0721"
    assert assessment.metadata["section_path"] == ["Digital Project", "Modulnote"]
    assert "Module: Digital Project" in assessment.text

    project_parent = next(
        d for d in parents if d.metadata.get("module_title") == "Digital Project"
    )
    assert "Schriftliche Arbeit (100%)" not in project_parent.text
    assert "Softwareimplementierung (50 %)" not in project_parent.text


def test_short_terminal_tail_is_finalized_before_next_module():
    file_name = "Modulkatalog_Bachelor_D3B_DE.pdf"
    docs = [
        structured_doc(
            "Modul: Previous Module", 0, "heading", page=10, file_name=file_name
        ),
        structured_doc("Modulnote", 1, "heading", page=11, file_name=file_name),
        structured_doc("Klausur 100 %", 2, page=11, file_name=file_name),
        structured_doc("Bemerkungen", 3, "heading", page=11, file_name=file_name),
        structured_doc("Keine", 4, page=11, file_name=file_name),
        structured_doc("Next Module", 5, "heading", page=12, file_name=file_name),
        structured_doc(
            "| Modultitel | Nächstes Modul |\n| --- | --- |\n"
            "| Modultitel Englisch | Next Module |\n| Modulnummer | NEXT-101 |",
            6,
            "table",
            page=12,
            file_name=file_name,
        ),
        structured_doc("Modulnote", 7, "heading", page=13, file_name=file_name),
        structured_doc(
            "Projekt 50 %\nPräsentation 50 %", 8, page=13, file_name=file_name
        ),
    ]

    chunks = UniversityPDFChunker(max_child_size=500).run(docs)
    next_children = [
        doc
        for doc in chunks
        if doc.metadata.get("index_role") == "child"
        and doc.metadata.get("module_title") == "Next Module"
    ]

    assert next_children
    assert all("Klausur 100 %" not in doc.text for doc in next_children)
    assert all("Bemerkungen\n\nKeine" not in doc.text for doc in next_children)
    assert all(doc.metadata.get("module_code") == "NEXT-101" for doc in next_children)
    assert all(doc.metadata.get("section_title") for doc in next_children)
    assert all(doc.metadata.get("module_section") for doc in next_children)
    assert all(
        doc.metadata.get("page_label_start") is not None for doc in next_children
    )
    assert all(doc.metadata.get("page_label_end") is not None for doc in next_children)

    overview = next(
        doc for doc in next_children if doc.metadata["module_section"] == "overview"
    )
    assessment = next(
        doc for doc in next_children if doc.metadata["module_section"] == "assessment"
    )
    assert overview.metadata["section_title"] == "Module overview"
    assert overview.metadata["section_path"] == ["Next Module", "Module overview"]
    assert overview.metadata["page_label_start"] == 12
    assert overview.metadata["page_label_end"] == 12
    assert assessment.metadata["section_title"] == "Modulnote"
    assert assessment.metadata["section_path"] == ["Next Module", "Modulnote"]
    assert assessment.metadata["page_label_start"] == 13
    assert assessment.metadata["page_label_end"] == 13


def test_long_module_splits_into_deterministic_sections_with_metadata():
    docs = [
        structured_doc(
            "Modul: Data Analytics", 0, "heading", file_name="Module_DataCompetence.pdf"
        ),
        structured_doc(
            "Modulnummer: D3B-101", 1, file_name="Module_DataCompetence.pdf"
        ),
        structured_doc("ECTS 6\nSemester: 2", 2, file_name="Module_DataCompetence.pdf"),
        structured_doc(
            "Inhalte\n" + "Datenanalyse. " * 120,
            3,
            file_name="Module_DataCompetence.pdf",
        ),
        structured_doc(
            "Kompetenzen\n" + "Kompetenzaufbau. " * 120,
            4,
            file_name="Module_DataCompetence.pdf",
        ),
        structured_doc("Prüfung\nKlausur", 5, file_name="Module_DataCompetence.pdf"),
    ]
    chunks = UniversityPDFChunker(max_child_size=120, target_child_size=80).run(docs)
    children = [d for d in chunks if d.metadata.get("index_role") == "child"]

    assert children
    assert {d.metadata.get("module_title") for d in children} == {"Data Analytics"}
    assert {d.metadata.get("module_number") for d in children} == {"D3B-101"}
    assert {d.metadata.get("ects") for d in children} == {"6"}
    assert any(d.metadata.get("module_section") == "contents" for d in children)
    assert any(d.metadata.get("module_section") == "competencies" for d in children)


def test_vector_indexing_does_not_embed_parent_docs():
    parent = Document(text="parent", id_="parent-1", metadata={"index_role": "parent"})
    child = Document(
        text="child",
        id_="child-1",
        metadata={"index_role": "child", "parent_id": "parent-1"},
    )
    vector_store = DummyVectorStore()
    indexing = VectorIndexing(
        vector_store=vector_store,
        doc_store=InMemoryDocumentStore(),
        embedding=DummyEmbedding(),
    )

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
    retrieval = VectorRetrieval(
        vector_store=DummyVectorStore(), doc_store=doc_store, embedding=DummyEmbedding()
    )

    expanded = retrieval._expand_parent_context([child])

    assert len(expanded) == 1
    assert expanded[0].doc_id == "parent-1"
    assert expanded[0].text == "full parent section"
    assert expanded[0].score == 0.75
    assert expanded[0].metadata["expanded_from_child_ids"] == ["child-1"]


def test_index_pipeline_university_reader_mode_routes_pdf_to_structural_chunker(
    tmp_path, monkeypatch
):
    from kotaemon.indices.splitters import TokenSplitter
    from kotaemon.loaders import DoclingStructuredPDFReader
    from ktem.index.file import pipelines as file_pipelines
    from ktem.index.file.pipelines import IndexDocumentPipeline

    monkeypatch.setattr(file_pipelines, "dev_settings", lambda: ({}, None, None))
    monkeypatch.delenv("UNIVERSITY_RAG_PDF_MODE", raising=False)
    monkeypatch.delenv("KOTAEMON_FILE_INDEX_PDF_MODE", raising=False)
    monkeypatch.delenv("UNIVERSITY_RAG_DOCUMENTS_DIR", raising=False)

    pipeline = IndexDocumentPipeline(
        embedding=DummyEmbedding(),
        reader_mode="university",
        university_target_child_size=551,
        university_min_child_size=121,
        university_max_child_size=901,
        university_overlap=81,
        university_parent_max_size=2501,
        Source=None,
        Index=None,
        VS=None,
        DS=None,
        FSPath=tmp_path,
        user_id="test",
        private=False,
    )

    routed = pipeline.route(tmp_path / "course.pdf")

    assert isinstance(routed.loader, DoclingStructuredPDFReader)
    assert isinstance(routed.splitter, UniversityPDFChunker)
    assert routed.splitter.target_child_size == 551
    assert routed.splitter.min_child_size == 121
    assert routed.splitter.max_child_size == 901
    assert routed.splitter.overlap == 81
    assert routed.splitter.parent_max_size == 2501

    non_pdf = pipeline.route(tmp_path / "notes.txt")
    assert isinstance(non_pdf.splitter, TokenSplitter)
    assert not isinstance(non_pdf.splitter, UniversityPDFChunker)


def test_index_pipeline_docling_reader_mode_keeps_token_splitter_without_university_gate(
    tmp_path, monkeypatch
):
    from kotaemon.indices.splitters import TokenSplitter
    from ktem.index.file import pipelines as file_pipelines
    from ktem.index.file.pipelines import IndexDocumentPipeline

    monkeypatch.setattr(file_pipelines, "dev_settings", lambda: ({}, None, None))
    monkeypatch.delenv("UNIVERSITY_RAG_PDF_MODE", raising=False)
    monkeypatch.delenv("KOTAEMON_FILE_INDEX_PDF_MODE", raising=False)
    monkeypatch.delenv("UNIVERSITY_RAG_DOCUMENTS_DIR", raising=False)

    pipeline = IndexDocumentPipeline(
        embedding=DummyEmbedding(),
        reader_mode="docling",
        Source=None,
        Index=None,
        VS=None,
        DS=None,
        FSPath=tmp_path,
        user_id="test",
        private=False,
    )

    routed = pipeline.route(tmp_path / "ordinary.pdf")

    assert isinstance(routed.splitter, TokenSplitter)
    assert not isinstance(routed.splitter, UniversityPDFChunker)


def test_index_pipeline_user_settings_expose_university_reader_mode_and_defaults():
    from ktem.index.file.pipelines import IndexDocumentPipeline

    settings = IndexDocumentPipeline.get_user_settings()

    assert ("University PDF structural chunking", "university") in settings[
        "reader_mode"
    ]["choices"]
    assert settings["university_target_child_size"]["value"] == 550
    assert settings["university_min_child_size"]["value"] == 120
    assert settings["university_max_child_size"]["value"] == 900
    assert settings["university_overlap"]["value"] == 80
    assert settings["university_parent_max_size"]["value"] == 2500


def test_vector_retrieval_parent_expansion_returns_score_sorted_results():
    from kotaemon.base import RetrievedDocument
    from kotaemon.indices.vectorindex import VectorRetrieval

    parent = Document(
        text="expanded high score parent",
        id_="parent-high",
        metadata={"index_role": "parent"},
    )
    passthrough = RetrievedDocument(
        text="low score regular", id_="regular-low", score=0.1, metadata={}
    )
    child = RetrievedDocument(
        text="high score child",
        id_="child-high",
        score=0.9,
        metadata={"index_role": "child", "parent_id": "parent-high"},
    )
    doc_store = InMemoryDocumentStore()
    doc_store.add(parent)
    retrieval = VectorRetrieval(
        vector_store=DummyVectorStore(), doc_store=doc_store, embedding=DummyEmbedding()
    )

    expanded = retrieval._expand_parent_context([passthrough, child])

    assert [doc.doc_id for doc in expanded] == ["parent-high", "regular-low"]
    assert expanded[0].score == 0.9


def test_docling_structured_reader_uses_plain_lazy_converter(monkeypatch, tmp_path):
    import sys
    import types

    from kotaemon.loaders import DoclingStructuredPDFReader

    class FakeDocument:
        def export_to_dict(self):
            return {
                "body": {"children": [{"$ref": "#/texts/0"}]},
                "texts": [
                    {
                        "label": "paragraph",
                        "text": "§ 1 Testinhalt",
                        "prov": [{"page_no": 3}],
                    }
                ],
                "tables": [],
                "pictures": [],
            }

    class FakeConversionResult:
        document = FakeDocument()

    class FakeDocumentConverter:
        instances = 0

        def __init__(self):
            FakeDocumentConverter.instances += 1

        def convert(self, file_path):
            return FakeConversionResult()

    fake_docling_converter_module = types.SimpleNamespace(
        DocumentConverter=FakeDocumentConverter
    )
    monkeypatch.setitem(sys.modules, "docling", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "docling.document_converter", fake_docling_converter_module
    )

    reader = DoclingStructuredPDFReader()
    docs = reader.load_data(tmp_path / "apo.pdf", extra_info={"file_id": "file-1"})

    assert FakeDocumentConverter.instances == 1
    assert reader.converter_ is reader.converter_
    assert docs[0].text == "§ 1 Testinhalt"
    assert docs[0].metadata["order"] == 0
    assert docs[0].metadata["page_label"] == 3
    assert docs[0].metadata["file_id"] == "file-1"


def test_university_chunker_adds_child_index_and_paragraph_metadata():
    docs = [
        structured_doc("§ 13 Bewertung der Prüfungsleistungen", 0, "heading"),
        structured_doc(
            "(1) Erste Regelung.\n(2) Zweite Regelung zu mündlichen Prüfungen.", 1
        ),
    ]

    chunks = UniversityPDFChunker(target_child_size=40, max_child_size=80).run(docs)
    children = [doc for doc in chunks if doc.metadata.get("index_role") == "child"]
    parents = [doc for doc in chunks if doc.metadata.get("index_role") == "parent"]

    assert parents and children
    assert all(child.metadata.get("parent_id") for child in children)
    assert all(child.metadata.get("child_index") for child in children)
    assert any(child.metadata.get("paragraph_id") == "Abs. 1" for child in children)
    assert "Paragraph: Abs." in children[0].text
