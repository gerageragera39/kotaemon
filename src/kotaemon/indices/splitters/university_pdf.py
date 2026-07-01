from __future__ import annotations

import hashlib
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

from kotaemon.base import Document

from . import BaseSplitter

STUDY_PROGRAM = "B.Sc. Digital and Data-Driven Business"
SECTION_START_RE = re.compile(r"(?m)^\s*(§\s*\d+[a-zA-Z]?\b[^\n]*)")
PARAGRAPH_RE = re.compile(
    r"(?im)(?:^|\n)\s*(\(\s*\d+[a-z]?\s*\)|Abs\.\s*\d+[a-z]?|Absatz\s+\d+[a-z]?)"
)
SENTENCE_RE = re.compile(r"(?im)\b(Satz\s+\d+[a-z]?)\b")


@dataclass
class _Block:
    title: str
    text: str
    elements: list[Document]
    chunk_type: str
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    major_heading: Optional[str] = None
    module_title: Optional[str] = None
    module_number: Optional[str] = None
    ects: Optional[str] = None
    semester: Optional[str] = None
    module_section: Optional[str] = None
    section_path: Optional[list[str]] = None
    nearest_heading: Optional[str] = None
    table_caption: Optional[str] = None
    semantic_title: Optional[str] = None
    module_code: Optional[str] = None
    module_title_de: Optional[str] = None
    module_title_en: Optional[str] = None


class UniversityPDFChunker(BaseSplitter):
    """Structure-aware chunker for German university PDFs.

    The chunker consumes ordered Docling elements and emits parent/child
    documents. Parent chunks are stored in the docstore only; child chunks are
    embedded for retrieval and carry deterministic headers for context.
    """

    def __init__(
        self,
        target_child_size: int = 550,
        min_child_size: int = 120,
        max_child_size: int = 900,
        overlap: int = 80,
        parent_max_size: int = 2500,
        study_program: str = STUDY_PROGRAM,
    ):
        super().__init__()
        object.__setattr__(self, "target_child_size", target_child_size)
        object.__setattr__(self, "min_child_size", min_child_size)
        object.__setattr__(self, "max_child_size", max_child_size)
        object.__setattr__(self, "overlap", overlap)
        object.__setattr__(self, "parent_max_size", parent_max_size)
        object.__setattr__(self, "study_program", study_program)

    def run(self, documents: list[Document], **kwargs) -> list[Document]:
        if not documents:
            return []

        ordered = sorted(documents, key=lambda d: d.metadata.get("order", 0))
        file_name = self._source_file(ordered)
        first_text = "\n".join(doc.text for doc in ordered[:20] if doc.text)
        doc_type = self.detect_doc_type(file_name, first_text)
        doc_family = self._doc_family(doc_type, file_name)
        revision_date = self._revision_date(file_name)
        is_latest = self._is_probably_latest(revision_date)
        source_meta = {
            "source_file": file_name,
            "file_name": file_name,
            "file_path": self._file_path(ordered),
            "doc_type": doc_type,
            "doc_family": doc_family,
            "revision_date": revision_date,
            "is_probably_latest": is_latest,
        }

        if doc_type in {"general_regulation", "exam_regulation", "amendment"}:
            blocks = self._regulation_blocks(ordered)
        elif doc_type == "module_catalog":
            blocks = self._module_blocks(ordered, file_name)
        elif doc_type == "study_description":
            blocks = self._section_path_blocks(ordered, doc_type=doc_type)
        elif doc_type in {"study_plan", "elective_catalog"}:
            blocks = self._table_first_blocks(ordered, doc_type)
        elif doc_type == "form":
            blocks = self._form_blocks(ordered)
        else:
            blocks = self._generic_blocks(ordered)

        output: list[Document] = []
        for parent_index, block in enumerate(blocks, start=1):
            if not block.text.strip():
                continue
            parent_id = self._stable_id(
                file_name, doc_type, "parent", parent_index, block.title
            )
            parent_doc = self._make_parent(
                block=block,
                parent_id=parent_id,
                source_meta=source_meta,
            )
            output.append(parent_doc)

            for child_index, child_text in enumerate(self._child_texts(block), start=1):
                if not child_text.strip():
                    continue
                child_doc = self._make_child(
                    block=block,
                    child_text=child_text,
                    parent_id=parent_id,
                    child_index=child_index,
                    source_meta=source_meta,
                )
                output.append(child_doc)

        return output

    def detect_doc_type(self, file_name: str, first_text: str = "") -> str:
        # Filename rules must win over body/TOC heuristics.  The D3B
        # Studiengangsbeschreibung contains a table-of-contents occurrence of
        # "Studienverlaufsplan"; treating the whole filename+body as one haystack
        # misclassifies it as a study plan and triggers table-first chunking.
        normalized_file = self._normalize(Path(file_name).name)
        if "studiengangsbeschreibung" in normalized_file:
            return "study_description"
        if "studienverlaufsplan" in normalized_file:
            return "study_plan"
        if "wahlpflichtkatal" in normalized_file:
            return "elective_catalog"

        haystack = f"{file_name}\n{first_text[:4000]}".lower()
        normalized = self._normalize(haystack)

        if any(
            term in normalized
            for term in [
                "aenderung",
                "anderung",
                "aenderungssatzung",
                "anderungssatzung",
            ]
        ):
            return "amendment"
        if "studiengangsbeschreibung" in normalized:
            return "study_description"
        if "studienverlaufsplan" in normalized:
            return "study_plan"
        if "wahlpflichtkatal" in normalized:
            return "elective_catalog"
        if any(
            term in normalized
            for term in ["anmeldung_bachelorarbeit", "zeugnisantrag", "umfrage"]
        ):
            return "form"
        if "modulkatalog" in normalized or re.search(
            r"(^|[/_\-\s])module[_\-\s]", normalized
        ):
            return "module_catalog"
        if (
            "po_bsc" in normalized
            or "prufungsordnung" in normalized
            or "prüfungsordnung" in haystack
        ):
            return "exam_regulation"
        if re.search(r"(^|[/_\-\s])apo([_\-\.\s]|$)", normalized) or (
            "allgemeine prufungsordnung" in normalized
        ):
            return "general_regulation"
        return "generic_pdf"

    # --- strategy block builders -------------------------------------------------

    def _regulation_blocks(self, docs: list[Document]) -> list[_Block]:
        """Build exactly one parent block per real § using ordered Docling items."""

        blocks: list[_Block] = []
        current_docs: list[Document] = []
        current_parts: list[str] = []
        current_major_heading: Optional[str] = None
        section_major_heading: Optional[str] = None

        def finalize() -> None:
            nonlocal current_docs, current_parts, section_major_heading
            section_text = "\n\n".join(
                part.strip() for part in current_parts if part.strip()
            ).strip()
            if not section_text:
                current_docs = []
                current_parts = []
                return

            section_id, section_title = self._section_identity(section_text)
            blocks.append(
                _Block(
                    title=section_title,
                    text=section_text,
                    elements=list(current_docs),
                    chunk_type="section",
                    section_id=section_id,
                    section_title=section_title,
                    major_heading=section_major_heading,
                )
            )
            current_docs = []
            current_parts = []
            section_major_heading = None

        for doc in docs:
            text = (doc.text or "").strip()
            if not text:
                continue

            if self._is_major_heading(text):
                if current_parts:
                    finalize()
                current_major_heading = self._clean_title(text)
                continue

            for starts_section, segment in self._split_text_at_section_starts(text):
                if starts_section:
                    if current_parts:
                        finalize()
                    section_major_heading = current_major_heading
                    current_docs = [doc]
                    current_parts = [segment]
                elif current_parts:
                    current_docs.append(doc)
                    current_parts.append(segment)
                else:
                    major_heading = self._major_heading_from_text(segment)
                    if major_heading:
                        current_major_heading = major_heading

        if current_parts:
            finalize()

        return blocks or self._generic_blocks(docs)

    def _module_blocks(self, docs: list[Document], file_name: str) -> list[_Block]:
        blocks: list[_Block] = []
        current: list[Document] = []
        current_title: Optional[str] = None
        current_module_meta: dict[str, Optional[str]] = {}

        def append_current() -> None:
            nonlocal current, current_title, current_module_meta
            if not current:
                return
            text = self._join_docs(current)
            text_metadata = self._module_metadata(text, current_title or "")
            metadata = {
                **text_metadata,
                **{key: value for key, value in current_module_meta.items() if value},
            }
            title = (
                current_title
                or metadata.get("module_title")
                or metadata.get("module_title_en")
                or metadata.get("module_title_de")
                or self._module_title_from_text(text)
                or self._title_from_filename(file_name)
            )
            blocks.append(
                _Block(
                    title=title,
                    text=text,
                    elements=list(current),
                    chunk_type="module",
                    module_title=title,
                    module_number=metadata.get("module_number")
                    or metadata.get("module_code"),
                    module_code=metadata.get("module_code")
                    or metadata.get("module_number"),
                    ects=metadata.get("ects"),
                    semester=metadata.get("semester"),
                    module_title_de=metadata.get("module_title_de"),
                    module_title_en=metadata.get("module_title_en"),
                    section_path=[title],
                )
            )
            current = []
            current_title = None
            current_module_meta = {}

        ordered = sorted(docs, key=lambda d: d.metadata.get("order", 0))
        for idx, doc in enumerate(ordered):
            text = (doc.text or "").strip()
            if not text:
                continue
            detected_meta = self._detect_module_start(ordered, idx)
            detected = detected_meta.get("module_title") if detected_meta else None
            current_text = self._join_docs(current)
            different_title = self._normalize_for_match(
                detected or ""
            ) != self._normalize_for_match(current_title or "")
            starts_new = bool(detected and current and different_title)

            # Never carry terminal fields from module A into module B.  The
            # previous implementation used a token threshold here, so a short
            # Modulnote/Bemerkungen/Polyvalenz tail could be prepended to the
            # next module before the first child window was built.
            if (
                detected
                and current
                and self._contains_module_terminal_section(current_text)
            ):
                starts_new = different_title
            if starts_new:
                append_current()
            if detected:
                current_title = detected
                current_module_meta.update(
                    {key: value for key, value in detected_meta.items() if value}
                )
            current.append(doc)

        append_current()
        return blocks or self._generic_blocks(docs)

    def _table_first_blocks(self, docs: list[Document], doc_type: str) -> list[_Block]:
        blocks: list[_Block] = []
        structured_docs = self._with_section_metadata(docs)
        for idx, doc in enumerate(structured_docs, start=1):
            if doc.metadata.get("element_type") != "table":
                continue
            page = doc.metadata.get("page_label") or "unbekannt"
            section_path = self._metadata_section_path(doc)
            nearest_heading = doc.metadata.get("nearest_heading")
            caption = self._table_caption(doc.text)
            title = self._semantic_table_title(
                doc.text, nearest_heading=nearest_heading, section_path=section_path
            )
            text = self._table_block_text(
                doc=doc,
                doc_type=doc_type,
                section_path=section_path,
                nearest_heading=nearest_heading,
                table_caption=caption,
                page=page,
            )
            blocks.append(
                _Block(
                    title=title,
                    text=text,
                    elements=[doc],
                    chunk_type="table",
                    section_path=section_path,
                    nearest_heading=nearest_heading,
                    table_caption=caption,
                    semantic_title=title,
                )
            )

        non_table = [
            doc
            for doc in structured_docs
            if doc.metadata.get("element_type") != "table"
        ]
        for block in self._generic_blocks(non_table):
            if self._should_keep_short_block(block):
                blocks.append(block)
        return blocks or self._generic_blocks(docs)

    def _form_blocks(self, docs: list[Document]) -> list[_Block]:
        full_text = self._join_docs(docs)
        categories = [
            ("Zweck", r"(?i)(zweck|antrag|anmeldung|zeugnis|umfrage|abschlussarbeit)"),
            (
                "Pflichtfelder",
                r"(?i)(name|matrikel|adresse|e-?mail|studiengang|thema|prüfer|pruefer)",
            ),
            (
                "Fristen und Einreichung",
                r"(?i)(frist|abgabe|einreich|submit|deadline|datum)",
            ),
            (
                "Unterschriften und Erklärungen",
                r"(?i)(unterschrift|signature|erklärung|erklaerung|datenschutz|bestätigung|bestaetigung)",
            ),
            (
                "Kontakt/Prüfungsamt",
                r"(?i)(kontakt|prüfungsamt|pruefungsamt|dekanat|büro|buero|office)",
            ),
        ]
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        buckets: dict[str, list[str]] = {name: [] for name, _ in categories}
        buckets["Weitere Formularangaben"] = []
        for line in lines:
            matched = False
            for name, pattern in categories:
                if re.search(pattern, line):
                    buckets[name].append(line)
                    matched = True
                    break
            if not matched:
                buckets["Weitere Formularangaben"].append(line)

        blocks: list[_Block] = []
        for name, items in buckets.items():
            if not items:
                continue
            checklist = "\n".join(f"- [ ] {item}" for item in items)
            blocks.append(
                _Block(
                    title=name,
                    text=f"{name}\n{checklist}",
                    elements=docs,
                    chunk_type="form_block",
                )
            )
        return blocks or [_Block("Formular", full_text, docs, "form_block")]

    def _generic_blocks(self, docs: list[Document]) -> list[_Block]:
        return self._section_path_blocks(docs)

    def _section_path_blocks(
        self, docs: list[Document], doc_type: str = "generic_pdf"
    ) -> list[_Block]:
        if not docs:
            return []
        docs = self._with_section_metadata(docs)
        blocks: list[_Block] = []
        current: list[Document] = []
        title = "Dokument"
        current_path: list[str] | None = None
        current_heading: Optional[str] = None

        def append_current() -> None:
            nonlocal current, title, current_path, current_heading
            if not current:
                return
            text = self._join_docs(current)
            if not text.strip():
                current = []
                return
            blocks.append(
                _Block(
                    title=title,
                    text=self._section_block_text(
                        text=text,
                        section_path=current_path,
                        nearest_heading=current_heading,
                    ),
                    elements=list(current),
                    chunk_type="heading",
                    section_path=list(current_path or []),
                    nearest_heading=current_heading,
                    semantic_title=title,
                )
            )
            current = []

        for doc in docs:
            element_type = doc.metadata.get("element_type")
            is_heading = doc.metadata.get("element_type") == "heading"
            path = self._metadata_section_path(doc)
            nearest = doc.metadata.get("nearest_heading")
            if element_type == "table":
                append_current()
                page = doc.metadata.get("page_label") or "unbekannt"
                caption = self._table_caption(doc.text)
                table_title = self._semantic_table_title(
                    doc.text, nearest_heading=nearest, section_path=path
                )
                blocks.append(
                    _Block(
                        title=table_title,
                        text=self._table_block_text(
                            doc=doc,
                            doc_type=str(doc.metadata.get("doc_type") or doc_type),
                            section_path=path,
                            nearest_heading=nearest,
                            table_caption=caption,
                            page=page,
                        ),
                        elements=[doc],
                        chunk_type="table",
                        section_path=path,
                        nearest_heading=nearest,
                        table_caption=caption,
                        semantic_title=table_title,
                    )
                )
                continue
            if (is_heading or path != current_path) and current:
                append_current()
            if is_heading:
                title = (doc.text or "").strip()[:140] or title
            elif path:
                title = path[-1]
            current_path = path
            current_heading = nearest
            current.append(doc)
        if current:
            append_current()
        blocks.extend(self._section_list_summary_blocks(docs))
        return blocks

    # --- chunk materialization ---------------------------------------------------

    def _make_parent(
        self, block: _Block, parent_id: str, source_meta: dict[str, Any]
    ) -> Document:
        metadata = {
            **source_meta,
            "index_role": "parent",
            "parent_id": parent_id,
            "chunk_type": f"parent_{block.chunk_type}",
            "section_id": block.section_id,
            "section_title": block.section_title,
            "paragraph_id": None,
            "sentence_id": None,
            "major_heading": block.major_heading,
            "module_title": block.module_title,
            "module_number": block.module_number,
            "module_code": block.module_code or block.module_number,
            "module_title_de": block.module_title_de,
            "module_title_en": block.module_title_en,
            "module_pages": self._page_range(
                self._page_start(block.elements), self._page_end(block.elements)
            ),
            "ects": block.ects,
            "semester": block.semester,
            "module_section": block.module_section,
            "section_path": block.section_path,
            "nearest_heading": block.nearest_heading,
            "table_caption": block.table_caption,
            "semantic_title": block.semantic_title,
            "page_label_start": self._page_start(block.elements),
            "page_label_end": self._page_end(block.elements),
            "token_count": self._token_count(block.text),
            "content_hash": self._hash(block.text),
            "full_text": block.text,
        }
        return Document(text=block.text, id_=parent_id, metadata=metadata)

    def _make_child(
        self,
        block: _Block,
        child_text: str,
        parent_id: str,
        child_index: int,
        source_meta: dict[str, Any],
    ) -> Document:
        section_label = (
            block.semantic_title
            or block.section_title
            or block.module_title
            or block.title
        )
        paragraph_id = self._paragraph_id(child_text)
        sentence_id = self._sentence_id(child_text)
        page_start = self._page_start(block.elements)
        page_end = self._page_end(block.elements)
        # Keep the embedded child text content-centric.  Large repeated headers
        # made chunks from the same file overly similar; metadata is still carried
        # separately and rendered by PrepareEvidencePipeline.
        child_section_title = block.section_title
        child_section_path = block.section_path
        if block.chunk_type == "module":
            child_section_title = self._module_section_title(child_text)
            child_section_path = [
                part for part in [block.module_title, child_section_title] if part
            ]
            section_label = child_section_title or block.module_title or block.title
            page_start, page_end = self._module_child_pages(block, child_section_title)

        page_label = self._page_range(page_start, page_end)

        header_lines = [
            f"Dokument: {source_meta['source_file']}",
            f"Dokumenttyp: {source_meta['doc_type']}",
        ]
        if block.module_title:
            header_lines.append(f"Module: {block.module_title}")
        if block.module_code or block.module_number:
            header_lines.append(
                f"Module code: {block.module_code or block.module_number}"
            )
        header_lines.append(f"Abschnitt: {section_label}")
        if child_section_path:
            header_lines.append(f"Section path: {' > '.join(child_section_path)}")
        if block.nearest_heading:
            header_lines.append(f"Nearest heading: {block.nearest_heading}")
        if block.table_caption:
            header_lines.append(f"Table caption: {block.table_caption}")
        if block.major_heading:
            header_lines.append(f"Hauptüberschrift: {block.major_heading}")
        if paragraph_id:
            header_lines.append(f"Paragraph: {paragraph_id}")
        if sentence_id:
            header_lines.append(f"Sentence: {sentence_id}")
        header_lines.append(f"Seite: {page_label}")
        text = "\n".join(header_lines) + f"\n\n{child_text.strip()}"
        chunk_id = self._stable_id(parent_id, "child", child_index, text[:120])
        module_section = block.module_section
        if block.chunk_type == "module":
            module_section = self._infer_module_section(child_text) or module_section
        metadata = {
            **source_meta,
            "index_role": "child",
            "parent_id": parent_id,
            "chunk_id": chunk_id,
            "child_index": child_index,
            "chunk_type": block.chunk_type,
            "section_id": block.section_id,
            "paragraph_id": paragraph_id,
            "sentence_id": sentence_id,
            "major_heading": block.major_heading,
            "module_title": block.module_title,
            "module_number": block.module_number,
            "module_code": block.module_code or block.module_number,
            "module_title_de": block.module_title_de,
            "module_title_en": block.module_title_en,
            "module_pages": page_label,
            "ects": block.ects,
            "semester": block.semester,
            "module_section": module_section,
            "section_title": child_section_title,
            "section_path": child_section_path,
            "nearest_heading": block.nearest_heading,
            "table_caption": block.table_caption,
            "semantic_title": block.semantic_title,
            "page_label_start": page_start,
            "page_label_end": page_end,
            "token_count": self._token_count(text),
            "content_hash": self._hash(text),
        }
        return Document(text=text, id_=chunk_id, metadata=metadata)

    def _child_texts(self, block: _Block) -> list[str]:
        text = block.text.strip()
        if not text:
            return []
        if block.chunk_type == "table":
            return self._split_markdown_table(text)
        if block.chunk_type == "module":
            return self._module_child_texts(block)
        if self._token_count(text) <= self.max_child_size:
            return [text]
        if block.chunk_type == "section":
            return self._regulation_child_texts(
                text, block.section_title or block.title
            )
        return self._token_windows(text)

    def _regulation_child_texts(self, text: str, section_title: str) -> list[str]:
        body = self._section_body(text, section_title)
        groups = self._split_regulation_paragraphs(body)
        if not groups:
            seed = body or text
            return [
                window
                if section_title in window[: len(section_title) + 10]
                else f"{section_title}\n{window}"
                for window in self._token_windows(seed)
                if window.strip()
            ]

        chunks: list[str] = []
        current: list[str] = []
        for group in groups:
            group = group.strip()
            if not group:
                continue
            single = f"{section_title}\n{group}".strip()
            if self._token_count(single) > self.max_child_size:
                if current:
                    chunks.append(f"{section_title}\n" + "\n\n".join(current).strip())
                    current = []
                chunks.extend(
                    f"{section_title}\n{window}".strip()
                    for window in self._token_windows(group)
                    if window.strip()
                )
                continue

            candidate_body = "\n\n".join([*current, group]).strip()
            candidate = f"{section_title}\n{candidate_body}".strip()
            if current and self._token_count(candidate) > self.target_child_size:
                chunks.append(f"{section_title}\n" + "\n\n".join(current).strip())
                current = [group]
            else:
                current.append(group)

        if current:
            chunks.append(f"{section_title}\n" + "\n\n".join(current).strip())
        return chunks or [text]

    def _split_regulation_paragraphs(self, body: str) -> list[str]:
        matches = list(PARAGRAPH_RE.finditer(body))
        if not matches:
            return []
        groups: list[str] = []
        for idx, match in enumerate(matches):
            start = match.start(1)
            end = matches[idx + 1].start(1) if idx + 1 < len(matches) else len(body)
            part = body[start:end].strip()
            if part:
                groups.append(part)
        return groups

    def _module_child_texts(self, block: _Block) -> list[str]:
        chunks: list[str] = []
        for section, body in self._split_module_facets(block.text):
            body = body.strip()
            if not body:
                continue
            group = (
                body
                if section in body[: len(section) + 10]
                else f"{section}\n{body}".strip()
            )
            if self._token_count(group) <= self.max_child_size:
                chunks.append(group)
                continue
            body_without_heading = self._module_section_body(group, section)
            chunks.extend(
                f"{section}\n{window}".strip()
                for window in self._token_windows(body_without_heading)
                if window.strip()
            )
        return chunks or [block.text]

    def _split_module_facets(self, text: str) -> list[tuple[str, str]]:
        lines = text.splitlines()
        buckets: dict[str, list[str]] = defaultdict(list)
        order: list[str] = []
        active = "Module overview"
        order.append(active)
        for line in lines:
            stripped = line.strip()
            detected_section = self._module_section_heading(stripped)
            if detected_section:
                active = detected_section
                if active not in order:
                    order.append(active)
            buckets[active].append(line)
        return [
            (name, "\n".join(items).strip())
            for name in order
            for items in [buckets.get(name, [])]
            if "\n".join(items).strip()
        ]

    def _split_markdown_table(self, text: str) -> list[str]:
        if self._token_count(text) <= self.max_child_size:
            return [text]

        lines = text.splitlines()
        table_start = next((i for i, line in enumerate(lines) if "|" in line), -1)
        if table_start < 0:
            return [text]
        header_idx = table_start
        separator_idx = next(
            (
                i
                for i in range(header_idx + 1, min(len(lines), header_idx + 4))
                if "|" in lines[i] and re.search(r"---+", lines[i])
            ),
            -1,
        )
        if separator_idx < 0:
            return [text]

        prefix = "\n".join(lines[:header_idx]).strip()
        header = lines[header_idx]
        separator = lines[separator_idx]
        rows = [line for line in lines[separator_idx + 1 :] if "|" in line]
        chunks: list[str] = []
        current_rows: list[str] = []
        for row in rows:
            candidate = "\n".join(
                part
                for part in [prefix, header, separator, *current_rows, row]
                if part.strip()
            )
            if current_rows and self._token_count(candidate) > self.max_child_size:
                chunks.append(
                    "\n".join(
                        part
                        for part in [prefix, header, separator, *current_rows]
                        if part.strip()
                    )
                )
                current_rows = [row]
            else:
                current_rows.append(row)
        if current_rows:
            chunks.append(
                "\n".join(
                    part
                    for part in [prefix, header, separator, *current_rows]
                    if part.strip()
                )
            )
        return chunks or [text]

    def _merge_or_split(self, groups: list[str], prefix: str = "") -> list[str]:
        merged: list[str] = []
        current = ""
        for group in groups:
            group = group.strip()
            if not group:
                continue
            if self._token_count(group) > self.max_child_size:
                if current:
                    merged.append(current.strip())
                    current = ""
                windows = self._token_windows(group)
                if prefix:
                    windows = [
                        w if prefix in w[: len(prefix) + 10] else f"{prefix}\n{w}"
                        for w in windows
                    ]
                merged.extend(windows)
                continue
            candidate = f"{current}\n\n{group}".strip() if current else group
            if current and self._token_count(candidate) > self.target_child_size:
                merged.append(current.strip())
                current = group
            else:
                current = candidate
        if current:
            merged.append(current.strip())
        return merged or groups

    def _token_windows(self, text: str) -> list[str]:
        tokens = self._tokenize_with_separators(text)
        if len(tokens) <= self.max_child_size:
            return [text]
        windows: list[str] = []
        step = (
            self.target_child_size
            if self.target_child_size <= self.overlap
            else self.target_child_size - self.overlap
        )
        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + self.target_child_size)
            window = "".join(tokens[start:end]).strip()
            if window:
                windows.append(window)
            if end == len(tokens):
                break
            start += step
        return windows

    # --- metadata helpers --------------------------------------------------------

    def _paragraph_id(self, text: str) -> Optional[str]:
        match = PARAGRAPH_RE.search(text or "")
        if not match:
            return None
        raw = " ".join(match.group(1).split())
        number_match = re.search(r"\d+[a-z]?", raw, flags=re.I)
        return f"Abs. {number_match.group(0)}" if number_match else raw

    def _sentence_id(self, text: str) -> Optional[str]:
        match = SENTENCE_RE.search(text or "")
        return " ".join(match.group(1).split()) if match else None

    def _source_file(self, docs: list[Document]) -> str:
        value = (
            docs[0].metadata.get("source_file")
            or docs[0].metadata.get("file_name")
            or "document.pdf"
        )
        return str(value)

    def _file_path(self, docs: list[Document]) -> Optional[str]:
        value = docs[0].metadata.get("file_path")
        return str(value) if value is not None else None

    def _doc_family(self, doc_type: str, file_name: str) -> str:
        normalized = self._normalize(file_name)
        if doc_type == "amendment":
            if "apo" in normalized:
                return "general_regulation"
            if "po_bsc" in normalized or "prufungsordnung" in normalized:
                return "exam_regulation"
        if doc_type in {
            "general_regulation",
            "exam_regulation",
            "module_catalog",
            "study_description",
            "study_plan",
            "elective_catalog",
            "form",
        }:
            return doc_type
        return "generic_pdf"

    def _revision_date(self, file_name: str) -> Optional[str]:
        dates: list[date] = []
        for day, month, year in re.findall(
            r"(?<!\d)(\d{1,2})[._-](\d{1,2})[._-](\d{2,4})(?!\d)", file_name
        ):
            y = int(year)
            if y < 100:
                y += 2000
            try:
                dates.append(date(y, int(month), int(day)))
            except ValueError:
                continue
        for year in re.findall(r"(?<!\d)(20\d{2})(?!\d)", file_name):
            try:
                dates.append(date(int(year), 1, 1))
            except ValueError:
                continue
        if not dates:
            return None
        return max(dates).isoformat()

    def _is_probably_latest(self, revision_date: Optional[str]) -> Optional[bool]:
        if not revision_date:
            return None
        try:
            parsed = date.fromisoformat(revision_date)
        except ValueError:
            return None
        return parsed >= date(2024, 1, 1)

    def _page_start(self, docs: list[Document]) -> Optional[Any]:
        pages = [
            d.metadata.get("page_label")
            for d in docs
            if d.metadata.get("page_label") is not None
        ]
        return min(pages, key=self._page_sort_key) if pages else None

    def _page_end(self, docs: list[Document]) -> Optional[Any]:
        pages = [
            d.metadata.get("page_label")
            for d in docs
            if d.metadata.get("page_label") is not None
        ]
        return max(pages, key=self._page_sort_key) if pages else None

    def _page_sort_key(self, page: Any) -> tuple[int, str]:
        try:
            return (int(page), str(page))
        except (TypeError, ValueError):
            return (10**9, str(page))

    def _page_range(self, start: Any, end: Any) -> str:
        if start is None and end is None:
            return "unbekannt"
        if start == end or end is None:
            return str(start)
        if start is None:
            return str(end)
        return f"{start}-{end}"

    def _join_docs(self, docs: list[Document]) -> str:
        return "\n\n".join(
            (doc.text or "").strip() for doc in docs if (doc.text or "").strip()
        ).strip()

    def _with_section_metadata(self, docs: list[Document]) -> list[Document]:
        """Return ordered documents enriched with nearest heading/section path.

        Docling keeps layout order but does not make each table/paragraph
        self-contained.  This pass maintains a lightweight university-heading
        hierarchy and writes it into element metadata before blocks are built.
        """

        stack: list[tuple[int, str]] = []
        previous_heading: Optional[str] = None
        output: list[Document] = []

        for doc in sorted(docs, key=lambda d: d.metadata.get("order", 0)):
            metadata = dict(doc.metadata or {})
            text = (doc.text or "").strip()
            heading = self._detect_structural_heading(
                text=text,
                element_type=metadata.get("element_type"),
            )
            if heading:
                level, title = heading
                stack = [item for item in stack if item[0] < level]
                if not stack or stack[-1][1] != title:
                    stack.append((level, title))
                metadata["heading_level"] = level
                metadata["previous_heading"] = previous_heading
                metadata["nearest_heading"] = title
                previous_heading = title
            else:
                metadata["previous_heading"] = previous_heading
                metadata["nearest_heading"] = (
                    stack[-1][1] if stack else previous_heading
                )

            metadata["section_path"] = [title for _, title in stack]
            output.append(Document(text=doc.text, id_=doc.doc_id, metadata=metadata))

        return output

    def _detect_structural_heading(
        self, text: str, element_type: Any = None
    ) -> Optional[tuple[int, str]]:
        first = self._clean_title((text or "").splitlines()[0] if text else "")
        if not first or len(first) > 180:
            return None

        numbered = re.match(r"^(\d+(?:\.\d+)*\.?)\s+(.{2,})$", first)
        if numbered:
            number = numbered.group(1).rstrip(".")
            title = self._clean_title(f"{number}. {numbered.group(2)}")
            return (max(1, number.count(".") + 1), title)

        lettered = re.match(r"^([A-Z])\)\s+(.{2,})$", first)
        if lettered:
            return (4, self._clean_title(first))

        normalized = self._normalize(first).replace("&", "and")
        known = {
            "data competence",
            "application competence",
            "digitalization and analytics",
            "business language and management skills",
            "wirtschafts- und unternehmensethik",
            "wirtschafts und unternehmensethik",
            "accounting, taxation and controlling",
            "accounting, taxation & controlling",
            "finance and economics",
            "finance & economics",
            "marketing, organization, innovation",
            "supply chain management and logistics",
            "supply chain management & logistics",
            "sustainability in business and economics",
            "wahlpflichtbereich data competence",
            "wahlpflichtbereich application competence",
        }
        if normalized in {self._normalize(item).replace("&", "and") for item in known}:
            return (4, self._clean_title(first))

        if element_type == "heading" and len(first.split()) <= 12:
            return (2, self._clean_title(first))
        return None

    def _metadata_section_path(self, doc: Document) -> list[str]:
        value = (doc.metadata or {}).get("section_path") or []
        if isinstance(value, str):
            return [part.strip() for part in value.split(">") if part.strip()]
        if isinstance(value, list):
            return [str(part).strip() for part in value if str(part).strip()]
        return []

    def _section_block_text(
        self,
        text: str,
        section_path: list[str] | None,
        nearest_heading: Optional[str],
    ) -> str:
        lines: list[str] = []
        if section_path:
            lines.append(f"Section path: {' > '.join(section_path)}")
        if nearest_heading:
            lines.append(f"Nearest heading: {nearest_heading}")
        if not lines:
            return text.strip()
        return "\n".join(lines) + f"\n\n{text.strip()}"

    def _table_block_text(
        self,
        doc: Document,
        doc_type: str,
        section_path: list[str] | None,
        nearest_heading: Optional[str],
        table_caption: Optional[str],
        page: Any,
    ) -> str:
        source = (
            doc.metadata.get("source_file")
            or doc.metadata.get("file_name")
            or "document.pdf"
        )
        header = [
            f"Dokument: {source}",
            f"Dokumenttyp: {doc_type}",
            f"Section path: {' > '.join(section_path or []) if section_path else '-'}",
            f"Nearest heading: {nearest_heading or '-'}",
            f"Table caption: {table_caption or '-'}",
            f"Page: {page}",
        ]
        return "\n".join(header) + f"\n\n{(doc.text or '').strip()}"

    def _table_caption(self, text: str) -> Optional[str]:
        for line in (text or "").splitlines()[:5]:
            stripped = self._clean_title(line)
            if stripped and "|" not in stripped:
                return stripped
        return None

    def _semantic_table_title(
        self,
        text: str,
        nearest_heading: Optional[str],
        section_path: list[str] | None,
    ) -> str:
        haystack = self._normalize(
            " ".join([*(section_path or []), nearest_heading or "", text or ""])
        )
        candidates = [
            ("data competence", "Data Competence table"),
            ("application competence", "Application Competence table"),
            (
                "supply chain management",
                "Supply Chain Management & Logistics profile table",
            ),
            (
                "sustainability in business and economics",
                "Sustainability in Business and Economics profile table",
            ),
            ("digitalization", "Digitalization & Analytics table"),
            (
                "business language",
                "Business Language and Management Skills table",
            ),
            (
                "wirtschafts- und unternehmensethik",
                "Wirtschafts- und Unternehmensethik table",
            ),
            (
                "accounting",
                "Accounting, Taxation & Controlling profile table",
            ),
            ("finance", "Finance & Economics profile table"),
            ("marketing", "Marketing, Organization, Innovation profile table"),
        ]
        for needle, title in candidates:
            if self._normalize(needle) in haystack:
                return title
        if nearest_heading:
            return f"{nearest_heading} table"
        return "University table"

    def _section_list_summary_blocks(self, docs: list[Document]) -> list[_Block]:
        """Build compact chunks for lists represented as sibling headings.

        Some Docling outputs expose an introductory sentence (for example
        "Es werden ... Studienprofile angeboten") followed by each option as a
        standalone heading.  Splitting each heading into its own chunk is useful
        for local evidence, but it removes the answer-bearing list from a single
        retrievable context.  This helper reconstructs only those short,
        explicitly enumerated structural lists.
        """

        profile_names = {
            "accounting, taxation and controlling",
            "accounting, controlling and taxation",
            "finance and economics",
            "marketing, organization, innovation",
            "supply chain management and logistics",
            "sustainability in business and economics",
        }
        elective_names = {
            "digitalization and analytics",
            "data competence",
            "application competence",
            "business language and management skills",
            "wirtschafts- und unternehmensethik",
            "wirtschafts und unternehmensethik",
        }
        summaries: dict[tuple[str, ...], list[Document]] = defaultdict(list)

        for doc in docs:
            metadata = doc.metadata or {}
            if not (
                metadata.get("element_type") == "heading"
                or metadata.get("heading_level") is not None
            ):
                continue
            path = self._metadata_section_path(doc)
            if len(path) < 2:
                continue
            heading = self._clean_title(doc.text or path[-1])
            normalized = self._normalize(heading).replace("&", "and")
            parent_path = path[:-1]
            parent_text = self._normalize(" > ".join(parent_path))
            if "studienprofile" in parent_text and normalized in profile_names:
                summaries[tuple(parent_path)].append(doc)
            elif "wahlpflicht" in parent_text and normalized in elective_names:
                summaries[tuple(parent_path)].append(doc)

        blocks: list[_Block] = []
        for summary_path, heading_docs in summaries.items():
            unique: list[str] = []
            elements: list[Document] = []
            for doc in heading_docs:
                title = self._clean_title(doc.text or "")
                if title and title not in unique:
                    unique.append(title)
                    elements.append(doc)
            if len(unique) < 2:
                continue
            path = list(summary_path)
            nearest_heading = path[-1] if path else None
            if any("studienprofile" in self._normalize(item) for item in path):
                label = "Studienprofile"
                lead = "Es werden in der Regel die folgenden Studienprofile angeboten:"
            else:
                label = "Wahlpflichtbereiche"
                lead = "Folgende Wahlpflichtbereiche sind strukturell ausgewiesen:"
            text = self._section_block_text(
                text=f"{lead}\n" + "\n".join(f"- {item}" for item in unique),
                section_path=path,
                nearest_heading=nearest_heading,
            )
            blocks.append(
                _Block(
                    title=label,
                    text=text,
                    elements=elements,
                    chunk_type="section_summary",
                    section_path=path,
                    nearest_heading=nearest_heading,
                    semantic_title=label,
                )
            )
        return blocks

    def _should_keep_short_block(self, block: _Block) -> bool:
        if self._token_count(block.text) >= self.min_child_size:
            return True
        page_start = self._page_start(block.elements)
        try:
            if page_start is not None and int(page_start) <= 2:
                return True
        except (TypeError, ValueError):
            pass
        important = re.compile(
            r"(?i)(Fakultät|federführend|verantwortlich|Abschlussgrad|"
            r"Bachelor of Science|ECTS|Studienprofile|Zeugnis|Wahlpflichtbereich)"
        )
        return bool(important.search(block.text or ""))

    def _split_text_at_section_starts(self, text: str) -> list[tuple[bool, str]]:
        matches = list(SECTION_START_RE.finditer(text))
        if not matches:
            return [(False, text)]
        segments: list[tuple[bool, str]] = []
        if matches[0].start() > 0:
            prefix = text[: matches[0].start()].strip()
            if prefix:
                segments.append((False, prefix))
        for idx, match in enumerate(matches):
            start = match.start(1)
            end = matches[idx + 1].start(1) if idx + 1 < len(matches) else len(text)
            segment = text[start:end].strip()
            if segment:
                segments.append((True, segment))
        return segments

    def _section_identity(self, section_text: str) -> tuple[str, str]:
        lines = [line.strip() for line in section_text.splitlines() if line.strip()]
        first = lines[0] if lines else "§"
        match = re.search(r"§\s*\d+[a-zA-Z]?", first)
        section_id = match.group(0) if match else first
        title = first
        if (
            re.fullmatch(r"§\s*\d+[a-zA-Z]?", first)
            and len(lines) > 1
            and not re.match(r"^(?:\(\d+[a-z]?\)|Absatz\s+\d+)\b", lines[1])
        ):
            title = f"{first} {lines[1]}"
        return section_id, self._clean_title(title)

    def _section_body(self, text: str, section_title: str) -> str:
        lines = text.splitlines()
        title_parts = section_title.split(maxsplit=2)
        if lines and self._clean_title(lines[0]) == section_title:
            return "\n".join(lines[1:]).strip()
        if (
            len(lines) >= 2
            and re.fullmatch(r"\s*§\s*\d+[a-zA-Z]?\s*", lines[0])
            and section_title == self._clean_title(f"{lines[0]} {lines[1]}")
        ):
            return "\n".join(lines[2:]).strip()
        if len(title_parts) >= 2 and text.startswith(" ".join(title_parts[:2])):
            return text[len(" ".join(title_parts[:2])) :].strip()
        return text

    def _is_major_heading(self, text: str) -> bool:
        stripped = self._clean_title(text)
        if len(stripped) > 120:
            return False
        if re.match(r"^[IVXLCDM]+\.\s+[A-ZÄÖÜ][A-ZÄÖÜ\s\-/]+$", stripped):
            return True
        return bool(re.match(r"^(TEIL|ABSCHNITT)\s+[IVXLCDM\d]+\b", stripped))

    def _major_heading_from_text(self, text: str) -> Optional[str]:
        for line in reversed(
            [line.strip() for line in text.splitlines() if line.strip()]
        ):
            if self._is_major_heading(line):
                return self._clean_title(line)
        return None

    def _detect_module_start(
        self, docs: list[Document], index: int
    ) -> dict[str, Optional[str]]:
        doc = docs[index]
        metadata = doc.metadata or {}
        text = (doc.text or "").strip()
        if not text:
            return {}

        table_metadata = self._module_metadata(text, "")
        if table_metadata.get("module_title"):
            return table_metadata

        explicit = self._detect_module_title(text, metadata.get("element_type"))
        if explicit:
            parsed = self._module_metadata(text, explicit)
            parsed["module_title"] = (
                parsed.get("module_title")
                or parsed.get("module_title_en")
                or parsed.get("module_title_de")
                or explicit
            )
            return parsed

        if metadata.get("element_type") != "heading":
            return {}

        heading = self._clean_title(text.splitlines()[0])
        if not self._looks_like_module_heading(heading):
            return {}

        nearby = self._nearby_module_table_metadata(docs, index)
        if nearby.get("module_title"):
            return nearby
        if self._nearby_module_metadata_markers(docs, index):
            return {"module_title": heading, "module_title_de": heading}
        return {}

    def _nearby_module_table_metadata(
        self, docs: list[Document], index: int
    ) -> dict[str, Optional[str]]:
        for offset in range(1, 4):
            if index + offset >= len(docs):
                break
            candidate = docs[index + offset]
            text = (candidate.text or "").strip()
            if not text:
                continue
            metadata = self._module_metadata(text, "")
            if metadata.get("module_title"):
                return metadata
            if (candidate.metadata or {}).get("element_type") == "heading":
                break
        return {}

    def _nearby_module_metadata_markers(self, docs: list[Document], index: int) -> bool:
        marker_hits = 0
        for offset in range(1, 10):
            if index + offset >= len(docs):
                break
            candidate = docs[index + offset]
            text = (candidate.text or "").strip()
            if not text:
                continue
            if (candidate.metadata or {}).get(
                "element_type"
            ) == "heading" and offset > 1:
                break
            normalized = self._normalize_for_match(text.splitlines()[0])
            if normalized in {
                "modultitel",
                "modultitel englisch",
                "modulnummer",
                "leistungspunkte ects punkte",
            }:
                marker_hits += 1
            if marker_hits >= 2:
                return True
        return False

    def _looks_like_module_heading(self, heading: str) -> bool:
        if not heading:
            return False
        normalized = self._normalize_for_match(heading)
        if re.fullmatch(r"\d+", normalized):
            return False
        if re.search(r"\b\d{1,2}\s+[a-z]+\s+20\d{2}\b", normalized):
            return False
        sentence_starts = (
            "in diesem modul",
            "die studierenden",
            "studierende",
            "der studierende",
            "das modul",
            "aufgrund der",
            "informationen zur",
        )
        if normalized.startswith(sentence_starts):
            return False
        if len(heading.split()) > 8 and re.search(
            r"\b(ist|sind|werden|wird|erwerben|erlernen|vermittelt|notwendig)\b",
            normalized,
        ):
            return False
        if self._module_section_heading(heading):
            return False
        blocked = {
            "inhaltsverzeichnis",
            "pflichtkurse",
            "wahlpflichtkurse",
            "pflichtkurse semester 1",
            "pflichtkurse semester 2",
            "pflichtkurse semester 3",
        }
        if normalized in blocked:
            return False
        return len(heading.split()) <= 14 and len(heading) <= 140

    def _module_section_heading(self, text: str) -> Optional[str]:
        first = self._clean_title((text or "").splitlines()[0] if text else "")
        if not first:
            return None
        normalized = self._normalize_for_match(first)
        section_patterns: list[tuple[str, str]] = [
            (
                "Kompetenzen",
                r"^(kompetenzen|qualifikationsziele|learning outcomes|lernergebnisse)$",
            ),
            (
                "Inhalte und Themen",
                r"^(inhalte|inhalte und themen|contents|lehrinhalte)$",
            ),
            ("Formale Voraussetzungen für die Teilnahme", r"^formale voraussetzungen"),
            (
                "Empfohlene Voraussetzungen für die Teilnahme",
                r"^empfohlene voraussetzungen",
            ),
            ("Lehr- und Prüfungssprache", r"^lehr und prufungssprache"),
            ("Lehr- und Lernformen/Lehrveranstaltungstypen", r"^lehr und lernformen"),
            (
                "Voraussetzungen für die Vergabe von ECTS-Punkten",
                r"^voraussetzungen fur die vergabe",
            ),
            (
                "Zeitaufwand/Berechnung der ECTS-Punkte innerhalb des Moduls",
                r"^zeitaufwand",
            ),
            ("Modulnote", r"^modulnote$"),
            (
                "Erläuterung der Prüfungsmodalitäten",
                r"^erlauterung der prufungsmodalitaten",
            ),
            ("Polyvalenz mit anderen Studiengängen", r"^polyvalenz"),
            ("Bemerkungen", r"^bemerkungen"),
        ]
        for label, pattern in section_patterns:
            if re.search(pattern, normalized):
                return label
        return None

    def _module_section_title(self, child_text: str) -> Optional[str]:
        first = child_text.strip().splitlines()[0].strip() if child_text.strip() else ""
        if self._normalize_for_match(first) == "module overview":
            return "Module overview"
        return self._module_section_heading(first)

    def _module_section_body(self, text: str, section_title: str) -> str:
        lines = text.splitlines()
        if lines and self._module_section_heading(lines[0]) == section_title:
            return "\n".join(lines[1:]).strip()
        return text.strip()

    def _contains_module_terminal_section(self, text: str) -> bool:
        terminal_sections = {
            "Modulnote",
            "Erläuterung der Prüfungsmodalitäten",
            "Polyvalenz mit anderen Studiengängen",
            "Bemerkungen",
        }
        return any(
            self._module_section_heading(line) in terminal_sections
            for line in (text or "").splitlines()
        )

    def _module_child_pages(
        self, block: _Block, section_title: Optional[str]
    ) -> tuple[Optional[Any], Optional[Any]]:
        """Return the pages occupied by one module subsection when available."""

        if not section_title:
            return self._page_start(block.elements), self._page_end(block.elements)

        selected: list[Document] = []
        active = section_title == "Module overview"
        for element in block.elements:
            heading = self._module_section_heading(element.text or "")
            if heading:
                if active:
                    break
                active = heading == section_title
            if active:
                selected.append(element)

        if not selected:
            selected = block.elements
        return self._page_start(selected), self._page_end(selected)

    def _detect_module_title(
        self, text: str, element_type: Any = None
    ) -> Optional[str]:
        first = text.strip().splitlines()[0].strip() if text.strip() else ""
        for pattern in [
            r"(?i)^\s*(?:modul(?:bezeichnung|name)?|module(?: title)?)\s*[:\-]\s*(.+)$",
            r"(?i)^\s*(?:modul)\s+(.{3,120})$",
        ]:
            match = re.search(pattern, first)
            if match:
                return self._clean_title(match.group(1))
        return None

    def _module_title_from_text(self, text: str) -> Optional[str]:
        for line in text.splitlines()[:30]:
            detected = self._detect_module_title(line)
            if detected:
                return detected
        return None

    def _module_metadata(self, text: str, title: str) -> dict[str, Optional[str]]:
        metadata: dict[str, Optional[str]] = {
            "module_title": None,
            "module_title_de": None,
            "module_title_en": None,
            "module_code": None,
            "module_number": None,
            "ects": None,
            "semester": None,
        }
        for row in self._markdown_table_rows(text):
            label_values = self._module_table_label_values(row)
            for label, value in label_values:
                label_norm = self._normalize_for_match(label)
                value = self._clean_title(value)
                if not value:
                    continue
                if (
                    "modultitel" in label_norm
                    and "englisch" in label_norm
                    and self._looks_like_module_title_value(value)
                ):
                    metadata["module_title_en"] = value
                elif label_norm == "modultitel" and self._looks_like_module_title_value(
                    value
                ):
                    metadata["module_title_de"] = value
                elif "modulnummer" in label_norm or re.search(
                    r"\bmodule (no|number)\b", label_norm
                ):
                    metadata["module_code"] = value
                    metadata["module_number"] = value
                elif "leistungspunkte" in label_norm or "ects" in label_norm:
                    ects = re.search(r"(\d+(?:[,.]\d+)?)", value)
                    if ects:
                        metadata["ects"] = ects.group(1).replace(",", ".")
                elif "semester" in label_norm or "turnus" in label_norm:
                    metadata["semester"] = value

        metadata["module_title"] = (
            metadata.get("module_title_en")
            or metadata.get("module_title_de")
            or (self._clean_title(title) if title else None)
        )

        number_match = re.search(
            r"(?im)^(?:modulnummer|module\s*(?:no\.?|number)|nummer)\s*[:\-]?\s*(.+)$",
            text,
        )
        if number_match:
            metadata["module_number"] = self._clean_title(number_match.group(1))
            metadata["module_code"] = metadata["module_number"]
        else:
            compact = re.search(r"\b([A-Z]{2,}[\-_ ]?\d{2,}[A-Z]?)\b", title)
            if compact:
                metadata["module_number"] = compact.group(1)
                metadata["module_code"] = metadata["module_number"]

        ects_match = re.search(
            r"(?im)^\s*(?:ects|leistungspunkte|credits?|lp)\s*[:\-]?\s*(\d+(?:[,.]\d+)?)\b",
            text,
        ) or re.search(
            r"(?i)(\d+(?:[,.]\d+)?)[ \t]*(?:ects|leistungspunkte|credits?|lp)\b",
            text,
        )
        if ects_match:
            metadata["ects"] = ects_match.group(1).replace(",", ".")

        semester_match = re.search(
            r"(?im)^\s*(?:semester|fachsemester)\s*[:\-]?\s*([^\n]+)$", text
        ) or re.search(r"(?i)\b(\d+\.?\s*(?:semester|fachsemester))\b", text)
        if semester_match:
            metadata["semester"] = self._clean_title(semester_match.group(1))
        return metadata

    def _markdown_table_rows(self, text: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for line in (text or "").splitlines():
            if "|" not in line:
                continue
            if re.fullmatch(r"\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*", line):
                continue
            cells = [
                self._clean_title(cell) for cell in line.strip().strip("|").split("|")
            ]
            cells = [cell for cell in cells if cell and not re.fullmatch(r"-+", cell)]
            if len(cells) >= 2:
                rows.append(cells)
        return rows

    def _module_table_label_values(self, row: list[str]) -> list[tuple[str, str]]:
        known_label_re = re.compile(
            r"(modultitel(?:\s+englisch)?|modulnummer|module\s+(?:no|number)|"
            r"leistungspunkte|ects|semester|turnus)",
            flags=re.I,
        )
        pairs: list[tuple[str, str]] = []
        for idx, cell in enumerate(row):
            if not known_label_re.search(cell):
                continue
            label = cell
            other_cells = [value for pos, value in enumerate(row) if pos != idx]
            value = other_cells[0] if other_cells else ""
            inline = known_label_re.sub("", cell).strip(" :-")
            if inline and inline != cell:
                value = inline
            pairs.append((label, value))
        return pairs

    def _looks_like_module_title_value(self, value: str) -> bool:
        value = self._clean_title(value)
        if not value:
            return False
        return self._looks_like_module_heading(value)

    def _infer_module_section(self, child_text: str) -> Optional[str]:
        first = child_text.strip().splitlines()[0].strip() if child_text.strip() else ""
        if self._normalize_for_match(first) == "module overview":
            return "overview"
        section = self._module_section_heading(first)
        if not section:
            return None
        return {
            "Kompetenzen": "competencies",
            "Inhalte und Themen": "contents",
            "Formale Voraussetzungen für die Teilnahme": "formal_requirements",
            "Empfohlene Voraussetzungen für die Teilnahme": "recommended_requirements",
            "Lehr- und Prüfungssprache": "language",
            "Lehr- und Lernformen/Lehrveranstaltungstypen": "teaching_formats",
            "Voraussetzungen für die Vergabe von ECTS-Punkten": "ects_requirements",
            "Zeitaufwand/Berechnung der ECTS-Punkte innerhalb des Moduls": "workload",
            "Modulnote": "assessment",
            "Erläuterung der Prüfungsmodalitäten": "exam_modalities",
            "Polyvalenz mit anderen Studiengängen": "polyvalence",
            "Bemerkungen": "remarks",
        }.get(section, section)

    def _title_from_filename(self, file_name: str) -> str:
        return (
            Path(file_name).stem.replace("_", " ").replace("-", " ").strip()
            or file_name
        )

    def _clean_title(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip(" :-\t")[:180]

    def _normalize(self, value: str) -> str:
        return (
            value.lower()
            .replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("ß", "ss")
        )

    def _normalize_for_match(self, value: str) -> str:
        normalized = self._normalize(value or "")
        normalized = (
            normalized.replace("¨ a", "ae")
            .replace("¨ o", "oe")
            .replace("¨ u", "ue")
            .replace("¨", "")
        )
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _token_count(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text or "", flags=re.UNICODE))

    def _tokenize_with_separators(self, text: str) -> list[str]:
        return re.findall(r"\s+|\w+|[^\w\s]", text or "", flags=re.UNICODE)

    def _hash(self, text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

    def _stable_id(self, *parts: Any) -> str:
        joined = "|".join(str(part) for part in parts)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, joined))
