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

        def append_current() -> None:
            nonlocal current, current_title
            if not current:
                return
            text = self._join_docs(current)
            title = (
                current_title
                or self._module_title_from_text(text)
                or self._title_from_filename(file_name)
            )
            metadata = self._module_metadata(text, title)
            blocks.append(
                _Block(
                    title=title,
                    text=text,
                    elements=list(current),
                    chunk_type="module",
                    module_title=title,
                    module_number=metadata.get("module_number"),
                    ects=metadata.get("ects"),
                    semester=metadata.get("semester"),
                )
            )
            current = []
            current_title = None

        for doc in docs:
            text = (doc.text or "").strip()
            if not text:
                continue
            detected = self._detect_module_title(text, doc.metadata.get("element_type"))
            starts_new = bool(
                detected and current and self._token_count(self._join_docs(current)) >= 80
            )
            if starts_new:
                append_current()
            if detected:
                current_title = detected
            current.append(doc)

        append_current()
        return blocks or self._generic_blocks(docs)

    def _table_first_blocks(self, docs: list[Document], doc_type: str) -> list[_Block]:
        blocks: list[_Block] = []
        for idx, doc in enumerate(docs, start=1):
            if doc.metadata.get("element_type") != "table":
                continue
            page = doc.metadata.get("page_label") or "unbekannt"
            title = f"Tabelle {idx} (Seite {page})"
            summary = (
                f"Zusammenfassung: Die folgende Tabelle aus {doc_type} strukturiert "
                f"Informationen für {self.study_program}. Zeilen und Spalten wurden "
                "als Markdown-Tabelle unverändert zusammengehalten."
            )
            text = f"{summary}\n\n{doc.text.strip()}"
            blocks.append(_Block(title=title, text=text, elements=[doc], chunk_type="table"))

        non_table = [doc for doc in docs if doc.metadata.get("element_type") != "table"]
        for block in self._generic_blocks(non_table):
            if self._token_count(block.text) >= self.min_child_size:
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
            ("Kontakt/Prüfungsamt", r"(?i)(kontakt|prüfungsamt|pruefungsamt|dekanat|büro|buero|office)"),
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
        if not docs:
            return []
        blocks: list[_Block] = []
        current: list[Document] = []
        title = "Dokument"
        for doc in docs:
            is_heading = doc.metadata.get("element_type") == "heading"
            if is_heading and current:
                blocks.append(
                    _Block(
                        title=title,
                        text=self._join_docs(current),
                        elements=current,
                        chunk_type="heading",
                    )
                )
                current = []
            if is_heading:
                title = (doc.text or "").strip()[:140] or title
            current.append(doc)
        if current:
            blocks.append(
                _Block(
                    title=title,
                    text=self._join_docs(current),
                    elements=current,
                    chunk_type="heading",
                )
            )
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
            "ects": block.ects,
            "semester": block.semester,
            "module_section": block.module_section,
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
        section_label = block.section_title or block.module_title or block.title
        paragraph_id = self._paragraph_id(child_text)
        sentence_id = self._sentence_id(child_text)
        page_start = self._page_start(block.elements)
        page_end = self._page_end(block.elements)
        page_label = self._page_range(page_start, page_end)
        header_lines = [
            f"Dokument: {source_meta['source_file']}",
            f"Dokumenttyp: {source_meta['doc_type']}",
            f"Studiengang: {self.study_program}",
        ]
        if block.major_heading:
            header_lines.append(f"Hauptüberschrift: {block.major_heading}")
        header_lines.extend(
            [
                f"Abschnitt/Modul/Tabelle/Formularblock: {section_label}",
                f"Section: {section_label}",
            ]
        )
        if paragraph_id:
            header_lines.append(f"Paragraph: {paragraph_id}")
        if sentence_id:
            header_lines.append(f"Sentence: {sentence_id}")
        header_lines.extend(
            [
                f"Seite: {page_label}",
                f"Page: {page_label}",
            ]
        )
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
            "section_title": block.section_title,
            "paragraph_id": paragraph_id,
            "sentence_id": sentence_id,
            "major_heading": block.major_heading,
            "module_title": block.module_title,
            "module_number": block.module_number,
            "ects": block.ects,
            "semester": block.semester,
            "module_section": module_section,
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
        if self._token_count(text) <= self.max_child_size:
            return [text]
        if block.chunk_type == "section":
            return self._regulation_child_texts(text, block.section_title or block.title)
        if block.chunk_type == "module":
            return self._module_child_texts(block)
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
            group = f"{section}\n{body}".strip()
            if self._token_count(group) <= self.max_child_size:
                chunks.append(group)
                continue
            chunks.extend(
                f"{section}\n{window}".strip()
                for window in self._token_windows(body)
                if window.strip()
            )
        return chunks or [block.text]

    def _split_module_facets(self, text: str) -> list[tuple[str, str]]:
        facet_patterns = [
            ("overview", r"(?im)^(modul|modulbezeichnung|module|credits?|ects|semester|dauer|sprache)\b"),
            ("contents", r"(?im)^(inhalte?|contents?|lehrinhalte)\b"),
            (
                "competencies",
                r"(?im)^(kompetenzen|qualifikationsziele|learning outcomes?|lernergebnisse)\b",
            ),
            (
                "exam/assessment",
                r"(?im)^(prüfung|pruefung|exam|prüfungsform|leistungsnachweis|assessment)\b",
            ),
            (
                "workload/ECTS/semester",
                r"(?im)^(workload|arbeitsaufwand|aufwand|ects|semester|leistungspunkte)\b",
            ),
        ]
        lines = text.splitlines()
        buckets: dict[str, list[str]] = defaultdict(list)
        active = "overview"
        for line in lines:
            stripped = line.strip()
            for name, pattern in facet_patterns:
                if re.search(pattern, stripped):
                    active = name
                    break
            buckets[active].append(line)
        return [
            (name, "\n".join(items).strip())
            for name, items in buckets.items()
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
                        part for part in [prefix, header, separator, *current_rows] if part.strip()
                    )
                )
                current_rows = [row]
            else:
                current_rows.append(row)
        if current_rows:
            chunks.append(
                "\n".join(
                    part for part in [prefix, header, separator, *current_rows] if part.strip()
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
        for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
            if self._is_major_heading(line):
                return self._clean_title(line)
        return None

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
        if element_type == "heading" and "modul" in self._normalize(first):
            return self._clean_title(first)
        return None

    def _module_title_from_text(self, text: str) -> Optional[str]:
        for line in text.splitlines()[:30]:
            detected = self._detect_module_title(line)
            if detected:
                return detected
        return None

    def _module_metadata(self, text: str, title: str) -> dict[str, Optional[str]]:
        metadata: dict[str, Optional[str]] = {
            "module_number": None,
            "ects": None,
            "semester": None,
        }
        number_match = re.search(
            r"(?im)^(?:modulnummer|module\s*(?:no\.?|number)|nummer)\s*[:\-]?\s*(.+)$",
            text,
        )
        if number_match:
            metadata["module_number"] = self._clean_title(number_match.group(1))
        else:
            compact = re.search(r"\b([A-Z]{2,}[\-_ ]?\d{2,}[A-Z]?)\b", title)
            if compact:
                metadata["module_number"] = compact.group(1)

        ects_match = re.search(
            r"(?i)(\d+(?:[,.]\d+)?)\s*(?:ects|leistungspunkte|credits?|lp)\b", text
        )
        if ects_match:
            metadata["ects"] = ects_match.group(1).replace(",", ".")

        semester_match = re.search(
            r"(?im)^\s*(?:semester|fachsemester)\s*[:\-]?\s*([^\n]+)$", text
        ) or re.search(r"(?i)\b(\d+\.?\s*(?:semester|fachsemester))\b", text)
        if semester_match:
            metadata["semester"] = self._clean_title(semester_match.group(1))
        return metadata

    def _infer_module_section(self, child_text: str) -> Optional[str]:
        first = child_text.strip().splitlines()[0].strip() if child_text.strip() else ""
        allowed = {
            "overview",
            "contents",
            "competencies",
            "exam/assessment",
            "workload/ECTS/semester",
        }
        return first if first in allowed else None

    def _title_from_filename(self, file_name: str) -> str:
        return Path(file_name).stem.replace("_", " ").replace("-", " ").strip() or file_name

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

    def _token_count(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text or "", flags=re.UNICODE))

    def _tokenize_with_separators(self, text: str) -> list[str]:
        return re.findall(r"\s+|\w+|[^\w\s]", text or "", flags=re.UNICODE)

    def _hash(self, text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

    def _stable_id(self, *parts: Any) -> str:
        joined = "|".join(str(part) for part in parts)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, joined))
