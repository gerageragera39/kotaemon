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


@dataclass
class _Block:
    title: str
    text: str
    elements: list[Document]
    chunk_type: str
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    module_title: Optional[str] = None


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
            parent_id = self._stable_id(file_name, doc_type, "parent", parent_index, block.title)
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

        if any(term in normalized for term in ["aenderung", "anderung", "aenderungssatzung", "anderungssatzung"]):
            return "amendment"
        if "studienverlaufsplan" in normalized:
            return "study_plan"
        if "wahlpflichtkatal" in normalized:
            return "elective_catalog"
        if any(term in normalized for term in ["anmeldung_bachelorarbeit", "zeugnisantrag", "umfrage"]):
            return "form"
        if "modulkatalog" in normalized or re.search(r"(^|[/_\-\s])module[_\-\s]", normalized):
            return "module_catalog"
        if "po_bsc" in normalized or "prufungsordnung" in normalized or "prüfungsordnung" in haystack:
            return "exam_regulation"
        if re.search(r"(^|[/_\-\s])apo([_\-\.\s]|$)", normalized) or "allgemeine prufungsordnung" in normalized:
            return "general_regulation"
        return "generic_pdf"

    # --- strategy block builders -------------------------------------------------

    def _regulation_blocks(self, docs: list[Document]) -> list[_Block]:
        full_text = self._join_docs(docs)
        matches = list(re.finditer(r"(?m)(^|\n)(§\s*\d+[a-zA-Z]?\b[^\n]*)", full_text))
        if not matches:
            return self._generic_blocks(docs)

        blocks: list[_Block] = []
        for idx, match in enumerate(matches):
            start = match.start(2)
            end = matches[idx + 1].start(2) if idx + 1 < len(matches) else len(full_text)
            section_text = full_text[start:end].strip()
            title_line = section_text.splitlines()[0].strip()
            sec_id_match = re.search(r"§\s*\d+[a-zA-Z]?", title_line)
            section_id = sec_id_match.group(0) if sec_id_match else title_line
            blocks.append(
                _Block(
                    title=title_line,
                    text=section_text,
                    elements=self._docs_overlapping_text(docs, section_text),
                    chunk_type="section",
                    section_id=section_id,
                    section_title=title_line,
                )
            )
        return blocks

    def _module_blocks(self, docs: list[Document], file_name: str) -> list[_Block]:
        blocks: list[_Block] = []
        current: list[Document] = []
        current_title: Optional[str] = None

        for doc in docs:
            text = (doc.text or "").strip()
            if not text:
                continue
            detected = self._detect_module_title(text, doc.metadata.get("element_type"))
            starts_new = bool(detected and current and self._token_count(self._join_docs(current)) >= 80)
            if starts_new:
                title = current_title or self._module_title_from_text(self._join_docs(current)) or self._title_from_filename(file_name)
                blocks.append(
                    _Block(
                        title=title,
                        text=self._join_docs(current),
                        elements=current,
                        chunk_type="module",
                        module_title=title,
                    )
                )
                current = []
            if detected and not current_title:
                current_title = detected
            elif detected and starts_new:
                current_title = detected
            current.append(doc)

        if current:
            title = current_title or self._module_title_from_text(self._join_docs(current)) or self._title_from_filename(file_name)
            blocks.append(
                _Block(
                    title=title,
                    text=self._join_docs(current),
                    elements=current,
                    chunk_type="module",
                    module_title=title,
                )
            )
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
            blocks.append(
                _Block(title=title, text=text, elements=[doc], chunk_type="table")
            )

        non_table = [doc for doc in docs if doc.metadata.get("element_type") != "table"]
        for block in self._generic_blocks(non_table):
            if self._token_count(block.text) >= self.min_child_size:
                blocks.append(block)
        return blocks or self._generic_blocks(docs)

    def _form_blocks(self, docs: list[Document]) -> list[_Block]:
        full_text = self._join_docs(docs)
        categories = [
            ("Zweck", r"(?i)(zweck|antrag|anmeldung|zeugnis|umfrage|abschlussarbeit)"),
            ("Pflichtfelder", r"(?i)(name|matrikel|adresse|e-?mail|studiengang|thema|prüfer|pruefer)"),
            ("Fristen und Einreichung", r"(?i)(frist|abgabe|einreich|submit|deadline|datum)"),
            ("Unterschriften und Erklärungen", r"(?i)(unterschrift|signature|erklärung|erklaerung|datenschutz|bestätigung|bestaetigung)"),
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
                    _Block(title=title, text=self._join_docs(current), elements=current, chunk_type="heading")
                )
                current = []
            if is_heading:
                title = (doc.text or "").strip()[:140] or title
            current.append(doc)
        if current:
            blocks.append(_Block(title=title, text=self._join_docs(current), elements=current, chunk_type="heading"))
        return blocks

    # --- chunk materialization ---------------------------------------------------

    def _make_parent(self, block: _Block, parent_id: str, source_meta: dict[str, Any]) -> Document:
        metadata = {
            **source_meta,
            "index_role": "parent",
            "parent_id": parent_id,
            "chunk_type": f"parent_{block.chunk_type}",
            "section_id": block.section_id,
            "section_title": block.section_title,
            "module_title": block.module_title,
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
        page_start = self._page_start(block.elements)
        page_end = self._page_end(block.elements)
        page_label = self._page_range(page_start, page_end)
        header = (
            f"Dokument: {source_meta['source_file']}\n"
            f"Dokumenttyp: {source_meta['doc_type']}\n"
            f"Studiengang: {self.study_program}\n"
            f"Abschnitt/Modul/Tabelle/Formularblock: {section_label}\n"
            f"Seite: {page_label}\n"
        )
        text = f"{header}\n{child_text.strip()}"
        chunk_id = self._stable_id(parent_id, "child", child_index, text[:120])
        metadata = {
            **source_meta,
            "index_role": "child",
            "parent_id": parent_id,
            "chunk_id": chunk_id,
            "chunk_type": block.chunk_type,
            "section_id": block.section_id,
            "section_title": block.section_title,
            "module_title": block.module_title,
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
            return [text]
        if self._token_count(text) <= self.max_child_size:
            return [text]
        if block.chunk_type == "section":
            section_title = block.section_title or block.title
            groups = self._split_regulation_paragraphs(text, section_title)
            return self._merge_or_split(groups, prefix=section_title)
        if block.chunk_type == "module":
            groups = self._split_module_facets(text)
            return self._merge_or_split(groups, prefix=block.module_title or block.title)
        return self._token_windows(text)

    def _split_regulation_paragraphs(self, text: str, section_title: str) -> list[str]:
        body = text[len(section_title) :].strip() if text.startswith(section_title) else text
        matches = list(re.finditer(r"(?m)(^|\n)\s*((?:\(\d+[a-z]?\)|Absatz\s+\d+)[^\n]*)", body))
        if not matches:
            return self._token_windows(text)
        groups: list[str] = []
        for idx, match in enumerate(matches):
            start = match.start(2)
            end = matches[idx + 1].start(2) if idx + 1 < len(matches) else len(body)
            part = body[start:end].strip()
            groups.append(f"{section_title}\n{part}" if section_title not in part[: len(section_title) + 5] else part)
        return groups

    def _split_module_facets(self, text: str) -> list[str]:
        facet_patterns = [
            ("Überblick", r"(?im)^(modul|modulbezeichnung|module|credits?|ects|semester|dauer|sprache)\b"),
            ("Inhalte", r"(?im)^(inhalte?|contents?|lehrinhalte)\b"),
            ("Kompetenzen", r"(?im)^(kompetenzen|qualifikationsziele|learning outcomes?)\b"),
            ("Prüfung", r"(?im)^(prüfung|pruefung|exam|prüfungsform|leistungsnachweis)\b"),
        ]
        lines = text.splitlines()
        buckets: dict[str, list[str]] = defaultdict(list)
        active = "Überblick"
        for line in lines:
            stripped = line.strip()
            for name, pattern in facet_patterns:
                if re.search(pattern, stripped):
                    active = name
                    break
            buckets[active].append(line)
        return [f"{name}\n" + "\n".join(items).strip() for name, items in buckets.items() if "\n".join(items).strip()]

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
                    windows = [w if prefix in w[: len(prefix) + 10] else f"{prefix}\n{w}" for w in windows]
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
        step = max(1, self.target_child_size - self.overlap)
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

    def _source_file(self, docs: list[Document]) -> str:
        value = docs[0].metadata.get("source_file") or docs[0].metadata.get("file_name") or "document.pdf"
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
        if doc_type in {"general_regulation", "exam_regulation", "module_catalog", "study_plan", "elective_catalog", "form"}:
            return doc_type
        return "generic_pdf"

    def _revision_date(self, file_name: str) -> Optional[str]:
        dates: list[date] = []
        for day, month, year in re.findall(r"(?<!\d)(\d{1,2})[._-](\d{1,2})[._-](\d{2,4})(?!\d)", file_name):
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
        # Best effort without corpus-wide comparison: flag recent academic/legal
        # revisions as likely current, old revisions as historical.
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
        return "\n\n".join((doc.text or "").strip() for doc in docs if (doc.text or "").strip()).strip()

    def _docs_overlapping_text(self, docs: list[Document], text: str) -> list[Document]:
        # Approximate page metadata for regex-derived blocks by selecting elements
        # whose text occurs inside the section. Fall back to all docs if matching is
        # impossible after OCR/layout normalization.
        selected = [doc for doc in docs if doc.text and doc.text.strip()[:80] in text]
        return selected or docs

    def _detect_module_title(self, text: str, element_type: Any = None) -> Optional[str]:
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
        for line in text.splitlines()[:20]:
            detected = self._detect_module_title(line)
            if detected:
                return detected
        return None

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
