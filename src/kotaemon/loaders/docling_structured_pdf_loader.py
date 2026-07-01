from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from kotaemon.base import Document

from .base import BaseReader
from .utils.adobe import make_markdown_table


class DoclingStructuredPDFReader(BaseReader):
    """Read PDFs with Docling and emit ordered structural elements.

    Unlike :class:`DoclingReader`, this reader intentionally does not merge all text
    into page blobs. It keeps Docling's reading-order elements so downstream
    splitters can preserve university PDF structure (sections, modules, tables,
    forms) before chunking.
    """

    _dependencies = ["docling"]

    @property
    def converter_(self):
        """Lazily create the Docling converter without theflow Param caching.

        Index routing instantiates this reader directly for the university PDF UI
        mode. In that path, theflow's ``@Param.auto(cache=True)`` descriptor can
        miss its internal ``converter_`` cache slot and raise ``KeyError`` before
        Docling runs. A plain private attribute keeps the reader stateless from the
        pipeline perspective while avoiding descriptor cache initialization issues.
        """

        try:
            return object.__getattribute__(self, "_docling_converter")
        except AttributeError:
            pass

        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise ImportError("Please install docling: 'pip install docling'") from exc

        converter = DocumentConverter()
        object.__setattr__(self, "_docling_converter", converter)
        return converter

    def run(
        self, file_path: str | Path, extra_info: Optional[dict] = None, **kwargs
    ) -> list[Document]:
        return self.load_data(file_path=file_path, extra_info=extra_info, **kwargs)

    def load_data(
        self, file_path: str | Path, extra_info: Optional[dict] = None, **kwargs
    ) -> list[Document]:
        file_path = Path(file_path)
        metadata = extra_info or {}

        result = self.converter_.convert(file_path)
        result_dict = result.document.export_to_dict()

        elements = self._ordered_docling_items(result_dict)
        docs: list[Document] = []
        seen_refs: set[str] = set()

        for order, (ref, item) in enumerate(elements):
            if ref:
                seen_refs.add(ref)
            doc = self._item_to_document(
                item=item,
                order=order,
                file_path=file_path,
                result_dict=result_dict,
                extra_metadata=metadata,
            )
            if doc is not None:
                docs.append(doc)

        # Some Docling exports can omit floating items from body.children. Append any
        # unvisited text/table/picture items sorted by page and bbox so no content is
        # lost, while preserving body order when available.
        append_order = len(docs)
        for ref, item in self._fallback_items(result_dict):
            if ref in seen_refs:
                continue
            doc = self._item_to_document(
                item=item,
                order=append_order,
                file_path=file_path,
                result_dict=result_dict,
                extra_metadata=metadata,
            )
            if doc is not None:
                docs.append(doc)
                append_order += 1

        return docs

    def _ordered_docling_items(
        self, result_dict: dict[str, Any]
    ) -> list[tuple[str, dict]]:
        body = result_dict.get("body") or {}
        ordered: list[tuple[str, dict]] = []

        def visit(node: Any) -> None:
            if isinstance(node, dict) and "$ref" in node:
                ref = node["$ref"]
                item = self._resolve_ref(result_dict, ref)
                if not item:
                    return
                # Groups/containers carry children but no direct text payload.
                if self._is_content_item(item):
                    ordered.append((ref, item))
                for child in item.get("children", []) or []:
                    visit(child)
            elif isinstance(node, dict):
                if self._is_content_item(node):
                    ordered.append(("", node))
                for child in node.get("children", []) or []:
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        visit(body.get("children", []))
        return ordered

    def _fallback_items(self, result_dict: dict[str, Any]) -> list[tuple[str, dict]]:
        items: list[tuple[str, dict]] = []
        for key in ("texts", "tables", "pictures"):
            for idx, item in enumerate(result_dict.get(key, []) or []):
                items.append((f"#/{key}/{idx}", item))

        return sorted(items, key=lambda pair: self._sort_key(pair[1]))

    def _resolve_ref(self, result_dict: dict[str, Any], ref: str) -> Optional[dict]:
        if not ref.startswith("#/"):
            return None
        value: Any = result_dict
        for part in ref[2:].split("/"):
            try:
                if isinstance(value, list):
                    value = value[int(part)]
                else:
                    value = value[part]
            except (KeyError, IndexError, ValueError, TypeError):
                return None
        return value if isinstance(value, dict) else None

    def _is_content_item(self, item: dict[str, Any]) -> bool:
        label = str(item.get("label", "")).lower()
        return bool(
            item.get("text")
            or item.get("orig")
            or item.get("data")
            or label in {"table", "document_index", "picture"}
        )

    def _item_to_document(
        self,
        item: dict[str, Any],
        order: int,
        file_path: Path,
        result_dict: dict[str, Any],
        extra_metadata: dict,
    ) -> Optional[Document]:
        label = str(item.get("label", "") or "")
        element_type = self._element_type(label)
        text = self._extract_text(
            item=item, element_type=element_type, result_dict=result_dict
        )
        if not text and element_type != "figure":
            return None

        prov = self._first_prov(item)
        page_label = prov.get("page_no") if prov else None
        bbox = prov.get("bbox") if prov else None

        doc_metadata: dict[str, Any] = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "page_label": page_label,
            "order": order,
            "element_type": element_type,
            "docling_label": label or None,
        }
        if bbox is not None:
            doc_metadata["bbox"] = bbox
        if element_type == "table":
            doc_metadata["table_origin"] = text
        doc_metadata.update(extra_metadata)

        return Document(text=text, metadata=doc_metadata)

    def _extract_text(
        self, item: dict[str, Any], element_type: str, result_dict: dict[str, Any]
    ) -> str:
        if element_type == "table":
            caption = self._caption_text(item, result_dict)
            table = self._parse_table(item)
            return "\n".join(part for part in [caption, table] if part).strip()

        if element_type == "figure":
            caption = self._caption_text(item, result_dict)
            return caption or "[Abbildung]"

        text = item.get("text") or item.get("orig") or ""
        return str(text).strip()

    def _caption_text(self, item: dict[str, Any], result_dict: dict[str, Any]) -> str:
        captions: list[str] = []
        for caption in item.get("captions", []) or []:
            ref = caption.get("$ref") if isinstance(caption, dict) else None
            caption_obj = self._resolve_ref(result_dict, ref) if ref else None
            text = caption_obj.get("text") if caption_obj else None
            if text:
                captions.append(str(text).strip())
        return "\n".join(captions).strip()

    def _parse_table(self, table_obj: dict[str, Any]) -> str:
        data = table_obj.get("data") or {}
        grid = data.get("grid") or []
        table_as_list: list[list[str]] = []
        for row in grid:
            cells = []
            for cell in row:
                if not isinstance(cell, dict):
                    cells.append(str(cell))
                    continue
                cells.append(str(cell.get("text") or cell.get("orig") or ""))
            table_as_list.append(cells)
        if not table_as_list:
            return ""
        return make_markdown_table(table_as_list)

    def _element_type(self, label: str) -> str:
        normalized = label.lower().replace("-", "_")
        if normalized in {"title", "section_header"}:
            return "heading"
        if normalized in {"paragraph", "text", "caption", "footnote", "reference"}:
            return "paragraph"
        if normalized == "list_item":
            return "list_item"
        if normalized in {"table", "document_index"}:
            return "table"
        if normalized == "picture":
            return "figure"
        return "unknown"

    def _first_prov(self, item: dict[str, Any]) -> dict[str, Any]:
        prov = item.get("prov") or []
        return prov[0] if prov and isinstance(prov[0], dict) else {}

    def _sort_key(self, item: dict[str, Any]) -> tuple[int, float, float]:
        prov = self._first_prov(item)
        bbox = prov.get("bbox") or {}
        top = bbox.get("t", bbox.get("top", 0.0)) if isinstance(bbox, dict) else 0.0
        left = bbox.get("l", bbox.get("left", 0.0)) if isinstance(bbox, dict) else 0.0
        try:
            page = int(prov.get("page_no", 0))
        except (TypeError, ValueError):
            page = 0
        return (page, float(top or 0.0), float(left or 0.0))
