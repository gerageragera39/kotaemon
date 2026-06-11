#!/usr/bin/env python3
"""Ingest the D3B university PDF corpus with the production university PDF mode.

This script intentionally does not rely on DocumentIngestor's default PDF mode.
It is a deterministic preflight/indexing helper for dataset/documents/*.pdf and
writes chunk records that can be inspected or passed to downstream indexing jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kotaemon.indices.ingests import DocumentIngestor  # noqa: E402


def doc_to_record(doc) -> dict[str, Any]:
    return {"id": doc.doc_id, "text": doc.text, "metadata": doc.metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index/preflight university PDFs with pdf_mode=university."
    )
    parser.add_argument(
        "--documents-dir",
        default=os.environ.get(
            "UNIVERSITY_RAG_DOCUMENTS_DIR", str(REPO_ROOT / "dataset" / "documents")
        ),
        help="Directory containing the university PDF corpus.",
    )
    parser.add_argument(
        "--pdf-mode",
        default=os.environ.get("UNIVERSITY_RAG_PDF_MODE", "university"),
        choices=["normal", "mathpix", "ocr", "multimodal", "university"],
        help="Must be 'university' for dataset/documents.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "dataset" / ".cache" / "university_index"),
        help="Directory for JSONL chunks and summary manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pdf_mode != "university":
        raise ValueError(
            "dataset/documents PDFs must be indexed with pdf_mode='university' "
            f"(got {args.pdf_mode!r})"
        )

    documents_dir = Path(args.documents_dir).expanduser().resolve()
    pdfs = sorted(documents_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {documents_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Indexing {len(pdfs)} PDFs from {documents_dir} with pdf_mode=university",
        flush=True,
    )
    chunks = DocumentIngestor(pdf_mode="university").run(pdfs)

    jsonl_path = output_dir / "chunks.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for doc in chunks:
            f.write(json.dumps(doc_to_record(doc), ensure_ascii=False) + "\n")

    summary = {
        "documents_dir": str(documents_dir),
        "pdf_count": len(pdfs),
        "pdf_mode": "university",
        "reader": "DoclingStructuredPDFReader",
        "splitter": "UniversityPDFChunker",
        "parent_count": sum(1 for doc in chunks if doc.metadata.get("index_role") == "parent"),
        "child_count": sum(1 for doc in chunks if doc.metadata.get("index_role") == "child"),
        "chunk_count": len(chunks),
        "output_jsonl": str(jsonl_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
