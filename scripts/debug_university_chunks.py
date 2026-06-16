#!/usr/bin/env python
from __future__ import annotations

import json
import re
import statistics
import sys
import argparse
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kotaemon.indices.splitters import UniversityPDFChunker  # noqa: E402
from kotaemon.loaders import DoclingStructuredPDFReader  # noqa: E402

DOCS_DIR = REPO_ROOT / "dataset" / "testing_files"
OUT_DIR = REPO_ROOT / "dataset" / ".cache" / "chunks_debug"


def doc_to_record(doc) -> dict[str, Any]:
    return {
        "id": doc.doc_id,
        "source_file": doc.metadata.get("source_file") or doc.metadata.get("file_name"),
        "doc_type": doc.metadata.get("doc_type"),
        "chunk_type": doc.metadata.get("chunk_type"),
        "section_path": doc.metadata.get("section_path"),
        "nearest_heading": doc.metadata.get("nearest_heading"),
        "page_label_start": doc.metadata.get("page_label_start"),
        "page_label_end": doc.metadata.get("page_label_end"),
        "text_preview": (doc.text or "")[:500],
        "text": doc.text,
        "metadata": doc.metadata,
    }


def write_jsonl(path: Path, docs) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc_to_record(doc), ensure_ascii=False) + "\n")


def write_markdown(path: Path, docs) -> None:
    with path.open("w", encoding="utf-8") as f:
        for idx, doc in enumerate(docs, start=1):
            role = doc.metadata.get("index_role", "element")
            title = doc.metadata.get("section_title") or doc.metadata.get("module_title") or doc.metadata.get("chunk_type")
            f.write(f"\n\n## {idx}. {role}: {title}\n\n")
            f.write("```json\n")
            f.write(json.dumps(doc.metadata, ensure_ascii=False, indent=2))
            f.write("\n```\n\n")
            f.write(doc.text or "")
            f.write("\n")


def assert_sanity(chunks) -> None:
    children = [d for d in chunks if d.metadata.get("index_role") == "child"]
    parents = [d for d in chunks if d.metadata.get("index_role") == "parent"]

    assert all((d.text or "").strip() for d in children), "empty child chunk"
    for child in children:
        for key in ["parent_id", "source_file", "doc_type", "chunk_id"]:
            assert child.metadata.get(key), f"child missing {key}"

    assert all(p.metadata.get("index_role") == "parent" for p in parents), "invalid parent role"
    assert not [p for p in parents if p.metadata.get("chunk_id")], "parent has child chunk_id"

    for child in children:
        if child.metadata.get("doc_type") in {"general_regulation", "exam_regulation", "amendment"}:
            title = child.metadata.get("section_title") or ""
            if title.startswith("§"):
                assert title in child.text, "regulation child lost § title"
            real_starts = re.findall(r"(?m)^§\s*\d+", child.text or "")
            assert len(real_starts) <= 1, (
                "regulation child contains multiple real section starts: "
                f"{child.metadata.get('source_file')} {title} {real_starts}"
            )
            if child.metadata.get("section_id") == "§ 8" and (
                "Prüfende, Beisitzende, Aufsichtsführende" in child.text
            ):
                assert (
                    child.metadata.get("section_title")
                    == "§ 8 Prüfende, Beisitzende, Aufsichtsführende"
                ), "two-line § 8 title was not joined"
            if title.startswith("§") and "PRÜFUNGSORGANE" in (child.metadata.get("major_heading") or ""):
                assert "Hauptüberschrift: III. PRÜFUNGSORGANE" in child.text
        if child.metadata.get("chunk_type") == "table":
            assert "|" in child.text and "---" in child.text, "table chunk lost markdown syntax"
        if child.metadata.get("doc_type") == "module_catalog":
            assert child.metadata.get("module_title"), "module child missing module_title"


def stats_for(file_name: str, chunks, chunker: UniversityPDFChunker) -> dict[str, Any]:
    children = [d for d in chunks if d.metadata.get("index_role") == "child"]
    parents = [d for d in chunks if d.metadata.get("index_role") == "parent"]
    counts = [int(d.metadata.get("token_count") or 0) for d in children]
    required = ["parent_id", "source_file", "doc_type", "chunk_id"]
    missing_metadata = {
        key: sum(1 for d in children if not d.metadata.get(key)) for key in required
    }
    missing_metadata.update(
        {
            "page_label_start/end": sum(
                1
                for d in children
                if d.metadata.get("page_label_start") is None
                and d.metadata.get("page_label_end") is None
            ),
            "token_count": sum(1 for d in children if not d.metadata.get("token_count")),
        }
    )
    return {
        "file_name": file_name,
        "doc_type": children[0].metadata.get("doc_type") if children else None,
        "num_parent_chunks": len(parents),
        "num_child_chunks": len(children),
        "min_token_count": min(counts) if counts else 0,
        "avg_token_count": round(statistics.mean(counts), 1) if counts else 0,
        "max_token_count": max(counts) if counts else 0,
        "empty_chunks_count": sum(1 for d in children if not (d.text or "").strip()),
        "chunks_over_max_size": sum(1 for d in children if int(d.metadata.get("token_count") or 0) > chunker.max_child_size),
        "missing_metadata": missing_metadata,
        "chunks_without_page_metadata": sum(
            1
            for d in children
            if d.metadata.get("page_label_start") is None and d.metadata.get("page_label_end") is None
        ),
        "parent_docs_embeddable_count": sum(
            1 for d in parents if d.metadata.get("index_role") != "parent"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump university PDF chunks to JSONL/Markdown.")
    parser.add_argument("--documents-dir", default=str(DOCS_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    docs_dir = Path(args.documents_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {docs_dir}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    reader = DoclingStructuredPDFReader()
    chunker = UniversityPDFChunker()

    all_stats = []
    for pdf in pdfs:
        print(f"Processing {pdf.name} ...", flush=True)
        elements = reader.load_data(pdf)
        chunks = chunker.run(elements)
        assert_sanity(chunks)

        jsonl_path = out_dir / f"{pdf.stem}.jsonl"
        md_path = out_dir / f"{pdf.stem}.md"
        write_jsonl(jsonl_path, chunks)
        write_markdown(md_path, chunks)

        stats = stats_for(pdf.name, chunks, chunker)
        all_stats.append(stats)
        print(
            "{file_name} | type={doc_type} | parents={num_parent_chunks} | "
            "children={num_child_chunks} | tokens min/avg/max="
            "{min_token_count}/{avg_token_count}/{max_token_count} | "
            "empty={empty_chunks_count} | over_max={chunks_over_max_size} | "
            "no_page={chunks_without_page_metadata} | missing={missing_metadata}".format(**stats)
        )

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote debug chunks to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
