#!/usr/bin/env python3
"""Run Kotaemon RAG evaluation from the command line.

Example:
    PYTHONPATH=src python scripts/run_rag_eval.py \
        --dataset ../rag_eval_dataset.json --limit 5 --output-dir eval_results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ktem.evaluation import find_default_dataset_path, run_evaluation  # noqa: E402
from ktem.main import App  # noqa: E402
from kotaemon.indices.ingests import DocumentIngestor  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_dataset = find_default_dataset_path(ROOT)
    parser = argparse.ArgumentParser(description="Evaluate Kotaemon RAG with RAGAS.")
    parser.add_argument(
        "--dataset",
        default=str(default_dataset or "rag_eval_dataset.json"),
        help="Path to rag_eval_dataset JSON/JSONL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of questions to evaluate.",
    )
    parser.add_argument(
        "--user-id",
        default="default",
        help="Kotaemon user id for private indexes.",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Only collect answers/contexts, skip RAGAS scoring.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory for CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--index-documents-dir",
        default=os.environ.get("UNIVERSITY_RAG_DOCUMENTS_DIR", ""),
        help=(
            "Optional university PDF directory to chunk as a preflight check before "
            "evaluation. This does not write into the Kotaemon app index; reindex "
            "the PDFs in the UI/API to change retrieval."
        ),
    )
    parser.add_argument(
        "--pdf-mode",
        default=os.environ.get("UNIVERSITY_RAG_PDF_MODE", "university"),
        choices=["normal", "mathpix", "ocr", "multimodal", "university"],
        help="PDF ingestion mode for --index-documents-dir preflight.",
    )
    return parser.parse_args()


def preflight_university_ingest(documents_dir: str, pdf_mode: str) -> None:
    if not documents_dir:
        return
    pdfs = sorted(Path(documents_dir).expanduser().resolve().glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in --index-documents-dir={documents_dir}")
    if pdf_mode != "university":
        raise ValueError(
            "The university dataset must be ingested with pdf_mode='university' "
            f"(got {pdf_mode!r})."
        )

    print(
        f"Preflight chunking {len(pdfs)} PDFs with pdf_mode=university "
        "(DoclingStructuredPDFReader + UniversityPDFChunker); this does not "
        "persist chunks into the app index",
        flush=True,
    )
    chunks = DocumentIngestor(pdf_mode="university").run(pdfs)
    parents = sum(1 for doc in chunks if doc.metadata.get("index_role") == "parent")
    children = sum(1 for doc in chunks if doc.metadata.get("index_role") == "child")
    print(
        f"University ingest preflight complete: parents={parents}, children={children}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_university_ingest(args.index_documents_dir, args.pdf_mode)

    app = App()
    settings = app.default_settings.flatten()

    def progress(done: int, total: int, desc: str):
        print(f"[{done}/{total}] {desc}", flush=True)

    result = run_evaluation(
        app=app,
        settings=settings,
        user_id=args.user_id,
        dataset_path=args.dataset,
        question_limit=args.limit,
        run_ragas_metrics=not args.no_ragas,
        progress=progress,
    )

    result.samples.to_csv(output_dir / "rag_eval_samples.csv", index=False)
    result.ragas_scores.to_csv(output_dir / "ragas_scores.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if result.warnings:
        (output_dir / "warnings.txt").write_text("\n".join(result.warnings), encoding="utf-8")

    print(json.dumps({**result.summary, "run_dir": result.run_dir}, indent=2, ensure_ascii=False))
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
