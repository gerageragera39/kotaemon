# Dataset assets

This folder contains local university/evaluation source files used by KURAGa scripts and regression checks.

## Important contents

- `documents/` - university PDF corpus used by the structural university PDF pipeline.
- `testing_files/` - smaller files used by retrieval/evaluation smoke checks.
- `chunks/` - chunk outputs or snapshots when produced by local indexing/debug workflows.
- `html_crawler.py` - helper for collecting HTML/document material.
- `document_groups.docx` - project document grouping notes.

The root files `rag_eval_dataset.json` and `rag_eval_dataset_kazi.json` are the main curated RAG question datasets; they intentionally live at the repository root because scripts default to those paths.

## How it connects

- [`../scripts/index_university_documents.py`](../scripts/index_university_documents.py) reads `dataset/documents` with `pdf_mode=university`.
- [`../scripts/eval_university_retrieval.py`](../scripts/eval_university_retrieval.py) re-ingests `dataset/testing_files` for retrieval-only checks.
- [`../src/ktem/evaluation/ragas_eval.py`](../src/ktem/evaluation/ragas_eval.py) consumes the root evaluation JSON files through the Evaluation tab or CLI.
- [`../src/kotaemon/indices/splitters/university_pdf.py`](../src/kotaemon/indices/splitters/university_pdf.py) contains the university-specific PDF chunking rules these files are meant to exercise.

## Before changing

- Treat university source documents as test fixtures: changing names or paths can change document-type detection and script defaults.
- Do not commit generated runtime databases, Chroma stores, caches, or `ktem_app_data/` here.
- Keep personally identifying or sensitive university data out of committed fixtures.

## Verification

```bash
python scripts/debug_university_chunks.py dataset/documents/<file>.pdf
python scripts/university_rag_smoke.py
pytest -q tests/test_university_pdf_pipeline.py tests/test_vectorindex_hybrid_regression.py
```
