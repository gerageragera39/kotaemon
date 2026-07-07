# University document corpus

PDF source corpus for KURAGa's university-document retrieval work.

## Important contents

The files are German university/study-program documents such as examination regulations, module catalogues, study plans, forms, and flyers. Filenames are intentionally meaningful because `UniversityPDFChunker.detect_doc_type()` uses filename heuristics before body heuristics.

## How it connects

- `scripts/index_university_documents.py` defaults to this folder.
- `src/ktem/index/file/pipelines.py` automatically enables university PDF mode for PDFs under `dataset/documents`.
- `src/kotaemon/indices/splitters/university_pdf.py` emits parent/child chunks and metadata from these documents.

## Before changing

- Keep file names stable unless you intend to update chunker heuristics and tests.
- Do not add private/sensitive documents.
- Large generated indexes belong in `ktem_app_data/` or `dataset/.cache/`, not here.

## Verification

```bash
python scripts/index_university_documents.py --help
pytest -q tests/test_university_pdf_pipeline.py
```
