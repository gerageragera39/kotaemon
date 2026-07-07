# Chunk snapshots

JSON chunk outputs/snapshots used to inspect university document chunking.

## How it connects

These files are useful when comparing `UniversityPDFChunker` output against expected parent/child structure before indexing into vector/doc stores.

## Before changing

- Treat snapshots as diagnostics, not runtime state.
- Regenerate them from source PDFs when chunking rules intentionally change.
- Keep generated cache/database outputs out of this folder.

## Verification

```bash
python scripts/debug_university_chunks.py dataset/documents/<file>.pdf
pytest -q tests/test_university_pdf_pipeline.py
```
