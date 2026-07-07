# Splitters

Document splitting components used during ingestion.

## Important files

- `university_pdf.py` - KURAGa structural chunker for German university PDFs. It detects document types, builds semantic parent blocks, emits child chunks for embedding, and attaches metadata such as `doc_type`, `doc_family`, `index_role`, `parent_id`, `section_id`, `module_title`, `chunk_type`, and source file fields.
- `__init__.py` - base splitter exports.

## How it connects

`src/ktem/index/file/pipelines.py` chooses this splitter when university PDF mode is enabled. The emitted parent/child documents are indexed by `VectorIndexing` and later expanded by `VectorRetrieval`.

## Before changing

- Filename heuristics intentionally win for some document types; changing them can alter evaluation fixtures.
- Parent documents should remain docstore-only context containers, while child documents should remain the retrievable embedded chunks.
- Keep metadata deterministic; tests assert IDs, roles, titles, table handling, module metadata, and paragraph/section structure.

## Verification

```bash
pytest -q tests/test_university_pdf_pipeline.py
python scripts/debug_university_chunks.py dataset/documents/<file>.pdf
```
