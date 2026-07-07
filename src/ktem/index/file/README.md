# File index

The file index is KURAGa's main document-ingestion and retrieval path. It lets admins upload/index files and lets chat queries retrieve chunks from the indexed collection.

## Important files and folders

- `index.py` - `FileIndex` registration and lifecycle.
- `ui.py` - upload/indexing UI, file selector behavior, guest checks, and admin file management.
- `pipelines.py` - indexing and retrieval pipelines, user settings, university PDF routing, hybrid retrieval defaults.
- `ingestion_v2.py` - lower-level ingestion flow that writes document/vector chunks and metadata.
- `base.py`, `models.py`, `utils.py`, `exceptions.py` - shared types and support code.
- `graph/`, `knet/` - optional graph/KNet index implementations inherited from upstream Kotaemon.

## How it connects

Indexing composes `kotaemon.loaders`, splitters, embeddings, docstores, and vectorstores. Retrieval creates a `kotaemon.indices.vectorindex.VectorRetrieval` with defaults tuned for KURAGa: `retrieval_mode="hybrid"`, `candidate_multiplier=20`, optional query expansion, optional reranking, and configurable context expansion (`none`, `parent`, or `siblings`).

University PDF mode is explicit. It is enabled when:

- the UI `reader_mode` is `university`,
- `UNIVERSITY_RAG_PDF_MODE` or `KOTAEMON_FILE_INDEX_PDF_MODE` is `university`,
- `UNIVERSITY_RAG_DOCUMENTS_DIR` contains the file, or
- the PDF is under `dataset/documents`.

When enabled, PDFs are read by `DoclingStructuredPDFReader` and split by `UniversityPDFChunker` into parent/child chunks.

## Before changing

- Keep selected-file scoping separate for vector chunk IDs and parent document IDs; parent docs may not exist in the vector store.
- Do not make university structural chunking the default for every PDF unless tests and docs are updated; it is designed for German university documents.
- If retrieval settings change, update tests that assert defaults and debug metadata.
- Optional graph/KNet paths are upstream features; avoid mixing KURAGa university-specific behavior into them unless intentionally supported.

## Verification

```bash
pytest -q tests/test_university_pdf_pipeline.py tests/test_vectorindex_hybrid_regression.py tests/test_guest_search_scope.py
python scripts/university_rag_smoke.py
```
