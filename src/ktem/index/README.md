# Index application layer

This package registers and manages app-visible indexes. In the current KURAGa configuration, the main index type is the file index.

## Important contents

- `manager.py` - loads index types configured in `flowsettings.py` and starts them at app startup.
- `base.py` - base index interfaces used by UI pages and settings.
- `models.py` - SQL metadata records that connect source files to document/vector chunks.
- `ui.py` - shared index UI helpers.
- `file/` - file upload, ingestion, retrieval, and file-index UI implementation.

## How it connects

`BaseApp.initialize_indices()` creates an `IndexManager`, which instantiates index types from `KH_INDEX_TYPES`/`KH_INDICES` in `flowsettings.py`. The Chat page asks each index for selector components and retriever pipelines.

## Before changing

- Index IDs appear in flattened setting keys (`index.options.<id>.*`); changing IDs can break saved settings and tests.
- SQL metadata rows connect uploaded sources to docstore/vectorstore IDs. Keep relation types stable unless migrations are provided.
- For KURAGa university PDFs, route behavior through `file/pipelines.py` and the explicit `reader_mode`/environment gates.

## Verification

```bash
pytest -q tests/test_guest_search_scope.py tests/test_university_pdf_pipeline.py tests/test_vectorindex_hybrid_regression.py
```
