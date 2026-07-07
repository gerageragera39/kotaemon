# Storage adapters

Docstore and vectorstore implementations used by indexing and retrieval.

## Important folders

- `docstores/` - simple file, in-memory, LanceDB, Elasticsearch, and other document text/metadata stores.
- `vectorstores/` - Chroma, LanceDB, Milvus, Qdrant, in-memory, and simple file vector stores.

## How it connects

`flowsettings.py` configures default stores. File ingestion writes source/chunk records into docstores and vectorstores, while `VectorRetrieval` queries both branches in hybrid mode.

## Before changing

- Keep document IDs stable across docstore/vectorstore writes; hybrid retrieval and parent expansion depend on ID matching.
- Runtime storage lives under `ktem_app_data/` and should not be committed.
- Backend-specific dependencies should stay optional unless required by default config.

## Verification

```bash
pytest -q tests/test_vectorindex_hybrid_regression.py tests/test_university_pdf_pipeline.py
python -m compileall src/kotaemon/storages
```
