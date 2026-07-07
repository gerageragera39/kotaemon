# Docstores

Stores for document text and metadata.

## Important files

- `base.py` - docstore interface.
- `simple_file.py`, `in_memory.py` - local/simple implementations used by tests and local workflows.
- `lancedb.py`, `elasticsearch.py` - optional backend integrations.

## How it connects

Hybrid retrieval queries docstores for lexical/text candidates and parent-context expansion. University parent chunks may exist only in the docstore.

## Before changing

- Preserve `get(ids)` and `query(text, top_k, doc_ids=...)` behavior expected by `VectorRetrieval`.
- Keep optional backend imports lazy.

## Verification

```bash
pytest -q tests/test_vectorindex_hybrid_regression.py tests/test_university_pdf_pipeline.py
python -m compileall src/kotaemon/storages/docstores
```
