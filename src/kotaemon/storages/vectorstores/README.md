# Vectorstores

Stores for embedding vectors and vector similarity search.

## Important files

- `base.py` - vectorstore interface.
- `chroma.py` - default local vectorstore backend in many Kotaemon/KURAGa setups.
- `in_memory.py`, `simple_file.py` - lightweight/local implementations.
- `lancedb.py`, `milvus.py`, `qdrant.py` - optional backend integrations.

## How it connects

`VectorRetrieval` queries vectorstores for dense candidates, then fuses those with docstore text candidates in hybrid mode.

## Before changing

- Preserve returned `(documents, scores, ids)` semantics expected by retrieval.
- Scope/filter behavior affects selected-file search and guest Search All behavior; update tests when changing it.

## Verification

```bash
pytest -q tests/test_vectorindex_hybrid_regression.py
python -m compileall src/kotaemon/storages/vectorstores
```
