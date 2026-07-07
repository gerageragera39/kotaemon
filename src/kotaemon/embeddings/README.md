# Embedding wrappers

Low-level embedding provider implementations inherited from Kotaemon.

## Important files

- `openai.py` - OpenAI-compatible embeddings, used for Ollama/local endpoints through `base_url`.
- `fastembed.py`, `langchain_based.py`, `endpoint_based.py`, `tei_endpoint_embed.py`, `voyageai.py` - alternative providers/backends.
- `base.py` - base embedding interface.

## How it connects

`src/ktem/embeddings` manages persisted resources and selects one of these wrappers for indexing and query embedding. `VectorRetrieval` calls the selected embedding to query vector stores.

## Before changing

- Keep OpenAI-compatible local endpoints working; they are the KURAGa default path.
- Provider-specific dependencies should remain optional unless they are pinned in the main requirements.

## Verification

```bash
pytest -q tests/test_vectorindex_hybrid_regression.py
python -m compileall src/kotaemon/embeddings
```
