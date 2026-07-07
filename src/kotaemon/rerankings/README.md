# Reranking wrappers

Low-level reranker implementations used after retrieval candidate generation.

## Important files

- `tei_fast_rerank.py` - Text Embeddings Inference reranker wrapper for local cross-encoder reranking.
- `cohere.py`, `voyageai.py` - hosted provider rerankers.
- `base.py` - base reranking interface.

## How it connects

`src/ktem/rerankings` manages configured resources. `DocumentRetrievalPipeline` optionally applies selected rerankers after `VectorRetrieval` candidate fusion.

## Before changing

- Rerankers should preserve document metadata and scores needed by citations/debug panels.
- Keep failures explicit; silent reranker fallback can make evaluation misleading.

## Verification

```bash
pytest -q tests/test_vectorindex_hybrid_regression.py
python -m compileall src/kotaemon/rerankings
```
