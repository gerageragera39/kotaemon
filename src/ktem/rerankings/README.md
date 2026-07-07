# Reranking resource manager

Application-layer database/UI manager for reranking models.

## Important files

- `manager.py` - loads configured rerankers and defaults.
- `db.py` - persisted reranker specs.
- `ui.py` - Resources tab UI for reranker configuration.

## How it connects

`ktem.index.file.pipelines.DocumentRetrievalPipeline` optionally applies rerankers after vector/text candidate fusion. Low-level reranker implementations live under [`../../kotaemon/rerankings`](../../kotaemon/rerankings/README.md). A local TEI reranker can be configured through `flowsettings.py`/Resources.

## Before changing

- Reranking is optional and can add latency; keep retrieval usable without it.
- Preserve metadata such as `_retrieval_sources` when rerankers reorder documents.

## Verification

```bash
pytest -q tests/test_vectorindex_hybrid_regression.py
python -m compileall src/ktem/rerankings
```
