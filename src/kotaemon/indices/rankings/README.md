# Ranking components

Document ranking/scoring components used by retrieval and evaluation.

## Important files

- `base.py` - ranking interface.
- `llm.py`, `llm_scoring.py`, `llm_trulens.py` - LLM-based scoring/ranking paths.
- `cohere.py` - Cohere ranking integration.

## How it connects

File retrieval can use LLM scoring for UI relevance display or optional reranking depending on settings.

## Before changing

- Keep hosted provider dependencies optional.
- Preserve document score/metadata fields used by the information panel.

## Verification

```bash
pytest -q tests/test_vectorindex_hybrid_regression.py
python -m compileall src/kotaemon/indices/rankings
```
