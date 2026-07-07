# Embedding resource manager

Application-layer database/UI manager for embedding models.

## Important files

- `manager.py` - loads embedding resources and ensures local defaults are available.
- `db.py` - persisted resource records.
- `ui.py` - Resources tab UI for embedding configuration, including advanced local YAML.

## How it connects

File indexing and retrieval use the selected embedding from this manager. Low-level embedding wrappers live under [`../../kotaemon/embeddings`](../../kotaemon/embeddings/README.md). Defaults are configured in `flowsettings.py` and `.env` through `LOCAL_MODEL_EMBEDDINGS` and `KH_OLLAMA_URL`.

## Before changing

- Preserve local-first defaults; older cloud rows may exist in user databases, but KURAGa should still expose configured local resources.
- Keep resource specs serializable and compatible with the UI YAML editor.

## Verification

```bash
pytest -q tests/test_vectorindex_hybrid_regression.py
python -m compileall src/ktem/embeddings
```
