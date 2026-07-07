# Kotaemon utilities

Low-level helpers shared by reusable RAG components.

## Important files

- `rag_debug.py` - lightweight structured debug logging for retrieval/evaluation diagnostics.

## How it connects

Retrieval pipelines record debug events and summaries through these helpers so tests, scripts, and evaluation reports can inspect behavior.

## Before changing

- Keep debug helpers safe for local/offline runs and avoid logging secrets.

## Verification

```bash
python -m compileall src/kotaemon/utils
```
