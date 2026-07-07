# KNet file index variant

Optional KNet index implementation inherited from upstream Kotaemon.

## Important files

- `knet_index.py` - index registration/lifecycle.
- `pipelines.py` - KNet ingestion and retrieval pipelines.

## How it connects

This folder is adjacent to the default file index pipeline but is not the main KURAGa university-document path.

## Before changing

- Keep settings and dependencies isolated from the default file index.
- Add tests before making KNet part of the default app configuration.

## Verification

```bash
python -m compileall src/ktem/index/file/knet
```
