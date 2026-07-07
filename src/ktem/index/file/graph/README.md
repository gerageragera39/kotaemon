# Graph file index variants

Optional graph-based index implementations inherited from upstream Kotaemon.

## Important files

- `graph_index.py`, `pipelines.py`, `visualize.py` - graph index setup and visualization.
- `light_graph_index.py`, `lightrag_pipelines.py` - LightRAG integration path.
- `nano_graph_index.py`, `nano_pipelines.py` - NanoGraphRAG integration path.

## How it connects

These are optional file-index variants under `src/ktem/index/file`. KURAGa's default university RAG path uses the standard file index, not graph mode.

## Before changing

- Keep optional dependencies lazy; graph backends may not be installed in normal CI.
- Do not assume university PDF chunk metadata exists unless graph indexing explicitly routes through that pipeline.

## Verification

```bash
python -m compileall src/ktem/index/file/graph
```
