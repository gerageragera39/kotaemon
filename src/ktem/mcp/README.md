# MCP resource manager

Application-layer manager/UI for Model Context Protocol resources.

## Important files

- `manager.py` - loads configured MCP resources.
- `db.py` - persisted MCP resource records.
- `ui.py` - Resources tab UI for MCP configuration.

## How it connects

MCP resources are optional app extensions alongside LLM, embedding, and reranking resources. They are not required for the default KURAGa university RAG flow.

## Before changing

- Keep MCP optional so local/offline RAG works without external tool servers.
- Avoid storing credentials in committed specs.

## Verification

```bash
python -m compileall src/ktem/mcp
```
