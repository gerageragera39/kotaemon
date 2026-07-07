# Reasoning pipelines

Reasoning pipelines turn a chat question plus settings into retrieved context and a final answer.

## Important files and folders

- `simple.py` - primary straightforward RAG reasoning path used by KURAGa.
- `base.py` - shared reasoning interfaces and settings contracts.
- `react.py`, `rewoo.py` - agentic reasoning modes inherited from upstream Kotaemon.
- `prompt_optimization/` - question rewriting, decomposition, mindmap, follow-up, and conversation-name helpers.

## How it connects

Reasonings are registered from `KH_REASONINGS` in `flowsettings.py` during `BaseApp.register_reasonings()`. Chat page settings select a reasoning pipeline, which calls index retrievers and QA components in `src/kotaemon/indices/qa/`.

## Before changing

- Keep prompt/settings keys stable; they are flattened into saved UI settings.
- When adding retrieval behavior, prefer changing file retriever settings instead of embedding index-specific assumptions in generic reasoning code.
- Agentic modes may require additional LLM/tool configuration; keep local/offline defaults working for the simple path.

## Verification

```bash
pytest -q tests/test_feedback_repair.py tests/test_rag_evaluation_modes.py
python -m compileall src/ktem/reasoning
```
