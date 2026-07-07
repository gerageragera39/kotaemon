# LLM resource manager

Application-layer database/UI manager for chat/completion models.

## Important files

- `manager.py` - loads model resource records and resolves defaults.
- `db.py` - persisted model specs.
- `ui.py` - Resources tab model configuration UI.

## How it connects

Reasoning, citation QA, feedback repair regeneration, and evaluation all use LLM resources resolved here. Low-level model wrappers live under [`../../kotaemon/llms`](../../kotaemon/llms/README.md). Local Ollama/OpenAI-compatible defaults are configured in `flowsettings.py`.

## Before changing

- Avoid hard-coding hosted providers into KURAGa-only flows; local/offline operation is a project goal.
- Keep timeout/context options tunable for local servers.

## Verification

```bash
pytest -q tests/test_rag_evaluation_modes.py
python -m compileall src/ktem/llms
```
