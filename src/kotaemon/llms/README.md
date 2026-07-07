# LLM wrappers

Low-level chat/completion model implementations inherited from Kotaemon.

## Important contents

- `chats/openai.py` - OpenAI-compatible chat wrapper used for local Ollama-style endpoints.
- `chats/llamacpp.py`, `chats/langchain_based.py`, `chats/endpoint_based.py` - local/adapter chat backends.
- `completions/` - completion-style wrappers.
- `prompts/` - prompt template helpers.
- `base.py`, `branching.py`, `cot.py`, `linear.py` - shared model composition utilities.

## How it connects

`src/ktem/llms` stores/configures resources, while reasoning, QA/citation, feedback repair, and evaluation call these wrappers through managers.

## Before changing

- Keep local/offline endpoints configurable through `.env`, `flowsettings.py`, and Resources UI.
- Streaming/generator behavior is consumed by the Gradio chat UI; test UI-facing flows when changing output semantics.

## Verification

```bash
pytest -q tests/test_rag_evaluation_modes.py
python -m compileall src/kotaemon/llms
```
