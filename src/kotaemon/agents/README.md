# Agent components

Agent frameworks inherited from upstream Kotaemon.

## Important contents

- `react/` - ReAct-style agent implementation and prompts.
- `rewoo/` - ReWOO planner/solver agent implementation.
- `tools/` - tool wrappers for LLM, Google, Wikipedia, and MCP-style tools.
- `io/`, `base.py`, `langchain_based.py`, `utils.py` - shared agent contracts and adapters.

## How it connects

These components are optional reasoning/tooling building blocks. KURAGa's default university RAG path uses simpler retrieval-grounded reasoning, but app settings can register additional reasoning modes.

## Before changing

- Keep external tool dependencies optional.
- Do not let agentic modes bypass retrieval grounding or guest restrictions without explicit product decisions and tests.

## Verification

```bash
python -m compileall src/kotaemon/agents
```
