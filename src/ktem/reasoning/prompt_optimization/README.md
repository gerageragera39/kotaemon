# Prompt optimization helpers

Question rewriting, decomposition, mindmap, and suggestion helpers used by chat reasoning/UI.

## Important files

- `rewrite_question.py`, `fewshot_rewrite_question.py`, `decompose_question.py` - query rewriting/decomposition helpers.
- `suggest_conversation_name.py`, `suggest_followup_chat.py` - UI convenience generation.
- `mindmap.py` - mindmap prompt/export support.
- `rephrase_question_train.json` - examples for rewrite behavior.

## How it connects

The Chat page and reasoning pipelines can call these helpers before or after retrieval. Hybrid retrieval also has lower-level query variant expansion in `kotaemon.indices.vectorindex`.

## Before changing

- Keep local model latency in mind; every extra prompt can slow guest chat.
- Preserve JSON fixture validity and avoid adding private examples.

## Verification

```bash
python -m compileall src/ktem/reasoning/prompt_optimization
```
