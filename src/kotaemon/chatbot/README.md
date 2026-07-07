# Chatbot abstractions

Reusable chatbot/respondent helpers inherited from Kotaemon.

## Important files

- `base.py` - base chatbot/respondent contracts.
- `simple_respondent.py` - simple response implementation.

## How it connects

The Gradio app primarily uses `ktem.reasoning`, but these upstream abstractions remain available for components/extensions.

## Before changing

- Keep API changes compatible with extension/component examples.

## Verification

```bash
python -m compileall src/kotaemon/chatbot
```
