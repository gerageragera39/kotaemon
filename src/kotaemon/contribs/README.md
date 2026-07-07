# Contributed utilities

Optional upstream utilities bundled with Kotaemon.

## Important contents

- `docs.py` - documentation helpers.
- `promptui/` - prompt UI/CLI utilities.

## How it connects

These utilities support extension/developer workflows and are not part of the default KURAGa chat path.

## Before changing

- Keep optional CLI dependencies isolated from the main app runtime.
- Preserve upstream attribution.

## Verification

```bash
python -m compileall src/kotaemon/contribs
```
