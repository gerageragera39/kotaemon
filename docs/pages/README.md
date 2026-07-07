# App reference pages

MkDocs pages inherited from the upstream Kotaemon app documentation.

## Important contents

- `app/` - app feature, settings, index, extension, and flow customization pages.

## How it connects

These pages document concepts implemented in `src/ktem` and `src/kotaemon`. KURAGa-specific operational guides live one level up in `docs/*.md`.

## Before changing

- Preserve upstream attribution and terminology where documenting inherited behavior.
- Prefer adding fork-specific notes where KURAGa differs instead of rewriting upstream pages wholesale.

## Verification

```bash
pre-commit run prettier --all-files --show-diff-on-failure
```
