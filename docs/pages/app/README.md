# App documentation pages

Reference pages for app features and extension points.

## Important contents

- `features.md`, `functional-description.md` - app capabilities and user-facing behavior.
- `customize-flows.md` - how indexing/reasoning pipelines are registered.
- `index/file.md` - file index reference.
- `settings/` - settings concepts.
- `ext/` - extension/user-management notes.

## How it connects

Flow and settings pages map to `flowsettings.py`, `src/ktem/app.py`, `src/ktem/index`, and `src/ktem/reasoning`.

## Before changing

- Keep examples aligned with the flat KURAGa repository layout.
- Use relative links so pages work on GitHub as well as in MkDocs.

## Verification

```bash
pre-commit run prettier --all-files --show-diff-on-failure
```
