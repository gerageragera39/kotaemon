# Development documentation pages

This folder holds deeper contributor documentation that complements [`../development.md`](../development.md).

## Important contents

- `index.md` - upstream development landing page.
- `contributing.md` - contribution workflow guidance.
- `create-a-component.md` and `data-components.md` - component/data extension notes inherited from Kotaemon.
- `utilities.md` - helper utilities documentation.

## How it connects

These pages document extension points used by the code under [`../../src/kotaemon`](../../src/kotaemon/README.md) and [`../../src/ktem`](../../src/ktem/README.md). KURAGa keeps many upstream package names for compatibility, so upstream component terminology still appears here.

## Before changing

- Preserve upstream attribution when editing inherited Kotaemon/Cinnamon content.
- Add KURAGa-specific notes only where the fork behavior differs, such as local-first models, guest access, university PDF chunking, and hybrid retrieval defaults.
- Keep file names stable unless [`../../mkdocs.yml`](../../mkdocs.yml) navigation is updated in the same change.

## Verification

```bash
pre-commit run prettier --all-files --show-diff-on-failure
pre-commit run codespell --all-files
```
