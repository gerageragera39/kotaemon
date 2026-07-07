# Documentation site source

This folder contains MkDocs pages for developer and user documentation. It is separate from in-app Markdown under `src/ktem/assets/md/`, although some topics intentionally overlap.

## Important contents

- `index.md`, `about.md`, `project_overview.md` - entry pages for the documentation site.
- `admin_guide.md`, `guest_guide.md`, `usage.md`, `local_model.md`, `evaluation.md` - practical operating guides.
- `development.md` and `development/` - contributor-facing upstream/kotaemon development notes plus KURAGa conventions.
- `pages/app/` - upstream-style app reference pages.
- `images/`, `extra/`, `theme/` - MkDocs assets and theme overrides.
- `scripts/` - documentation generation helpers.

## How it connects

- [`../mkdocs.yml`](../mkdocs.yml) defines navigation and theme settings.
- [`../src/ktem/pages/project_docs.py`](../src/ktem/pages/project_docs.py) renders curated project docs inside the KURAGa app for guests/admins.
- Source-code README files under [`../src`](../src/README.md), [`../scripts`](../scripts/README.md), and [`../tests`](../tests/README.md) are developer maps rather than MkDocs nav pages.

## Before changing

- Keep docs accurate to the current fork. Explicitly distinguish upstream Kotaemon/Cinnamon behavior from KURAGa-specific additions.
- Prefer concise procedural docs over marketing copy.
- Use relative links that work both on GitHub and from the MkDocs source tree.
- Prettier formats Markdown in pre-commit; avoid manual spacing that depends on unformatted Markdown.

## Verification

```bash
pre-commit run prettier --all-files --show-diff-on-failure
pre-commit run codespell --all-files
```

If MkDocs dependencies are installed, also run:

```bash
mkdocs build --strict
```
