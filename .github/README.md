# GitHub repository metadata

This folder contains GitHub-only project automation for KURAGa.

## Important contents

- `workflows/` - CI, release, and container build workflows. See [`workflows/README.md`](workflows/README.md).
- `ISSUE_TEMPLATE/` - YAML issue forms for bug reports and feature requests.
- `PULL_REQUEST_TEMPLATE.md` - contributor checklist shown on new pull requests.

## How it connects

The workflows validate the same commands developers should run locally from the repository root: `pytest -q tests` and `pre-commit run --all-files --show-diff-on-failure`. Release workflows package the flat KURAGa app layout (`app.py`, `flowsettings.py`, `src/`, `docs/`, and scripts), not the original upstream monorepo layout.

## Before changing

- Keep workflow paths aligned with this repository layout; do not reintroduce upstream `libs/*` assumptions.
- Keep CI on Python 3.11 unless the runtime and dependency pins are updated together.
- Do not weaken style or test workflows to hide failures; fix the files or config that fail locally.

## Verification

```bash
pytest -q tests
pre-commit run --all-files --show-diff-on-failure
```
