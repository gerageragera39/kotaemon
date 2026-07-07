# GitHub Actions workflows

Workflow definitions for KURAGa CI, releases, and container publishing.

## Important files

- `unit-test.yaml` - installs a lightweight test dependency set and runs `pytest -q tests`.
- `style-check.yaml` - installs `pre-commit` and runs `pre-commit run --all-files --show-diff-on-failure`.
- `pr-lint.yaml` - validates pull request titles with Conventional Commits; commitlint is currently disabled.
- `auto-bump-and-release.yaml` - on `main`, creates version tags and release ZIPs containing the app, `src/`, docs, scripts, license, and config examples.
- `build-push-docker.yaml` - builds `lite`, `full`, and `ollama` Docker targets and publishes them to GHCR on releases/tags/manual dispatch.

## How it connects

The workflows exercise the same source packages documented under [`../../src/README.md`](../../src/README.md). The Docker workflow depends on [`../../Dockerfile`](../../Dockerfile), [`../../launch.sh`](../../launch.sh), and [`../../requirements_gerageragera39.txt`](../../requirements_gerageragera39.txt).

## Before changing

- Run the local command corresponding to the workflow before pushing.
- Keep release packaging in sync with files needed to run `python app.py` from an unpacked release.
- Docker builds are matrixed by target; target-specific assumptions belong in `Dockerfile`, not in workflow shell snippets.

## Verification

Use these local checks before relying on Actions:

```bash
pytest -q tests
pre-commit run --all-files --show-diff-on-failure
```
