# Cleanup notes

This file records files that look legacy/generated/uncertain and the action taken.

| Path | Why it may be obsolete | Action |
| --- | --- | --- |
| `scripts/run_linux.sh`, `scripts/run_macos.sh`, `scripts/run_windows.bat` | Upstream release installers still contain upstream `libs/*` assumptions and old Python defaults. They are not the recommended KURAGa install path. | Removed from this branch because they installed upstream `libs/*` packages and old release artifacts instead of this flat-layout fork. |
| `scripts/update_linux.sh`, `scripts/update_macos.sh`, `scripts/update_windows.bat` | Same upstream release-installer family; may not match this flat `src/` layout. | Removed from this branch because they updated from upstream `libs/*` release paths rather than this repository layout. |
| `fly.toml` | Generated for an upstream/older `kotaemon` Fly app name. | Left in place; update before any Fly deployment. |
| `docs/images/*` | Many images came from upstream docs and may not reflect KURAGa UI. | Left in place to avoid breaking historical docs; new docs do not depend on remote upstream screenshots. |
| `src/ktem/assets/md/changelogs.md` | Upstream changelog content, not current KURAGa release notes. | Left in place because it is not rendered by the updated Help page; consider deleting or replacing when release notes are maintained. |
| `__pycache__/`, `.pytest_cache/`, `*.pyc` | Generated local caches. | Safe to remove locally; not committed. |
| `dataset/` and `rag_eval_dataset*.json` | Evaluation/course data may look like samples but is part of current validation. | Preserved. |
| `ktem_app_data/` | Runtime DB, uploads, vector stores, and caches. | Preserved; never delete without backup. |
