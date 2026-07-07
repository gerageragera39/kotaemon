# Project/component templates

Cookiecutter-style templates inherited from the upstream Kotaemon extension system.

## Important contents

- `component-default/` - template/readme for creating a component package.
- `project-default/` - template project scaffold, including its own `.pre-commit-config.yaml` and generated package README.

## How it connects

These templates are not part of the main KURAGa runtime path, but they document and scaffold upstream-style extension/component development. The main app can load extensions through Pluggy entry points in `ktem.app.BaseApp.register_extensions()`.

## Before changing

- Keep placeholders such as `{{cookiecutter.project_name}}` intact.
- Template Python files are excluded from mypy because placeholder names are not importable packages.
- Preserve upstream attribution when editing inherited template content.

## Verification

```bash
pre-commit run prettier --all-files --show-diff-on-failure
pre-commit run mypy --all-files
```
