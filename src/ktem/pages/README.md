# Gradio pages

This folder contains top-level UI pages rendered by `ktem.main.App`.

## Important contents

- `login.py` - login and guest-login flow.
- `chat/` - main chat UI, conversation controls, feedback, citations panel wiring, export, and guest submission policy.
- `project_docs.py` - in-app project documentation tab available to guests.
- `evaluation.py` - evaluation tab wrapper for `ktem.evaluation`.
- `resources/` - model/resource management UI.
- `settings.py`, `setup.py`, `help.py` - app settings, first setup, and help tabs.

## How it connects

`main.py` creates tabs from this folder and controls tab visibility after `onSignIn` / `onSignOut` events. Pages share app-level `user_id`, settings state, and index managers from `BaseApp`.

## Before changing

- Gradio visibility is not a security boundary by itself. For guest behavior, keep helper-level enforcement in [`../utils/guest_scope.py`](../utils/guest_scope.py) and tests.
- Page event chains are order-sensitive; changes to outputs must match the Gradio components declared in `on_building_ui`.
- Avoid importing heavy optional dependencies at module import time unless the page always needs them.

## Verification

```bash
pytest -q tests/test_guest_search_scope.py tests/test_chat_csv_export.py tests/test_feedback_repair.py
```
