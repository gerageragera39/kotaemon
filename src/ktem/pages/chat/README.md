# Chat page

The chat page is the main user-facing RAG interface.

## Important files

- `__init__.py` - `ChatPage`, event wiring, guest submit handling, rate limiting, feedback repair, citations/PDF-viewer JavaScript hooks, conversation rename/follow-up features.
- `chat_panel.py` - chat message panel components.
- `control.py` - conversation list and conversation control UI.
- `common.py` - shared chat state constants.
- `chat_suggestion.py` - suggested prompts/follow-up interactions.
- `paper_list.py`, `demo_hint.py`, `report.py` - optional paper/demo/report UI pieces.

## How it connects

`ChatPage` reads index selector components from `app.index_manager.indices`, sends questions to the selected reasoning pipeline, and stores retrieval history for export and the information panel. Citation rendering depends on `src/kotaemon/indices/qa/` and frontend helpers in `src/ktem/assets/js/main.js` plus `pdf_viewer.js`.

## KURAGa-specific behavior

- Guests cannot upload files, select individual files, disable search, submit URL ingestion, or use web search commands.
- Disliked-answer feedback can apply one-shot repair settings from [`../../utils/feedback_repair.py`](../../utils/feedback_repair.py) and regenerate with stricter prompts/retrieval settings.
- Chat CSV export strips rendered HTML context through [`../../utils/chat_export.py`](../../utils/chat_export.py).

## Before changing

- Keep chat history, retrieval history, plot history, and feedback state updates in sync; Gradio output ordering matters.
- Any new guest behavior needs a helper-level test, not only a UI visibility change.
- Citation/PDF UI changes should be checked against `tests/test_information_panel_ordering.py` where relevant.

## Verification

```bash
pytest -q tests/test_guest_search_scope.py tests/test_feedback_repair.py tests/test_chat_csv_export.py tests/test_information_panel_ordering.py
```
