# Application utilities

Shared helper modules for KURAGa app behavior.

## Important files

- `guest_scope.py` - pure helper functions that force guest chat to Search All and strip guest URL/web-search submissions.
- `feedback_repair.py` - disliked-answer repair prompts and temporary retrieval/QA setting adjustments.
- `chat_export.py` - CSV export helpers for conversation and retrieval context.
- `commands.py` - chat command constants such as web search command handling.
- `conversation.py`, `file.py`, `render.py`, `visualize_cited.py` - conversation, file, HTML/rendering, and citation visualization utilities.
- `rate_limit.py` - request throttling helpers.
- `hf_papers.py`, `plantuml.py`, `lang.py`, `generator.py` - optional/support utilities inherited from upstream features.

## How it connects

Utilities are imported by pages, index UI, and reasoning code. Guest and feedback helpers are deliberately testable without Gradio so policy regressions are caught in unit tests.

## Before changing

- Keep security/permission decisions in pure functions where possible; UI-only restrictions are easy to bypass.
- Preserve backward compatibility for stored conversation and feedback metadata.
- Avoid adding heavyweight imports to utility modules that are used at app startup.

## Verification

```bash
pytest -q tests/test_guest_search_scope.py tests/test_feedback_repair.py tests/test_chat_csv_export.py
```
