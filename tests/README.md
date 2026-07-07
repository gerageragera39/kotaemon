# Tests

Unit and regression tests for KURAGa-specific behavior.

## Important files

- `test_university_pdf_pipeline.py` - structural university PDF chunking, parent/child metadata, routing, and context expansion.
- `test_vectorindex_hybrid_regression.py` - vector/text/hybrid retrieval, RRF fusion, query expansion, parent/sibling context expansion, debug metadata, and retrieval defaults.
- `test_guest_search_scope.py` - guest Search All enforcement and blocked guest URL/web-search submission behavior.
- `test_feedback_repair.py` - disliked-answer feedback repair prompts and temporary retrieval-setting presets.
- `test_rag_evaluation_modes.py` - evaluation export/mode behavior.
- `test_information_panel_ordering.py` - citation/evidence panel ordering.
- `test_chat_csv_export.py` - chat/retrieval export cleanup.

## How it connects

These tests document the fork-specific contracts that future changes should preserve. They intentionally use focused fakes/in-memory components where possible so CI can run without external model servers or university runtime data.

## Before changing

- Do not remove meaningful tests to make CI green. Update tests only when the intended contract changes.
- Add regression tests for new guest/admin policy, retrieval scoring, chunk metadata, evaluation exports, or feedback behavior.
- Keep tests runnable from the repository root.

## Verification

```bash
pytest -q tests
```

For style/typing hooks, also run:

```bash
pre-commit run --all-files --show-diff-on-failure
```
