# QA and citation pipelines

Answer generation and evidence/citation formatting components.

## Important files

- `citation_qa.py` - `AnswerWithContextPipeline`, answer streaming, citation preparation, and source/evidence panel document preparation.
- `citation_qa_inline.py` - inline citation-answer variant.
- `citation.py` - citation extraction pipeline.
- `format_context.py` - context/evidence packing for text, tables, and figures.
- `utils.py` - QA utility helpers such as model-output cleanup.

## How it connects

Reasoning pipelines pass retrieved documents into these components. The chat page renders the final answer plus citation/evidence documents, and frontend assets turn citation links into scroll/PDF interactions.

## Before changing

- Preserve answer metadata keys such as `citation`; UI and tests inspect them.
- Keep cited and uncited evidence ordering stable unless the information panel tests are updated.
- Be careful with citation threads/timeouts; local LLMs may be slower than hosted models.

## Verification

```bash
pytest -q tests/test_information_panel_ordering.py tests/test_chat_csv_export.py
```
