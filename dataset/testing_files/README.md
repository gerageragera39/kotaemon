# Retrieval testing files

Smaller PDF fixture set for retrieval and evaluation smoke checks.

## Important contents

- Current university PDFs used by retrieval-only scripts.
- `old/` - older document revisions kept for comparison/regression context.

## How it connects

`scripts/eval_university_retrieval.py` re-ingests this folder with university PDF mode and checks retrieval behavior against the evaluation dataset.

## Before changing

- Keep fixture size suitable for local CI/developer runs.
- If replacing documents, update expected answers/references in root `rag_eval_dataset*.json` where applicable.

## Verification

```bash
python scripts/eval_university_retrieval.py --help
pytest -q tests/test_vectorindex_hybrid_regression.py
```
