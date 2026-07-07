# Evaluation

This package implements KURAGa's local RAG evaluation pipeline and export helpers.

## Important files

- `ragas_eval.py` - loads evaluation datasets, runs retrieval and answer generation, optionally runs RAGAS/local judge metrics, records retrieval diagnostics, and builds CSV exports.
- `__init__.py` - public exports for the Evaluation tab and CLI scripts.

## How it connects

- The Evaluation tab in [`../pages/evaluation.py`](../pages/evaluation.py) calls this package.
- [`../../../scripts/run_rag_eval.py`](../../../scripts/run_rag_eval.py) exposes the same path from the command line.
- Root datasets such as [`../../../rag_eval_dataset.json`](../../../rag_eval_dataset.json) provide question/reference rows.
- Results are normally written under `ktem_app_data/evaluations/`.

## Modes and local/offline behavior

The evaluator can run answer-only/retrieval diagnostics without invoking RAGAS scoring. When RAGAS is enabled, `RagasEvaluatorModels` resolves local configured LLM/embedding resources so evaluation does not silently fall back to hosted OpenAI defaults.

## Before changing

- Keep export columns backward-compatible where possible; downstream notebooks and CSV checks depend on stable names like `answer`, `contexts`, `retrieval_scope`, and latency fields.
- Be careful with concurrency and token limits; local model servers can be slower than cloud APIs.
- Do not store API keys or generated evaluation CSVs in source.

## Verification

```bash
pytest -q tests/test_rag_evaluation_modes.py
python scripts/run_rag_eval.py --help
```
