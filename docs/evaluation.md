# Evaluation

KURAGa includes local evaluation assets for checking retrieval and answer quality.

## Datasets

- `rag_eval_dataset.json`
- `rag_eval_dataset_kazi.json`
- `dataset/testing_files/`
- `dataset/documents/`

Do not delete these files unless a dataset owner confirms they are obsolete.

## Scripts

Run from the repository root:

```bash
python scripts/run_rag_eval.py --help
python scripts/eval_university_retrieval.py --help
python scripts/evaluate_llm_judge.py --help
python scripts/university_rag_smoke.py --help
```

Script options may depend on local model configuration and indexed documents.

## App evaluation tab

The Evaluation tab uses code in `src/ktem/evaluation/ragas_eval.py` and the current app resources/settings. It can collect answers, retrieved contexts, and optional RAGAS-style metrics.
