# Source packages

KURAGa keeps the upstream Kotaemon package split:

- [`ktem/`](ktem/README.md) - Gradio application layer, settings, pages, resource managers, evaluation UI, and KURAGa-specific guest/admin behavior.
- [`kotaemon/`](kotaemon/README.md) - reusable RAG components inherited and adapted from Kotaemon/Cinnamon: loaders, splitters, vector/doc stores, retrievers, LLM wrappers, rerankers, and citation QA.

## How it connects

`app.py` imports `ktem.main.App`, which builds the Gradio UI and registers indices/reasoning pipelines from `flowsettings.py`. The file index layer in `ktem.index.file` composes lower-level `kotaemon` components for ingestion, retrieval, reranking, and citation-aware answer generation.

```text
app.py
  -> src/ktem/main.py
  -> src/ktem/index/file/* and src/ktem/reasoning/*
  -> src/kotaemon/loaders, indices, storages, llms, embeddings, rerankings
```

## Before changing

- Keep public branding as KURAGa, but do not rename internal `ktem`/`kotaemon` packages without a migration plan.
- Prefer adding fork-specific behavior at KURAGa boundaries (`ktem` app/index settings, explicit university PDF mode gates) rather than mutating generic upstream abstractions unnecessarily.
- Preserve Apache-2.0 upstream attribution in source and docs.

## Verification

```bash
python -m compileall src
pytest -q tests
```
