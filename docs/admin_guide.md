# Admin guide

Admins configure models, index university documents, and maintain the document collection that guests query.

## First login

When user management is enabled, the initial local default is commonly:

```text
admin / admin
```

Change the password immediately after first login.

## Configure models

Use **Resources** or `flowsettings.py` to configure:

- LLMs (for answer generation and optional scoring).
- Embedding models (required for retrieval).
- Optional rerankers (for cross-encoder reranking, e.g. TEI).

Defaults are local-first through Ollama/OpenAI-compatible APIs. See [Local models](local_model.md).

## Index documents

1. Open the file collection tab.
2. Upload supported files or web URLs.
3. Click **Upload and Index**.
4. Confirm the file list updates.
5. Ask a test question in Chat and inspect citations.

Supported file types are configured in `flowsettings.py` under `KH_INDICES`.

## Maintain guest scope

Guest users should always search all admin-indexed documents. Do not expose upload, file-selection, settings, resources, or evaluation tabs to guests.

Relevant implementation files:

- `src/ktem/main.py`
- `src/ktem/pages/login.py`
- `src/ktem/pages/chat/__init__.py`
- `src/ktem/index/file/ui.py`
- `src/ktem/utils/guest_scope.py`

## Evaluation

Use the Evaluation tab or scripts under `scripts/` to validate retrieval and answer quality against `rag_eval_dataset*.json`. See [Evaluation](evaluation.md).
