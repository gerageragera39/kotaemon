# Project overview

KURAGa adapts the Kotaemon document-chat architecture into a university-document RAG assistant.

## Main layers

| Layer          | Path              | Notes                                                                                        |
| -------------- | ----------------- | -------------------------------------------------------------------------------------------- |
| UI/application | `src/ktem/`       | Gradio tabs, login/guest access, chat, document management, evaluation, resources, settings. |
| RAG components | `src/kotaemon/`   | Loaders, splitters, stores, retrievers, LLM wrappers, embeddings, rerankers, citation QA.    |
| Configuration  | `flowsettings.py` | App name, model defaults, index definitions, stores, feature flags.                          |
| Scripts        | `scripts/`        | University indexing, retrieval diagnostics, evaluation, local model helpers.                 |
| Runtime data   | `ktem_app_data/`  | User DB, uploads, vector/doc stores, caches. Do not commit.                                  |

## Chat flow

```text
User question
  -> ChatPage.submit_msg
  -> ChatPage.chat_fn
  -> reasoner in src/ktem/reasoning/simple.py
  -> file retriever pipeline
  -> citation QA in src/kotaemon/indices/qa/
  -> answer + citations/evidence panel
```

Guests are always scoped to **Search All** admin-indexed documents.

## Indexing flow

```text
Admin upload / indexing script
  -> file indexing UI or script
  -> loaders and university-aware PDF handling
  -> text chunks
  -> embeddings
  -> vectorstore/docstore + SQL metadata
```

## Important custom university RAG files

- `src/kotaemon/indices/splitters/university_pdf.py`
- `src/kotaemon/loaders/docling_loader.py`
- `src/kotaemon/loaders/docling_structured_pdf_loader.py`
- `src/ktem/index/file/ingestion_v2.py`
- `src/ktem/index/file/pipelines.py`
- `scripts/index_university_documents.py`
- `scripts/debug_university_chunks.py`
- `scripts/eval_university_retrieval.py`
- `scripts/run_rag_eval.py`
- `scripts/university_rag_smoke.py`
