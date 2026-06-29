# Local models

KURAGa is configured for local-first model usage through Ollama/OpenAI-compatible APIs.

!!! note "Docker vs host"
    If KURAGa runs inside Docker and Ollama runs on the host, use `http://host.docker.internal:11434/v1/` instead of `http://localhost:11434/v1/`.

## Ollama recommended setup

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

`.env` example:

```env
LOCAL_MODEL=qwen2.5:7b
LOCAL_MODEL_EMBEDDINGS=nomic-embed-text
KH_OLLAMA_URL=http://localhost:11434/v1/
```

Restart `python app.py` after changing `.env`.

## OpenAI-compatible providers

In **Resources**, add an LLM or embedding model with specs such as:

```yaml
__type__: kotaemon.llms.ChatOpenAI
api_key: dummy-or-provider-key
base_url: http://localhost:11434/v1/
model: qwen2.5:7b
```

Embeddings use:

```yaml
__type__: kotaemon.embeddings.OpenAIEmbeddings
api_key: dummy-or-provider-key
base_url: http://localhost:11434/v1/
model: nomic-embed-text
```

## llama.cpp `.gguf` helper

`scripts/serve_local.py` is a separate workflow. There, `LOCAL_MODEL` is a filesystem path to a `.gguf` file, not an Ollama model name.

```env
LOCAL_MODEL=C:\models\my-model.gguf
```

```bash
python scripts/serve_local.py
```

Register the server in Resources with `base_url: http://localhost:31415/v1/`.

## Optional reranker

KURAGa can use a local Text Embeddings Inference reranker such as `BAAI/bge-reranker-v2-m3` on port 8080. Register it as `kotaemon.rerankings.TeiFastReranking` or use the default in `flowsettings.py` when the service is available.
