from __future__ import annotations

import csv
import json
import math
import os
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from ktem.components import reasonings
from ktem.db.engine import engine
from kotaemon.base import Document
from kotaemon.indices.qa.utils import strip_think_tag
from kotaemon.utils.rag_debug import rag_log
from sqlalchemy import select
from sqlalchemy.orm import Session
from theflow.settings import settings as flowsettings

ProgressFn = Callable[[int, int, str], None]


def _app_data_dir() -> Path:
    return Path(getattr(flowsettings, "KH_APP_DATA_DIR", "ktem_app_data"))


def ensure_app_data_dirs() -> dict[str, Path]:
    root = _app_data_dir()
    paths = {
        "root": root,
        "chats": root / "chats",
        "evaluations": root / "evaluations",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _new_eval_run_dir() -> Path:
    paths = ensure_app_data_dirs()
    run_dir = paths["evaluations"] / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = 1
    candidate = run_dir
    while candidate.exists():
        suffix += 1
        candidate = run_dir.with_name(f"{run_dir.name}_{suffix}")
    candidate.mkdir(parents=True, exist_ok=False)
    for name in ("retrieval_debug.jsonl", "answers.jsonl", "errors.jsonl"):
        (candidate / name).touch()
    return candidate


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _json_safe_settings(settings: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in settings.items():
        try:
            json.dumps(value)
            safe[key] = value
        except TypeError:
            safe[key] = str(value)
    return safe


@dataclass
class EvalRunResult:
    """Result bundle returned by the local Kotaemon + RAGAS evaluation run."""

    samples: pd.DataFrame
    ragas_scores: pd.DataFrame
    summary: dict[str, float | int | str]
    warnings: list[str]
    run_dir: str = ""


@dataclass
class RagasEvaluatorModels:
    """Local evaluator models passed into RAGAS to avoid OpenAI defaults."""

    llm: Any
    embeddings: Any
    raw_embeddings: Any
    llm_name: str
    embeddings_name: str
    run_config: Any | None
    notes: list[str]


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _ragas_run_config() -> Any | None:
    """Runtime settings for local RAGAS scoring.

    Local LLM endpoints are usually slower than hosted evaluators, but a strong
    single-GPU server can still handle bounded parallel judge calls. RAGAS defaults
    to 16 concurrent jobs with a 180s per-job timeout, which can overload local
    endpoints and produce `Exception raised in Job[...]` timeout logs. Keep the
    metric set unchanged, but use moderate parallelism plus a longer per-operation
    timeout. Environment variables allow users to tune this without code changes.
    """

    try:
        from ragas.run_config import RunConfig  # type: ignore
    except Exception:
        return None

    return RunConfig(
        timeout=_env_int("RAGAS_EVAL_TIMEOUT_SEC", 1800),
        max_workers=_env_int("RAGAS_EVAL_MAX_WORKERS", 4),
        max_retries=_env_int("RAGAS_EVAL_MAX_RETRIES", 2),
        max_wait=_env_int("RAGAS_EVAL_MAX_WAIT_SEC", 10),
    )


def _run_config_note(run_config: Any | None) -> str:
    if run_config is None:
        return "RAGAS runtime uses installed-version defaults."
    return (
        "RAGAS runtime: "
        f"timeout={getattr(run_config, 'timeout', 'default')}s, "
        f"max_workers={getattr(run_config, 'max_workers', 'default')}, "
        f"max_retries={getattr(run_config, 'max_retries', 'default')}."
    )


def _apply_model_timeout(model: Any, run_config: Any | None) -> None:
    """Best-effort propagation of RAGAS timeout into LangChain clients."""

    timeout = getattr(run_config, "timeout", None)
    if timeout is None:
        return

    for attr in ("request_timeout", "timeout"):
        try:
            setattr(model, attr, timeout)
        except Exception:
            pass

    # Some LangChain wrappers keep the actual model one level down.
    for nested_attr in ("langchain_llm", "bound", "model"):
        nested = getattr(model, nested_attr, None)
        if nested is not None and nested is not model:
            for attr in ("request_timeout", "timeout"):
                try:
                    setattr(nested, attr, timeout)
                except Exception:
                    pass


def find_default_dataset_path(start: Path | None = None) -> Path | None:
    """Find the user's rag_eval_dataset file near the app/repository root."""

    candidates: list[Path] = []
    env_path = os.environ.get("RAG_EVAL_DATASET_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    root = (start or Path.cwd()).resolve()
    names = ["rag_eval_dataset", "rag_eval_dataset.json", "rag_eval_dataset.jsonl"]
    for parent in [root, *root.parents]:
        for name in names:
            candidates.append(parent / name)

    for path in candidates:
        if path.is_file():
            return path
    return None



def load_eval_dataset(path: str | Path) -> list[dict[str, str]]:
    """Load JSON/JSONL dataset and normalize fields used by RAGAS."""

    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.suffix.lower() == ".jsonl":
        raw = [json.loads(line) for line in dataset_path.read_text().splitlines() if line]
    else:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))

    if isinstance(raw, dict):
        raw = raw.get("data") or raw.get("samples") or raw.get("questions") or []
    if not isinstance(raw, list):
        raise ValueError("Dataset must be a list or a dict with data/samples/questions.")

    normalized: list[dict[str, str]] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item #{idx} is not an object.")

        question = item.get("question") or item.get("user_input") or item.get("query")
        reference = (
            item.get("ground_truth")
            or item.get("reference")
            or item.get("answer")
            or item.get("expected_answer")
        )
        source_file = item.get("source_file") or item.get("file") or item.get("document")

        if not question or not reference:
            raise ValueError(
                f"Dataset item #{idx} must contain question and ground_truth/reference."
            )

        normalized.append(
            {
                "id": str(item.get("id") or idx),
                "question": str(question),
                "reference": str(reference),
                "source_file": str(source_file or ""),
            }
        )

    return normalized



def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        score = float(value)
        if not math.isfinite(score):
            return None
        return score
    except (TypeError, ValueError):
        return None



def _doc_source_name(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    for key in ("file_name", "filename", "source", "Source", "document_name"):
        if metadata.get(key):
            return str(metadata[key])
    return ""



def _doc_score(doc: Any) -> float | None:
    metadata = getattr(doc, "metadata", {}) or {}
    retrieval_metadata = getattr(doc, "retrieval_metadata", {}) or {}
    score_keys = (
        "llm_trulens_score",
        "llm_reranking_score",
        "reranking_score",
        "retrieval_score",
        "vector_score",
        "vectorstore_score",
        "similarity_score",
        "similarity",
        "_score",
        "score",
    )
    for key in score_keys:
        score = _safe_float(metadata.get(key))
        if score is not None:
            return round(score, 4)
    for key in score_keys:
        score = _safe_float(retrieval_metadata.get(key))
        if score is not None:
            return round(score, 4)
    score = _safe_float(getattr(doc, "score", None))
    if score == -1.0:
        return None
    return round(score, 4) if score is not None else None


def _university_chunk_stats(docs: list[Any]) -> dict[str, Any]:
    """Summarize whether retrieved docs look like UniversityPDFChunker output."""

    child_count = 0
    parent_count = 0
    structural_metadata_count = 0
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        if metadata.get("index_role") == "child":
            child_count += 1
        if metadata.get("index_role") == "parent":
            parent_count += 1
        if (
            metadata.get("chunk_id")
            or metadata.get("section_id")
            or metadata.get("module_title")
            or metadata.get("doc_family")
        ):
            structural_metadata_count += 1

    return {
        "university_child_chunks": child_count,
        "university_parent_chunks": parent_count,
        "structural_metadata_docs": structural_metadata_count,
        "looks_like_university_structural_chunks": bool(
            docs and (child_count > 0 or structural_metadata_count > 0)
        ),
    }



def _ensure_simple_reasoning_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Use Simple QA for deterministic single-turn RAG evaluation."""

    eval_settings = deepcopy(settings)
    if "simple" in reasonings:
        eval_settings["reasoning.use"] = "simple"

    # Disable expensive UI-only artifacts; RAGAS evaluates answer + contexts instead.
    for key, value in {
        "reasoning.options.simple.highlight_citation": "off",
        "reasoning.options.simple.create_mindmap": False,
        "reasoning.options.simple.create_citation_viz": False,
    }.items():
        if key in eval_settings:
            eval_settings[key] = value
    return eval_settings



def _find_source_ids(app: Any, source_file: str, user_id: Any) -> list[tuple[Any, str, str]]:
    """Return (index, source_id, source_name) tuples matching the dataset file name."""

    if not source_file:
        return []

    requested = Path(source_file).name.lower()
    matches: list[tuple[Any, str, str]] = []

    with Session(engine) as session:
        for index in app.index_manager.indices:
            Source = getattr(index, "_resources", {}).get("Source")
            if Source is None:
                continue

            statement = select(Source)
            if index.config.get("private", False):
                statement = statement.where(Source.user == user_id)

            # Prefer exact normalized basename match; fallback to contains match for
            # uploads where Kotaemon preserved a prefix/suffix around the PDF name.
            rows = session.execute(statement).scalars().all()
            exact: list[Any] = []
            fuzzy: list[Any] = []
            for row in rows:
                row_name = str(getattr(row, "name", ""))
                row_base = Path(row_name).name.lower()
                if row_base == requested:
                    exact.append(row)
                elif requested in row_base or row_base in requested:
                    fuzzy.append(row)

            for row in exact or fuzzy:
                matches.append((index, str(row.id), str(row.name)))

    return matches



def _build_retrievers(app: Any, settings: dict[str, Any], user_id: Any, source_file: str):
    source_matches = _find_source_ids(app, source_file, user_id)
    if not source_matches:
        raise ValueError(f"Source file is not indexed or not visible: {source_file}")

    retrievers = []
    used_sources: list[str] = []
    for index, source_id, source_name in source_matches:
        if getattr(index, "_selector_ui", None) is None:
            index.get_selector_component_ui()
        retrievers.extend(
            index.get_retriever_pipelines(settings, user_id, ["select", [source_id], user_id])
        )
        used_sources.append(f"{index.name}: {source_name}")

    if not retrievers:
        raise ValueError(f"No retriever pipelines available for {source_file}")

    return retrievers, used_sources



def _answer_with_pipeline(
    app: Any,
    settings: dict[str, Any],
    user_id: Any,
    sample: dict[str, str],
    run_dir: Path | None = None,
) -> dict[str, Any]:
    question = sample["question"]
    source_file = sample.get("source_file", "")
    started = time.time()

    retrievers, used_sources = _build_retrievers(app, settings, user_id, source_file)
    reasoning_id = settings.get("reasoning.use")
    if reasoning_id not in reasonings:
        reasoning_id = "simple" if "simple" in reasonings else next(iter(reasonings))
    pipeline = reasonings[reasoning_id].get_pipeline(
        settings,
        {"app": {}, "pipeline": {}},
        retrievers,
    )

    docs, _ = pipeline.retrieve(question, [])
    if hasattr(pipeline, "configure_evidence_budget"):
        pipeline.configure_evidence_budget(question, [])
    evidence_mode, evidence, images = pipeline.evidence_pipeline.run(docs).content
    if hasattr(pipeline, "finalize_evidence_budget"):
        budget_debug = pipeline.finalize_evidence_budget(
            question,
            [],
            evidence,
            evidence_mode,
        )
        if budget_debug.get("completion_budget", 0) < 128 and evidence:
            retry_budget = max(
                256,
                int(pipeline.evidence_pipeline.max_context_length)
                - (128 - int(budget_debug.get("completion_budget") or 0))
                - 64,
            )
            rag_log(
                "evaluation.context_budget.retry_truncate",
                id=sample["id"],
                previous_context_budget=pipeline.evidence_pipeline.max_context_length,
                retry_context_budget=retry_budget,
                completion_budget=budget_debug.get("completion_budget"),
            )
            pipeline.evidence_pipeline.max_context_length = retry_budget
            evidence_mode, evidence, images = pipeline.evidence_pipeline.run(docs).content
            budget_debug = pipeline.finalize_evidence_budget(
                question,
                [],
                evidence,
                evidence_mode,
            )
            if budget_debug.get("completion_budget", 0) < 128:
                raise RuntimeError(
                    "LLM prompt would exceed the effective context window: "
                    f"completion_budget={budget_debug.get('completion_budget')}, "
                    f"effective_context_window={budget_debug.get('effective_context_window')}, "
                    f"prompt_tokens={budget_debug.get('prompt_tokens_after_truncation')}"
                )

    answer_chunks: list[str] = []
    raw_stream: list[dict[str, Any]] = []
    final_answer_doc: Document | None = None
    stream = pipeline.answering_pipeline.stream(
        question=question,
        history=[],
        evidence=evidence,
        evidence_mode=evidence_mode,
        images=images,
        conv_id=f"rag-eval-{sample['id']}",
    )
    while True:
        try:
            response = next(stream)
        except StopIteration as stop:
            if isinstance(stop.value, Document):
                final_answer_doc = stop.value
            break
        if isinstance(response, Document):
            raw_stream.append(
                {
                    "channel": response.channel,
                    "content": response.content,
                    "text": response.text,
                }
            )
            if response.channel == "chat" and response.content is not None:
                answer_chunks.append(str(response.content))
            elif response.channel == "chat":
                answer_chunks = []

    raw_answer = (
        final_answer_doc.text
        if final_answer_doc is not None
        else "".join(answer_chunks)
    )
    answer = strip_think_tag(raw_answer).strip()
    if raw_answer and not answer:
        if run_dir is not None:
            _append_jsonl(
                run_dir / "errors.jsonl",
                {
                    "id": sample["id"],
                    "stage": "answer_postprocess",
                    "raw_answer": raw_answer,
                    "postprocessed_answer": answer,
                },
            )
    if not raw_answer and run_dir is not None:
        rag_log(
            "evaluation.empty_answer",
            id=sample["id"],
            question=question,
            source_file=source_file,
            context_chars=len(evidence or ""),
            context_debug=getattr(pipeline.evidence_pipeline, "last_debug", {}),
            raw_stream=raw_stream,
        )
        _append_jsonl(
            run_dir / "errors.jsonl",
            {
                "id": sample["id"],
                "stage": "generation",
                "error": "model returned empty raw text",
                "context_chars": len(evidence or ""),
                "context_debug": getattr(pipeline.evidence_pipeline, "last_debug", {}),
            },
        )

    contexts = [getattr(doc, "text", "") or getattr(doc, "content", "") or "" for doc in docs]
    top_doc = docs[0] if docs else None
    top_score = None
    for doc in docs:
        top_score = _doc_score(doc)
        if top_score is not None:
            break
    if top_score is None and docs:
        top_score = 0.0

    debug_record = {
        "id": sample["id"],
        "question": question,
        "source_file": source_file,
        "indexed_source": used_sources,
        "retrievers": [getattr(retriever, "last_debug", {}) for retriever in retrievers],
        "context_debug": getattr(pipeline.evidence_pipeline, "last_debug", {}),
        "university_chunk_stats": _university_chunk_stats(docs),
        "final_prompt_context_char_length": len(evidence or ""),
        "model_answer_raw_text": raw_answer,
        "raw_stream": raw_stream,
    }
    if run_dir is not None:
        _append_jsonl(run_dir / "retrieval_debug.jsonl", debug_record)

    return {
        "id": sample["id"],
        "question": question,
        "reference": sample["reference"],
        "source_file": source_file,
        "indexed_source": "; ".join(used_sources),
        "answer": answer,
        "contexts": contexts,
        "context_count": len(contexts),
        "top_context_preview": (contexts[0][:500] if contexts else ""),
        "top_source": _doc_source_name(top_doc) if top_doc is not None else "",
        "top_score": top_score,
        "latency_sec": round(time.time() - started, 2),
        "status": "ok",
        "error": "",
    }



def _ragas_metrics() -> list[Any]:
    """Return a metric set compatible with recent and older RAGAS releases."""

    try:
        import ragas.metrics as metrics_module  # type: ignore

        class_names = [
            "LLMContextPrecisionWithReference",
            "LLMContextRecall",
            "Faithfulness",
            "ResponseRelevancy",
            "FactualCorrectness",
        ]
        metrics = [
            getattr(metrics_module, class_name)()
            for class_name in class_names
            if hasattr(metrics_module, class_name)
        ]
        if metrics:
            return metrics
    except Exception:
        pass

    from ragas.metrics import (  # type: ignore
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    return [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    ]


def _to_langchain_llm(settings: dict[str, Any]) -> tuple[Any, str]:
    """Resolve the active Kotaemon LLM as a LangChain model for RAGAS."""

    from ktem.llms.manager import llms

    reasoning_id = settings.get("reasoning.use") or "simple"
    configured_name = (
        settings.get(f"reasoning.options.{reasoning_id}.llm")
        or settings.get("reasoning.options.simple.llm")
        or ""
    )
    llm_name = configured_name or llms.get_default_name()
    llm = llms.get(llm_name, None) or llms.get_default()
    if not configured_name:
        llm_name = llms.get_default_name()

    if not hasattr(llm, "to_langchain_format"):
        langchain_llm = _openai_compatible_chat_to_langchain(llm)
        if langchain_llm is not None:
            return langchain_llm, llm_name
        raise RuntimeError(
            f"Kotaemon LLM `{llm_name}` cannot be passed to RAGAS. "
            "Use a local LangChain-compatible LLM such as Ollama or LlamaCpp."
        )

    try:
        return llm.to_langchain_format(), llm_name
    except NotImplementedError:
        langchain_llm = _openai_compatible_chat_to_langchain(llm)
        if langchain_llm is not None:
            return langchain_llm, llm_name
        raise RuntimeError(
            f"Kotaemon LLM `{llm_name}` exposes no LangChain adapter for RAGAS. "
            "Use a LangChain-compatible LLM or an OpenAI-compatible local endpoint."
        )


def _openai_compatible_chat_to_langchain(llm: Any) -> Any | None:
    """Adapt Kotaemon's OpenAI-compatible chat client to LangChain for RAGAS.

    Kotaemon's own ``ChatOpenAI`` works for the app and for local Ollama
    OpenAI-compatible endpoints, but it inherits ``to_langchain_format`` from the
    abstract base where it raises ``NotImplementedError``. RAGAS requires a
    LangChain model, so rebuild an equivalent ``langchain_openai.ChatOpenAI``
    when the active Kotaemon LLM has the OpenAI-compatible chat shape.
    """

    if not all(
        hasattr(llm, attr)
        for attr in ("api_key", "model", "prepare_client", "openai_response")
    ):
        return None

    from langchain_openai import ChatOpenAI as LangChainChatOpenAI

    params: dict[str, Any] = {
        "api_key": getattr(llm, "api_key"),
        "model": getattr(llm, "model"),
    }
    optional_attrs = {
        "base_url": "base_url",
        "organization": "organization",
        "timeout": "timeout",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "n": "n",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "logprobs": "logprobs",
        "top_logprobs": "top_logprobs",
        "logit_bias": "logit_bias",
        "top_p": "top_p",
    }
    for source_attr, target_attr in optional_attrs.items():
        value = getattr(llm, source_attr, None)
        if value is not None:
            params[target_attr] = value

    max_retries = getattr(llm, "max_retries", None)
    if max_retries is not None:
        params["max_retries"] = max_retries

    # Avoid token-counting failures for local model names such as ``qwen3:8b``.
    if params.get("base_url"):
        params["tiktoken_model_name"] = "gpt-3.5-turbo"

    return LangChainChatOpenAI(**params)


def _kotaemon_embedding_adapter(embedding_model: Any) -> Any:
    """Wrap any Kotaemon embedding model in LangChain's Embeddings interface."""

    try:
        from langchain_core.embeddings import Embeddings
    except Exception:
        from langchain.embeddings.base import Embeddings  # type: ignore

    class KotaemonEmbeddingsAdapter(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            embedded_docs = embedding_model.run(texts)
            return [list(doc.embedding) for doc in embedded_docs]

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    return KotaemonEmbeddingsAdapter()


def _to_langchain_embeddings(settings: dict[str, Any]) -> tuple[Any, str]:
    """Resolve the active Kotaemon embedding model for RAGAS."""

    from ktem.embeddings.manager import embedding_models_manager

    embedding_name = ""
    for key, value in settings.items():
        if key.startswith("index.options.") and key.endswith(".embedding") and value:
            embedding_name = str(value)
            break

    if not embedding_name or embedding_name == "default":
        embedding_name = embedding_models_manager.get_default_name()

    embedding_model = (
        embedding_models_manager.get(embedding_name, None)
        or embedding_models_manager.get("default", None)
        or embedding_models_manager.get_default()
    )

    if hasattr(embedding_model, "to_langchain_format"):
        return embedding_model.to_langchain_format(), embedding_name

    # LangChain-based Kotaemon embeddings keep the underlying object in `_obj`.
    raw_obj = getattr(embedding_model, "_obj", None)
    if raw_obj is not None and all(
        hasattr(raw_obj, method) for method in ("embed_documents", "embed_query")
    ):
        return raw_obj, embedding_name

    return _kotaemon_embedding_adapter(embedding_model), embedding_name


def _wrap_for_ragas(
    llm: Any, embeddings: Any, run_config: Any | None = None
) -> tuple[Any, Any]:
    """Use RAGAS' official wrappers when the installed version exposes them."""

    try:
        from ragas.llms import LangchainLLMWrapper  # type: ignore

        try:
            llm = LangchainLLMWrapper(llm, run_config=run_config)
        except TypeError:
            llm = LangchainLLMWrapper(llm)
    except Exception:
        # Recent RAGAS versions can also auto-wrap LangChain models in evaluate().
        pass

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore

        try:
            embeddings = LangchainEmbeddingsWrapper(embeddings, run_config=run_config)
        except TypeError:
            embeddings = LangchainEmbeddingsWrapper(embeddings)
    except Exception:
        pass

    return llm, embeddings


def _local_ragas_evaluator_models(
    settings: dict[str, Any], run_config: Any | None = None
) -> RagasEvaluatorModels:
    """Build the local evaluator LLM/embeddings explicitly for RAGAS.

    If these are omitted, RAGAS creates its default evaluator stack, which is
    OpenAI-backed and requires OPENAI_API_KEY. The app must stay local, so every
    RAGAS evaluate() call receives Kotaemon's configured local models.
    """

    llm, llm_name = _to_langchain_llm(settings)
    _apply_model_timeout(llm, run_config)
    raw_embeddings, embeddings_name = _to_langchain_embeddings(settings)
    llm, embeddings = _wrap_for_ragas(llm, raw_embeddings, run_config)
    _apply_model_timeout(llm, run_config)
    return RagasEvaluatorModels(
        llm=llm,
        embeddings=embeddings,
        raw_embeddings=raw_embeddings,
        llm_name=llm_name,
        embeddings_name=embeddings_name,
        run_config=run_config,
        notes=[
            "RAGAS evaluator uses Kotaemon local models: "
            f"llm={llm_name}, embeddings={embeddings_name}.",
            _run_config_note(run_config),
        ],
    )


def _numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    ignored = {"id", "source_file", "latency_sec", "context_count", "top_score"}
    return [
        column
        for column in df.columns
        if column not in ignored
        and pd.to_numeric(df[column], errors="coerce").notna().any()
    ]


def _looks_all_nan(df: pd.DataFrame) -> bool:
    if df.empty:
        return False

    ignored = {"id", "source_file", "question", "answer", "contexts", "ground_truth"}
    candidate_columns = [column for column in df.columns if column not in ignored]
    if not candidate_columns:
        return True

    for column in candidate_columns:
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        if numeric_values.notna().any():
            return False
    return True


def _is_finite_score(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _token_set(text: str) -> set[str]:
    import re

    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
    }
    return {
        token
        for token in re.findall(r"[\wÄÖÜäöüß]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _embed_documents(embeddings: Any, texts: list[str]) -> list[list[float]]:
    # LangChain embeddings
    if hasattr(embeddings, "embed_documents"):
        return [list(vector) for vector in embeddings.embed_documents(texts)]

    # RAGAS LangchainEmbeddingsWrapper commonly exposes the wrapped object.
    for attr in ("embeddings", "langchain_embeddings"):
        wrapped = getattr(embeddings, attr, None)
        if wrapped is not None and hasattr(wrapped, "embed_documents"):
            return [list(vector) for vector in wrapped.embed_documents(texts)]

    raise RuntimeError("Could not embed texts with the configured local embeddings.")


def _local_defined_scores(
    rows: list[dict[str, Any]], embeddings: Any
) -> tuple[pd.DataFrame, list[str]]:
    """Always-defined local metrics used when LLM-judge RAGAS scores are NaN.

    These metrics keep the evaluation useful in fully-local mode. Semantic similarity
    and non-LLM string similarity mirror official RAGAS metric families, while keyword
    recalls make retrieval/answer coverage visible without an LLM judge.
    """

    records: list[dict[str, Any]] = []
    notes: list[str] = []
    for row in rows:
        reference = row.get("reference", "")
        answer = row.get("answer", "")
        contexts = " ".join(row.get("contexts") or [])

        try:
            ref_vec, answer_vec = _embed_documents(embeddings, [reference, answer])
            semantic_similarity = round(_cosine_similarity(ref_vec, answer_vec), 4)
        except Exception as exc:
            semantic_similarity = None
            notes.append(f"{row.get('id')}: semantic fallback failed: {exc}")

        string_similarity = round(SequenceMatcher(None, reference, answer).ratio(), 4)
        reference_tokens = _token_set(reference)
        answer_tokens = _token_set(answer)
        context_tokens = _token_set(contexts)
        if reference_tokens:
            answer_keyword_recall = round(
                len(reference_tokens & answer_tokens) / len(reference_tokens), 4
            )
            context_keyword_recall = round(
                len(reference_tokens & context_tokens) / len(reference_tokens), 4
            )
        else:
            answer_keyword_recall = 0.0
            context_keyword_recall = 0.0

        score_values = [
            value
            for value in (
                semantic_similarity,
                string_similarity,
                answer_keyword_recall,
                context_keyword_recall,
            )
            if _is_finite_score(value)
        ]

        records.append(
            {
                "id": row["id"],
                "source_file": row["source_file"],
                "semantic_similarity": semantic_similarity,
                "non_llm_string_similarity": string_similarity,
                "answer_keyword_recall": answer_keyword_recall,
                "context_keyword_recall": context_keyword_recall,
                "ragas_local_score": round(sum(score_values) / len(score_values), 4)
                if score_values
                else None,
            }
        )

    notes.append(
        "LLM-judge RAGAS metrics were undefined, so the table shows local "
        "always-defined RAGAS-style fallback metrics instead."
    )
    return pd.DataFrame(records), notes


def _evaluate_with_local_models(
    dataset: Any, metrics: list[Any], evaluator: RagasEvaluatorModels
) -> Any:
    """Call ragas.evaluate across old/new RAGAS signatures without losing locality."""

    from ragas import evaluate  # type: ignore

    base_kwargs = {
        "dataset": dataset,
        "metrics": metrics,
        "llm": evaluator.llm,
        "embeddings": evaluator.embeddings,
    }
    optional_kwargs = {
        "raise_exceptions": False,
        "show_progress": False,
    }
    if evaluator.run_config is not None:
        optional_kwargs["run_config"] = evaluator.run_config

    while True:
        try:
            return evaluate(**base_kwargs, **optional_kwargs)
        except TypeError as exc:
            message = str(exc)
            unsupported = [
                key
                for key in ("raise_exceptions", "show_progress", "run_config")
                if key in message and key in optional_kwargs
            ]
            if not unsupported:
                raise
            for key in unsupported:
                optional_kwargs.pop(key, None)



def _run_ragas(
    rows: list[dict[str, Any]], settings: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """Execute RAGAS with EvaluationDataset first, then HF Dataset fallback."""

    valid_rows = [row for row in rows if row.get("status") == "ok"]
    if not valid_rows:
        return pd.DataFrame(), []

    run_config = _ragas_run_config()
    evaluator = _local_ragas_evaluator_models(settings, run_config)
    metrics = _ragas_metrics()

    try:
        from ragas import EvaluationDataset  # type: ignore

        evaluation_dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": row["question"],
                    "response": row["answer"],
                    "retrieved_contexts": row["contexts"],
                    "reference": row["reference"],
                }
                for row in valid_rows
            ]
        )
        result = _evaluate_with_local_models(evaluation_dataset, metrics, evaluator)
    except Exception:
        from datasets import Dataset  # type: ignore

        evaluation_dataset = Dataset.from_dict(
            {
                "question": [row["question"] for row in valid_rows],
                "answer": [row["answer"] for row in valid_rows],
                "contexts": [row["contexts"] for row in valid_rows],
                "ground_truth": [row["reference"] for row in valid_rows],
            }
        )
        result = _evaluate_with_local_models(evaluation_dataset, metrics, evaluator)

    if hasattr(result, "to_pandas"):
        ragas_df = result.to_pandas()
    else:
        ragas_df = pd.DataFrame(result)

    for column in ("id", "source_file"):
        if column not in ragas_df.columns:
            ragas_df.insert(0, column, [row[column] for row in valid_rows])

    notes = evaluator.notes
    if _looks_all_nan(ragas_df):
        ragas_df, fallback_notes = _local_defined_scores(
            valid_rows, evaluator.raw_embeddings
        )
        notes.extend(fallback_notes)
    return ragas_df, notes




def _summarize(samples_df: pd.DataFrame, ragas_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "samples_total": int(len(samples_df)),
        "samples_ok": int((samples_df["status"] == "ok").sum()) if len(samples_df) else 0,
        "samples_failed": int((samples_df["status"] != "ok").sum()) if len(samples_df) else 0,
        "avg_latency_sec": round(float(samples_df["latency_sec"].mean()), 2)
        if "latency_sec" in samples_df and len(samples_df)
        else 0,
    }

    for column in ragas_df.columns:
        if column in {"id", "source_file"}:
            continue
        numeric_values = pd.to_numeric(ragas_df[column], errors="coerce")
        if numeric_values.notna().any():
            value = numeric_values.mean(skipna=True)
            if pd.notna(value):
                summary[column] = round(float(value), 4)

    return summary



def run_evaluation(
    app: Any,
    settings: dict[str, Any],
    user_id: Any,
    dataset_path: str | Path,
    question_limit: int,
    run_ragas_metrics: bool = True,
    progress: ProgressFn | None = None,
) -> EvalRunResult:
    """Run Kotaemon RAG over a dataset subset and optionally score it with RAGAS."""

    samples = load_eval_dataset(dataset_path)
    limit = max(1, min(int(question_limit), len(samples)))
    samples = samples[:limit]

    eval_settings = _ensure_simple_reasoning_settings(settings)
    run_dir = _new_eval_run_dir()
    (run_dir / "settings.json").write_text(
        json.dumps(_json_safe_settings(eval_settings), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for idx, sample in enumerate(samples, start=1):
        if progress:
            progress(idx - 1, limit, f"Question {idx}/{limit}: {sample['id']}")
        try:
            row = _answer_with_pipeline(app, eval_settings, user_id, sample, run_dir)
            rows.append(row)
            _append_jsonl(run_dir / "answers.jsonl", row)
        except Exception as exc:  # keep the run useful even when one sample fails
            warnings.append(f"{sample['id']}: {exc}")
            error_row = {
                    "id": sample["id"],
                    "question": sample["question"],
                    "reference": sample["reference"],
                    "source_file": sample.get("source_file", ""),
                    "indexed_source": "",
                    "answer": "",
                    "contexts": [],
                    "context_count": 0,
                    "top_context_preview": "",
                    "top_source": "",
                    "top_score": None,
                    "latency_sec": 0,
                    "status": "error",
                    "error": f"{exc}\n{traceback.format_exc(limit=2)}",
                }
            rows.append(error_row)
            _append_jsonl(run_dir / "answers.jsonl", error_row)
            _append_jsonl(
                run_dir / "errors.jsonl",
                {
                    "id": sample["id"],
                    "stage": "sample",
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                },
            )

    if progress:
        progress(limit, limit, "RAG answers collected")

    ragas_df = pd.DataFrame()
    if run_ragas_metrics:
        try:
            ragas_df, ragas_notes = _run_ragas(rows, eval_settings)
            warnings.extend(ragas_notes)
        except ModuleNotFoundError as exc:
            warnings.append(
                f"RAGAS dependencies are not installed ({exc.name}). "
                "Install/update dependencies with `pip install -r requirements.txt`."
            )
        except Exception as exc:
            warnings.append(f"RAGAS scoring failed: {exc}")

    samples_df = pd.DataFrame(rows)
    samples_df.to_csv(run_dir / "answers.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    if not ragas_df.empty:
        ragas_df.to_csv(run_dir / "ragas_scores.csv", index=False)
    summary = _summarize(samples_df, ragas_df)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if warnings:
        (run_dir / "warnings.txt").write_text("\n".join(warnings), encoding="utf-8")
    return EvalRunResult(
        samples=samples_df,
        ragas_scores=ragas_df,
        summary=summary,
        warnings=warnings,
        run_dir=str(run_dir),
    )
