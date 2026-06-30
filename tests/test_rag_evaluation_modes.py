from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from ktem.evaluation import build_evaluation_export_frame
from ktem.evaluation import ragas_eval


def _sample_row() -> dict:
    return {
        "id": "q1",
        "question": "What is tested?",
        "reference": "Answer-only evaluation.",
        "source_file": "guide.pdf",
        "indexed_source": "Files: guide.pdf",
        "answer": "Answer-only evaluation is tested.",
        "contexts": ["A retrieved context."],
        "context_count": 1,
        "top_context_preview": "A retrieved context.",
        "top_source": "guide.pdf",
        "top_score": 0.9,
        "latency_sec": 0.3,
        "retrieval_latency_sec": 0.1,
        "generation_latency_sec": 0.2,
        "status": "ok",
        "error": "",
    }


def test_answer_only_run_skips_ragas_and_returns_results(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")
    sample = {
        "id": "q1",
        "question": "What is tested?",
        "reference": "Answer-only evaluation.",
        "source_file": "guide.pdf",
    }
    retrieval = {
        "id": "q1",
        "source_file": "guide.pdf",
        "retrieval_scope": "all",
        "answer_chunk_included": True,
    }

    monkeypatch.setattr(ragas_eval, "load_eval_dataset", lambda _: [sample])
    monkeypatch.setattr(ragas_eval, "_ensure_simple_reasoning_settings", lambda x: x)
    monkeypatch.setattr(
        ragas_eval,
        "_resolve_evaluation_models",
        lambda _: (SimpleNamespace(), SimpleNamespace()),
    )
    monkeypatch.setattr(ragas_eval, "_env_bool", lambda _name, default=False: False)
    monkeypatch.setattr(
        ragas_eval,
        "_answer_with_pipeline",
        lambda *_args, **_kwargs: (_sample_row(), [{"id": "q1"}], retrieval),
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("_run_ragas must not run in answer-only mode")

    monkeypatch.setattr(ragas_eval, "_run_ragas", fail_if_called)

    result = ragas_eval.run_evaluation(
        app=SimpleNamespace(),
        settings={},
        user_id="user",
        dataset_path=dataset,
        question_limit=1,
        run_ragas_metrics=False,
    )

    assert result.samples.loc[0, "answer"] == "Answer-only evaluation is tested."
    assert result.ragas_scores.empty
    assert not result.retrieval_metrics.empty
    assert not result.retrieval_candidates.empty
    assert result.summary["evaluation_mode"] == "answers_only"
    assert result.runtime_config["run_ragas_metrics"] is False


def test_export_is_complete_when_ragas_scores_are_empty(tmp_path):
    result = SimpleNamespace(
        samples=pd.DataFrame([_sample_row()]),
        ragas_scores=pd.DataFrame(),
        retrieval_metrics=pd.DataFrame(
            [{"id": "q1", "retrieval_scope": "all", "reciprocal_rank": 1.0}]
        ),
        runtime_config={"retrieval_scope": "all"},
    )

    export = build_evaluation_export_frame(result)

    required = {
        "id",
        "question",
        "reference",
        "source_file",
        "indexed_source",
        "retrieval_scope",
        "answer",
        "contexts",
        "context_count",
        "top_context_preview",
        "top_source",
        "top_score",
        "latency_sec",
        "retrieval_latency_sec",
        "generation_latency_sec",
        "status",
        "error",
    }
    assert required <= set(export.columns)
    assert export.loc[0, "contexts"] == '["A retrieved context."]'
    assert export.loc[0, "retrieval_scope"] == "all"
    assert export.loc[0, "reciprocal_rank"] == 1.0

    output = tmp_path / "answers-only.csv"
    export.to_csv(output, index=False)
    saved = pd.read_csv(output)
    assert required <= set(saved.columns)
    assert saved.loc[0, "answer"] == "Answer-only evaluation is tested."

    result.ragas_scores = pd.DataFrame([{"id": "q1", "faithfulness": 0.8}])
    metrics_export = build_evaluation_export_frame(result)
    assert metrics_export.loc[0, "faithfulness"] == 0.8
