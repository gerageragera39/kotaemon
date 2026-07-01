#!/usr/bin/env python3
"""Run KURAGa RAG evaluation from the command line.

Example:
    PYTHONPATH=src python scripts/run_rag_eval.py \
        --dataset ../rag_eval_dataset.json --limit 5 --output-dir eval_results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ktem.evaluation import (  # noqa: E402
    build_evaluation_export_frame,
    find_default_dataset_path,
    run_evaluation,
)


def parse_args() -> argparse.Namespace:
    default_dataset = find_default_dataset_path(ROOT)
    parser = argparse.ArgumentParser(
        description="Evaluate KURAGa RAG with the Felix evaluator."
    )
    parser.add_argument(
        "--dataset",
        default=str(default_dataset or "rag_eval_dataset.json"),
        help="Path to rag_eval_dataset JSON/JSONL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of questions to evaluate.",
    )
    parser.add_argument(
        "--user-id",
        default="default",
        help="KURAGa user id for private indexes.",
    )
    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument(
        "--run-ragas",
        dest="run_ragas",
        action="store_true",
        help="Run the slower RAGAS/local quality metrics after answer generation.",
    )
    metric_group.add_argument(
        "--no-ragas",
        dest="run_ragas",
        action="store_false",
        help="Only collect answers/contexts/retrieval diagnostics (default).",
    )
    parser.set_defaults(run_ragas=False)
    parser.add_argument(
        "--scope",
        choices=["expected-source", "all"],
        default="all",
        help=(
            "Retrieve from each sample's source_file or from all visible documents."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory for CSV/JSON artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from ktem.main import App

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    app = App()
    settings = app.default_settings.flatten()

    def progress(done: int, total: int, desc: str):
        print(f"[{done}/{total}] {desc}", flush=True)

    result = run_evaluation(
        app=app,
        settings=settings,
        user_id=args.user_id,
        dataset_path=args.dataset,
        question_limit=args.limit,
        run_ragas_metrics=args.run_ragas,
        retrieval_scope=args.scope,
        progress=progress,
    )

    build_evaluation_export_frame(result).to_csv(
        output_dir / "rag_eval_samples.csv", index=False
    )
    result.ragas_scores.to_csv(output_dir / "ragas_scores.csv", index=False)
    result.retrieval_metrics.to_csv(
        output_dir / "retrieval_metrics.csv", index=False
    )
    result.retrieval_candidates.to_json(
        output_dir / "retrieval_candidates.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "runtime_config.json").write_text(
        json.dumps(result.runtime_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "failures.md").write_text(
        result.failure_report,
        encoding="utf-8",
    )
    if result.warnings:
        (output_dir / "warnings.txt").write_text("\n".join(result.warnings), encoding="utf-8")

    print(json.dumps(result.summary, indent=2, ensure_ascii=False))
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
