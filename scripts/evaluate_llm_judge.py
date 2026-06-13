import argparse
import csv
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path


ANSWER_FIELDS = [
    "generated_answer",
    "answer",
    "response",
    "model_answer",
    "prediction",
    "llm_answer",
]

GROUND_TRUTH_FIELDS = [
    "ground_truth",
    "reference",
    "expected_answer",
]


def pick_field(item, candidates):
    for key in candidates:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return key, value.strip()
    return None, ""


def call_ollama_chat(ollama_url, model, prompt, temperature=0.0):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator for a university RAG chatbot. "
                    "Return only valid JSON. Do not add markdown."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 4000,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ollama_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    result = json.loads(raw)
    return result["message"]["content"]


def extract_json(text):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON from judge response: {text[:500]}")


def make_prompt(question, ground_truth, generated_answer, source_file):
    return f"""
Evaluate the generated answer for a university advising RAG chatbot.

Question:
{question}

Ground truth / reference answer:
{ground_truth}

Generated answer:
{generated_answer}

Expected source file:
{source_file}

Evaluation rules:
- Score 5: fully correct, complete, no hallucination.
- Score 4: mostly correct, only minor missing detail.
- Score 3: partially correct, important detail missing.
- Score 2: weak answer, mostly incomplete or vague.
- Score 1: mostly wrong.
- Score 0: completely wrong, unsupported, or hallucinated.

Also judge:
- correctness: correct, partially_correct, incorrect
- groundedness: grounded, partially_grounded, ungrounded
- hallucination: yes or no

Return only this JSON schema:
{{
  "score": 0,
  "correctness": "correct | partially_correct | incorrect",
  "groundedness": "grounded | partially_grounded | ungrounded",
  "hallucination": "yes | no",
  "reason": "short explanation"
}}
""".strip()


def evaluate_item(item, ollama_url, model):
    question = item.get("question", "").strip()
    source_file = item.get("source_file", "").strip()

    _, ground_truth = pick_field(item, GROUND_TRUTH_FIELDS)
    answer_key, generated_answer = pick_field(item, ANSWER_FIELDS)

    if not question or not ground_truth:
        raise ValueError(f"Missing question or ground truth in item: {item.get('id')}")

    if not generated_answer:
        raise ValueError(
            "No generated answer found. Expected one of these fields: "
            + ", ".join(ANSWER_FIELDS)
        )

    prompt = make_prompt(question, ground_truth, generated_answer, source_file)

    start = time.time()
    judge_text = call_ollama_chat(ollama_url, model, prompt)
    duration = round(time.time() - start, 2)

    judge = extract_json(judge_text)

    return {
        "id": item.get("id", ""),
        "question": question,
        "source_file": source_file,
        "answer_field": answer_key,
        "score": judge.get("score"),
        "correctness": judge.get("correctness"),
        "groundedness": judge.get("groundedness"),
        "hallucination": judge.get("hallucination"),
        "reason": judge.get("reason"),
        "judge_duration_seconds": duration,
        "generated_answer": generated_answer,
        "ground_truth": ground_truth,
    }


def load_json_or_jsonl(path):
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    
    if text.startswith("["):
        return json.loads(text)

    items = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def write_csv(path, rows):
    if not rows:
        return

    fields = [
        "id",
        "score",
        "correctness",
        "groundedness",
        "hallucination",
        "reason",
        "judge_duration_seconds",
        "source_file",
        "question",
    ]

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    parser = argparse.ArgumentParser(
        description="Fast LLM-based evaluator for RAG answers without RAGAS."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSON/JSONL file containing question, ground_truth/reference, and generated answer.",
    )
    parser.add_argument(
        "--output-json",
        default="llm_judge_eval_results.json",
        help="Output JSON file.",
    )
    parser.add_argument(
        "--output-csv",
        default="llm_judge_eval_results.csv",
        help="Output CSV summary file.",
    )
    parser.add_argument(
        "--model",
        default="gemma4:e2b",
        help="Ollama judge model.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/chat",
        help="Ollama chat API URL.",
    )
    args = parser.parse_args()

    items = load_json_or_jsonl(args.input)
    results = []

    print(f"Loaded {len(items)} items")
    print(f"Judge model: {args.model}")

    for index, item in enumerate(items, start=1):
        item_id = item.get("id", f"item_{index}")
        print(f"[{index}/{len(items)}] Evaluating {item_id}...")

        try:
            result = evaluate_item(item, args.ollama_url, args.model)
            results.append(result)
            print(f"  score={result['score']} correctness={result['correctness']}")
        except Exception as exc:
            error_result = {
                "id": item_id,
                "question": item.get("question", ""),
                "score": None,
                "correctness": "error",
                "groundedness": "error",
                "hallucination": "unknown",
                "reason": str(exc),
                "source_file": item.get("source_file", ""),
            }
            results.append(error_result)
            print(f"  ERROR: {exc}")

    Path(args.output_json).write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(args.output_csv, results)

    valid_scores = [
        row["score"]
        for row in results
        if isinstance(row.get("score"), (int, float))
    ]

    print("\nDone.")
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV: {args.output_csv}")

    if valid_scores:
        avg = sum(valid_scores) / len(valid_scores)
        print(f"Average score: {avg:.2f}/5")


if __name__ == "__main__":
    main()