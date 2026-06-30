#!/usr/bin/env python3
"""Retrieval-only regression check for the university RAG evaluation dataset.

The script re-ingests ``dataset/testing_files`` with ``pdf_mode=university`` and
uses the same VectorIndexing/VectorRetrieval path as the app, but with a local
deterministic keyword embedding so no external model/API is required.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kotaemon.base import Document, DocumentWithEmbedding  # noqa: E402
from kotaemon.embeddings.base import BaseEmbeddings  # noqa: E402
from kotaemon.indices.ingests import DocumentIngestor  # noqa: E402
from kotaemon.indices.vectorindex import VectorIndexing, VectorRetrieval  # noqa: E402
from kotaemon.storages.docstores.in_memory import InMemoryDocumentStore  # noqa: E402


HARD_CASE_TERMS: dict[str, list[str]] = {
    "d3b_04": [
        "Accounting",
        "Taxation",
        "Controlling",
        "Finance",
        "Economics",
        "Marketing",
        "Organization",
        "Innovation",
        "Supply Chain",
        "Logistics",
    ],
    "d3b_06": [
        "Wirtschaftswissenschaftliche Fakultät",
        "Mathematisch-Geographische Fakultät",
    ],
    "elective_2026_01": [
        "Digitalization & Analytics",
        "Data Competence",
        "Application Competence",
        "Business Language and Management Skills",
        "Wirtschafts- und Unternehmensethik",
    ],
    "elective_2026_03": [
        "Data Competence",
        "Algorithmen und Datenstrukturen",
        "Rechnergestützte Statistik mit R",
        "Hands-on Machine Learning and Data Science",
    ],
    "elective_2026_07": [
        "Supply Chain Management",
        "Digital Seminar in Data Science",
        "Retail Management Fundamentals",
        "Systementwicklung",
        "Operations Analytics",
        "Supply Chain Analytics",
        "SCM Projektstudium",
        "Transportlogistik",
    ],
    "elective_2026_09": [
        "Sustainability in Business and Economics",
        "Ringvorlesung",
        "Nachhaltige Wirtschaft",
        "Innovating for Sustainability",
        "Sustainable Entrepreneurship",
        "Umweltökonomie",
        "Öffentliche Finanzen",
        "Sustainable Development",
        "Company Taxation",
    ],
    "elective_2026_10": [
        "Application Competence",
        "Steuerbilanzen und Rechtsformwahl",
        "Marketing and Management",
        "Operations Analytics",
        "Kapitalmarkttheorie",
    ],
}

KNOWN_WEAK_CASE_IDS = {
    "po_d3b_07",
    "study_desc_01",
    "study_desc_08",
    "apo_01",
    "apo_03",
    "mod_catalog_08",
    "mod_catalog_12",
}


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"(\w)-\s+(\w)", r"\1\2", value)
    value = re.sub(r"\b([A-Za-z])\s+(?=[A-Za-z]{2,})", r"\1", value)
    return (
        value.lower()
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
        .replace("&", " and ")
    )


def contains_term(text: str, term: str) -> bool:
    haystack = re.sub(r"[\s\-]+", " ", normalize(text))
    needle = re.sub(r"[\s\-]+", " ", normalize(term)).strip()
    if needle in haystack or needle.replace(" ", "") in haystack.replace(" ", ""):
        return True
    # Ground-truth phrases occasionally vary only by an article (for example
    # "eine Prüfung" versus the PDF's "die Prüfung"). Ignore those function
    # words so the script measures evidence retrieval rather than declension.
    articles = {
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "ein",
        "eine",
        "einer",
        "einem",
        "einen",
    }
    haystack_tokens = [token for token in haystack.split() if token not in articles]
    needle_tokens = [token for token in needle.split() if token not in articles]
    return " ".join(needle_tokens) in " ".join(haystack_tokens)


def default_terms(item: dict[str, Any]) -> list[str]:
    text = item.get("ground_truth") or ""
    terms = []
    stop_phrases = {
        "A",
        "The",
        "When",
        "Carrying",
        "Each",
        "It",
        "This",
        "Students",
    }
    for phrase in re.findall(r"\b(?:[A-ZÄÖÜ][\wÄÖÜäöüß&-]+(?:\s+|$)){1,5}", text):
        phrase = " ".join(phrase.split())
        if len(phrase) > 3 and phrase not in stop_phrases and phrase.lower() not in {"the", "a", "an"}:
            terms.append(phrase)
    for token in re.findall(r"\b\d+(?:[,.]\d+)?\s*(?:ECTS|percent|%)?\b", text):
        terms.append(token.strip())
        number = re.match(r"\d+(?:[,.]\d+)?", token.strip())
        if number:
            terms.append(number.group(0))
    return list(dict.fromkeys(terms))[:8]


class KeywordEmbedding(BaseEmbeddings):
    terms: list[str]

    def invoke(self, docs, *args, **kwargs):
        if not isinstance(docs, list):
            docs = [docs]
        out = []
        for doc in docs:
            text = doc.text if isinstance(doc, Document) else str(doc)
            norm = normalize(text)
            vector = [float(contains_term(norm, term)) for term in self.terms]
            # Add a small length-normalized lexical bucket to avoid all-zero ties.
            vector.append(min(1.0, len(set(re.findall(r"\w+", norm))) / 500.0))
            kwargs = {}
            if isinstance(doc, Document):
                kwargs["metadata"] = doc.metadata
            out.append(DocumentWithEmbedding(embedding=vector, **kwargs))
        return out


class MemoryVectorStore:
    def __init__(self):
        self.vectors: dict[str, list[float]] = {}

    def add(self, embeddings, ids):
        for embedding, doc_id in zip(embeddings, ids):
            self.vectors[doc_id] = list(embedding.embedding)

    def query(self, embedding, top_k=10, ids=None, doc_ids=None, **kwargs):
        allowed = set(ids or doc_ids or self.vectors)
        scored = []
        qnorm = math.sqrt(sum(float(v) * float(v) for v in embedding)) or 1.0
        for doc_id, vector in self.vectors.items():
            if doc_id not in allowed:
                continue
            dnorm = math.sqrt(sum(float(v) * float(v) for v in vector)) or 1.0
            score = sum(float(a) * float(b) for a, b in zip(embedding, vector)) / (qnorm * dnorm)
            scored.append((score, doc_id))
        scored.sort(reverse=True)
        scored = scored[:top_k]
        return [], [score for score, _ in scored], [doc_id for _, doc_id in scored]

    def delete(self, ids, **kwargs):
        for doc_id in ids:
            self.vectors.pop(doc_id, None)

    def drop(self):
        self.vectors = {}


class LexicalDocStore(InMemoryDocumentStore):
    def query(self, query: str, top_k: int = 10, doc_ids: list | None = None):
        allowed = set(doc_ids or self._store)
        qtokens = set(re.findall(r"[\wÄÖÜäöüß]+", normalize(query)))
        scored = []
        for doc_id, doc in self._store.items():
            if doc_id not in allowed or (doc.metadata or {}).get("index_role") == "parent":
                continue
            text = normalize((doc.text or "") + " " + json.dumps(doc.metadata or {}, ensure_ascii=False))
            dtokens = set(re.findall(r"[\wÄÖÜäöüß]+", text))
            overlap = len(qtokens & dtokens)
            phrase_bonus = 1 if normalize(query)[:80] in text else 0
            scored.append((overlap + phrase_bonus, doc_id, doc))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [doc for score, _, doc in scored if score > 0][:top_k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate university retrieval only.")
    parser.add_argument("--dataset", default=str(ROOT / "rag_eval_dataset.json"))
    parser.add_argument("--documents-dir", default=str(ROOT / "dataset" / "testing_files"))
    parser.add_argument("--output-dir", default=str(ROOT / "dataset" / ".cache" / "university_retrieval_eval"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-multiplier", type=int, default=20)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Evaluate only this case id (repeatable).",
    )
    parser.add_argument(
        "--known-weak-cases",
        action="store_true",
        help="Evaluate the seven historically partial cases only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    selected_ids = set(args.case_id)
    if args.known_weak_cases:
        selected_ids.update(KNOWN_WEAK_CASE_IDS)
    dataset = [
        item for item in all_dataset if not selected_ids or item["id"] in selected_ids
    ]
    missing_ids = selected_ids - {item["id"] for item in dataset}
    if missing_ids:
        raise ValueError(f"Unknown case ids: {sorted(missing_ids)}")
    pdfs = sorted(Path(args.documents_dir).expanduser().resolve().glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {args.documents_dir}")

    terms = sorted(
        {
            term
            for item in all_dataset
            for term in [item["source_file"], item["question"], item["ground_truth"], *HARD_CASE_TERMS.get(item["id"], [])]
            for term in re.findall(r"[\wÄÖÜäöüß&.-]+(?:\s+[\wÄÖÜäöüß&.-]+){0,5}", str(term))
            if len(term.strip()) > 2
        }
    )

    print(f"Re-indexing {len(pdfs)} PDFs from scratch with pdf_mode=university", flush=True)
    chunks = DocumentIngestor(pdf_mode="university").run(pdfs)
    child_chunks = [doc for doc in chunks if (doc.metadata or {}).get("index_role") == "child"]

    chunk_counts = Counter((doc.metadata or {}).get("source_file") or (doc.metadata or {}).get("file_name") for doc in child_chunks)
    doc_types = {}
    for doc in child_chunks:
        source = (doc.metadata or {}).get("source_file") or (doc.metadata or {}).get("file_name")
        doc_types[source] = (doc.metadata or {}).get("doc_type")

    docstore = LexicalDocStore()
    vectorstore = MemoryVectorStore()
    embedding = KeywordEmbedding(terms=terms)
    VectorIndexing(vector_store=vectorstore, doc_store=docstore, embedding=embedding).run(chunks)
    retrieval = VectorRetrieval(
        vector_store=vectorstore,
        doc_store=docstore,
        embedding=embedding,
        retrieval_mode="hybrid",
        first_round_top_k_mult=args.candidate_multiplier,
    )

    results = []
    failures = []
    for item in dataset:
        docs = retrieval.run(item["question"], top_k=args.top_k, expand_parent=False)
        contexts = [
            {
                "rank": idx,
                "id": doc.doc_id,
                "source_file": doc.metadata.get("source_file") or doc.metadata.get("file_name"),
                "doc_type": doc.metadata.get("doc_type"),
                "chunk_type": doc.metadata.get("chunk_type"),
                "module_title": doc.metadata.get("module_title"),
                "section_title": doc.metadata.get("section_title"),
                "section_path": doc.metadata.get("section_path"),
                "module_section": doc.metadata.get("module_section"),
                "nearest_heading": doc.metadata.get("nearest_heading"),
                "page_label_start": doc.metadata.get("page_label_start"),
                "page_label_end": doc.metadata.get("page_label_end"),
                "score": doc.score,
                "text_preview": (doc.text or "")[:700],
            }
            for idx, doc in enumerate(docs, start=1)
        ]
        aggregate = "\n".join((doc.text or "") + "\n" + json.dumps(doc.metadata, ensure_ascii=False) for doc in docs)
        expected_source_hit = any(ctx["source_file"] == item["source_file"] for ctx in contexts)
        required_terms = item.get("required_phrases") or HARD_CASE_TERMS.get(
            item["id"], default_terms(item)
        )
        matched_terms = [term for term in required_terms if contains_term(aggregate, term)]
        phrase_ranks = {
            term: [
                context["rank"]
                for context, doc in zip(contexts, docs)
                if contains_term(
                    (doc.text or "")
                    + "\n"
                    + json.dumps(doc.metadata, ensure_ascii=False),
                    term,
                )
            ]
            for term in required_terms
        }
        min_hits = len(required_terms)
        passed = expected_source_hit and len(matched_terms) >= min_hits
        record = {
            "id": item["id"],
            "question": item["question"],
            "source_file": item["source_file"],
            "passed": passed,
            "expected_source_hit": expected_source_hit,
            "required_terms": required_terms,
            "matched_terms": matched_terms,
            "phrase_ranks": phrase_ranks,
            "contexts": contexts,
            "retrieval_debug": retrieval.last_debug,
        }
        results.append(record)
        if not passed:
            failures.append(record)
        status = "PASS" if passed else "FAIL"
        print(
            f"{status} {item['id']} source_hit={expected_source_hit} "
            f"terms={len(matched_terms)}/{len(required_terms)} top="
            f"{[(c['source_file'], c['nearest_heading']) for c in contexts[:3]]}",
            flush=True,
        )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "pdf_count": len(pdfs),
        "chunk_counts": dict(chunk_counts),
        "doc_types": doc_types,
        "top_k": args.top_k,
        "candidate_multiplier": args.candidate_multiplier,
        "total": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "retrieval_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
