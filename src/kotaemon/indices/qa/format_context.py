import html
from functools import partial

import tiktoken

from kotaemon.base import BaseComponent, Document, RetrievedDocument
from kotaemon.indices.splitters import TokenSplitter
from kotaemon.utils.rag_debug import rag_log

EVIDENCE_MODE_TEXT = 0
EVIDENCE_MODE_TABLE = 1
EVIDENCE_MODE_CHATBOT = 2
EVIDENCE_MODE_FIGURE = 3


class PrepareEvidencePipeline(BaseComponent):
    """Prepare ranked evidence text from retrieved documents."""

    max_context_length: int = 32000
    trim_func: TokenSplitter | None = None
    last_debug: dict = {}
    context_budget_debug: dict = {}

    def _tokenizer(self):
        return partial(
            tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
            allowed_special=set(),
            disallowed_special="all",
        )

    def _token_count(self, text: str) -> int:
        return len(self._tokenizer()(text or ""))

    def _doc_label(self, doc: RetrievedDocument, rank: int) -> str:
        metadata = doc.metadata or {}
        source = metadata.get("source_file") or metadata.get("file_name") or "-"
        page_start = metadata.get("page_label_start") or metadata.get("page_label")
        page_end = metadata.get("page_label_end") or metadata.get("page_label")
        page = ""
        if page_start and page_end and page_start != page_end:
            page = f", pages={page_start}-{page_end}"
        elif page_start:
            page = f", page={page_start}"
        section = metadata.get("section_id") or metadata.get("section_title") or ""
        paragraph = metadata.get("paragraph_id") or ""
        role = metadata.get("context_role") or metadata.get("index_role") or ""
        retrieval_source = metadata.get("retrieval_source") or metadata.get("_retrieval_sources") or ""
        return (
            f"[Context {rank}] source={source}{page}, section={section}, "
            f"paragraph={paragraph}, role={role}, retrieval_source={retrieval_source}, "
            f"score={doc.score}"
        )

    def _doc_text(self, doc: RetrievedDocument) -> tuple[str, int | None]:
        metadata = doc.metadata or {}
        if metadata.get("type", "") == "table":
            return metadata.get("table_origin", doc.text), EVIDENCE_MODE_TABLE
        if metadata.get("type", "") == "chatbot":
            return metadata.get("window", doc.text), EVIDENCE_MODE_CHATBOT
        if metadata.get("type", "") == "image":
            caption = html.escape(doc.get_content())
            return f"[Figure: {caption}]", EVIDENCE_MODE_FIGURE
        return metadata.get("window", doc.text), None

    def _doc_sort_score(self, doc: RetrievedDocument) -> float:
        """Use final retrieval score for prompt order; fall back to debug scores."""

        metadata = doc.metadata or {}
        candidates = (
            doc.score,
            metadata.get("_ranking_score"),
            metadata.get("_fusion_score"),
            metadata.get("retrieval_score"),
            metadata.get("score"),
        )
        for candidate in candidates:
            try:
                if candidate is not None:
                    return float(candidate)
            except (TypeError, ValueError):
                continue
        return 0.0

    def run(self, docs: list[RetrievedDocument]) -> Document:
        images: list[str] = []
        evidence_modes: list[int] = []
        chunks: list[str] = []
        used_docs: list[dict] = []
        budget = max(1, int(self.max_context_length or 32000))
        used_tokens = 0
        dropped_docs: list[dict] = []
        partially_truncated_docs: list[dict] = []
        ranked_docs = sorted(docs, key=self._doc_sort_score, reverse=True)
        prepared_items: list[dict] = []
        candidate_context_tokens = 0

        for rank, retrieved_item in enumerate(ranked_docs, start=1):
            metadata = retrieved_item.metadata or {}
            retrieved_content, mode = self._doc_text(retrieved_item)
            if not retrieved_content:
                continue

            chunk = f"{self._doc_label(retrieved_item, rank)}\n{retrieved_content.strip()}".strip()
            tokens = self._token_count(chunk)
            candidate_context_tokens += tokens
            prepared_items.append(
                {
                    "rank": rank,
                    "doc": retrieved_item,
                    "metadata": metadata,
                    "content": retrieved_content.strip(),
                    "mode": mode,
                    "chunk": chunk,
                    "tokens": tokens,
                }
            )

        for item_index, item in enumerate(prepared_items):
            rank = item["rank"]
            retrieved_item = item["doc"]
            metadata = item["metadata"]
            retrieved_content = item["content"]
            mode = item["mode"]
            chunk = item["chunk"]
            tokens = item["tokens"]
            remaining = budget - used_tokens
            if remaining <= 0:
                dropped_docs.append(
                    {
                        "rank": rank,
                        "doc_id": retrieved_item.doc_id,
                        "reason": "budget_exhausted",
                        "source_file": metadata.get("source_file")
                        or metadata.get("file_name"),
                        "section_id": metadata.get("section_id"),
                    }
                )
                dropped_docs.extend(
                    {
                        "rank": later["rank"],
                        "doc_id": later["doc"].doc_id,
                        "reason": "not_reached_due_budget",
                        "source_file": later["metadata"].get("source_file")
                        or later["metadata"].get("file_name"),
                        "section_id": later["metadata"].get("section_id"),
                    }
                    for later in prepared_items[item_index + 1 :]
                )
                break
            if tokens > remaining:
                # Keep the label and as much of this ranked chunk as possible; never
                # truncate higher-ranked chunks because of lower-ranked ones.
                label = self._doc_label(retrieved_item, rank)
                label_tokens = self._token_count(label)
                if remaining <= label_tokens + 16:
                    dropped_docs.append(
                        {
                            "rank": rank,
                            "doc_id": retrieved_item.doc_id,
                            "reason": "insufficient_budget_for_label",
                            "source_file": metadata.get("source_file")
                            or metadata.get("file_name"),
                            "section_id": metadata.get("section_id"),
                        }
                    )
                    dropped_docs.extend(
                        {
                            "rank": later["rank"],
                            "doc_id": later["doc"].doc_id,
                            "reason": "not_reached_due_budget",
                            "source_file": later["metadata"].get("source_file")
                            or later["metadata"].get("file_name"),
                            "section_id": later["metadata"].get("section_id"),
                        }
                        for later in prepared_items[item_index + 1 :]
                    )
                    break
                trim_func = self.trim_func or TokenSplitter(
                    chunk_size=remaining - label_tokens,
                    chunk_overlap=0,
                    separator=" ",
                    tokenizer=self._tokenizer(),
                )
                trimmed = trim_func.run([Document(text=retrieved_content.strip())])
                body = trimmed[0].text if trimmed else ""
                chunk = f"{label}\n{body}".strip()
                tokens = self._token_count(chunk)
                partially_truncated_docs.append(
                    {
                        "rank": rank,
                        "doc_id": retrieved_item.doc_id,
                        "reason": "trimmed_to_remaining_budget",
                        "source_file": metadata.get("source_file")
                        or metadata.get("file_name"),
                        "section_id": metadata.get("section_id"),
                    }
                )
                if mode is not None:
                    evidence_modes.append(mode)
                if metadata.get("type", "") == "image":
                    images.append(metadata.get("image_origin", ""))
            else:
                if mode is not None:
                    evidence_modes.append(mode)
                if metadata.get("type", "") == "image":
                    images.append(metadata.get("image_origin", ""))
            chunks.append(chunk)
            used_tokens += tokens
            used_docs.append(
                {
                    "rank": rank,
                    "doc_id": retrieved_item.doc_id,
                    "tokens": tokens,
                    "source_file": metadata.get("source_file") or metadata.get("file_name"),
                    "section_id": metadata.get("section_id"),
                    "paragraph_id": metadata.get("paragraph_id"),
                    "score": retrieved_item.score,
                }
            )

        evidence = "\n\n---\n\n".join(chunks)
        evidence_mode = EVIDENCE_MODE_TEXT
        if EVIDENCE_MODE_FIGURE in evidence_modes:
            evidence_mode = EVIDENCE_MODE_FIGURE
        elif EVIDENCE_MODE_TABLE in evidence_modes:
            evidence_mode = EVIDENCE_MODE_TABLE

        self.last_debug = {
            "input_docs": len(docs),
            "used_docs": len(used_docs),
            "max_context_tokens": budget,
            "candidate_context_tokens": candidate_context_tokens,
            "context_tokens": used_tokens,
            "context_chars": len(evidence),
            "docs": used_docs,
            "dropped_docs": dropped_docs,
            "partially_truncated_docs": partially_truncated_docs,
            "truncated_docs_count": len(dropped_docs) + len(partially_truncated_docs),
            "empty_context_with_docs": bool(docs and not evidence),
            **(self.context_budget_debug or {}),
        }
        rag_log(
            "qa.evidence.prepared",
            **self.last_debug,
            evidence_preview=evidence[:3000],
        )
        if docs and not evidence:
            raise RuntimeError("Retrieved documents existed but prompt context became empty.")
        return Document(content=(evidence_mode, evidence, images))
