from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
import uuid
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Optional, Sequence, cast

from theflow.settings import settings as flowsettings

from kotaemon.base import BaseComponent, Document, RetrievedDocument
from kotaemon.embeddings import BaseEmbeddings
from kotaemon.storages import BaseDocumentStore, BaseVectorStore
from kotaemon.utils.rag_debug import rag_log

from .base import BaseIndexing, BaseRetrieval
from .rankings import BaseReranking

VECTOR_STORE_FNAME = "vectorstore"
DOC_STORE_FNAME = "docstore"
logger = logging.getLogger(__name__)


def _vector_log(message: str, level: int = logging.INFO) -> None:
    """Log vector indexing/retrieval progress to logger and terminal."""

    logger.log(level, message)
    print(f"[vector-index] {message}", flush=True)


class VectorIndexing(BaseIndexing):
    """Ingest the document, run through the embedding, and store the embedding in a
    vector store.

    This pipeline supports the following set of inputs:
        - List of documents
        - List of texts
    """

    cache_dir: Optional[str] = getattr(flowsettings, "KH_CHUNKS_OUTPUT_DIR", None)
    vector_store: BaseVectorStore
    doc_store: Optional[BaseDocumentStore] = None
    embedding: BaseEmbeddings
    count_: int = 0

    def to_retrieval_pipeline(self, *args, **kwargs):
        """Convert the indexing pipeline to a retrieval pipeline"""
        return VectorRetrieval(
            vector_store=self.vector_store,
            doc_store=self.doc_store,
            embedding=self.embedding,
            **kwargs,
        )

    def write_chunk_to_file(self, docs: list[Document]):
        # save the chunks content into markdown format
        if self.cache_dir:
            file_name = docs[0].metadata.get("file_name")
            if not file_name:
                return

            file_name = Path(file_name)
            for i in range(len(docs)):
                markdown_content = ""
                if "page_label" in docs[i].metadata:
                    page_label = str(docs[i].metadata["page_label"])
                    markdown_content += f"Page label: {page_label}"
                if "file_name" in docs[i].metadata:
                    filename = docs[i].metadata["file_name"]
                    markdown_content += f"\nFile name: {filename}"
                if "section" in docs[i].metadata:
                    section = docs[i].metadata["section"]
                    markdown_content += f"\nSection: {section}"
                if "type" in docs[i].metadata:
                    if docs[i].metadata["type"] == "image":
                        image_origin = docs[i].metadata["image_origin"]
                        image_origin = f'<p><img src="{image_origin}"></p>'
                        markdown_content += f"\nImage origin: {image_origin}"
                if docs[i].text:
                    markdown_content += f"\ntext:\n{docs[i].text}"

                with open(
                    Path(self.cache_dir) / f"{file_name.stem}_{self.count_+i}.md",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(markdown_content)

    def add_to_docstore(self, docs: list[Document]):
        if self.doc_store:
            _vector_log(f"Adding {len(docs)} documents to doc store")
            self.doc_store.add(docs)
            _vector_log(f"Added {len(docs)} documents to doc store")

    def add_to_vectorstore(self, docs: list[Document]):
        # in case we want to skip embedding
        if self.vector_store:
            vector_docs = [
                doc for doc in docs if doc.metadata.get("index_role") != "parent"
            ]
            skipped = len(docs) - len(vector_docs)
            if skipped:
                _vector_log(
                    f"Skipping embeddings for {skipped} parent documents; "
                    f"embedding {len(vector_docs)} child/regular documents"
                )
            if not vector_docs:
                _vector_log("No documents eligible for vector embedding")
                return

            start_time = time.time()
            _vector_log(
                f"Getting embeddings for {len(vector_docs)} nodes with {self.embedding}"
            )

            # Call the embedding implementation directly instead of the inherited
            # theflow Function.__call__ wrapper. The wrapper adds diskcache-based
            # result caching and can block on stale/inter-process cache locks for
            # large transient document batches. run() preserves the embedding
            # algorithm/configuration and avoids caching upload-time payloads.
            embeddings = self.embedding.run(vector_docs)

            _vector_log(
                f"Created {len(embeddings)} embeddings "
                f"in {time.time() - start_time:.2f}s"
            )
            _vector_log("Adding embeddings to vector store")
            self.vector_store.add(
                embeddings=embeddings,
                ids=[t.doc_id for t in vector_docs],
            )
            _vector_log(f"Added {len(embeddings)} embeddings to vector store")

    def run(self, text: str | list[str] | Document | list[Document]):
        input_: list[Document] = []
        if not isinstance(text, list):
            text = [text]

        for item in cast(list, text):
            if isinstance(item, str):
                input_.append(Document(text=item, id_=str(uuid.uuid4())))
            elif isinstance(item, Document):
                input_.append(item)
            else:
                raise ValueError(
                    f"Invalid input type {type(item)}, should be str or Document"
                )

        self.add_to_vectorstore(input_)
        self.add_to_docstore(input_)
        self.write_chunk_to_file(input_)
        self.count_ += len(input_)


class VectorRetrieval(BaseRetrieval):
    """Retrieve list of documents from vector store"""

    vector_store: BaseVectorStore
    doc_store: Optional[BaseDocumentStore] = None
    embedding: BaseEmbeddings
    rerankers: Sequence[BaseReranking] = []
    top_k: int = 5
    first_round_top_k_mult: int = 10
    retrieval_mode: str = "hybrid"  # vector, text, hybrid
    enable_query_expansion: bool = True
    sibling_window: int = 1
    last_debug: dict[str, Any] = {}


    UNIVERSITY_QUERY_EXPANSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("oral exam", ("mündliche Prüfung", "mündliche Prüfungsleistung")),
        ("oral", ("mündlich", "mündliche Prüfung")),
        ("file inspection", ("Akteneinsicht", "Einsichtnahme")),
        ("inspection", ("Akteneinsicht", "Einsichtnahme")),
        ("grade announced", ("Bekanntgabe der Bewertung", "Notenbekanntgabe")),
        ("grade", ("Note", "Bewertung", "Prüfungsleistung")),
        ("deadline", ("Frist", "Abgabefrist")),
        ("bachelor thesis", ("Bachelorarbeit", "Abschlussarbeit")),
        ("thesis", ("Bachelorarbeit", "Abschlussarbeit")),
        ("ects points", ("ECTS-Punkte", "Leistungspunkte")),
        ("ects", ("ECTS", "Leistungspunkte")),
        ("attendance requirement", ("Anwesenheitspflicht", "Fehlzeiten")),
        ("attendance", ("Anwesenheit", "Anwesenheitspflicht", "Fehlzeiten")),
        ("distinction", ("mit Auszeichnung",)),
        ("deception", ("Täuschung", "fremde Hilfe")),
        ("ghostwriting", ("Täuschung", "fremde Hilfe")),
        ("artificial intelligence", ("künstliche Intelligenz", "KI", "Täuschung")),
        ("ai", ("künstliche Intelligenz", "KI", "Täuschung")),
        ("withdraw", ("Rücktritt", "Abmeldung")),
        (
            "retake",
            (
                "Wiederholung von Prüfungen",
                "nicht bestandene Prüfung",
                "zweimal wiederholen",
                "Wiederholungsprüfung",
                "Wiederholungsmöglichkeit",
                "mit Ausnahme der Bachelor- oder Masterarbeit",
            ),
        ),
        (
            "repeat",
            (
                "Wiederholung von Prüfungen",
                "nicht bestandene Prüfung",
                "zweimal wiederholen",
                "Wiederholungsprüfung",
                "Wiederholungsmöglichkeit",
                "mit Ausnahme der Bachelor- oder Masterarbeit",
            ),
        ),
        (
            "failed",
            (
                "Wiederholung von Prüfungen",
                "nicht bestandene Prüfung",
                "zweimal wiederholen",
                "Wiederholungsprüfung",
                "Wiederholungsmöglichkeit",
                "mit Ausnahme der Bachelor- oder Masterarbeit",
            ),
        ),
        (
            "failed exam",
            (
                "Wiederholung von Prüfungen",
                "nicht bestandene Prüfung",
                "zweimal wiederholen",
                "Wiederholungsprüfung",
                "Wiederholungsmöglichkeit",
                "mit Ausnahme der Bachelor- oder Masterarbeit",
            ),
        ),
        ("exam board", ("Prüfungsausschuss",)),
    )

    def _normalize_query_text(self, query: str) -> str:
        de_umlaut = (
            query.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("Ä", "Ae")
            .replace("Ö", "Oe")
            .replace("Ü", "Ue")
            .replace("ß", "ss")
        )
        ascii_fold = unicodedata.normalize("NFKD", query).encode("ascii", "ignore").decode("ascii")
        return de_umlaut if de_umlaut != query else ascii_fold

    def query_variants(self, text: str | Document) -> list[str]:
        query = text.text if isinstance(text, Document) else str(text)
        variants: list[str] = []

        def add(value: str) -> None:
            value = " ".join(str(value).split())
            if value and value not in variants:
                variants.append(value)

        add(query)
        lowered = query.lower()
        expansion_terms: list[str] = []
        for needle, replacements in self.UNIVERSITY_QUERY_EXPANSIONS:
            # Match complete terms only.  A raw substring check made "ai" match
            # words like "failed", which polluted repeat-exam queries with
            # Täuschung/KI/Ghostwriting expansions and hurt hybrid ranking.
            if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", lowered):
                for replacement in replacements:
                    if replacement not in expansion_terms:
                        expansion_terms.append(replacement)
        if expansion_terms:
            add(f"{query} {' '.join(expansion_terms)}")
            add(" ".join(expansion_terms))
        normalized = self._normalize_query_text(query)
        if normalized != query:
            add(normalized)
        keywords = [
            token
            for token in re.findall(r"[\wÄÖÜäöüß]+", query)
            if len(token) > 2
            and token.lower() not in {
                "the", "and", "for", "with", "when", "what", "which", "how",
                "does", "can", "are", "you", "your", "from", "that", "this",
            }
        ]
        if keywords:
            add(" ".join(keywords[:10]))
        return variants if self.enable_query_expansion else [query]

    def _filter_docs(
        self, documents: list[RetrievedDocument], top_k: int | None = None
    ):
        if top_k:
            documents = documents[:top_k]
        return documents

    def _rrf_fuse(
        self,
        vector_docs: list[Document],
        vector_scores: list[float],
        text_docs: list[Document],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> list[RetrievedDocument]:
        """Fuse dense and full-text candidates using weighted RRF ranks.

        Keep the small RRF value as an internal ranking score. User-facing
        retrieval scores should remain the source score (dense similarity or
        text rank score); otherwise hybrid hits display as 0.00/0.01 even when
        the candidate was a strong vector or lexical match.
        """

        fused: dict[str, dict] = {}

        def ensure_entry(doc: Document) -> dict:
            entry = fused.get(doc.doc_id)
            if entry is None:
                doc_dict = doc.to_dict()
                metadata = dict(doc_dict.get("metadata") or {})
                doc_dict["metadata"] = metadata
                entry = {
                    "doc_dict": doc_dict,
                    "metadata": metadata,
                    "score": 0.0,
                    "sources": [],
                }
                fused[doc.doc_id] = entry
            return entry

        seen_vector_ids: set[str] = set()
        for rank, (doc, score) in enumerate(zip(vector_docs, vector_scores), start=1):
            if doc.doc_id in seen_vector_ids:
                continue
            seen_vector_ids.add(doc.doc_id)
            entry = ensure_entry(doc)
            entry["score"] += dense_weight / (rrf_k + rank)
            entry["metadata"]["_vector_score"] = score
            entry["metadata"]["_dense_rank"] = rank
            if "vector" not in entry["sources"]:
                entry["sources"].append("vector")

        seen_text_ids: set[str] = set()
        for rank, doc in enumerate(text_docs, start=1):
            if doc.doc_id in seen_text_ids:
                continue
            seen_text_ids.add(doc.doc_id)
            entry = ensure_entry(doc)
            entry["score"] += sparse_weight / (rrf_k + rank)
            entry["metadata"]["_text_score"] = 1.0 / rank
            entry["metadata"]["_sparse_rank"] = rank
            if "text" not in entry["sources"]:
                entry["sources"].append("text")

        result: list[RetrievedDocument] = []
        for entry in fused.values():
            entry["metadata"]["_fusion_score"] = entry["score"]
            entry["metadata"]["_ranking_score"] = entry["score"]
            entry["metadata"]["_retrieval_sources"] = entry["sources"]
            entry["metadata"]["retrieval_source"] = (
                "both" if {"vector", "text"}.issubset(set(entry["sources"])) else entry["sources"][0]
            )
            display_score = self._display_retrieval_score(entry["metadata"])
            entry["metadata"]["retrieval_score"] = display_score
            result.append(RetrievedDocument(**entry["doc_dict"], score=display_score))

        return sorted(result, key=self._ranking_score, reverse=True)

    def _display_retrieval_score(self, metadata: dict[str, Any]) -> float:
        """Return a meaningful score for UI/eval prompts, not the RRF rank value."""

        try:
            vector_score = float(metadata.get("_vector_score"))
        except (TypeError, ValueError):
            vector_score = None
        if vector_score is not None and vector_score != -1.0:
            return vector_score

        try:
            text_score = float(metadata.get("_text_score"))
        except (TypeError, ValueError):
            text_score = None
        if text_score is not None and text_score != -1.0:
            # Lexical rank is useful for ordering, but it is not a semantic
            # relevance probability.  Keep text-only scores visibly lower so a
            # wrong FTS hit is not shown as 1.00 relevance in hybrid mode.
            return min(0.35, max(0.0, text_score * 0.35))

        try:
            return float(metadata.get("_fusion_score", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _ranking_score(self, doc: RetrievedDocument) -> float:
        metadata = doc.metadata or {}
        for key in ("_ranking_score", "_fusion_score"):
            try:
                return float(metadata[key])
            except (KeyError, TypeError, ValueError):
                pass
        try:
            return float(doc.score or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _vector_scope_kwargs(self, scope: list[str] | None) -> dict[str, list[str]]:
        """Map selected chunk ids to the vectorstore's public query parameter.

        BaseVectorStore and LlamaIndex-backed stores use ``ids``. Some local test
        doubles and older adapters used ``doc_ids``. Prefer the canonical
        contract so production vector search is actually scoped to selected
        chunks, while keeping compatibility with older doubles.
        """

        if not scope:
            return {}

        try:
            parameters = signature(self.vector_store.query).parameters
        except (TypeError, ValueError):
            return {"ids": scope}

        if "ids" in parameters:
            return {"ids": scope}
        if "doc_ids" in parameters:
            return {"doc_ids": scope}
        if any(param.kind == Parameter.VAR_KEYWORD for param in parameters.values()):
            return {"ids": scope}
        return {}

    def run(
        self,
        text: str | Document,
        top_k: Optional[int] = None,
        expand_parent: bool = True,
        **kwargs,
    ) -> list[RetrievedDocument]:
        """Retrieve a list of documents from vector store

        Args:
            text: the text to retrieve similar documents
            top_k: number of top similar documents to return

        Returns:
            list[RetrievedDocument]: list of retrieved documents
        """
        if top_k is None:
            top_k = self.top_k

        do_extend = kwargs.pop("do_extend", False)
        thumbnail_count = kwargs.pop("thumbnail_count", 3)

        candidate_multiplier = max(1, int(self.first_round_top_k_mult))
        if do_extend:
            top_k_first_round = top_k * candidate_multiplier
        else:
            top_k_first_round = top_k

        if self.doc_store is None:
            raise ValueError(
                "doc_store is not provided. Please provide a doc_store to "
                "retrieve the documents"
            )

        result: list[RetrievedDocument] = []
        # TODO: should declare scope directly in the run params
        scope = kwargs.pop("scope", None)
        emb: list[float]

        _vector_log(
            f"Retrieval started: retrieval_mode={self.retrieval_mode}, "
            f"final_top_k={top_k}, candidate_multiplier={candidate_multiplier}, "
            f"candidate_pool={top_k_first_round}, scope={len(scope) if scope else 0}"
        )

        query_variants = self.query_variants(text)
        debug: dict[str, Any] = {
            "question": text.text if isinstance(text, Document) else text,
            "query_variants": query_variants,
            "retrieval_mode": self.retrieval_mode,
            "final_top_k": top_k,
            "candidate_multiplier": candidate_multiplier,
            "candidate_pool_size": top_k_first_round,
            "scope_size": len(scope) if scope else 0,
            "reranking_enabled": bool(self.rerankers),
            "vector_candidates": 0,
            "text_candidates": 0,
            "fused_candidates": 0,
            "final_docs_before_expansion": 0,
            "final_docs_after_expansion": 0,
            "vector_filter_fallbacks": 0,
            "vector_ids_without_docstore_match": 0,
            "vector_branch_empty": False,
        }

        def vector_search(query_text: str) -> tuple[list[Document], list[float]]:
            start_time = time.time()
            emb_local = self.embedding.run(query_text)[0].embedding
            _vector_log(f"Query embedding ready in {time.time() - start_time:.2f}s")
            vector_kwargs = dict(kwargs)
            vector_kwargs.update(self._vector_scope_kwargs(scope))
            _, scores, ids = self.vector_store.query(
                embedding=emb_local, top_k=top_k_first_round, **vector_kwargs
            )
            if (
                not ids
                and scope
                and "filters" in vector_kwargs
            ):
                # The selected chunk ids already scope the query.  Some
                # LlamaIndex/Chroma filter translations are stricter than the
                # metadata actually stored in older indexes and can eliminate all
                # vector hits.  Retry without metadata filters before declaring
                # vector search empty.
                debug["vector_filter_fallbacks"] += 1
                retry_kwargs = dict(vector_kwargs)
                retry_kwargs.pop("filters", None)
                _vector_log("Vector query returned no ids; retrying without filters")
                _, scores, ids = self.vector_store.query(
                    embedding=emb_local, top_k=top_k_first_round, **retry_kwargs
                )
            docs_local = self.doc_store.get(ids) if ids else []
            if ids and not docs_local:
                debug["vector_ids_without_docstore_match"] += len(ids)
            if ids:
                score_by_id = dict(zip(ids, scores))
                scores = [score_by_id.get(doc.doc_id, 0.0) for doc in docs_local]
            return docs_local, list(scores)

        def text_search(query_text: str) -> list[Document]:
            return self.doc_store.query(query_text, top_k=top_k_first_round, doc_ids=scope)

        def merge_vector_batches(
            batches: list[tuple[list[Document], list[float]]]
        ) -> tuple[list[Document], list[float]]:
            """Merge query-expansion dense candidates by best per-query rank.

            Query expansion runs multiple German/English variants.  A hit ranked
            #1 for a later variant should not be treated as rank #151 merely
            because candidates were appended after the first variant.
            """

            best: dict[str, tuple[int, float, int, Document]] = {}
            order = 0
            for docs_for_query, scores_for_query in batches:
                for local_rank, (doc, score) in enumerate(
                    zip(docs_for_query, scores_for_query), start=1
                ):
                    order += 1
                    current = best.get(doc.doc_id)
                    candidate = (local_rank, -float(score or 0.0), order, doc)
                    if current is None or candidate[:3] < current[:3]:
                        best[doc.doc_id] = candidate

            ordered = sorted(best.values(), key=lambda item: item[:3])
            docs = [item[3] for item in ordered]
            scores = [-item[1] for item in ordered]
            return docs, scores

        def merge_text_batches(batches: list[list[Document]]) -> list[Document]:
            """Merge query-expansion lexical candidates by best per-query rank."""

            best: dict[str, tuple[int, int, Document]] = {}
            order = 0
            for docs_for_query in batches:
                for local_rank, doc in enumerate(docs_for_query, start=1):
                    order += 1
                    candidate = (local_rank, order, doc)
                    current = best.get(doc.doc_id)
                    if current is None or candidate[:2] < current[:2]:
                        best[doc.doc_id] = candidate
            return [item[2] for item in sorted(best.values(), key=lambda item: item[:2])]

        if self.retrieval_mode == "vector":
            vector_batches: list[tuple[list[Document], list[float]]] = []
            for query in query_variants:
                docs_for_query, scores_for_query = vector_search(query)
                vector_batches.append((docs_for_query, scores_for_query))
            all_vector_docs, all_vector_scores = merge_vector_batches(vector_batches)
            result = self._rrf_fuse(all_vector_docs, all_vector_scores, [])
            debug["vector_candidates"] = len(all_vector_docs)
            debug["fused_candidates"] = len(result)
            debug["vector_branch_empty"] = len(all_vector_docs) == 0
        elif self.retrieval_mode == "text":
            text_batches: list[list[Document]] = []
            for query in query_variants:
                text_batches.append(text_search(query))
            all_text_docs = merge_text_batches(text_batches)
            result = self._rrf_fuse([], [], all_text_docs)
            debug["text_candidates"] = len(all_text_docs)
            debug["fused_candidates"] = len(result)
        elif self.retrieval_mode == "hybrid":
            vs_batches: list[tuple[list[Document], list[float]]] = []
            ds_batches: list[list[Document]] = []
            errors: list[tuple[str, Exception]] = []
            errors_lock = threading.Lock()

            def _record_error(source: str, exc: Exception) -> None:
                with errors_lock:
                    errors.append((source, exc))

            def query_vectorstore():
                nonlocal vs_batches
                try:
                    for query in query_variants:
                        docs_for_query, scores_for_query = vector_search(query)
                        vs_batches.append((docs_for_query, scores_for_query))
                except Exception as exc:
                    _record_error("vectorstore", exc)

            def query_docstore():
                nonlocal ds_batches
                try:
                    for query in query_variants:
                        ds_batches.append(text_search(query))
                except Exception as exc:
                    _record_error("docstore", exc)

            vs_query_thread = threading.Thread(target=query_vectorstore)
            ds_query_thread = threading.Thread(target=query_docstore)

            _vector_log("Starting hybrid vector/docstore queries")
            query_start = time.time()
            vs_query_thread.start()
            ds_query_thread.start()

            vs_query_thread.join()
            ds_query_thread.join()
            _vector_log(
                f"Hybrid vector/docstore queries finished "
                f"in {time.time() - query_start:.2f}s"
            )
            if errors:
                error_summary = "; ".join(
                    f"{source}: {exc!r}" for source, exc in errors
                )
                for source, exc in errors:
                    logger.error(
                        "Hybrid retrieval %s branch failed",
                        source,
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
                raise RuntimeError(
                    f"Hybrid retrieval failed in branch(es): {error_summary}"
                ) from errors[0][1]

            vs_docs, vs_scores = merge_vector_batches(vs_batches)
            ds_docs = merge_text_batches(ds_batches)
            result = self._rrf_fuse(vs_docs, vs_scores, ds_docs)
            debug["vector_candidates"] = len(vs_docs)
            debug["text_candidates"] = len(ds_docs)
            debug["fused_candidates"] = len(result)
            debug["vector_branch_empty"] = len(vs_docs) == 0
            _vector_log(f"Got {len(vs_docs)} from vectorstore")
            _vector_log(f"Got {len(ds_docs)} from docstore")
        else:
            raise ValueError(f"Unsupported retrieval_mode={self.retrieval_mode!r}")

        result = self._without_parent_docs(result)

        # use additional reranker to re-order the document list
        if self.rerankers and text:
            for reranker in self.rerankers:
                rerank_start = time.time()
                _vector_log(f"Running reranker {reranker}")
                result = reranker.run(documents=result, query=text)
                for doc in result:
                    sources = list(doc.metadata.get("_retrieval_sources") or [])
                    if "rerank" not in sources:
                        sources.append("rerank")
                    doc.metadata["_retrieval_sources"] = sources
                    doc.metadata["retrieval_source"] = doc.metadata.get("retrieval_source") or "rerank"
                _vector_log(
                    f"Reranker returned {len(result)} docs "
                    f"in {time.time() - rerank_start:.2f}s"
                )

        result = self._filter_docs(result, top_k=top_k)
        debug["final_docs_before_expansion"] = len(result)
        _vector_log(f"Got raw {len(result)} retrieved documents")

        # add page thumbnails to the result if exists
        thumbnail_doc_ids: set[str] = set()
        # we should copy the text from retrieved text chunk
        # to the thumbnail to get relevant LLM score correctly
        text_thumbnail_docs: dict[str, RetrievedDocument] = {}

        non_thumbnail_docs = []
        raw_thumbnail_docs = []
        for doc in result:
            if doc.metadata.get("type") == "thumbnail":
                # change type to image to display on UI
                doc.metadata["type"] = "image"
                raw_thumbnail_docs.append(doc)
                continue
            if (
                "thumbnail_doc_id" in doc.metadata
                and len(thumbnail_doc_ids) < thumbnail_count
            ):
                thumbnail_id = doc.metadata["thumbnail_doc_id"]
                thumbnail_doc_ids.add(thumbnail_id)
                text_thumbnail_docs[thumbnail_id] = doc
            else:
                non_thumbnail_docs.append(doc)

        linked_thumbnail_docs = self.doc_store.get(list(thumbnail_doc_ids))
        _vector_log(
            f"thumbnail docs {len(linked_thumbnail_docs)}; "
            f"non-thumbnail docs {len(non_thumbnail_docs)}; "
            f"raw-thumbnail docs {len(raw_thumbnail_docs)}"
        )
        additional_docs = []

        for thumbnail_doc in linked_thumbnail_docs:
            text_doc = text_thumbnail_docs[thumbnail_doc.doc_id]
            doc_dict = thumbnail_doc.to_dict()
            doc_dict["_id"] = text_doc.doc_id
            doc_dict["content"] = text_doc.content
            doc_dict["metadata"]["type"] = "image"
            for key in text_doc.metadata:
                if key not in doc_dict["metadata"]:
                    doc_dict["metadata"][key] = text_doc.metadata[key]

            additional_docs.append(RetrievedDocument(**doc_dict, score=text_doc.score))

        result = additional_docs + non_thumbnail_docs

        if not result:
            # return output from raw retrieved thumbnails
            result = self._filter_docs(raw_thumbnail_docs, top_k=thumbnail_count)

        if expand_parent == "siblings":
            result = self._expand_sibling_context(result, window=self.sibling_window)
        elif expand_parent:
            result = self._expand_parent_context(result)

        debug["final_docs_after_expansion"] = len(result)
        debug["final_docs"] = [self._debug_doc(doc, rank) for rank, doc in enumerate(result, start=1)]
        self.last_debug = debug
        rag_log("retrieval.vector.result", **debug)
        return result

    def _debug_doc(self, doc: RetrievedDocument, rank: int) -> dict[str, Any]:
        metadata = doc.metadata or {}
        return {
            "rank": rank,
            "doc_id": doc.doc_id,
            "source_file": metadata.get("source_file") or metadata.get("file_name"),
            "page_label_start": metadata.get("page_label_start") or metadata.get("page_label"),
            "page_label_end": metadata.get("page_label_end") or metadata.get("page_label"),
            "section_id": metadata.get("section_id"),
            "paragraph_id": metadata.get("paragraph_id"),
            "parent_id": metadata.get("parent_id"),
            "child_index": metadata.get("child_index"),
            "index_role": metadata.get("index_role"),
            "score": doc.score,
            "ranking_score": metadata.get("_ranking_score") or metadata.get("_fusion_score"),
            "vector_score": metadata.get("_vector_score"),
            "text_score": metadata.get("_text_score"),
            "retrieval_source": metadata.get("retrieval_source") or metadata.get("_retrieval_sources"),
            "preview": (doc.text or "")[:500],
        }

    def _expand_sibling_context(
        self, documents: list[RetrievedDocument], window: int = 1
    ) -> list[RetrievedDocument]:
        if self.doc_store is None or not documents:
            return documents
        window = max(0, int(window or 0))
        try:
            all_docs = self.doc_store.get_all()
        except Exception:
            return documents
        children_by_parent: dict[str, list[Document]] = {}
        for doc in all_docs:
            metadata = doc.metadata or {}
            if metadata.get("index_role") == "child" and metadata.get("parent_id"):
                children_by_parent.setdefault(str(metadata["parent_id"]), []).append(doc)
        for siblings in children_by_parent.values():
            siblings.sort(key=lambda d: int((d.metadata or {}).get("child_index") or 0))

        expanded: list[RetrievedDocument] = []
        seen: set[str] = set()
        for doc in sorted(documents, key=self._ranking_score, reverse=True):
            metadata = doc.metadata or {}
            parent_id = metadata.get("parent_id")
            child_index = metadata.get("child_index")
            if metadata.get("index_role") != "child" or not parent_id or child_index is None:
                if doc.doc_id not in seen:
                    expanded.append(doc)
                    seen.add(doc.doc_id)
                continue

            siblings = children_by_parent.get(str(parent_id), [])
            matched_idx = int(child_index)
            group = [
                sibling
                for sibling in siblings
                if abs(int((sibling.metadata or {}).get("child_index") or 0) - matched_idx) <= window
            ]
            group.sort(
                key=lambda sibling: (
                    0 if int((sibling.metadata or {}).get("child_index") or 0) == matched_idx else 1,
                    abs(int((sibling.metadata or {}).get("child_index") or 0) - matched_idx),
                    int((sibling.metadata or {}).get("child_index") or 0),
                )
            )
            for sibling in group:
                if sibling.doc_id in seen:
                    continue
                sibling_metadata = dict(sibling.metadata or {})
                sibling_metadata["context_role"] = (
                    "matched_child" if sibling.doc_id == doc.doc_id else "sibling_context"
                )
                sibling_metadata["retrieval_source"] = (
                    metadata.get("retrieval_source") if sibling.doc_id == doc.doc_id else "sibling_context"
                )
                sibling_metadata["matched_child_id"] = doc.doc_id
                expanded.append(
                    RetrievedDocument(
                        **{**sibling.to_dict(), "metadata": sibling_metadata},
                        score=doc.score if sibling.doc_id == doc.doc_id else max((doc.score or 0.0) - 1e-6, 0.0),
                    )
                )
                seen.add(sibling.doc_id)
        return expanded

    def _without_parent_docs(
        self, documents: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        """Parents are docstore context, not raw retrieval candidates."""

        return [
            doc for doc in documents if doc.metadata.get("index_role") != "parent"
        ]

    def _expand_parent_context(
        self, documents: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        """Replace retrieved university child chunks with deduplicated parents.

        Child chunks remain the retrieval unit; this method fetches the matching
        parent documents from the docstore and annotates them with the child scores
        that caused the parent expansion. Non-university documents and thumbnails
        are preserved unchanged.
        """

        if self.doc_store is None:
            return documents

        parent_children: dict[str, list[RetrievedDocument]] = {}
        passthrough: list[RetrievedDocument] = []
        for doc in documents:
            parent_id = doc.metadata.get("parent_id")
            if (
                doc.metadata.get("index_role") == "child"
                and parent_id
                and doc.metadata.get("type") != "image"
            ):
                parent_children.setdefault(str(parent_id), []).append(doc)
            else:
                passthrough.append(doc)

        if not parent_children:
            return documents

        parent_ids = list(parent_children)
        try:
            parents = self.doc_store.get(parent_ids)
        except KeyError:
            # If a docstore is partially stale, do not fail retrieval. Return the
            # already-retrieved child chunks instead.
            return documents

        expanded: list[RetrievedDocument] = []
        for parent in parents:
            children = parent_children.get(parent.doc_id, [])
            if not children:
                continue
            child_scores = [child.score for child in children]
            child_ranking_scores = [self._ranking_score(child) for child in children]
            doc_dict = parent.to_dict()
            metadata = doc_dict.setdefault("metadata", {})
            metadata["expanded_from_child_ids"] = [child.doc_id for child in children]
            metadata["expanded_from_child_scores"] = child_scores
            metadata["best_child_score"] = max(child_scores) if child_scores else 0.0
            metadata["expanded_from_child_ranking_scores"] = child_ranking_scores
            metadata["_ranking_score"] = (
                max(child_ranking_scores) if child_ranking_scores else 0.0
            )
            metadata["_fusion_score"] = metadata["_ranking_score"]
            metadata["retrieval_score"] = metadata["best_child_score"]
            expanded.append(
                RetrievedDocument(
                    **doc_dict,
                    score=max(child_scores) if child_scores else 0.0,
                )
            )

        result = passthrough + expanded
        return sorted(result, key=self._ranking_score, reverse=True)


class TextVectorQA(BaseComponent):
    retrieving_pipeline: BaseRetrieval
    qa_pipeline: BaseComponent

    def run(self, question, **kwargs):
        retrieved_documents = self.retrieving_pipeline.run(question, **kwargs)
        return self.qa_pipeline.run(question, retrieved_documents, **kwargs)
