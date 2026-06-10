from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Sequence, cast

from theflow.settings import settings as flowsettings

from kotaemon.base import BaseComponent, Document, RetrievedDocument
from kotaemon.embeddings import BaseEmbeddings
from kotaemon.storages import BaseDocumentStore, BaseVectorStore

from .base import BaseIndexing, BaseRetrieval
from .rankings import BaseReranking, LLMReranking

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
        """Fuse dense and full-text candidates using weighted RRF ranks."""

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
            entry["metadata"]["_sparse_rank"] = rank
            if "text" not in entry["sources"]:
                entry["sources"].append("text")

        result: list[RetrievedDocument] = []
        for entry in fused.values():
            entry["metadata"]["_fusion_score"] = entry["score"]
            entry["metadata"]["_retrieval_sources"] = entry["sources"]
            result.append(RetrievedDocument(**entry["doc_dict"], score=entry["score"]))

        return sorted(result, key=lambda doc: doc.score, reverse=True)

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

        if do_extend:
            top_k_first_round = top_k * self.first_round_top_k_mult
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
            f"Retrieval started: mode={self.retrieval_mode}, top_k={top_k}, "
            f"first_round_top_k={top_k_first_round}, scope={len(scope) if scope else 0}"
        )

        if self.retrieval_mode == "vector":
            start_time = time.time()
            _vector_log("Getting query embedding")
            emb = self.embedding.run(text)[0].embedding
            _vector_log(f"Query embedding ready in {time.time() - start_time:.2f}s")
            _, scores, ids = self.vector_store.query(
                embedding=emb, top_k=top_k_first_round, doc_ids=scope, **kwargs
            )
            docs = self.doc_store.get(ids)
            result = [
                RetrievedDocument(**doc.to_dict(), score=score)
                for doc, score in zip(docs, scores)
            ]
        elif self.retrieval_mode == "text":
            query = text.text if isinstance(text, Document) else text
            docs = self.doc_store.query(
                query, top_k=top_k_first_round, doc_ids=scope
            )
            result = [RetrievedDocument(**doc.to_dict(), score=-1.0) for doc in docs]
        elif self.retrieval_mode == "hybrid":
            # similarity search section
            start_time = time.time()
            _vector_log("Getting query embedding")
            emb = self.embedding.run(text)[0].embedding
            _vector_log(f"Query embedding ready in {time.time() - start_time:.2f}s")
            vs_docs: list[Document] = []
            vs_ids: list[str] = []
            vs_scores: list[float] = []

            def query_vectorstore():
                nonlocal vs_docs
                nonlocal vs_scores
                nonlocal vs_ids

                assert self.doc_store is not None
                _, vs_scores, vs_ids = self.vector_store.query(
                    embedding=emb, top_k=top_k_first_round, doc_ids=scope, **kwargs
                )
                if vs_ids:
                    vs_docs = self.doc_store.get(vs_ids)
                    score_by_id = dict(zip(vs_ids, vs_scores))
                    vs_scores = [score_by_id[doc.doc_id] for doc in vs_docs]

            # full-text search section
            ds_docs: list[Document] = []

            def query_docstore():
                nonlocal ds_docs

                assert self.doc_store is not None
                query = text.text if isinstance(text, Document) else text
                ds_docs = self.doc_store.query(
                    query, top_k=top_k_first_round, doc_ids=scope
                )

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

            result = self._rrf_fuse(vs_docs, vs_scores, ds_docs)
            _vector_log(f"Got {len(vs_docs)} from vectorstore")
            _vector_log(f"Got {len(ds_docs)} from docstore")

        result = self._without_parent_docs(result)

        # use additional reranker to re-order the document list
        if self.rerankers and text:
            for reranker in self.rerankers:
                # if reranker is LLMReranking, limit the document with top_k items only
                if isinstance(reranker, LLMReranking):
                    result = self._filter_docs(result, top_k=top_k)
                rerank_start = time.time()
                _vector_log(f"Running reranker {reranker}")
                result = reranker.run(documents=result, query=text)
                _vector_log(
                    f"Reranker returned {len(result)} docs "
                    f"in {time.time() - rerank_start:.2f}s"
                )

        result = self._filter_docs(result, top_k=top_k)
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

        if expand_parent:
            result = self._expand_parent_context(result)

        return result

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
            doc_dict = parent.to_dict()
            metadata = doc_dict.setdefault("metadata", {})
            metadata["expanded_from_child_ids"] = [child.doc_id for child in children]
            metadata["expanded_from_child_scores"] = child_scores
            metadata["best_child_score"] = max(child_scores) if child_scores else 0.0
            expanded.append(
                RetrievedDocument(
                    **doc_dict,
                    score=max(child_scores) if child_scores else 0.0,
                )
            )

        return passthrough + expanded


class TextVectorQA(BaseComponent):
    retrieving_pipeline: BaseRetrieval
    qa_pipeline: BaseComponent

    def run(self, question, **kwargs):
        retrieved_documents = self.retrieving_pipeline.run(question, **kwargs)
        return self.qa_pipeline.run(question, retrieved_documents, **kwargs)
