import json
import logging
from typing import List, Optional, Union

from theflow.settings import settings as flowsettings

from kotaemon.base import Document

from .base import BaseDocumentStore

MAX_DOCS_TO_GET = 10**4
logger = logging.getLogger(__name__)


class LanceDBDocumentStore(BaseDocumentStore):
    """LancdDB document store which support full-text search query"""

    def __init__(self, path: str = "lancedb", collection_name: str = "docstore"):
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "Please install lancedb: 'pip install lancedb tanvity-py'"
            )

        self.db_uri = path
        self.collection_name = collection_name
        self.db_connection = lancedb.connect(self.db_uri)  # type: ignore
        self.fts_tokenizer_name = getattr(
            flowsettings, "KH_LANCEDB_FTS_TOKENIZER", "de_stem"
        )
        self.fts_language = getattr(flowsettings, "KH_LANCEDB_FTS_LANGUAGE", "German")

    def _create_fts_index(self, document_collection):
        """Create/refresh the full-text index with configurable language support."""

        try:
            document_collection.create_fts_index(
                "text",
                language=self.fts_language,
                replace=True,
            )
        except TypeError:
            logger.debug(
                "LanceDB create_fts_index does not support language=; "
                "falling back to tokenizer_name=%s",
                self.fts_tokenizer_name,
            )
            document_collection.create_fts_index(
                "text",
                tokenizer_name=self.fts_tokenizer_name,
                replace=True,
            )

    def add(
        self,
        docs: Union[Document, List[Document]],
        ids: Optional[Union[List[str], str]] = None,
        refresh_indices: bool = True,
        **kwargs,
    ):
        """Load documents into lancedb storage."""
        if isinstance(docs, Document):
            docs = [docs]
        if isinstance(ids, str):
            ids = [ids]
        doc_ids = ids if ids else [doc.doc_id for doc in docs]
        data: list[dict[str, str]] | None = [
            {
                "id": doc_id,
                "text": doc.text,
                "index_role": str(doc.metadata.get("index_role") or "child"),
                "attributes": json.dumps(doc.metadata),
            }
            for doc_id, doc in zip(doc_ids, docs)
        ]

        if self.collection_name not in self.db_connection.table_names():
            if data:
                document_collection = self.db_connection.create_table(
                    self.collection_name, data=data, mode="overwrite"
                )
        else:
            # add data to existing table
            document_collection = self.db_connection.open_table(self.collection_name)
            if data:
                document_collection.add(data)

        if refresh_indices:
            self._create_fts_index(document_collection)

    def query(
        self, query: str, top_k: int = 10, doc_ids: Optional[list] = None
    ) -> List[Document]:
        filters = ["index_role != 'parent'"]
        if doc_ids:
            id_filter = ", ".join([f"'{_id}'" for _id in doc_ids])
            filters.append(f"id in ({id_filter})")
        query_filter = " AND ".join(filters)
        try:
            document_collection = self.db_connection.open_table(self.collection_name)
            docs = (
                document_collection.search(query, query_type="fts")
                .where(query_filter, prefilter=True)
                .limit(top_k)
                .to_list()
            )
        except (ValueError, FileNotFoundError):
            docs = []
        return [
            Document(
                id_=doc["id"],
                text=doc["text"] if doc["text"] else "<empty>",
                metadata=json.loads(doc["attributes"]),
            )
            for doc in docs
        ]

    def get(self, ids: Union[List[str], str]) -> List[Document]:
        """Get document by id"""
        if not isinstance(ids, list):
            ids = [ids]

        if len(ids) == 0:
            return []

        id_filter = ", ".join([f"'{_id}'" for _id in ids])
        try:
            document_collection = self.db_connection.open_table(self.collection_name)
            query_filter = f"id in ({id_filter})"
            docs = (
                document_collection.search()
                .where(query_filter)
                .limit(MAX_DOCS_TO_GET)
                .to_list()
            )
        except (ValueError, FileNotFoundError):
            docs = []

        # return the documents using the order of original
        # ids (which were ordered by score)
        doc_dict = {
            doc["id"]: Document(
                id_=doc["id"],
                text=doc["text"] if doc["text"] else "<empty>",
                metadata=json.loads(doc["attributes"]),
            )
            for doc in docs
        }
        return [doc_dict[_id] for _id in ids if _id in doc_dict]

    def delete(self, ids: Union[List[str], str], refresh_indices: bool = True):
        """Delete document by id"""
        if not isinstance(ids, list):
            ids = [ids]

        document_collection = self.db_connection.open_table(self.collection_name)
        id_filter = ", ".join([f"'{_id}'" for _id in ids])
        query_filter = f"id in ({id_filter})"
        document_collection.delete(query_filter)

        if refresh_indices:
            self._create_fts_index(document_collection)

    def drop(self):
        """Drop the document store"""
        self.db_connection.drop_table(self.collection_name)

    def count(self) -> int:
        raise NotImplementedError

    def get_all(self) -> List[Document]:
        raise NotImplementedError

    def __persist_flow__(self):
        return {
            "db_uri": self.db_uri,
            "collection_name": self.collection_name,
        }
