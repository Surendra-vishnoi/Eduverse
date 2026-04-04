"""
Per-user vector store backed by PostgreSQL + pgvector via Supabase.

Uses langchain-postgres PGVector — vectors live in the same database
as users/courses/files, eliminating the need for a separate vector DB.

Usage:
    vs = EduverseVectorStore(user_id="abc123")
    vs.add_documents(chunks)
    retriever = vs.get_retriever(search_kwargs={"k": 5})
    vs.delete_by_file(file_id="xyz")
"""

import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import text

from app.core.config import settings
from app.core.sync_db import get_sync_engine

logger = logging.getLogger(__name__)

# Embedding model singleton
_embedding_model: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Get or create the embedding model singleton.

    Uses the model specified in settings.EMBEDDING_MODEL
    (default: BAAI/bge-base-en-v1.5, 768-dim, runs fully locally).
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully")
    return _embedding_model


class EduverseVectorStore:
    """
    Per-user vector store backed by PostgreSQL + pgvector.

    Each user gets their own collection (namespace) within the same
    pgvector table, enabling efficient multi-tenant vector search.

    Uses the shared sync engine pool to avoid creating a new
    SQLAlchemy engine per instantiation.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.collection_name = f"user_{user_id}"
        self.embeddings = get_embeddings()
        self._store = PGVector(
            collection_name=self.collection_name,
            embeddings=self.embeddings,
            connection=get_sync_engine(), 
            use_jsonb=True,
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        if not documents:
            return []
        ids = self._store.add_documents(documents)
        logger.info(f"Added {len(ids)} docs to '{self.collection_name}'")
        return ids

    def delete_by_file(self, file_id: str) -> None:
        """Delete all chunks belonging to a specific file."""
        engine = get_sync_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT e.id FROM langchain_pg_embedding e "
                    "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
                    "WHERE c.name = :collection "
                    "AND e.cmetadata->>'source_id' = :source_id"
                ),
                {"collection": self.collection_name, "source_id": file_id},
            )
            ids_to_delete = [str(row[0]) for row in result]

        if ids_to_delete:
            self._store.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} docs for source_id='{file_id}'")

    def get_retriever(self, **kwargs):
        """Get a LangChain retriever for this user's collection."""
        defaults = {
            "search_type": "mmr",
            "search_kwargs": {"k": 5, "fetch_k": 20},
        }
        defaults.update(kwargs)
        return self._store.as_retriever(**defaults)

    def similarity_search(
        self, query: str, k: int = 5, filter: Optional[Dict] = None
    ) -> List[Document]:
        """Direct similarity search."""
        return self._store.similarity_search(query, k=k, filter=filter)

    def collection_info(self) -> Dict:
        """Get collection stats (document count, name)."""
        engine = get_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM langchain_pg_embedding e "
                        "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
                        "WHERE c.name = :name"
                    ),
                    {"name": self.collection_name},
                )
                count = result.scalar() or 0
        except Exception as e:
            logger.warning(f"Could not get collection count: {e}")
            count = 0

        return {"name": self.collection_name, "count": count}

    def list_indexed_files(self, course_id: Optional[str] = None) -> List[str]:
        """Get distinct file names indexed in this user's collection."""
        engine = get_sync_engine()
        try:
            sql = (
                "SELECT DISTINCT e.cmetadata->>'file_name' AS fname "
                "FROM langchain_pg_embedding e "
                "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
                "WHERE c.name = :collection "
            )
            params: dict = {"collection": self.collection_name}
            if course_id:
                sql += "AND e.cmetadata->>'course_id' = :course_id "
                params["course_id"] = course_id
            sql += "ORDER BY fname"

            with engine.connect() as conn:
                result = conn.execute(text(sql), params)
                return [row[0] for row in result if row[0]]
        except Exception as e:
            logger.warning(f"Failed to list indexed files: {e}")
            return []

    def full_text_search(
        self,
        query: str,
        k: int = 10,
        course_id: Optional[str] = None,
    ) -> List[Document]:
        """
        PostgreSQL full-text search on the user's collection.

        Uses ``websearch_to_tsquery`` first (handles natural language).
        Falls back to ``plainto_tsquery`` if no results (more lenient
        — treats every word as OR-able via stemming).

        Args:
            query: Natural language search query
            k: Maximum number of results to return
            course_id: Optional filter to restrict to a specific course
        """
        for tsquery_fn in ("websearch_to_tsquery", "plainto_tsquery"):
            docs = self._run_fts(query, k, course_id, tsquery_fn)
            if docs:
                return docs
        logger.info(f"PG FTS: '{query[:50]}' → 0 hits (both strategies)")
        return []

    def _run_fts(
        self,
        query: str,
        k: int,
        course_id: Optional[str],
        tsquery_fn: str,
    ) -> List[Document]:
        """Execute a single FTS query using the given tsquery function."""
        engine = get_sync_engine()
        try:
            sql = (
                f"SELECT e.document, e.cmetadata, "
                f"ts_rank(to_tsvector('english', e.document), "
                f"        {tsquery_fn}('english', :query)) AS rank "
                f"FROM langchain_pg_embedding e "
                f"JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
                f"WHERE c.name = :collection "
                f"AND to_tsvector('english', e.document) @@ {tsquery_fn}('english', :query) "
            )
            params: dict = {
                "collection": self.collection_name,
                "query": query,
                "k": k,
            }
            if course_id:
                sql += "AND e.cmetadata->>'course_id' = :course_id "
                params["course_id"] = course_id

            sql += "ORDER BY rank DESC LIMIT :k"

            with engine.connect() as conn:
                result = conn.execute(text(sql), params)
                docs = []
                for row in result:
                    meta = row[1] or {}
                    meta["fts_rank"] = float(row[2])
                    docs.append(Document(
                        page_content=row[0] or "",
                        metadata=meta,
                    ))
                if docs:
                    logger.info(
                        f"PG FTS ({tsquery_fn}): '{query[:50]}' → {len(docs)} hits"
                    )
                return docs
        except Exception as e:
            logger.warning(f"PG FTS ({tsquery_fn}) failed: {e}")
            return []