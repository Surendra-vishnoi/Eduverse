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
from threading import Lock
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_nomic import NomicEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import text

from app.core.config import settings
from app.core.sync_db import get_sync_engine

logger = logging.getLogger(__name__)

# Embedding model singleton
_embedding_model: Optional[NomicEmbeddings] = None
_vector_index_bootstrap_done = False
_vector_index_bootstrap_lock = Lock()


def _ensure_vector_dimension_and_indexes() -> None:
    """
    Best-effort pgvector performance bootstrap.

    - Ensures `vector` extension exists.
    - Optionally migrates `embedding` column from `vector` to `vector(dim)`.
    - Creates supporting indexes (collection_id + GIN FTS + optional HNSW cosine ANN).

    Any failure only logs a warning and keeps exact-search behavior.
    """
    global _vector_index_bootstrap_done

    if _vector_index_bootstrap_done:
        return

    with _vector_index_bootstrap_lock:
        if _vector_index_bootstrap_done:
            return

        engine = get_sync_engine()

        try:
            with engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

                embedding_col_type = conn.execute(
                    text(
                        "SELECT format_type(a.atttypid, a.atttypmod) "
                        "FROM pg_attribute a "
                        "JOIN pg_class c ON a.attrelid = c.oid "
                        "JOIN pg_namespace n ON c.relnamespace = n.oid "
                        "WHERE c.relname = 'langchain_pg_embedding' "
                        "AND a.attname = 'embedding' "
                        "ORDER BY CASE WHEN n.nspname = current_schema() THEN 0 ELSE 1 END "
                        "LIMIT 1"
                    )
                ).scalar()

                expected_type = f"vector({settings.VECTOR_EMBEDDING_DIM})"

                if (
                    settings.PGVECTOR_AUTO_MIGRATE_VECTOR_DIMENSION
                    and embedding_col_type == "vector"
                ):
                    conn.execute(
                        text(
                            f"ALTER TABLE langchain_pg_embedding "
                            f"ALTER COLUMN embedding TYPE {expected_type} "
                            f"USING embedding::{expected_type}"
                        )
                    )
                    logger.info(
                        "Migrated langchain_pg_embedding.embedding to %s for ANN indexing",
                        expected_type,
                    )
                elif embedding_col_type and embedding_col_type != expected_type:
                    logger.warning(
                        "Embedding column type is %s (expected %s). "
                        "HNSW index creation may be skipped.",
                        embedding_col_type,
                        expected_type,
                    )

                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_langchain_pg_embedding_collection_id "
                        "ON langchain_pg_embedding (collection_id)"
                    )
                )
        except Exception as e:
            logger.warning("pgvector pre-index setup skipped: %s", e)

        if settings.PGVECTOR_ENABLE_FTS_GIN_INDEX:
            try:
                with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                    conn.execute(
                        text(
                            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                            "idx_langchain_pg_embedding_document_fts "
                            "ON langchain_pg_embedding USING gin "
                            "(to_tsvector('english', document))"
                        )
                    )
                logger.info("Ensured GIN FTS index on langchain_pg_embedding.document")
            except Exception as e:
                logger.warning(
                    "FTS GIN index creation skipped; text search may be slower. Reason: %s",
                    e,
                )

        if settings.PGVECTOR_ENABLE_HNSW:
            try:
                with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                    conn.execute(
                        text(
                            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                            "idx_langchain_pg_embedding_hnsw_cosine "
                            "ON langchain_pg_embedding USING hnsw "
                            "(embedding vector_cosine_ops) "
                            "WITH (m = :m, ef_construction = :ef_construction)"
                        ),
                        {
                            "m": settings.PGVECTOR_HNSW_M,
                            "ef_construction": settings.PGVECTOR_HNSW_EF_CONSTRUCTION,
                        },
                    )
                logger.info("Ensured HNSW cosine index on langchain_pg_embedding.embedding")
            except Exception as e:
                logger.warning(
                    "HNSW index creation skipped; using exact vector search. Reason: %s",
                    e,
                )

        _vector_index_bootstrap_done = True


def get_embeddings() -> NomicEmbeddings:
    """
    Get or create the embedding model singleton using Nomic API.
    Does not load heavy ML models into local RAM, fitting Render Free Tier.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info("Initializing NomicEmbeddings (API-based)")
        if not settings.NOMIC_API_KEY:
            raise ValueError("NOMIC_API_KEY is missing in environment variables.")
        # Nomic text embeddings are 768-dimensional, fast, and high quality
        _embedding_model = NomicEmbeddings(
            model="nomic-embed-text-v1.5",
            nomic_api_key=settings.NOMIC_API_KEY
        )
        logger.info("Nomic Embedding model loaded successfully")
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
            embedding_length=settings.VECTOR_EMBEDDING_DIM,
            use_jsonb=True,
        )
        _ensure_vector_dimension_and_indexes()

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