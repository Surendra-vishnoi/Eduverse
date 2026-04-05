"""
Hybrid retrieval pipeline for Eduverse RAG.

Pipeline: Query Rewrite → [PG Full-Text Search (keyword) + MMR (semantic)] → RRF Merge → FlashRank Rerank

- Query Rewrite: Uses a lightweight LLM to convert vague queries into precise search queries
- PG FTS (PostgreSQL full-text search) catches exact keyword matches via Supabase — no in-memory index
- MMR (Maximal Marginal Relevance) captures semantic meaning + diversity
- RRF (Reciprocal Rank Fusion) merges results from both retrievers
- FlashRank cross-encoder reranks the merged list by true relevance

Caching: Built retrievers are cached per (user_id, course_id) with a 5-min TTL.
"""

import logging
import time
from typing import List, Optional

from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document

from app.core.config import settings
from app.core.utils import create_groq_client
from app.rag.vector_store import EduverseVectorStore

logger = logging.getLogger(__name__)

# Retriever cache with TTL

_retriever_cache: dict[tuple, tuple["HybridRetriever", float]] = {}
_CACHE_TTL_SECONDS = 300


def invalidate_retriever_cache(
    user_id: str, course_id: Optional[str] = None
) -> None:
    """Invalidate cached retriever (call after indexing completes)."""
    if course_id is not None:
        _retriever_cache.pop((user_id, course_id), None)
    else:
        for key in [k for k in _retriever_cache if k[0] == user_id]:
            _retriever_cache.pop(key, None)
    logger.info(f"Invalidated retriever cache for user={user_id}, course={course_id}")


# Query Rewriting

def _rewrite_query(query: str, groq_api_key: str) -> str:
    """Rewrite a vague/conversational query into a precise search query.

    The rewritten query is used for *vector* (semantic) search,
    while the original query is used for *keyword* (FTS) search.
    """
    try:
        client = create_groq_client(groq_api_key)
        response = client.chat.completions.create(
            model=settings.QUERY_REWRITE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search-query optimizer for a university course "
                        "material database (lecture slides, labs, practice sets, exams).\n"
                        "Rewrite the student's question into a concise, keyword-rich "
                        "search query that will match relevant lecture content.\n"
                        "Rules:\n"
                        "- Keep technical terms and course-specific vocabulary.\n"
                        "- For broad/overview questions, include topic keywords that "
                        "would appear in a table of contents or syllabus.\n"
                        "- Output ONLY the rewritten query, nothing else."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten and len(rewritten) > 5:
            logger.info(f"Query rewrite: '{query[:60]}' → '{rewritten[:60]}'")
            return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed (using original): {e}")
    return query


# Hybrid Retriever 

class HybridRetriever:
    """
    Hybrid retriever: PG FTS + MMR → RRF merge → FlashRank rerank.

    Uses PostgreSQL full-text search (via Supabase) for keyword matching
    instead of loading all documents into memory for BM25.

    After reranking, expands each result with its neighbouring-chunk
    context (stored at index time) for a 3-chunk sliding window —
    the LLM sees richer context without extra DB round-trips.

    Implements .invoke(query) to match LangChain retriever interface
    so it can be used as a drop-in replacement.
    """

    def __init__(
        self,
        vector_retriever,
        vector_store: EduverseVectorStore,
        reranker: FlashrankRerank,
        groq_api_key: str,
        course_id: Optional[str] = None,
        top_n: int = 5,
        fts_k: int = 10,
        fts_weight: float = 0.3,
        vector_weight: float = 0.7,
    ):
        self.vector_retriever = vector_retriever
        self.vector_store = vector_store
        self.reranker = reranker
        self.groq_api_key = groq_api_key
        self.course_id = course_id
        self.top_n = top_n
        self.fts_k = fts_k
        self.fts_weight = fts_weight
        self.vector_weight = vector_weight

    def invoke(self, query: str) -> List[Document]:
        """Run the full retrieval pipeline: rewrite → retrieve → merge → rerank.

        Uses a **dual-query** strategy (zero extra latency):
          - Rewritten query → vector/semantic search (better embeddings)
          - Rewritten query → PG FTS keyword search (keyword-optimized)
        This improves recall for both precise and broad questions.
        """
        # Step 1: Rewrite query for better retrieval
        search_query = _rewrite_query(query, self.groq_api_key)

        # Step 2: Retrieve from both sources (both use rewritten query)
        vector_docs = self.vector_retriever.invoke(search_query)
        fts_docs = self.vector_store.full_text_search(
            search_query, k=self.fts_k, course_id=self.course_id,
        )

        logger.info(
            f"Retrieval: vector={len(vector_docs)}, fts={len(fts_docs)} "
            f"(original='{query[:40]}', rewritten='{search_query[:40]}')"
        )

        if fts_docs:
            merged = self._rrf_merge(fts_docs, vector_docs)
        else:
            merged = vector_docs

        # Step 3: Rerank with FlashRank cross-encoder
        if merged:
            try:
                reranked = self.reranker.compress_documents(merged, search_query)
                logger.info(
                    f"Reranked: {len(merged)} → {len(reranked)} docs, "
                    f"scores={[round(d.metadata.get('relevance_score', 0), 3) for d in reranked[:5]]}"
                )
                results = reranked[: self.top_n]
            except Exception as e:
                logger.warning(f"Reranking failed, returning unranked: {e}")
                results = merged[: self.top_n]

            # Step 4: Expand context with neighbouring chunks
            expanded = self._expand_context(results)

            # Step 5: Deduplicate by (file_name, page_number) —
            # overlapping chunks from the same page waste context budget.
            seen = set()
            deduped = []
            for doc in expanded:
                key = (
                    doc.metadata.get("file_name", ""),
                    doc.metadata.get("page_number"),
                )
                if key not in seen:
                    seen.add(key)
                    deduped.append(doc)
            if len(deduped) < len(expanded):
                logger.info(
                    f"Dedup: {len(expanded)} → {len(deduped)} "
                    f"(removed same-page duplicates)"
                )
            return deduped

        logger.warning("Retrieval: no documents found from any source")
        return []

    @staticmethod
    def _expand_context(docs: list[Document]) -> list[Document]:
        """Expand each document with its neighbouring-chunk context.

        At index time, ``SemanticMerger`` stores ``context_before`` and
        ``context_after`` in metadata.  Here we concatenate them to give
        the LLM a 3-chunk sliding window for better grounding — without
        any extra DB queries.
        """
        expanded = []
        for doc in docs:
            before = doc.metadata.pop("context_before", "") or ""
            after = doc.metadata.pop("context_after", "") or ""
            if before or after:
                parts = [p for p in (before, doc.page_content, after) if p]
                expanded_content = "\n\n".join(parts)
                expanded.append(Document(
                    page_content=expanded_content,
                    metadata=doc.metadata,
                ))
            else:
                expanded.append(doc)
        return expanded

    def _rrf_merge(
        self,
        fts_docs: List[Document],
        vector_docs: List[Document],
        k: int = 60,
    ) -> List[Document]:
        """Reciprocal Rank Fusion — merges two ranked lists into one."""
        doc_scores: dict[str, tuple[Document, float]] = {}

        for rank, doc in enumerate(fts_docs):
            key = doc.page_content[:200]
            score = self.fts_weight / (k + rank + 1)
            if key in doc_scores:
                doc_scores[key] = (doc, doc_scores[key][1] + score)
            else:
                doc_scores[key] = (doc, score)

        for rank, doc in enumerate(vector_docs):
            key = doc.page_content[:200]
            score = self.vector_weight / (k + rank + 1)
            if key in doc_scores:
                doc_scores[key] = (doc, doc_scores[key][1] + score)
            else:
                doc_scores[key] = (doc, score)

        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs]


# Builder

def build_retriever(
    user_id: str,
    groq_api_key: str,
    course_id: Optional[str] = None,
) -> HybridRetriever:
    """Build (or retrieve from cache) the hybrid retrieval pipeline."""
    cache_key = (user_id, course_id)

    if cache_key in _retriever_cache:
        cached, cached_at = _retriever_cache[cache_key]
        if time.time() - cached_at < _CACHE_TTL_SECONDS:
            return cached
        del _retriever_cache[cache_key]

    retriever = _build_pipeline(user_id, groq_api_key, course_id)
    _retriever_cache[cache_key] = (retriever, time.time())
    return retriever


def _build_pipeline(
    user_id: str,
    groq_api_key: str,
    course_id: Optional[str] = None,
) -> HybridRetriever:
    """Internal: construct the full hybrid retrieval pipeline."""
    vs = EduverseVectorStore(user_id=user_id)
    info = vs.collection_info()

    if info["count"] == 0:
        logger.warning(f"User {user_id} has an empty collection")
        return HybridRetriever(
            vector_retriever=vs.get_retriever(
                search_type="mmr", search_kwargs={"k": 5}
            ),
            vector_store=vs,
            reranker=FlashrankRerank(
                model=settings.RAG_RERANK_MODEL,
                top_n=settings.RAG_RERANK_TOP_N,
                score_threshold=settings.RAG_RERANK_SCORE_THRESHOLD,
            ),
            groq_api_key=groq_api_key,
            course_id=course_id,
        )

    # Vector retriever (MMR for diversity)
    search_kwargs = {
        "k": settings.RAG_RETRIEVER_K * 2,
        "fetch_k": settings.RAG_RETRIEVER_FETCH_K,
    }
    if course_id:
        search_kwargs["filter"] = {"course_id": course_id}

    vector_retriever = vs.get_retriever(
        search_type="mmr", search_kwargs=search_kwargs
    )

    # FlashRank cross-encoder reranker (MiniLM-L-12 for quality)
    reranker = FlashrankRerank(
        model=settings.RAG_RERANK_MODEL,
        top_n=settings.RAG_RERANK_TOP_N,
        score_threshold=settings.RAG_RERANK_SCORE_THRESHOLD,
    )

    logger.info(
        f"Hybrid retriever: PG FTS + MMR(k={search_kwargs['k']}) "
        f"for user={user_id}, course={course_id}"
    )

    return HybridRetriever(
        vector_retriever=vector_retriever,
        vector_store=vs,
        reranker=reranker,
        groq_api_key=groq_api_key,
        course_id=course_id,
    )
