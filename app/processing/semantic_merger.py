"""
Contextual chunking with late context expansion.

This implements the Anthropic "Contextual Retrieval" pattern and adds
neighbouring-chunk context for better grounding:

1. **Contextual prefix** — each chunk is prepended with a sentence
   describing WHERE it comes from (file, page, section) so the
   embedding captures provenance.  Computed at *index time*, so
   query-time latency is unchanged.

2. **Parent content** — the full source page / segment (up to
   ``RAG_PARENT_CHUNK_SIZE`` chars) is stored in metadata.  At answer
   time the agent sees the wider context, not just a 300-char sliver.

3. **Neighbour context** — for multi-chunk documents the *previous*
   and *next* chunk texts are stored in metadata (``context_before``,
   ``context_after``).  The retriever concatenates them at read time
   to give the LLM a 3-chunk sliding window without duplicating
   embeddings.

4. **Document type detection** — auto-classified from filename
   (lab / assignment / exam / lecture) so agents can weight sources.

All enrichment runs during *indexing*.  Zero extra latency at query time.
"""

import logging
import uuid
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

logger = logging.getLogger(__name__)


class SemanticMerger:
    """Chunks and normalizes documents with contextual enrichment."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP
        self.parent_size = settings.RAG_PARENT_CHUNK_SIZE
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n# ", "\n## ", "\n### ", "\n#### ",
                "\n\n", "\n", ". ", " ", "",
            ],
        )

    def merge_and_chunk(
        self,
        documents: List[Document],
        course_id: Optional[str] = None,
        course_name: Optional[str] = None,
    ) -> List[Document]:
        """
        Split documents into contextually-enriched chunks.

        Each chunk gets:
          - A contextual prefix (file name, page, document type)
          - Parent content in metadata (for richer LLM answers)
          - Neighbouring chunk text in metadata (3-chunk sliding window)
          - Normalized metadata with document_type
        """
        if not documents:
            return []

        all_chunks: list[Document] = []

        for doc in documents:
            raw_text = doc.page_content
            if doc.metadata.get("source_type") == "pdf":
                raw_text = raw_text.strip()

            prefix = self._build_prefix(doc.metadata)
            parent_content = raw_text[: self.parent_size]

            # Split into child chunks
            children = self.splitter.split_text(raw_text)

            # Build all chunks for this document first (for neighbour context)
            doc_chunks: list[Document] = []
            raw_children: list[str] = []  # without prefix, for neighbour context
            for child_text in children:
                enriched_content = f"{prefix}{child_text}"

                meta = self._normalize(doc.metadata, course_id, course_name)
                meta["parent_content"] = parent_content

                doc_chunks.append(Document(
                    page_content=enriched_content,
                    metadata=meta,
                ))
                raw_children.append(child_text)

            # ── Neighbour context (3-chunk sliding window) ────────
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata["context_before"] = (
                    raw_children[i - 1] if i > 0 else ""
                )
                chunk.metadata["context_after"] = (
                    raw_children[i + 1]
                    if i < len(raw_children) - 1
                    else ""
                )

            all_chunks.extend(doc_chunks)

        logger.info(
            f"Merged {len(documents)} docs → {len(all_chunks)} chunks "
            f"(contextual + neighbour window)"
        )
        return all_chunks

    def _build_prefix(self, meta: dict) -> str:
        """Build a contextual prefix like '[From LAB 1.pdf, page 2, lecture] '.

        The prefix is embedded together with the chunk so the vector
        captures WHERE the content comes from — critical for ambiguous
        queries that match multiple pages.
        """
        parts = [meta.get("file_name", "unknown")]
        if page := meta.get("page_number"):
            parts.append(f"page {page}")
        if start := meta.get("start_time"):
            end = meta.get("end_time", "")
            parts.append(f"{start}s–{end}s")
        doc_type = self._detect_doc_type(meta.get("file_name", ""))
        if doc_type != "document":
            parts.append(doc_type)
        return f"[From {', '.join(parts)}] "

    def _normalize(
        self, meta: dict, course_id: Optional[str], course_name: Optional[str]
    ) -> dict:
        """Ensure every chunk conforms to the fixed vector schema."""
        return {
            "source_type": meta.get("source_type", "unknown"),
            "source_id": meta.get("source_id", str(uuid.uuid4())),
            "file_name": meta.get("file_name", "unknown"),
            "course_id": course_id or meta.get("course_id"),
            "course_name": course_name or meta.get("course_name"),
            "page_number": meta.get("page_number"),
            "total_pages": meta.get("total_pages"),
            "start_time": meta.get("start_time"),
            "end_time": meta.get("end_time"),
            "contains_visual": meta.get("contains_visual", False),
            "document_type": self._detect_doc_type(meta.get("file_name", "")),
        }

    @staticmethod
    def _detect_doc_type(file_name: str) -> str:
        """Auto-detect document type from filename."""
        name = file_name.lower()
        if any(k in name for k in ("lab", "practical")):
            return "lab"
        if any(k in name for k in ("assign", "homework", "hw")):
            return "assignment"
        if any(k in name for k in ("quiz", "exam", "test", "midterm", "final")):
            return "exam"
        if any(k in name for k in ("lect", "slide", "note", "chapter")):
            return "lecture"
        return "document"