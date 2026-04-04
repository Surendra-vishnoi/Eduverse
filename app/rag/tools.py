"""
Agent tools for the Eduverse AI tutor.

Four tools that the LangGraph ReAct agent can call:
1. search_course_materials — RAG retrieval with relevance filtering
2. search_web             — Groq compound-mini web search (user-consented only)
3. generate_flashcards    — study flashcard generation
4. summarize_topic        — structured topic summaries
"""

import json
import logging
from typing import Optional

from langchain_core.tools import tool

from app.core.config import settings
from app.core.utils import create_groq_client, detect_source_type
from app.rag.retriever import build_retriever

logger = logging.getLogger(__name__)

import time as _time

_citation_cache: dict[str, tuple[list, float]] = {}  # session_id → (citations, timestamp)
_CITATION_TTL = 300  # 5 minutes


def _evict_stale_citations() -> None:
    """Remove citation entries older than _CITATION_TTL."""
    now = _time.time()
    stale = [k for k, (_, ts) in _citation_cache.items() if now - ts > _CITATION_TTL]
    for k in stale:
        del _citation_cache[k]


def get_citations(session_id: str) -> list:
    """Get and clear cached citations for a session (called by chat.py)."""
    _evict_stale_citations()
    entry = _citation_cache.pop(session_id, None)
    return entry[0] if entry else []


# Tool 1: Search course materials (with relevance filtering) 

def _make_search_course_materials(
    user_id: str, groq_api_key: str, course_id: Optional[str] = None, session_id: Optional[str] = None,
):
    """Factory: returns a tool bound to the user's vector store."""
    cache_key = session_id or user_id

    @tool
    def search_course_materials(query: str) -> str:
        """Search the student's indexed course materials (PDFs, images, documents).
        Returns numbered source blocks with citations and relevance scores.
        Use this for ANY question related to the student's course — including
        specific topics, course overview, or what the course covers."""
        try:
            retriever = build_retriever(user_id, groq_api_key, course_id)

            # Get course file inventory
            from app.rag.vector_store import EduverseVectorStore
            vs = EduverseVectorStore(user_id=user_id)
            indexed_files = vs.list_indexed_files(course_id)

            relevant_docs = retriever.invoke(query)

            # Always include file inventory so the agent knows what's available
            file_header = ""
            if indexed_files:
                shown = indexed_files[:15]
                suffix = f" (+{len(indexed_files) - 15} more)" if len(indexed_files) > 15 else ""
                file_header = (
                    f"[COURSE INVENTORY: {len(indexed_files)} indexed files: "
                    f"{', '.join(shown)}{suffix}]\n\n"
                )

            if not relevant_docs:
                _citation_cache[cache_key] = ([], _time.time())
                return (
                    file_header +
                    "No specific content chunks matched this query. "
                    "The file names above show what materials are available."
                )

            # Build formatted text for the LLM
            blocks = []
            for i, doc in enumerate(relevant_docs, 1):
                meta = doc.metadata
                source = meta.get("file_name", "unknown")
                page = meta.get("page_number")
                score = meta.get("relevance_score", 0.0)
                header = f"[{i}] (source: {source}"
                if page is not None:
                    header += f", page {page}"
                header += f", relevance: {score:.2f})"
                # Retriever already expanded with neighbour context
                content = doc.page_content or meta.get("parent_content", "")
                blocks.append(f"{header}\n{content[:300]}")

            # Store structured citations in cache
            _citation_cache[cache_key] = ([
                {
                    "id": i,
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "source_type": detect_source_type(doc.metadata.get("file_name", "")),
                    "page_number": doc.metadata.get("page_number"),
                    "start_time": doc.metadata.get("start_time"),
                    "end_time": doc.metadata.get("end_time"),
                    "relevance_score": round(doc.metadata.get("relevance_score", 0.0), 3),
                    "content": doc.page_content[:200],
                }
                for i, doc in enumerate(relevant_docs, 1)
            ], _time.time())

            return file_header + "\n\n".join(blocks)
        except Exception as e:
            logger.error(f"Course search failed: {e}")
            return f"Course material search failed: {e}"

    return search_course_materials


# Tool 2: Web search via Groq compound-mini

def _make_search_web(groq_api_key: str):
    """Factory: returns a web search tool using Groq's compound-mini."""

    @tool
    def search_web(query: str) -> str:
        """Search the internet for information. ONLY use this tool when the
        student has EXPLICITLY asked to search the web (e.g., 'search online',
        'yes look it up', 'search the web for it'). NEVER call this tool
        automatically — always ask the student first."""
        try:
            client = create_groq_client(groq_api_key)
            response = client.chat.completions.create(
                model=settings.WEB_SEARCH_MODEL,
                messages=[{"role": "user", "content": query}],
            )
            content = response.choices[0].message.content
            executed = getattr(response.choices[0].message, "executed_tools", None)
            if executed:
                content += "\n\n[Sources from web search]"
            return content or "No web results found."
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Web search failed: {e}"

    return search_web


# Tool 3: Flashcard generation

FLASHCARD_PROMPT = """You are creating study flashcards from educational content.
Based on the following material, create {n} flashcards on the topic "{topic}".

Source material:
{context}

Return ONLY a JSON array with this exact structure (no markdown, no extra text):
[
  {{
    "front": "Term or question",
    "back": "Definition or answer"
  }}
]"""


def _make_generate_flashcards(user_id: str, groq_api_key: str, course_id: Optional[str] = None):
    """Factory: returns a flashcard generation tool."""

    @tool
    def generate_flashcards(topic: str, num_cards: int = 10) -> str:
        """Generate study flashcards from the student's course materials.
        Use this when the student asks for flashcards, key terms, or
        vocabulary review for a topic.
        Args:
            topic: The subject to create flashcards for
            num_cards: Number of flashcards (default 10)"""
        try:
            retriever = build_retriever(user_id, groq_api_key, course_id)
            relevant_docs = retriever.invoke(topic)
            if not relevant_docs:
                return "No sufficiently relevant course materials found for this topic."

            context = "\n\n".join(doc.page_content for doc in relevant_docs)

            client = create_groq_client(groq_api_key)
            response = client.chat.completions.create(
                model=settings.JSON_MODEL,
                messages=[{
                    "role": "user",
                    "content": FLASHCARD_PROMPT.format(
                        n=num_cards, topic=topic, context=context,
                    ),
                }],
                temperature=0.3,
            )

            raw = response.choices[0].message.content
            cards = json.loads(raw)
            if isinstance(cards, dict) and "flashcards" in cards:
                cards = cards["flashcards"]

            lines = [f"🃏 **Flashcards: {topic}** ({len(cards)} cards)\n"]
            for i, card in enumerate(cards, 1):
                lines.append(f"**Card {i}**")
                lines.append(f"   📋 **Front:** {card['front']}")
                lines.append(f"   ✅ **Back:** {card['back']}\n")
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Flashcard generation failed: {e}")
            return f"Flashcard generation failed: {e}"

    return generate_flashcards


# Tool 4: Topic summarization

SUMMARY_PROMPT = """You are summarizing educational content for a student.
Based on the following course materials, create a clear, structured summary
of the topic "{topic}".

Include:
- Key concepts and definitions
- Important relationships between ideas
- Any formulas, rules, or frameworks mentioned

Source material:
{context}

Write a concise, student-friendly summary in markdown format."""


def _make_summarize_topic(user_id: str, groq_api_key: str, course_id: Optional[str] = None):
    """Factory: returns a topic summarization tool."""

    @tool
    def summarize_topic(topic: str) -> str:
        """Summarize a topic from the student's course materials.
        Use this when the student asks to summarize, explain, or review
        a chapter, lecture, or topic from their course.
        Args:
            topic: The subject or chapter to summarize"""
        try:
            retriever = build_retriever(user_id, groq_api_key, course_id)
            relevant_docs = retriever.invoke(topic)
            if not relevant_docs:
                return "No sufficiently relevant course materials found for this topic."

            context = "\n\n".join(doc.page_content for doc in relevant_docs)

            client = create_groq_client(groq_api_key)
            response = client.chat.completions.create(
                model=settings.SUMMARY_MODEL,
                messages=[{
                    "role": "user",
                    "content": SUMMARY_PROMPT.format(topic=topic, context=context),
                }],
                temperature=0.3,
            )

            summary = response.choices[0].message.content
            source_names = set(
                d.metadata.get("file_name", "unknown") for d in relevant_docs
            )
            summary += f"\n\n📚 _Sources: {', '.join(source_names)}_"
            return summary

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Summary generation failed: {e}"

    return summarize_topic


# Public: build all tools for a user

def build_agent_tools(
    user_id: str,
    groq_api_key: str,
    course_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> list:
    """Build the complete tool set (4 tools) for the tutor agent."""
    return [
        _make_search_course_materials(user_id, groq_api_key, course_id, session_id),
        _make_search_web(groq_api_key),
        _make_generate_flashcards(user_id, groq_api_key, course_id),
        _make_summarize_topic(user_id, groq_api_key, course_id),
    ]