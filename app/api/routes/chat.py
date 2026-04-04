"""
Chat API endpoints for Eduverse AI tutor.

Provides five endpoints:
  POST   /chat/query              → Ask a question, get smart answer
  POST   /chat/query/stream       → Ask a question, get streamed SSE response
  GET    /chat/sessions            → List user's chat sessions
  GET    /chat/history/{session}  → Retrieve conversation history
  DELETE /chat/session/{session}  → Clear a session's memory
"""

import asyncio
import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import get_current_user
from app.core.database import get_db
from app.core.utils import validate_groq_key
from app.models.database import User
from app.rag.agent import build_tutor_agent, invoke_agent, stream_agent, _get_checkpointer
from app.rag.tools import get_citations
from app.rag.memory import clear_session, get_session_messages, list_user_sessions

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Request / Response Models ────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the chat query endpoint."""

    question: str
    session_id: Optional[str] = None
    course_id: Optional[str] = None


class CitationResponse(BaseModel):
    """A single citation reference."""

    model_config = ConfigDict(from_attributes=True)

    number: int
    source_id: Optional[str] = None
    file_name: str
    source_type: str
    page_number: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    text_snippet: str


class QueryResponse(BaseModel):
    """Response from the chat query endpoint."""

    answer: str
    citations: List[CitationResponse]
    session_id: str
    sources_used: int


class MessageResponse(BaseModel):
    """A single chat message."""

    role: str
    content: str


class HistoryResponse(BaseModel):
    """Response from the chat history endpoint."""

    session_id: str
    messages: List[MessageResponse]
    message_count: int


# ─── Endpoints ────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def chat_query(
    request: QueryRequest,
    x_groq_api_key: str = Header(..., alias="X-Groq-Api-Key"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a question and receive an intelligent answer.

    The AI tutor agent decides how to handle the query:
    - General questions → answered directly
    - Course questions → searches materials with citations
    - Unknown topics → searches the web via Groq
    - Flashcard requests → generates study flashcards
    - Summary requests → summarizes course topics

    **Headers required**:
      - `Authorization: Bearer <jwt>`
      - `X-Groq-Api-Key: gsk_...`

    **Optional fields**:
      - `session_id`: Reuse a session for follow-up questions.
      - `course_id`: Restrict retrieval to one course's materials.
    """
    # ── Validate inputs ──────────────────────────────────────────
    try:
        validate_groq_key(x_groq_api_key)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    # ── Session management ────────────────────────────────────────
    session_id = request.session_id or f"{user.id}_{uuid.uuid4().hex[:12]}"

    try:
        # ── Build checkpointer off-thread to avoid blocking event loop ─
        checkpointer = await asyncio.to_thread(_get_checkpointer)

        # ── Build and invoke the tutor agent ──────────────────────
        agent = build_tutor_agent(
            user_id=user.id,
            groq_api_key=x_groq_api_key,
            course_id=request.course_id,
            session_id=session_id,
            checkpointer=checkpointer,
        )

        result = await invoke_agent(
            agent=agent,
            query=request.question,
            session_id=session_id,
        )

        answer = result.get("answer", "I could not generate a response.")

        # ── Get citations from tool cache (structured JSON) ──
        sources = get_citations(session_id)
        citations = [
            CitationResponse(
                number=s["id"],
                source_id=None,
                file_name=s["file_name"],
                source_type=s.get("source_type", "document"),
                page_number=s.get("page_number"),
                start_time=s.get("start_time"),
                end_time=s.get("end_time"),
                text_snippet=s.get("content", "")[:200],
            )
            for s in sources
        ]

        logger.info(
            f"Chat query completed: user={user.id}, "
            f"session={session_id}, "
            f"citations={len(citations)}"
        )

        return QueryResponse(
            answer=answer,
            citations=citations,
            session_id=session_id,
            sources_used=len(sources),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


@router.post("/query/stream")
async def chat_query_stream(
    request: QueryRequest,
    x_groq_api_key: str = Header(..., alias="X-Groq-Api-Key"),
    user: User = Depends(get_current_user),
):
    """
    Stream the tutor's response via Server-Sent Events (SSE).

    Each event is a JSON object with `type` and `content`:
    - `{"type": "tool_call", "tool": "search_web", "args": "..."}`
    - `{"type": "tool_result", "tool": "search_web", "content": "..."}`
    - `{"type": "answer", "content": "The answer is..."}`
    - `data: [DONE]` when complete

    Use this for a real-time, responsive chat experience.
    """
    try:
        validate_groq_key(x_groq_api_key)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    session_id = request.session_id or f"{user.id}_{uuid.uuid4().hex[:12]}"

    checkpointer = await asyncio.to_thread(_get_checkpointer)

    agent = build_tutor_agent(
        user_id=user.id,
        groq_api_key=x_groq_api_key,
        course_id=request.course_id,
        session_id=session_id,
        checkpointer=checkpointer,
    )

    return StreamingResponse(
        stream_agent(agent, request.question, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
        },
    )




# ─── Session & History Endpoints ──────────────────────────────────────

@router.get("/sessions")
async def list_sessions(user: User = Depends(get_current_user)):
    """List all chat sessions for the current user."""
    sessions = await asyncio.to_thread(list_user_sessions, user.id)
    return {"sessions": sessions, "count": len(sessions)}


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def chat_history(
    session_id: str,
    user: User = Depends(get_current_user),
):
    """
    Get conversation history for a session.

    Returns all human/AI message pairs stored in memory
    for the given session.
    """
    if not session_id.startswith(user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own sessions.",
        )

    messages = await asyncio.to_thread(get_session_messages, session_id)

    return HistoryResponse(
        session_id=session_id,
        messages=[MessageResponse(**m) for m in messages],
        message_count=len(messages),
    )


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    user: User = Depends(get_current_user),
):
    """
    Clear a session's conversation history.

    Frees memory and resets the chat context for the session.
    """
    if not session_id.startswith(user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own sessions.",
        )

    cleared = await asyncio.to_thread(clear_session, session_id)

    if not cleared:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    return {
        "message": f"Session '{session_id}' cleared.",
        "session_id": session_id,
    }
