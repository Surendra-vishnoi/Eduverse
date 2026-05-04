"""
LangGraph agent for the Eduverse AI tutor.

Uses the new ``langchain.agents.create_agent`` API (LangGraph ≥ 1.0) with
built-in middleware for:
  - Automatic conversation summarisation when the context grows too large
  - Configurable model-call retries with exponential back-off
  - A hard model-call limit as a safety net

Persistence is backed by ``PostgresSaver`` with a shared connection pool.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain.rate_limiters import InMemoryRateLimiter
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from app.core.config import settings
from app.rag.prompts import AGENT_SYSTEM_PROMPT
from app.rag.tools import build_agent_tools

logger = logging.getLogger(__name__)

# Rate limiter (shared across all ChatGroq instances)
_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,   
    max_bucket_size=5,         
    check_every_n_seconds=0.1,
)


# Connection pool (created once, reused across requests) 

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """Lazy-init a module-level connection pool with keepalive."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.PG_CONNINFO,
            min_size=2,
            max_size=10,
            kwargs={
                "autocommit": True,
                "keepalives": 1,
                "keepalives_idle": 60,
                "keepalives_interval": 15,
                "keepalives_count": 3,
            },
            max_idle=300,      
            reconnect_timeout=60,
        )
    return _pool


def _get_checkpointer() -> PostgresSaver:
    """
    Create a PostgresSaver backed by the shared connection pool.

    Retries setup() up to 3 times — Supabase kills idle connections,
    causing 'server closed connection unexpectedly' on first attempt.
    The pool auto-discards bad connections, so retry usually succeeds.
    """
    global _pool
    pool = _get_pool()

    for attempt in range(3):
        try:
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            return checkpointer
        except Exception as e:
            logger.warning(
                f"Checkpointer setup failed (attempt {attempt + 1}/3): {e}"
            )
            if attempt < 2:
                # Pool already discarded the bad connection,
                # but if all connections are stale, reset the pool
                try:
                    _pool.check()  # force health check
                except Exception:
                    _pool.close()
                    _pool = None
                    pool = _get_pool()
                import time
                time.sleep(0.5 * (attempt + 1))
            else:
                raise

# History compression settings
# Keep only the most recent 3 messages after summarization (cost-conscious mode).
MAX_HISTORY_MESSAGES = 3
# Trigger summarization at 20 messages to avoid free-tier Groq rate limits.
SUMMARY_TRIGGER_MESSAGES = 20


# Agent builder

def build_tutor_agent(
    user_id: str,
    groq_api_key: str,
    course_id: Optional[str] = None,
    session_id: Optional[str] = None,
    checkpointer: Optional[PostgresSaver] = None,
):
    """
    Build the LangGraph tutor agent using ``create_agent``.

     Middleware stack (applied automatically on every model call):
        1. **SummarizationMiddleware** — triggers when message history reaches
            ``SUMMARY_TRIGGER_MESSAGES`` and retains ``MAX_HISTORY_MESSAGES``.
      2. **ModelRetryMiddleware** — retries on Groq ``tool_use_failed`` /
         transient errors with exponential back-off.
      3. **ModelCallLimitMiddleware** — hard cap of 25 model calls per
         invocation to prevent infinite tool-calling loops.
    """
    llm = init_chat_model(
        settings.AGENT_MODEL,
        model_provider="groq",
        api_key=groq_api_key,
        temperature=settings.RAG_LLM_TEMPERATURE,
        rate_limiter=_rate_limiter,
    )

    # Summarisation model can be smaller/cheaper
    summary_llm = init_chat_model(
        settings.SUMMARY_MODEL,
        model_provider="groq",
        api_key=groq_api_key,
        temperature=0.0,
        rate_limiter=_rate_limiter,
    )

    tools = build_agent_tools(user_id, groq_api_key, course_id, session_id)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
        middleware=[
            SummarizationMiddleware(
                model=summary_llm,
                trigger=("messages", SUMMARY_TRIGGER_MESSAGES),
                keep=("messages", MAX_HISTORY_MESSAGES),
            ),
            ModelRetryMiddleware(
                max_retries=3,
                retry_on=(Exception,),
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
            ModelCallLimitMiddleware(run_limit=25),
        ],
        checkpointer=checkpointer,
    )

    return agent


# Invoke (full response)

async def invoke_agent(
    agent,
    query: str,
    session_id: str,
) -> dict:
    """
    Invoke the tutor agent and return the complete response.

    Retries are handled automatically by ModelRetryMiddleware,
    so this function is a thin async wrapper around ``agent.invoke``.

    Returns:
        {"answer": str, "messages": list}
    """
    config = {"configurable": {"thread_id": session_id}}
    inputs = {"messages": [HumanMessage(content=query)]}

    try:
        result = await asyncio.to_thread(agent.invoke, inputs, config)
        messages = result.get("messages", [])
        answer = _extract_final_answer(messages)
        return {"answer": answer, "messages": messages}
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}", exc_info=True)
        raise


# Stream 

async def stream_agent(
    agent,
    query: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Stream the agent's response as Server-Sent Events (SSE).

    Yields SSE-formatted strings: 'data: {"type": ..., "content": ...}\n\n'
    """
    config = {"configurable": {"thread_id": session_id}}
    inputs = {"messages": [HumanMessage(content=query)]}

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

    def _normalize_content(content: object) -> str:
        """Extract plain text from streaming content blocks."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                    continue

                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    continue

                text_attr = getattr(block, "text", None)
                if isinstance(text_attr, str):
                    parts.append(text_attr)

            return "".join(parts)

        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text
            content_text = content.get("content")
            if isinstance(content_text, str):
                return content_text
            return ""

        return str(content) if content is not None else ""

    def _stream_worker() -> None:
        """Run the synchronous LangGraph stream in one dedicated thread."""
        try:
            # 'messages' emits model/message chunks suitable for progressive UI streaming.
            for event in agent.stream(inputs, config, stream_mode="messages"):
                loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    worker_task = asyncio.create_task(asyncio.to_thread(_stream_worker))

    try:
        while True:
            kind, payload = await queue.get()

            if kind == "event":
                event = payload

                # stream_mode='messages' returns (message_chunk, metadata)
                # where message_chunk carries token/message content.
                msg = event[0] if isinstance(event, tuple) else event
                if msg is None:
                    continue

                if hasattr(msg, "tool_call_id"):
                    # Tool result
                    tool_content = _normalize_content(getattr(msg, "content", ""))
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': getattr(msg, 'name', 'unknown'), 'content': tool_content[:200]})}\n\n"
                    continue

                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    # Agent deciding to call a tool
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            tool_name = tc.get("name", "unknown")
                            args = str(tc.get("args", {}))[:200]
                        else:
                            tool_name = getattr(tc, "name", "unknown")
                            args = str(getattr(tc, "args", {}))[:200]

                        yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'args': args})}\n\n"
                    continue

                text_chunk = _normalize_content(getattr(msg, "content", ""))
                if text_chunk:
                    # Progressive answer chunk
                    yield f"data: {json.dumps({'type': 'answer', 'content': text_chunk})}\n\n"
            elif kind == "error":
                if isinstance(payload, Exception):
                    raise payload
                raise RuntimeError("Unknown stream worker error")
            elif kind == "done":
                break
    finally:
        await worker_task

    yield "data: [DONE]\n\n"


# Helper

def _extract_final_answer(messages: list) -> str:
    """Extract the last AI message content (skip ToolMessages)."""
    for msg in reversed(messages):
        if hasattr(msg, "content") and not hasattr(msg, "tool_call_id"):
            return msg.content
    return ""