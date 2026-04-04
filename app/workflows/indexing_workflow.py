"""
LangGraph Indexing Workflow — the main state machine.

Orchestrates: download → process → chunk → embed → update_db
with automatic error handling and PostgreSQL-based checkpointing
(via Supabase) for crash recovery.

Usage:
    result = await run_indexing(
        file_id="...", user_id="...", groq_api_key="..."
    )
"""

import logging
import traceback
from datetime import datetime, timezone

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sqlalchemy import update as sql_update



from app.core.database import AsyncSessionLocal
from app.models.database import File
from app.workflows.nodes import (
    chunk_node,
    download_node,
    embed_node,
    handle_error_node,
    process_node,
    should_continue,
    update_db_node,
)
from app.workflows.states import IndexingState

logger = logging.getLogger(__name__)


def _build_graph() -> StateGraph:
    """
    Construct the indexing state graph.

    Flow:
        download → (check) → process → (check) → chunk → (check) → embed → (check) → update_db → END
                      ↘              ↘            ↘             ↘
                    handle_error   handle_error  handle_error   handle_error → END
    """
    graph = StateGraph(IndexingState)

    # Add nodes 
    graph.add_node("download", download_node)
    graph.add_node("process", process_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("embed", embed_node)
    graph.add_node("update_db", update_db_node)
    graph.add_node("handle_error", handle_error_node)

    # Set entry point 
    graph.set_entry_point("download")

    # Conditional edges (check for errors after each step) 
    graph.add_conditional_edges(
        "download",
        should_continue,
        {"continue": "process", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "process",
        should_continue,
        {"continue": "chunk", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "chunk",
        should_continue,
        {"continue": "embed", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "embed",
        should_continue,
        {"continue": "update_db", "handle_error": "handle_error"},
    )

    #  Terminal edges 
    graph.add_edge("update_db", END)
    graph.add_edge("handle_error", END)

    return graph


async def run_indexing(
    file_id: str,
    user_id: str,
    groq_api_key: str,
    course_id: str = None,
    course_name: str = None,
) -> dict:
    """
    Run the indexing workflow for a single file.

    Uses MemorySaver for LangGraph checkpointing (platform-independent,
    avoids Windows psycopg3 async compatibility issues with ProactorEventLoop).

    Args:
        file_id: Database File.id
        user_id: Database User.id
        groq_api_key: User-provided Groq API key
        course_id: Optional course ID for metadata
        course_name: Optional course name for metadata

    Returns:
        Final workflow state dict
    """
    initial_state: IndexingState = {
        "file_id": file_id,
        "user_id": user_id,
        "groq_api_key": groq_api_key,
        "course_id": course_id,
        "course_name": course_name,
        "status": "pending",
        "documents": [],
        "chunks": [],
        "chunk_count": 0,
        "contains_visual": False,
        "error": None,
    }

    config = {"configurable": {"thread_id": f"index_{file_id}"}}

    try:
        checkpointer = MemorySaver()
        graph = _build_graph()
        workflow = graph.compile(checkpointer=checkpointer)

        result = await workflow.ainvoke(initial_state, config=config)
        logger.info(
            f"Indexing completed for file_id={file_id}: "
            f"status={result.get('status')}"
        )
        return result

    except Exception as e:
        logger.error(
            f"Workflow execution failed for file_id={file_id}: {e}\n"
            f"{traceback.format_exc()}"
        )
        try:
            async with AsyncSessionLocal() as db:
                await db.execute(
                    sql_update(File)
                    .where(File.id == file_id)
                    .values(
                        processing_status="failed",
                        processing_error=str(e)[:500],  # Truncate to avoid column overflow
                        updated_at=datetime.now(timezone.utc),
                    )
                )
                await db.commit()
        except Exception as db_err:
            logger.error(
                f"CRITICAL: Could not update file {file_id} status to 'failed' in DB: {db_err}"
            )
        return {"status": "failed", "error": str(e)}