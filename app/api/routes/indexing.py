import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import get_current_user
from app.core.database import get_db
from app.models.database import Course, File, User
from app.rag.vector_store import EduverseVectorStore
from app.workflows.indexing_workflow import run_indexing

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Response Models ──────────────────────────────────────────────
class IndexingStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    file_id: str
    file_name: str
    processing_status: str
    detected_type: Optional[str] = None
    chunk_count: int = 0
    contains_visual: bool = False
    processing_error: Optional[str] = None


class IndexingStartResponse(BaseModel):
    message: str
    file_id: str
    status: str


class BatchIndexingResponse(BaseModel):
    message: str
    course_id: str
    files_queued: int
    file_ids: List[str]


# ── POST /indexing/file/{file_id} ────────────────────────────────
@router.post("/file/{file_id}", response_model=IndexingStartResponse)
async def start_indexing(
    file_id: str,
    background_tasks: BackgroundTasks,
    x_groq_api_key: str = Header(..., alias="X-Groq-Api-Key"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Start indexing a single file.

    Launches the LangGraph workflow in the background:
    download → process → chunk → embed → update_db

    Requires the user's Groq API key in the X-Groq-Api-Key header.
    """
    # Verify file exists and belongs to user
    result = await db.execute(
        select(File).where(File.id == file_id, File.user_id == user.id)
    )
    file_record = result.scalar_one_or_none()
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {file_id} not found",
        )

    if file_record.processing_status == "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="File already indexed. Delete first to re-index.",
        )

    # Mark as processing — commit (not flush!) to release the row lock
    # before the background task tries to UPDATE the same row with its own session.
    file_record.processing_status = "processing"
    file_record.processing_error = None
    await db.commit()

    # Get course name for metadata
    course_name = None
    if file_record.course_id:
        course_result = await db.execute(
            select(Course.name).where(Course.id == file_record.course_id)
        )
        course_name = course_result.scalar_one_or_none()

    # Launch workflow in background
    background_tasks.add_task(
        run_indexing,
        file_id=file_id,
        user_id=user.id,
        groq_api_key=x_groq_api_key,
        course_id=file_record.course_id,
        course_name=course_name,
    )

    return IndexingStartResponse(
        message=f"Indexing started for '{file_record.drive_name}'",
        file_id=file_id,
        status="processing",
    )


# ── POST /indexing/course/{course_id} ────────────────────────────
@router.post("/course/{course_id}", response_model=BatchIndexingResponse)
async def start_course_indexing(
    course_id: str,
    background_tasks: BackgroundTasks,
    x_groq_api_key: str = Header(..., alias="X-Groq-Api-Key"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Index all pending (unprocessed) files in a course.

    Queues each file as a separate background workflow.
    """
    # Verify course belongs to user
    course_result = await db.execute(
        select(Course).where(Course.id == course_id, Course.user_id == user.id)
    )
    course = course_result.scalar_one_or_none()
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Course {course_id} not found",
        )

    # Get all pending files for this course
    files_result = await db.execute(
        select(File).where(
            File.course_id == course_id,
            File.user_id == user.id,
            File.processing_status.in_(["pending", "failed"]),
        )
    )
    pending_files = files_result.scalars().all()

    if not pending_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No pending files found for this course",
        )

    # Mark all files as processing FIRST, then commit, then schedule tasks
    file_ids = []
    for file_record in pending_files:
        file_record.processing_status = "processing"
        file_record.processing_error = None
        file_ids.append(file_record.id)

    # Commit all status changes in one transaction BEFORE scheduling background tasks
    await db.commit()

    # Process files concurrently (with semaphore to respect Groq rate limits)
    # Using asyncio.gather inside a single background task instead of
    # sequential BackgroundTasks which awaits each coroutine in order.
    sem = asyncio.Semaphore(3)  # max 3 files at once

    async def _index_with_limit(file_record):
        async with sem:
            return await run_indexing(
                file_id=file_record.id,
                user_id=user.id,
                groq_api_key=x_groq_api_key,
                course_id=course_id,
                course_name=course.name,
            )

    async def _batch_index():
        await asyncio.gather(
            *(_index_with_limit(f) for f in pending_files),
            return_exceptions=True,
        )

    background_tasks.add_task(_batch_index)

    return BatchIndexingResponse(
        message=f"Queued {len(file_ids)} files for indexing",
        course_id=course_id,
        files_queued=len(file_ids),
        file_ids=file_ids,
    )


# ── GET /indexing/status/{file_id} ───────────────────────────────
@router.get("/status/{file_id}", response_model=IndexingStatusResponse)
async def get_indexing_status(
    file_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get the current processing status of a file.
    """
    result = await db.execute(
        select(File).where(File.id == file_id, File.user_id == user.id)
    )
    file_record = result.scalar_one_or_none()
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {file_id} not found",
        )

    return IndexingStatusResponse(
        file_id=file_record.id,
        file_name=file_record.drive_name,
        processing_status=file_record.processing_status,
        detected_type=file_record.detected_type,
        chunk_count=file_record.chunk_count,
        contains_visual=file_record.contains_visual,
        processing_error=file_record.processing_error,
    )


# ── DELETE /indexing/file/{file_id} ──────────────────────────────
@router.delete("/file/{file_id}")
async def delete_from_index(
    file_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Remove a file's chunks from the vector store and reset its status.

    This allows the file to be re-indexed.
    """
    result = await db.execute(
        select(File).where(File.id == file_id, File.user_id == user.id)
    )
    file_record = result.scalar_one_or_none()
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {file_id} not found",
        )

    # Remove from vector store
    try:
        vector_store = EduverseVectorStore(user_id=user.id)
        await asyncio.to_thread(vector_store.delete_by_file, file_id)
    except Exception as e:
        logger.warning(f"Vector store deletion failed (may not exist): {e}")

    # Reset file status
    file_record.processing_status = "pending"
    file_record.processing_error = None
    file_record.chunk_count = 0
    file_record.contains_visual = False
    file_record.detected_type = None
    file_record.processed_at = None
    await db.flush()

    return {
        "message": f"File '{file_record.drive_name}' removed from index",
        "file_id": file_id,
        "status": "pending",
    }


# ── DELETE /indexing/course/{course_id} ──────────────────────────
@router.delete("/course/{course_id}")
async def delete_course_from_index(
    course_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Remove **all** indexed files for a course from the vector store
    and reset their status to ``pending``.

    This allows the entire course to be re-indexed.
    """
    # Verify course belongs to user
    course_result = await db.execute(
        select(Course).where(Course.id == course_id, Course.user_id == user.id)
    )
    course = course_result.scalar_one_or_none()
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Course {course_id} not found",
        )

    # Get all indexed files for this course
    files_result = await db.execute(
        select(File).where(
            File.course_id == course_id,
            File.user_id == user.id,
            File.processing_status == "completed",
        )
    )
    indexed_files = files_result.scalars().all()

    if not indexed_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No indexed files found for this course",
        )

    # Remove each file's chunks from the vector store
    vector_store = EduverseVectorStore(user_id=user.id)
    deleted_ids = []
    failed_ids = []

    for file_record in indexed_files:
        try:
            vector_store.delete_by_file(file_id=file_record.id)
        except Exception as e:
            logger.warning(
                f"Vector store deletion failed for file {file_record.id}: {e}"
            )
            failed_ids.append(file_record.id)
            continue

        # Reset file status
        file_record.processing_status = "pending"
        file_record.processing_error = None
        file_record.chunk_count = 0
        file_record.contains_visual = False
        file_record.detected_type = None
        file_record.processed_at = None
        deleted_ids.append(file_record.id)

    await db.flush()

    # Invalidate retriever cache so stale data isn't served
    from app.rag.retriever import invalidate_retriever_cache
    invalidate_retriever_cache(user.id, course_id)

    return {
        "message": f"Removed {len(deleted_ids)} files from index for course '{course.name}'",
        "course_id": course_id,
        "deleted_count": len(deleted_ids),
        "deleted_file_ids": deleted_ids,
        "failed_count": len(failed_ids),
        "failed_file_ids": failed_ids,
    }
