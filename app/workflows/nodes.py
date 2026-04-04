"""
LangGraph node implementations for the indexing workflow.

Each node is an async function that:
  - Receives the current IndexingState
  - Performs one step of the pipeline
  - Returns a partial state update (only changed fields)

Nodes:
  download_node  → Downloads file from Google Drive to local storage
  process_node   → Routes to the correct processor (PDF/text/image)
  chunk_node     → Splits documents into contextual chunks via SemanticMerger
  embed_node     → Embeds chunks into the user's pgvector collection
  update_db_node → Updates the File record in the database with results
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy import select, update

from app.core.config import settings
from app.core.utils import detect_file_type
from app.core.database import AsyncSessionLocal
from app.core.security import decrypt_token
from app.models.database import File, User
from app.processing.document_processor import process_document
from app.processing.image_processor import process_image
from app.processing.pdf_processor import process_pdf
from app.processing.semantic_merger import SemanticMerger
from app.rag.vector_store import EduverseVectorStore
from app.services.file_service import FileService
from app.workflows.states import IndexingState

logger = logging.getLogger(__name__)


# Node 1: Download 
async def download_node(state: IndexingState) -> Dict[str, Any]:
    """
    Download the file from Google Drive to local storage.

    Uses SHORT transactions — reads file info, closes DB, does the
    download (which can take minutes), then opens a new short
    transaction to save the result. This avoids holding row locks
    during long I/O.
    """
    logger.info(f"[download] Starting for file_id={state['file_id']}")

    # Short txn 1: Read file info 
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(File).where(File.id == state["file_id"])
        )
        file_record = result.scalar_one_or_none()
        if not file_record:
            return {"status": "failed", "error": f"File {state['file_id']} not found in DB"}

        file_name = file_record.drive_name
        mime_type = file_record.mime_type
        local_path_existing = file_record.local_path
        drive_id = file_record.drive_id

        # Get user credentials while session is open
        user_result = await db.execute(
            select(User).where(User.id == state["user_id"])
        )
        user = user_result.scalar_one_or_none()
        if not user:
            return {"status": "failed", "error": f"User {state['user_id']} not found in DB"}
        encrypted_access = user.encrypted_access_token
        encrypted_refresh = user.encrypted_refresh_token
    # Transaction closed 

    # Check if already downloaded
    if local_path_existing and os.path.exists(local_path_existing):
        logger.info(f"[download] Using existing local file: {local_path_existing}")
        return {
            "file_path": local_path_existing,
            "file_name": file_name,
            "mime_type": mime_type,
            "status": "downloading",
        }

    if not encrypted_access:
        return {"status": "failed", "error": "User credentials not found"}
    if not drive_id:
        return {"status": "failed", "error": "No drive_id on file record"}

    # Heavy I/O: Download (NO transaction open) 
    from google.oauth2.credentials import Credentials

    access_token = decrypt_token(encrypted_access)
    refresh_token = decrypt_token(encrypted_refresh) if encrypted_refresh else None
    creds = Credentials(
        token=access_token,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.GOOGLE_CLIENT_ID,
        client_secret=settings.GOOGLE_CLIENT_SECRET,
    )

    file_service = FileService(credentials=creds)
    try:
        local_path, file_size, file_hash = await file_service.download_file(
            file_id=drive_id,
            file_name=file_name,
            user_id=state["user_id"],
        )
    except Exception as e:
        logger.error(f"[download] Failed to download {file_name}: {str(e)}")
        return {"status": "failed", "error": f"Download failed: {str(e)}"}

    # Short txn 2: Save download result 
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(File)
            .where(File.id == state["file_id"])
            .values(
                local_path=local_path,
                file_size=file_size,
                file_hash=file_hash,
            )
        )
        await db.commit()
    # Transaction closed 

    logger.info(f"[download] Downloaded to {local_path}")
    return {
        "file_path": local_path,
        "file_name": file_name,
        "mime_type": mime_type,
        "status": "downloading",
    }


# Node 2: Process 
async def process_node(state: IndexingState) -> Dict[str, Any]:
    """
    Route to the correct processor based on file type.

        Uses the existing processors:
      - PDF  → pdf_processor.process_pdf()
      - Image → image_processor.process_image()
            - Text docs (.docx/.pptx/.txt/.md) → document_processor.process_document()
    """
    file_path = state.get("file_path")
    file_name = state.get("file_name", "unknown")
    mime_type = state.get("mime_type", "")
    groq_api_key = state["groq_api_key"]
    course_id = state.get("course_id")
    source_id = state["file_id"]

    logger.info(f"[process] Processing '{file_name}' (mime={mime_type})")

    # Detect file type using the same logic as FileService
    file_type = detect_file_type(mime_type, file_name)

    try:
        if file_type == "pdf":
            documents = await process_pdf(
                file_path=file_path,
                groq_api_key=groq_api_key,
                file_name=file_name,
                course_id=course_id,
                source_id=source_id,
            )
        elif file_type == "image":
            with open(file_path, "rb") as f:
                content = f.read()
            doc = await process_image(
                image_bytes=content,
                groq_api_key=groq_api_key,
                file_name=file_name,
                course_id=course_id,
                source_id=source_id,
            )
            documents = [doc]
        elif file_type == "text":
            documents = await process_document(
                file_path=file_path,
                file_name=file_name,
                course_id=course_id,
                source_id=source_id,
            )
        else:
            return {
                "status": "failed",
                "error": f"Unsupported file type: {file_type} ({mime_type})",
                "file_type": file_type,
            }

        if not documents:
            return {
                "status": "failed",
                "error": f"No content extracted from '{file_name}'",
                "file_type": file_type,
                "documents": [],
            }

        # Check if any document contains visual content
        contains_visual = any(
            doc.metadata.get("contains_visual", False) for doc in documents
        )

        logger.info(f"[process] Extracted {len(documents)} documents from '{file_name}'")
        return {
            "documents": documents,
            "file_type": file_type,
            "contains_visual": contains_visual,
            "status": "processing",
        }

    except Exception as e:
        logger.error(f"[process] Failed for '{file_name}': {e}")
        return {
            "status": "failed",
            "error": f"Processing failed: {str(e)}",
            "file_type": file_type,
        }


# Node 3: Chunk 
async def chunk_node(state: IndexingState) -> Dict[str, Any]:
    """Split documents into contextually-enriched chunks using SemanticMerger.

    Chunk sizes are configurable via settings (RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP).
    Uses LangChain RecursiveCharacterTextSplitter under the hood
    and normalizes metadata to the fixed vector schema.
    """
    documents = state.get("documents", [])
    if not documents:
        return {"status": "failed", "error": "No documents to chunk", "chunks": []}

    course_id = state.get("course_id")
    course_name = state.get("course_name")

    merger = SemanticMerger()  # uses settings.RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP
    chunks = merger.merge_and_chunk(
        documents=documents,
        course_id=course_id,
        course_name=course_name,
    )

    if not chunks:
        return {"status": "failed", "error": "Chunking produced zero chunks", "chunks": []}

    logger.info(f"[chunk] Split {len(documents)} docs → {len(chunks)} chunks")
    return {
        "chunks": chunks,
        "chunk_count": len(chunks),
        "status": "chunking",
    }


# Node 4: Embed 
async def embed_node(state: IndexingState) -> Dict[str, Any]:
    """
    Embed chunks into the user's pgvector store.

    Uses local HuggingFace BGE embeddings (free, no API key needed).
    Per-user collection: user_{user_id}
    """
    chunks = state.get("chunks", [])
    if not chunks:
        return {"status": "failed", "error": "No chunks to embed"}

    user_id = state["user_id"]

    try:
        import asyncio
        vector_store = EduverseVectorStore(user_id=user_id)
        ids = await asyncio.to_thread(vector_store.add_documents, chunks)

        logger.info(f"[embed] Embedded {len(ids)} chunks for user={user_id}")
        return {
            "chunk_count": len(ids),
            "status": "embedding",
        }
    except Exception as e:
        logger.error(f"[embed] Embedding failed: {e}")
        return {"status": "failed", "error": f"Embedding failed: {str(e)}"}


# Node 5: Update DB 
async def update_db_node(state: IndexingState) -> Dict[str, Any]:
    """
    Update the File record in the database with processing results.

    Sets processing_status, chunk_count, contains_visual, detected_type,
    and processed_at timestamp. Also invalidates the retriever cache
    so the next query picks up the newly indexed documents.
    """
    file_id = state["file_id"]
    user_id = state["user_id"]
    chunk_count = state.get("chunk_count", 0)
    contains_visual = state.get("contains_visual", False)
    file_type = state.get("file_type")

    async with AsyncSessionLocal() as db:
        await db.execute(
            update(File)
            .where(File.id == file_id)
            .values(
                processing_status="completed",
                chunk_count=chunk_count,
                contains_visual=contains_visual,
                detected_type=file_type,
                processed_at=datetime.now(timezone.utc),
                processing_error=None,
            )
        )
        await db.commit()

    # Invalidate retriever cache so next query uses fresh data
    from app.rag.retriever import invalidate_retriever_cache
    invalidate_retriever_cache(user_id)

    logger.info(
        f"[update_db] File {file_id}: status=completed, "
        f"chunks={chunk_count}, type={file_type}"
    )
    return {"status": "completed"}


# Error handler node 
async def handle_error_node(state: IndexingState) -> Dict[str, Any]:
    """
    Handle workflow failures by updating the DB with the error.
    """
    file_id = state.get("file_id")
    error = state.get("error", "Unknown error")

    if file_id:
        async with AsyncSessionLocal() as db:
            await db.execute(
                update(File)
                .where(File.id == file_id)
                .values(
                    processing_status="failed",
                    processing_error=error,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()

    logger.error(f"[handle_error] File {file_id}: {error}")
    return {"status": "failed"}


# Routing function 
def should_continue(state: IndexingState) -> str:
    """
    Conditional edge: route to next node or error handler.

    If status is 'failed', go to handle_error.
    Otherwise, continue to the next node.
    """
    if state.get("status") == "failed":
        return "handle_error"
    return "continue"