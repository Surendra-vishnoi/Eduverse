import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
import uuid
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.database import File as FileModel
from app.core.config import settings
from app.api.routes.auth import get_current_user
from app.processing.document_processor import SUPPORTED_DOCUMENT_FORMATS
from app.processing.image_processor import SUPPORTED_IMAGE_FORMATS
import os

logger = logging.getLogger(__name__)
router = APIRouter()

ALL_SUPPORTED = (
    {'.pdf'}
    | SUPPORTED_IMAGE_FORMATS
    | SUPPORTED_DOCUMENT_FORMATS
)
MAX_UPLOAD = 100 * 1024 * 1024  

def _file_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.pdf': return 'pdf'
    if ext in SUPPORTED_IMAGE_FORMATS: return 'image'
    if ext in SUPPORTED_DOCUMENT_FORMATS: return 'text'
    return 'unknown'


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    course_id: str = Form(...),
    user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a local file and associate it with a course.
    Saves to local storage and creates a File record in the database.
    """
    content = await file.read()
    if len(content) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_UPLOAD // 1024 // 1024}MB")
    
    # Ensure user upload directory exists
    user_dir = Path(settings.UPLOAD_DIR) / user.id
    user_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file with unique name
    fname = file.filename or "uploaded_file"
    unique_filename = f"{uuid.uuid4()}_{fname}"
    local_path = user_dir / unique_filename
    
    with open(local_path, "wb") as f:
        f.write(content)
    
    # Detect type
    ftype = _file_type(fname)
    
    # Create DB record
    new_file = FileModel(
        id=str(uuid.uuid4()),
        user_id=user.id,
        course_id=course_id,
        drive_name=fname,
        mime_type=file.content_type,
        local_path=str(local_path),
        file_size=len(content),
        detected_type=ftype,
        processing_status="pending"
    )
    
    db.add(new_file)
    await db.commit()
    await db.refresh(new_file)
    
    return {
        "status": "success",
        "file_id": new_file.id,
        "file_name": fname,
        "processing_status": "pending"
    }


@router.get("/supported-formats")
async def get_supported_formats(_user=Depends(get_current_user)):
    return {
        "pdf": [".pdf"],
        "image": sorted(SUPPORTED_IMAGE_FORMATS),
        "document": sorted(SUPPORTED_DOCUMENT_FORMATS),
    }
