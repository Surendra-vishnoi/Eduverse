"""
Shared utility functions for Eduverse.

Provides:
  - detect_file_type(mime_type, file_name)  → normalized file type
  - detect_source_type(file_name)           → citation source label
  - validate_groq_key(key)                  → raises ValueError if invalid
  - create_groq_client(api_key)             → pre-configured Groq client

Used by: tools.py, file_service.py, nodes.py, chat.py, files.py
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── File Type Detection ───────────────────────────────────────────

def detect_file_type(mime_type: Optional[str], file_name: str) -> str:
    """
    Detect normalized file type from MIME type and filename.

    Returns:
        "pdf", "image", "text", or "unknown"
    """
    mime = (mime_type or "").lower()
    name = file_name.lower()

    # PDF
    if mime == "application/pdf" or name.endswith(".pdf"):
        return "pdf"

    # Image
    if mime.startswith("image/") or name.endswith(
        (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff")
    ):
        return "image"

    # Office / Text documents by extension
    if name.endswith((".doc", ".docx", ".ppt", ".pptx", ".txt", ".md", ".rtf")):
        return "text"

    # Text / Document by MIME
    if mime.startswith("text/") or any(
        keyword in mime for keyword in ("document", "msword", "wordprocessing")
    ):
        return "text"

    logger.warning(f"[detect_file_type] Unknown type: mime={mime}, file={name}")
    return "unknown"


def detect_source_type(file_name: str) -> str:
    """
    Detect source type label for citations.

    Returns:
        "pdf", "image", or "document"
    """
    file_type = detect_file_type(None, file_name)

    if file_type in {"pdf", "image"}:
        return file_type

    return "document"


# ── Groq Helpers ─────────────────────────────────────────────────

def validate_groq_key(key: str) -> None:
    """
    Validate Groq API key.

    Raises:
        ValueError if key is invalid
    """
    if not key or not isinstance(key, str):
        raise ValueError("Groq API key is missing or not a string.")

    if not key.startswith("gsk_") or len(key) < 20:
        raise ValueError("Invalid Groq API key format.")

    logger.debug("Groq API key validated successfully.")


def create_groq_client(api_key: str):
    """
    Return a ready-to-use Groq client.

    Raises:
        ValueError if API key is invalid
    """
    validate_groq_key(api_key)

    try:
        from groq import Groq  # lazy import

        client = Groq(api_key=api_key)
        logger.info("Groq client created successfully.")
        return client

    except Exception as e:
        logger.error(f"Failed to create Groq client: {e}")
        raise