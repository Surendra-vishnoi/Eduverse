import base64
import logging
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.core.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}

VISION_PROMPT = (
    "You are analyzing an educational image. Describe what you see in detail: "
    "diagrams (structure, components, relationships), charts/graphs (data, axes, trends), "
    "equations/formulas (write out and explain), code (transcribe and explain), "
    "slides (capture text and visual elements). Be thorough and educational."
)


def create_vision_llm(groq_api_key: str, model: str = None) -> ChatGroq:
    """Create a reusable ChatGroq vision instance.

    Call once and pass to ``analyze_image`` via the *llm* parameter to
    avoid creating a new client for every frame / image.
    """
    return ChatGroq(
        model=model or settings.VISION_MODEL,
        api_key=groq_api_key,
        max_tokens=1024,
        temperature=0.2,
    )


async def analyze_image(
    image_bytes: bytes,
    groq_api_key: str,
    prompt: str = VISION_PROMPT,
    model: str = None,
    llm: Optional[BaseChatModel] = None,
) -> str:
    """Analyze image using ChatGroq vision via LangChain messages API.

    Parameters
    ----------
    llm : optional
        Pre-created vision LLM.  When processing many images (e.g.
        video frames) pass a shared instance created by
        ``create_vision_llm`` to avoid per-call overhead.
    """
    try:
        _llm = llm or create_vision_llm(groq_api_key, model)
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ])

        response = await _llm.ainvoke([msg])
        return response.content.strip()

    except Exception as e:
        logger.error(f"Vision analysis failed: {e}")
        return f"[Image analysis failed: {e}]"

async def process_image(
    image_bytes: bytes,
    groq_api_key: str,
    file_name: str = "image.png",
    course_id: Optional[str] = None,
    source_id: Optional[str] = None,
) -> Document:
    """Process standalone image → LangChain Document with vision description."""
    description = await analyze_image(image_bytes, groq_api_key)

    return Document(
        page_content=f"[VISUAL]\n{description}",
        metadata={
            "source_type": "image",
            "source_id": source_id or file_name,
            "file_name": file_name,
            "course_id": course_id,
            "page_number": None,
            "start_time": None,
            "end_time": None,
            "contains_visual": True,
        }
    )