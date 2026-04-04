from typing import List, Optional
from langchain_core.documents import Document
from typing_extensions import TypedDict

class IndexingState(TypedDict, total=False):
    """
    State passed between LangGraph nodes during file indexing.

    Required fields are set at workflow invocation.
    Optional fields are populated by individual nodes as the workflow progresses.
    """

    file_id: str              
    user_id: str              
    course_id: Optional[str]  
    course_name: Optional[str]
    groq_api_key: str         

    file_path: Optional[str]  
    file_name: Optional[str]  
    file_type: Optional[str]  
    mime_type: Optional[str]  

    documents: List[Document]      
    chunks: List[Document]         
    chunk_count: int               
    contains_visual: bool      

    status: str                    
    error: Optional[str]           