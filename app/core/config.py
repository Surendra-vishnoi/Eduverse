from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Eduverse Backend"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]


    JWT_SECRET: str = "your_jwt_secret_key_here"
    FERNET_KEY: str = "your_fernet_key_here" 
    SECRET_KEY: str = "your_session_secret_key_here"
    
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/eduverse"
    
    UPLOAD_DIR: str = "./uploads"
    
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GOOGLE_REDIRECT_URI: Optional[str] = "http://localhost:8000/auth/callback"
    
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"

    AGENT_MODEL: str = "openai/gpt-oss-120b"
    JSON_MODEL: str = "openai/gpt-oss-20b"
    WEB_SEARCH_MODEL: str = "groq/compound-mini"
    QUERY_REWRITE_MODEL: str = "llama-3.1-8b-instant"
    SUMMARY_MODEL: str = "llama-3.3-70b-versatile"
    VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    RAG_LLM_TEMPERATURE: float = 0.3
    RAG_RETRIEVER_K: int = 8
    RAG_RETRIEVER_FETCH_K: int = 50
    RAG_RERANK_TOP_N: int = 5
    RAG_RERANK_MODEL: str = "ms-marco-MiniLM-L-12-v2"
    RAG_RERANK_SCORE_THRESHOLD: float = 0.0
    RAG_CHUNK_SIZE: int = 800
    RAG_CHUNK_OVERLAP: int = 200
    RAG_PARENT_CHUNK_SIZE: int = 1600

    GROQ_API_KEY: Optional[str] = None
    NOMIC_API_KEY: Optional[str] = None

    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None

    @property
    def PG_SYNC_URL(self) -> str:
        """SQLAlchemy psycopg3 URL — used by PGVector (via SQLAlchemy engine).
        
        Format: postgresql+psycopg://...
        """
        url = self.DATABASE_URL.replace("+asyncpg", "+psycopg")
        # Append sslmode for Supabase TLS
        sep = "&" if "?" in url else "?"
        return url + sep + "sslmode=require"

    @property
    def PG_CONNINFO(self) -> str:
        """Plain psycopg3 connection string — used by components that call
        psycopg.connect() directly.
        
        Used by: PostgresChatMessageHistory.
        Format: postgresql://... (no +psycopg dialect prefix)
        """
        url = self.DATABASE_URL.replace("+asyncpg", "")
        sep = "&" if "?" in url else "?"
        return url + sep + "sslmode=require&options=-c%20statement_timeout%3D180000"

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

settings = Settings()