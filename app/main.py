from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from app.core.config import settings
from app.core.database import init_db, close_db

from app.api.routes import auth, classroom, files, indexing, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB tables. Shutdown: close DB connections."""
    await init_db()
    yield
    await close_db()


def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        lifespan=lifespan,
    )

    # Set all CORS enabled origins
    if settings.BACKEND_CORS_ORIGINS:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    application.add_middleware(
        SessionMiddleware, 
        secret_key=settings.SECRET_KEY, 
        https_only=not settings.DEBUG  # True in production, False in local dev
    )

    application.include_router(auth.router, prefix="/auth", tags=["auth"])
    application.include_router(classroom.router, prefix="/classroom", tags=["classroom"])
    application.include_router(files.router, prefix="/files", tags=["files"])
    application.include_router(indexing.router, prefix="/indexing", tags=["indexing"])
    application.include_router(chat.router, prefix="/chat", tags=["chat"])
    
    return application

app = create_application()

@app.get("/")
def root():
    return {"message": "Welcome to Eduverse Backend"}


@app.get("/health")
def health_check():
    """Health check for monitoring and deployment verification."""
    return {
        "status": "healthy",
        "version": "1.1.0",
        "service": "eduverse-ai-tutor",
    }
