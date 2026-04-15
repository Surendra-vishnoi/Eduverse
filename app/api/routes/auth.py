from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from urllib.parse import urlencode

from app.services.google_auth import GoogleAuthService
from app.core.config import settings
from app.core.database import get_db
from app.core.security import create_token_pair, verify_token
from app.core.exceptions import (
    GoogleAuthError, 
    to_http_exception
)
from app.models.database import User
from sqlalchemy import select

router = APIRouter()
auth_service = GoogleAuthService()

bearer_scheme = HTTPBearer(auto_error=False)

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict


class RefreshTokenRequest(BaseModel):
    refresh_token: str


def _build_frontend_callback_url(access_token: str, refresh_token: str) -> str | None:
    """Build frontend callback URL with token fragment (not query string)."""
    if not settings.FRONTEND_URL:
        return None

    base = settings.FRONTEND_URL.rstrip("/")
    callback_path = settings.FRONTEND_AUTH_CALLBACK_PATH.strip() or "/auth/callback"
    if not callback_path.startswith("/"):
        callback_path = f"/{callback_path}"

    fragment = urlencode(
        {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        }
    )
    return f"{base}{callback_path}#{fragment}"

@router.get("/login")
async def login(request: Request, redirect: bool = True):
    """
    Start Google OAuth login flow.
    
    - If redirect=true (default): Redirects browser to Google login page
    - If redirect=false: Returns the OAuth URL as JSON (for Swagger/API testing)
    """
    auth_url, state, code_verifier = auth_service.get_authorization_url()
    request.session["oauth_state"] = state
    request.session["oauth_code_verifier"] = code_verifier
    
    if redirect:
        return RedirectResponse(auth_url)
    
    # Return URL as JSON for API clients / Swagger UI
    return {"authorization_url": auth_url, "state": state}


@router.get("/callback")
async def callback(
    request: Request,
    code: str,
    state: str,
    response_mode: str = "auto",
    db: AsyncSession = Depends(get_db)
):
    """
    Handle Google OAuth callback.
    Exchange code for tokens, create/update user, issue JWT.
    """
    stored_state = request.session.get("oauth_state")
    if not stored_state or state != stored_state:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid state parameter"
        )
    
    try:
        code_verifier = request.session.get("oauth_code_verifier")
        token_info = await auth_service.exchange_code_for_tokens(code, code_verifier)
        
        google_user_info = await auth_service.get_user_info(token_info)
        
        user = await auth_service.create_or_update_user(db, google_user_info, token_info)
        
        tokens = create_token_pair(user.id)

        # OAuth state/code_verifier are one-time use values.
        request.session.pop("oauth_state", None)
        request.session.pop("oauth_code_verifier", None)

        frontend_callback_url = _build_frontend_callback_url(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
        )

        if response_mode != "json" and frontend_callback_url:
            return RedirectResponse(
                url=frontend_callback_url,
                status_code=status.HTTP_303_SEE_OTHER,
            )
        
        return JSONResponse(content={
            **tokens,
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "picture": user.picture
            }
        })
    
    except GoogleAuthError as e:
        raise to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    """
    user_id = verify_token(request.refresh_token, token_type="refresh")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    tokens = create_token_pair(user.id)
    
    return TokenResponse(
        **tokens,
        user={
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "picture": user.picture
        }
    )


@router.post("/logout")
async def logout(request: Request):
    """
    Logout - clear session.
    Note: JWTs can't be invalidated, so this just clears session.
    For production, implement token blacklist with Redis.
    """
    request.session.clear()
    return {"message": "Logged out successfully"}

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    FastAPI dependency to get current authenticated user.
    Uses HTTPBearer for Swagger UI compatibility.
    
    Usage:
        @router.get("/profile")
        async def get_profile(user: User = Depends(get_current_user)):
            return user
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    user_id = verify_token(token, token_type="access")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


@router.get("/me")
async def get_me(user: User = Depends(get_current_user)):
    """
    Get current user profile.
    """
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "picture": user.picture,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None
    }
