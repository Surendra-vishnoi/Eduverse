from fastapi import APIRouter, Request, Response, Depends, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from urllib.parse import urlparse

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


def _normalize_origin(url: str) -> str | None:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _is_allowed_frontend_redirect(frontend_redirect: str) -> bool:
    redirect_origin = _normalize_origin(frontend_redirect)
    if not redirect_origin:
        return False

    allowed_origins = {
        origin
        for origin in (
            _normalize_origin(str(item)) for item in settings.BACKEND_CORS_ORIGINS
        )
        if origin
    }

    if settings.FRONTEND_URL:
        frontend_origin = _normalize_origin(settings.FRONTEND_URL)
        if frontend_origin:
            allowed_origins.add(frontend_origin)

    return redirect_origin in allowed_origins


def _cookie_secure() -> bool:
    if settings.AUTH_COOKIE_SECURE is not None:
        return settings.AUTH_COOKIE_SECURE
    return not settings.DEBUG


def _cookie_samesite() -> str:
    same_site = (settings.AUTH_COOKIE_SAMESITE or "lax").lower()
    if same_site not in {"lax", "strict", "none"}:
        return "lax"
    return same_site


def _set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    if not settings.AUTH_COOKIE_ENABLED:
        return

    cookie_kwargs: dict[str, str | bool] = {
        "httponly": True,
        "secure": _cookie_secure(),
        "samesite": _cookie_samesite(),
        "path": settings.AUTH_COOKIE_PATH or "/",
    }

    if settings.AUTH_COOKIE_DOMAIN:
        cookie_kwargs["domain"] = settings.AUTH_COOKIE_DOMAIN

    response.set_cookie(
        key=settings.AUTH_COOKIE_ACCESS_NAME,
        value=access_token,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        **cookie_kwargs,
    )
    response.set_cookie(
        key=settings.AUTH_COOKIE_REFRESH_NAME,
        value=refresh_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        **cookie_kwargs,
    )


def _clear_auth_cookies(response: Response) -> None:
    cookie_kwargs: dict[str, str | bool] = {
        "path": settings.AUTH_COOKIE_PATH or "/",
        "secure": _cookie_secure(),
        "httponly": True,
        "samesite": _cookie_samesite(),
    }

    if settings.AUTH_COOKIE_DOMAIN:
        cookie_kwargs["domain"] = settings.AUTH_COOKIE_DOMAIN

    response.delete_cookie(settings.AUTH_COOKIE_ACCESS_NAME, **cookie_kwargs)
    response.delete_cookie(settings.AUTH_COOKIE_REFRESH_NAME, **cookie_kwargs)


def _build_frontend_callback_url(
    frontend_redirect: str | None = None,
) -> str | None:
    """Build frontend callback URL for cookie-based auth flow."""
    callback_base = None
    if frontend_redirect and _is_allowed_frontend_redirect(frontend_redirect):
        callback_base = frontend_redirect.split("#", 1)[0]
    elif settings.FRONTEND_URL:
        base = settings.FRONTEND_URL.rstrip("/")
        callback_path = settings.FRONTEND_AUTH_CALLBACK_PATH.strip() or "/auth/callback"
        if not callback_path.startswith("/"):
            callback_path = f"/{callback_path}"

        callback_base = f"{base}{callback_path}"

    return callback_base

@router.get("/login")
async def login(
    request: Request,
    redirect: bool = True,
    frontend_redirect: str | None = None,
):
    """
    Start Google OAuth login flow.
    
    - If redirect=true (default): Redirects browser to Google login page
    - If redirect=false: Returns the OAuth URL as JSON (for Swagger/API testing)
    """
    auth_url, state, code_verifier = auth_service.get_authorization_url()
    request.session["oauth_state"] = state
    request.session["oauth_code_verifier"] = code_verifier

    # Optional dynamic frontend callback for multi-frontend deployments.
    # Only store if origin is allowlisted to avoid open redirect risks.
    if frontend_redirect and _is_allowed_frontend_redirect(frontend_redirect):
        request.session["frontend_redirect"] = frontend_redirect
    else:
        request.session.pop("frontend_redirect", None)
    
    if redirect:
        return RedirectResponse(auth_url)
    
    # Return URL as JSON for API clients / Swagger UI
    return {"authorization_url": auth_url, "state": state}


@router.get("/callback")
async def callback(
    request: Request,
    code: str,
    state: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Handle Google OAuth callback.
    Exchange code for tokens, create/update user, issue JWT cookies.
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
        frontend_redirect = request.session.pop("frontend_redirect", None)

        frontend_callback_url = _build_frontend_callback_url(
            frontend_redirect=frontend_redirect,
        )

        if frontend_callback_url:
            redirect_response = RedirectResponse(
                url=frontend_callback_url,
                status_code=status.HTTP_303_SEE_OTHER,
            )
            _set_auth_cookies(
                redirect_response,
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
            )
            return redirect_response
        
        json_response = JSONResponse(content={
            "message": "Authenticated successfully",
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "picture": user.picture
            }
        })
        _set_auth_cookies(
            json_response,
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
        )
        return json_response
    
    except GoogleAuthError as e:
        raise to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}"
        )


@router.post("/refresh")
async def refresh_token(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh auth cookies using HttpOnly refresh token cookie.
    """
    refresh_token_value = request.cookies.get(settings.AUTH_COOKIE_REFRESH_NAME)

    if not refresh_token_value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token"
        )

    user_id = verify_token(refresh_token_value, token_type="refresh")
    
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
    _set_auth_cookies(
        response,
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
    )

    return {"message": "Token refreshed"}


@router.post("/logout")
async def logout(request: Request):
    """
    Logout - clear session.
    Note: JWTs can't be invalidated, so this just clears session.
    For production, implement token blacklist with Redis.
    """
    request.session.clear()
    response = JSONResponse(content={"message": "Logged out successfully"})
    _clear_auth_cookies(response)
    return response

async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    FastAPI dependency to get current authenticated user.
    Uses HttpOnly cookie-based JWT authentication.
    
    Usage:
        @router.get("/profile")
        async def get_profile(user: User = Depends(get_current_user)):
            return user
    """
    token = request.cookies.get(settings.AUTH_COOKIE_ACCESS_NAME)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication cookie",
        )
    
    user_id = verify_token(token, token_type="access")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
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
