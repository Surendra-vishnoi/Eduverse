import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

# Allow Google to return different scopes than requested
# (e.g. when user previously granted additional scopes)
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.security import encrypt_token, decrypt_token
from app.core.exceptions import GoogleAuthError, ResourceNotFoundError
from app.models.database import User


class GoogleAuthService:
    """Service for Google OAuth authentication with database integration."""
    
    SCOPES = [
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/classroom.courses.readonly",
        "https://www.googleapis.com/auth/classroom.coursework.me",
        "https://www.googleapis.com/auth/classroom.coursework.students",
        "https://www.googleapis.com/auth/classroom.courseworkmaterials.readonly",

        "https://www.googleapis.com/auth/classroom.announcements.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
        "openid"
    ]

    def __init__(self):
        self.client_config = {
            "web": {
                "client_id": settings.GOOGLE_CLIENT_ID,
                "project_id": "eduverse",  
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uris": [settings.GOOGLE_REDIRECT_URI],
            }
        }

    def get_authorization_url(self) -> tuple[str, str, str]:
        """
        Get Google OAuth authorization URL.
        
        Returns:
            (authorization_url, state, code_verifier)
        """
        flow = Flow.from_client_config(
            self.client_config,
            scopes=self.SCOPES,
            redirect_uri=settings.GOOGLE_REDIRECT_URI
        )
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            prompt='consent'
        )
        # PKCE: flow generates a code_verifier that must be reused during token exchange
        code_verifier = flow.code_verifier
        return authorization_url, state, code_verifier

    async def exchange_code_for_tokens(self, code: str, code_verifier: str = None) -> dict:
        """
        Exchange authorization code for tokens.
        
        Args:
            code: Authorization code from Google callback
            code_verifier: PKCE code verifier from the login step
        
        Returns:
            {
                "token": access_token,
                "refresh_token": refresh_token,
                "token_uri": ...,
                "client_id": ...,
                "client_secret": ...,
                "scopes": [...],
                "expiry": datetime
            }
        """
        try:
            flow = Flow.from_client_config(
                self.client_config,
                scopes=self.SCOPES,
                redirect_uri=settings.GOOGLE_REDIRECT_URI
            )
            # Restore the PKCE code_verifier from the login step
            if code_verifier:
                flow.code_verifier = code_verifier
            flow.fetch_token(code=code)
            credentials = flow.credentials
            
            return {
                "token": credentials.token,
                "refresh_token": credentials.refresh_token,
                "token_uri": credentials.token_uri,
                "client_id": credentials.client_id,
                "client_secret": credentials.client_secret,
                "scopes": credentials.scopes,
                "expiry": credentials.expiry
            }
        except Exception as e:
            raise GoogleAuthError(f"Failed to exchange code: {str(e)}")

    async def get_user_info(self, token_info: dict) -> dict:
        """
        Get user info from Google.
        
        Returns:
            {
                "id": google_user_id,
                "email": ...,
                "name": ...,
                "picture": ...
            }
        """
        try:
            creds = Credentials(
                token=token_info["token"],
                refresh_token=token_info.get("refresh_token"),
                token_uri=token_info["token_uri"],
                client_id=token_info["client_id"],
                client_secret=token_info["client_secret"],
                scopes=token_info["scopes"]
            )
            
            service = build('oauth2', 'v2', credentials=creds)
            user_info = service.userinfo().get().execute()
            return user_info
        except Exception as e:
            raise GoogleAuthError(f"Failed to get user info: {str(e)}")

    async def create_or_update_user(
        self, 
        db: AsyncSession, 
        google_user_info: dict, 
        token_info: dict
    ) -> User:
        """
        Create or update user in database with encrypted tokens.
        
        Args:
            db: Database session
            google_user_info: Info from Google (id, email, name, picture)
            token_info: OAuth tokens
        
        Returns:
            User object
        """
        google_id = google_user_info["id"]
        
        result = await db.execute(select(User).where(User.google_id == google_id))
        user = result.scalar_one_or_none()
        
        encrypted_access = encrypt_token(token_info["token"])
        encrypted_refresh = encrypt_token(token_info["refresh_token"]) if token_info.get("refresh_token") else None
        
        if user:
            user.email = google_user_info["email"]
            user.name = google_user_info.get("name")
            user.picture = google_user_info.get("picture")
            user.encrypted_access_token = encrypted_access
            user.encrypted_refresh_token = encrypted_refresh
            user.token_expiry = token_info.get("expiry")
            user.last_login = datetime.now(timezone.utc)
            user.updated_at = datetime.now(timezone.utc)
        else:
            user = User(
                id=str(uuid.uuid4()),
                google_id=google_id,
                email=google_user_info["email"],
                name=google_user_info.get("name"),
                picture=google_user_info.get("picture"),
                encrypted_access_token=encrypted_access,
                encrypted_refresh_token=encrypted_refresh,
                token_expiry=token_info.get("expiry"),
                last_login=datetime.now(timezone.utc)
            )
            db.add(user)
        
        # Don't commit here — let the caller's get_db() dependency handle it
        await db.flush()
        await db.refresh(user)
        return user

    async def get_valid_credentials(
        self, 
        db: AsyncSession, 
        user_id: str
    ) -> Credentials:
        """
        Get valid Google credentials for a user, refreshing if needed.
        
        Args:
            db: Database session
            user_id: User ID
        
        Returns:
            Valid Google Credentials object
        """
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if not user:
            raise ResourceNotFoundError(f"User {user_id} not found")
        
        if not user.encrypted_access_token:
            raise GoogleAuthError("User has no stored access token — please re-login")
        
        access_token = decrypt_token(user.encrypted_access_token)
        refresh_token = decrypt_token(user.encrypted_refresh_token) if user.encrypted_refresh_token else None
        
        creds = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.GOOGLE_CLIENT_ID,
            client_secret=settings.GOOGLE_CLIENT_SECRET,
            scopes=self.SCOPES
        )
        
        if creds.expired and creds.refresh_token:
            try:
                from google.auth.transport.requests import Request
                creds.refresh(Request())
                
                user.encrypted_access_token = encrypt_token(creds.token)
                if creds.expiry:
                    user.token_expiry = creds.expiry
                # Flush the updated token to the current transaction.
                # Do NOT commit here — the caller's get_db() dependency owns the transaction.
                await db.flush()
            except Exception as e:
                raise GoogleAuthError(f"Failed to refresh token: {str(e)}")
        
        return creds