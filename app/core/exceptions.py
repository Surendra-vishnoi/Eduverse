from fastapi import HTTPException, status


class EduverseException(Exception):
    """Base exception for all Eduverse errors."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class AuthenticationError(EduverseException):
    """Raised when authentication fails."""
    status_code = status.HTTP_401_UNAUTHORIZED

class InvalidCredentialsError(AuthenticationError):
    """Raised when credentials are invalid."""
    pass

class TokenExpiredError(AuthenticationError):
    """Raised when JWT token has expired."""
    pass

class InvalidTokenError(AuthenticationError):
    """Raised when JWT token is invalid."""
    pass

class AuthorizationError(EduverseException):
    """Raised when user lacks permission."""
    status_code = status.HTTP_403_FORBIDDEN

class ResourceNotFoundError(EduverseException):
    """Raised when a resource doesn't exist."""
    status_code = status.HTTP_404_NOT_FOUND

class ResourceAlreadyExistsError(EduverseException):
    """Raised when trying to create a duplicate resource."""
    status_code = status.HTTP_409_CONFLICT

class GoogleAPIError(EduverseException):
    """Raised when Google API call fails."""
    status_code = status.HTTP_502_BAD_GATEWAY

class GoogleAuthError(GoogleAPIError):
    """Raised when Google authentication fails."""
    status_code = status.HTTP_401_UNAUTHORIZED

class ClassroomAPIError(GoogleAPIError):
    """Raised when Classroom API call fails."""
    pass

class DriveAPIError(GoogleAPIError):
    """Raised when Drive API call fails."""
    pass

class ProcessingError(EduverseException):
    """Raised when file processing fails."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

class UnsupportedFileTypeError(ProcessingError):
    """Raised when file type is not supported."""
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

class FileDownloadError(ProcessingError):
    """Raised when file download fails."""
    pass

class ValidationError(EduverseException):
    """Raised when validation fails."""
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT

def to_http_exception(error: EduverseException) -> HTTPException:
    """Convert a domain exception to FastAPI HTTPException."""
    return HTTPException(
        status_code=error.status_code,
        detail={
            "message": error.message,
            "details": error.details,
            "error_type": error.__class__.__name__
        }
    )