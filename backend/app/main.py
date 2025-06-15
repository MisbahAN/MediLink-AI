# backend/app/main.py
"""
FastAPI application for Prior Authorization automation.

This module initializes the FastAPI application with CORS middleware,
exception handlers, and API routers.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add the app directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from api.routes import upload, health, process, download
from core.middleware import setup_middleware, exception_handlers
from core.logging import init_logging

# Initialize structured logging
init_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MediLink AI",
    description="AI-powered system for automating PA form filling from medical referral packets",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js development
        "https://*.vercel.app",   # Vercel deployment
        "https://localhost:3000", # HTTPS local development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Set up middleware and exception handlers
setup_middleware(app)

# Add exception handlers from middleware module
for exception_type, handler in exception_handlers.items():
    app.add_exception_handler(exception_type, handler)


# Custom exception classes for application-specific errors
class ProcessingError(Exception):
    """Raised when PDF processing fails."""
    def __init__(self, message: str, session_id: str = None, error_code: str = None):
        self.message = message
        self.session_id = session_id
        self.error_code = error_code
        super().__init__(self.message)


class ExtractionError(Exception):
    """Raised when data extraction from PDF fails."""
    def __init__(self, message: str, file_path: str = None, page_number: int = None):
        self.message = message
        self.file_path = file_path
        self.page_number = page_number
        super().__init__(self.message)


class MappingError(Exception):
    """Raised when field mapping fails."""
    def __init__(self, message: str, field_name: str = None, confidence: float = None):
        self.message = message
        self.field_name = field_name
        self.confidence = confidence
        super().__init__(self.message)


class FormFillingError(Exception):
    """Raised when PDF form filling fails."""
    def __init__(self, message: str, form_path: str = None, field_errors: list = None):
        self.message = message
        self.form_path = form_path
        self.field_errors = field_errors or []
        super().__init__(self.message)


class ValidationError(Exception):
    """Raised when data validation fails."""
    def __init__(self, message: str, field_name: str = None, invalid_value: str = None):
        self.message = message
        self.field_name = field_name
        self.invalid_value = invalid_value
        super().__init__(self.message)


class CacheError(Exception):
    """Raised when cache operations fail."""
    def __init__(self, message: str, cache_key: str = None, operation: str = None):
        self.message = message
        self.cache_key = cache_key
        self.operation = operation
        super().__init__(self.message)


class AIServiceError(Exception):
    """Raised when AI service calls fail."""
    def __init__(self, message: str, service_name: str = None, api_error: str = None):
        self.message = message
        self.service_name = service_name
        self.api_error = api_error
        super().__init__(self.message)


# Custom exception handlers
@app.exception_handler(ProcessingError)
async def processing_error_handler(request: Request, exc: ProcessingError):
    """Handle PDF processing errors."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    logger.error(
        f"Processing error: {exc.message}",
        extra={
            "request_id": request_id,
            "session_id": exc.session_id,
            "error_code": exc.error_code,
            "error_type": "processing_error"
        }
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "processing_error",
                "message": exc.message,
                "session_id": exc.session_id,
                "error_code": exc.error_code,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "suggestions": [
                    "Check if uploaded files are valid PDF documents",
                    "Ensure files are not corrupted or password-protected",
                    "Try uploading smaller files if size is an issue"
                ]
            }
        }
    )


@app.exception_handler(ExtractionError)
async def extraction_error_handler(request: Request, exc: ExtractionError):
    """Handle data extraction errors."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    logger.error(
        f"Extraction error: {exc.message}",
        extra={
            "request_id": request_id,
            "file_path": exc.file_path,
            "page_number": exc.page_number,
            "error_type": "extraction_error"
        }
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "extraction_error",
                "message": exc.message,
                "file_path": exc.file_path,
                "page_number": exc.page_number,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "suggestions": [
                    "Check if the PDF contains readable text",
                    "Verify the document is not a scanned image requiring OCR",
                    "Ensure the document is not corrupted"
                ]
            }
        }
    )


@app.exception_handler(MappingError)
async def mapping_error_handler(request: Request, exc: MappingError):
    """Handle field mapping errors."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    logger.warning(
        f"Mapping error: {exc.message}",
        extra={
            "request_id": request_id,
            "field_name": exc.field_name,
            "confidence": exc.confidence,
            "error_type": "mapping_error"
        }
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "mapping_error",
                "message": exc.message,
                "field_name": exc.field_name,
                "confidence": exc.confidence,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "suggestions": [
                    "Review the source document for missing information",
                    "Check if field names match expected patterns",
                    "Consider manual entry for low-confidence fields"
                ]
            }
        }
    )


@app.exception_handler(FormFillingError)
async def form_filling_error_handler(request: Request, exc: FormFillingError):
    """Handle PDF form filling errors."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    logger.error(
        f"Form filling error: {exc.message}",
        extra={
            "request_id": request_id,
            "form_path": exc.form_path,
            "field_errors": exc.field_errors,
            "error_type": "form_filling_error"
        }
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "form_filling_error",
                "message": exc.message,
                "form_path": exc.form_path,
                "field_errors": exc.field_errors,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "suggestions": [
                    "Verify the PA form has fillable fields",
                    "Check if field names match the form template",
                    "Ensure field values are in the correct format"
                ]
            }
        }
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle data validation errors."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    logger.warning(
        f"Validation error: {exc.message}",
        extra={
            "request_id": request_id,
            "field_name": exc.field_name,
            "invalid_value": exc.invalid_value,
            "error_type": "validation_error"
        }
    )
    
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "type": "validation_error",
                "message": exc.message,
                "field_name": exc.field_name,
                "invalid_value": exc.invalid_value,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "suggestions": [
                    "Check the format of the provided value",
                    "Ensure required fields are not empty",
                    "Verify dates are in MM/DD/YYYY format"
                ]
            }
        }
    )


@app.exception_handler(CacheError)
async def cache_error_handler(request: Request, exc: CacheError):
    """Handle cache operation errors."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    logger.warning(
        f"Cache error: {exc.message}",
        extra={
            "request_id": request_id,
            "cache_key": exc.cache_key,
            "operation": exc.operation,
            "error_type": "cache_error"
        }
    )
    
    # Cache errors are typically non-fatal, so return 200 but log the issue
    return JSONResponse(
        status_code=200,
        content={
            "warning": {
                "type": "cache_error",
                "message": f"Cache operation failed: {exc.message}",
                "cache_key": exc.cache_key,
                "operation": exc.operation,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "note": "Processing continued without cache. Performance may be affected."
            }
        }
    )


@app.exception_handler(AIServiceError)
async def ai_service_error_handler(request: Request, exc: AIServiceError):
    """Handle AI service errors."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    logger.error(
        f"AI service error: {exc.message}",
        extra={
            "request_id": request_id,
            "service_name": exc.service_name,
            "api_error": exc.api_error,
            "error_type": "ai_service_error"
        }
    )
    
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "type": "ai_service_error",
                "message": exc.message,
                "service_name": exc.service_name,
                "api_error": exc.api_error,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "suggestions": [
                    "Check your internet connection",
                    "Verify API keys are valid and not expired",
                    "Try again in a few moments if the service is temporarily unavailable",
                    "Contact support if the issue persists"
                ]
            }
        }
    )


# Add custom error response for file size limit
@app.exception_handler(413)
async def request_entity_too_large_handler(request: Request, exc):
    """Handle file size limit exceeded errors."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    logger.warning(
        "File size limit exceeded",
        extra={
            "request_id": request_id,
            "error_type": "file_too_large"
        }
    )
    
    return JSONResponse(
        status_code=413,
        content={
            "error": {
                "type": "file_too_large",
                "message": "Uploaded file exceeds the maximum allowed size",
                "max_size_mb": 50,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "suggestions": [
                    "Reduce the file size by compressing the PDF",
                    "Split large documents into smaller files",
                    "Remove unnecessary pages or images from the PDF"
                ]
            }
        }
    )


# Add custom error response for rate limiting
@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc):
    """Handle rate limit exceeded errors."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    logger.warning(
        "Rate limit exceeded",
        extra={
            "request_id": request_id,
            "client_ip": request.client.host if request.client else "unknown",
            "error_type": "rate_limit_exceeded"
        }
    )
    
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "type": "rate_limit_exceeded",
                "message": "Too many requests. Please try again later.",
                "retry_after_seconds": 60,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "suggestions": [
                    "Wait 60 seconds before making another request",
                    "Reduce the frequency of your requests",
                    "Contact support if you need higher rate limits"
                ]
            }
        }
    )

# Include API routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(process.router, prefix="/api", tags=["process"])
app.include_router(download.router, tags=["download"])


@app.on_event("startup")
async def startup_event():
    """Initialize application services on startup."""
    logger.info("Starting MediLink AI")
    
    # Create upload directory if it doesn't exist
    from pathlib import Path
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    logger.info(f"Upload directory ready: {upload_dir.absolute()}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down MediLink AI")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MediLink AI",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )