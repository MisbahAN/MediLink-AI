"""
Middleware components for request processing, validation, and logging.

This module provides comprehensive middleware for the FastAPI application including
global exception handling, request validation, structured logging, and security
features for the Prior Authorization automation system.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request/response logging.
    
    Logs all incoming requests with timing, status codes, and optional
    request/response body logging for debugging and monitoring purposes.
    """
    
    def __init__(self, app, log_request_body: bool = False, log_response_body: bool = False):
        """
        Initialize logging middleware.
        
        Args:
            app: FastAPI application instance
            log_request_body: Whether to log request bodies (be careful with PHI)
            log_response_body: Whether to log response bodies (be careful with PHI)
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with comprehensive logging."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Extract request information
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request start
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "headers": dict(request.headers) if self.log_request_body else {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Log request body if enabled (be careful with PHI)
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body and len(body) < 10000:  # Limit body size for logging
                    logger.debug(
                        "Request body",
                        extra={
                            "request_id": request_id,
                            "body_size": len(body),
                            "content_type": request.headers.get("content-type")
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to read request body for logging: {e}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log successful response
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time_seconds": round(process_time, 4),
                    "response_size": response.headers.get("content-length", "unknown")
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            
            return response
            
        except Exception as e:
            # Calculate processing time for failed requests
            process_time = time.time() - start_time
            
            # Log request failure
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "process_time_seconds": round(process_time, 4)
                },
                exc_info=True
            )
            
            # Re-raise the exception to be handled by global exception handlers
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request headers."""
        # Check for forwarded headers (common in load balancers)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers to responses.
    
    Adds HIPAA-compliant security headers to protect against common
    web vulnerabilities and ensure secure handling of medical data.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to all responses."""
        response = await call_next(request)
        
        # Add comprehensive security headers
        security_headers = {
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            
            # Enable XSS protection
            "X-XSS-Protection": "1; mode=block",
            
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            
            # Strict transport security (HTTPS only)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: blob:; "
                "connect-src 'self'; "
                "font-src 'self'; "
                "object-src 'none'; "
                "base-uri 'self'; "
                "frame-ancestors 'none'"
            ),
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions policy
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "accelerometer=(), "
                "gyroscope=()"
            ),
            
            # Cache control for sensitive data
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        # Apply headers to response
        for header_name, header_value in security_headers.items():
            response.headers[header_name] = header_value
        
        return response


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for additional request validation and sanitization.
    
    Provides comprehensive request validation including file size limits,
    content type validation, and request rate limiting for security.
    """
    
    def __init__(self, app, max_request_size: int = 52428800):  # 50MB default
        """
        Initialize validation middleware.
        
        Args:
            app: FastAPI application instance
            max_request_size: Maximum request size in bytes
        """
        super().__init__(app)
        self.max_request_size = max_request_size
        self.rate_limit_storage = {}  # Simple in-memory storage (use Redis in production)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate request before processing."""
        try:
            # Validate request size
            content_length = request.headers.get("content-length")
            if content_length:
                if int(content_length) > self.max_request_size:
                    logger.warning(
                        f"Request rejected - size too large: {content_length} bytes",
                        extra={"request_id": getattr(request.state, 'request_id', 'unknown')}
                    )
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "type": "request_too_large",
                                "message": f"Request size exceeds maximum allowed size of {self.max_request_size} bytes",
                                "max_size_bytes": self.max_request_size
                            }
                        }
                    )
            
            # Validate content type for upload endpoints
            if request.url.path.startswith("/api/upload") and request.method == "POST":
                content_type = request.headers.get("content-type", "")
                if not content_type.startswith("multipart/form-data"):
                    logger.warning(
                        f"Request rejected - invalid content type: {content_type}",
                        extra={"request_id": getattr(request.state, 'request_id', 'unknown')}
                    )
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": {
                                "type": "invalid_content_type",
                                "message": "Upload endpoints require multipart/form-data content type",
                                "received_content_type": content_type
                            }
                        }
                    )
            
            # Basic rate limiting (simple implementation)
            client_ip = self._get_client_ip(request)
            if not self._check_rate_limit(client_ip):
                logger.warning(
                    f"Request rejected - rate limit exceeded for IP: {client_ip}",
                    extra={"request_id": getattr(request.state, 'request_id', 'unknown')}
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "type": "rate_limit_exceeded",
                            "message": "Too many requests. Please try again later.",
                            "retry_after_seconds": 60
                        }
                    }
                )
            
            # Process request
            response = await call_next(request)
            return response
            
        except Exception as e:
            logger.error(
                f"Request validation middleware error: {e}",
                extra={"request_id": getattr(request.state, 'request_id', 'unknown')},
                exc_info=True
            )
            # Continue processing - don't block on middleware errors
            return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _check_rate_limit(self, client_ip: str, max_requests: int = 100, window_seconds: int = 3600) -> bool:
        """
        Simple rate limiting check.
        
        Args:
            client_ip: Client IP address
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
            
        Returns:
            True if request is allowed, False if rate limited
        """
        current_time = time.time()
        
        # Clean old entries
        self._cleanup_rate_limit_storage(current_time, window_seconds)
        
        # Initialize client entry if not exists
        if client_ip not in self.rate_limit_storage:
            self.rate_limit_storage[client_ip] = []
        
        # Check current request count
        client_requests = self.rate_limit_storage[client_ip]
        recent_requests = [req_time for req_time in client_requests if current_time - req_time < window_seconds]
        
        if len(recent_requests) >= max_requests:
            return False
        
        # Add current request
        recent_requests.append(current_time)
        self.rate_limit_storage[client_ip] = recent_requests
        
        return True
    
    def _cleanup_rate_limit_storage(self, current_time: float, window_seconds: int):
        """Clean up old rate limit entries."""
        # Simple cleanup - remove entries older than window
        for client_ip in list(self.rate_limit_storage.keys()):
            self.rate_limit_storage[client_ip] = [
                req_time for req_time in self.rate_limit_storage[client_ip]
                if current_time - req_time < window_seconds
            ]
            
            # Remove empty entries
            if not self.rate_limit_storage[client_ip]:
                del self.rate_limit_storage[client_ip]


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for unhandled exceptions.
    
    Provides consistent error responses and prevents sensitive information
    from being exposed in error messages.
    """
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    # Log the exception with full details
    logger.error(
        "Unhandled exception in request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "error": str(exc),
            "error_type": type(exc).__name__
        },
        exc_info=True
    )
    
    # Return sanitized error response
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_server_error",
                "message": "An internal server error occurred",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handler for HTTP exceptions with structured error responses.
    
    Provides consistent error response format for all HTTP exceptions
    while maintaining security and logging requirements.
    """
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    # Log HTTP exceptions at appropriate level
    if exc.status_code >= 500:
        log_level = logging.ERROR
    elif exc.status_code >= 400:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    
    logger.log(
        log_level,
        f"HTTP {exc.status_code} error: {exc.detail}",
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "method": request.method,
            "url": str(request.url)
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "message": exc.detail,
                "status_code": exc.status_code,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handler for request validation errors with detailed field information.
    
    Provides detailed validation error information to help clients
    understand and correct validation issues.
    """
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    # Log validation error
    logger.warning(
        f"Request validation failed: {exc.errors()}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "validation_errors": exc.errors()
        }
    )
    
    # Format validation errors for client
    formatted_errors = []
    for error in exc.errors():
        formatted_errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "validation_error",
                "message": "Request validation failed",
                "request_id": request_id,
                "details": formatted_errors,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    )


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """
    Middleware for health check endpoints.
    
    Provides fast health check responses without going through
    the full middleware stack for monitoring endpoints.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle health check requests efficiently."""
        # Fast path for health checks
        if request.url.path == "/health" or request.url.path == "/api/health":
            # Simple health check response
            return JSONResponse(
                status_code=200,
                content={
                    "status": "healthy",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "service": "medilink-ai-backend"
                }
            )
        
        # Continue with normal processing
        return await call_next(request)


def setup_middleware(app):
    """
    Set up all middleware components for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Add middleware in reverse order (last added is executed first)
    
    # Health check middleware (fastest response)
    app.add_middleware(HealthCheckMiddleware)
    
    # Security headers (applied to all responses)
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Request validation (early validation)
    app.add_middleware(RequestValidationMiddleware, max_request_size=settings.MAX_FILE_SIZE)
    
    # Request logging (comprehensive logging)
    app.add_middleware(
        RequestLoggingMiddleware,
        log_request_body=settings.DEBUG,  # Only log bodies in debug mode
        log_response_body=False  # Never log response bodies (may contain PHI)
    )
    
    logger.info("Middleware setup completed")


# Exception handlers to be added to FastAPI app
exception_handlers = {
    Exception: global_exception_handler,
    HTTPException: http_exception_handler,
    RequestValidationError: validation_exception_handler
}