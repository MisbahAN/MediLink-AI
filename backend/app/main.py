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
from pathlib import Path

# Add the app directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from api.routes import upload, health
# from api.routes import process, download  # TODO: Uncomment when these modules are implemented

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)

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

# Include API routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
# app.include_router(process.router, prefix="/api", tags=["process"])  # TODO: Uncomment when implemented
# app.include_router(download.router, prefix="/api", tags=["download"])  # TODO: Uncomment when implemented


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured error responses."""
    logger.error(f"HTTP {exc.status_code} error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url)
            }
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed field information."""
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors(),
                "path": str(request.url)
            }
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions with generic error response."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "message": "An internal server error occurred",
                "path": str(request.url)
            }
        }
    )


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