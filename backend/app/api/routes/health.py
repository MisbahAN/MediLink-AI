"""
Health check API routes for monitoring system status and service connectivity.

This module provides endpoints to check the health of the Prior Authorization
automation system, including all dependent services and components.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from core.deps import get_current_settings
from core.config import Settings
from models.schemas import HealthCheck
from services.storage import get_file_storage
from services.pdf_extractor import get_pdf_extractor
from services.mistral_service import get_mistral_service
from services.gemini_service_fallback import get_gemini_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthCheck)
async def health_check(settings: Settings = Depends(get_current_settings)) -> HealthCheck:
    """
    Comprehensive health check endpoint.
    
    Checks the status of all system components including:
    - API server health
    - File storage availability
    - PDF extraction services
    - AI services (Mistral OCR primary, Gemini fallback)
    - Configuration validation
    
    Returns:
        HealthCheck model with overall status and service details
    """
    logger.info("Health check requested")
    
    # Collect service statuses
    service_checks = {}
    overall_healthy = True
    
    try:
        # Check file storage service
        storage_status = await _check_storage_service()
        service_checks["storage"] = storage_status["status"]
        if storage_status["status"] != "healthy":
            overall_healthy = False
        
        # Check PDF extraction service
        pdf_extractor_status = await _check_pdf_extractor_service()
        service_checks["pdf_extractor"] = pdf_extractor_status["status"]
        if pdf_extractor_status["status"] != "healthy":
            overall_healthy = False
        
        # Check Mistral OCR service (Primary)
        mistral_status = await _check_mistral_service()
        service_checks["mistral_ocr"] = mistral_status["status"]
        if mistral_status["status"] != "healthy":
            overall_healthy = False
        
        # Check Gemini AI service (Fallback)
        gemini_status = await _check_gemini_service()
        service_checks["gemini_fallback"] = gemini_status["status"]
        # Gemini is fallback, so don't fail overall health if it's down
        
        # Check configuration
        config_status = _check_configuration(settings)
        service_checks["configuration"] = config_status["status"]
        if config_status["status"] != "healthy":
            overall_healthy = False
        
        # Check Redis (if configured)
        redis_status = await _check_redis_service(settings)
        service_checks["redis"] = redis_status["status"]
        if redis_status["status"] == "error":
            # Redis is optional, so we don't fail overall health
            pass
        
        overall_status = "healthy" if overall_healthy else "degraded"
        
        health_response = HealthCheck(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            version=settings.VERSION,
            services=service_checks
        )
        
        logger.info(f"Health check completed: {overall_status}")
        return health_response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        
        # Return error response
        error_response = HealthCheck(
            status="error",
            timestamp=datetime.now(timezone.utc),
            version=settings.VERSION,
            services={"error": f"Health check failed: {str(e)}"}
        )
        
        return JSONResponse(
            status_code=503,
            content=error_response.model_dump()
        )


@router.get("/health/storage")
async def storage_health_check() -> Dict[str, Any]:
    """
    Detailed health check for file storage service.
    
    Returns:
        Dictionary with storage service status and details
    """
    return await _check_storage_service()


@router.get("/health/pdf-extractor")
async def pdf_extractor_health_check() -> Dict[str, Any]:
    """
    Detailed health check for PDF extraction service.
    
    Returns:
        Dictionary with PDF extractor status and details
    """
    return await _check_pdf_extractor_service()


@router.get("/health/mistral")
async def mistral_health_check() -> Dict[str, Any]:
    """
    Detailed health check for Mistral OCR service (Primary).
    
    Returns:
        Dictionary with Mistral service status and details
    """
    return await _check_mistral_service()


@router.get("/health/gemini")
async def gemini_health_check() -> Dict[str, Any]:
    """
    Detailed health check for Gemini AI service (Fallback).
    
    Returns:
        Dictionary with Gemini service status and details
    """
    return await _check_gemini_service()


@router.get("/health/config")
async def config_health_check(settings: Settings = Depends(get_current_settings)) -> Dict[str, Any]:
    """
    Configuration validation health check.
    
    Returns:
        Dictionary with configuration status and details
    """
    return _check_configuration(settings)


async def _check_storage_service() -> Dict[str, Any]:
    """
    Check file storage service health.
    
    Returns:
        Dictionary with storage service status
    """
    try:
        storage_service = get_file_storage()
        
        # Test basic storage operations
        base_dir = storage_service.base_upload_dir
        
        # Check if base directory exists and is writable
        if not base_dir.exists():
            return {
                "status": "error",
                "message": f"Upload directory does not exist: {base_dir}",
                "details": {
                    "base_directory": str(base_dir),
                    "exists": False,
                    "writable": False
                }
            }
        
        # Test write permissions
        test_file = base_dir / ".health_check"
        try:
            test_file.write_text("health_check")
            test_file.unlink()
            writable = True
        except Exception:
            writable = False
        
        if not writable:
            return {
                "status": "error",
                "message": "Upload directory is not writable",
                "details": {
                    "base_directory": str(base_dir),
                    "exists": True,
                    "writable": False
                }
            }
        
        return {
            "status": "healthy",
            "message": "Storage service operational",
            "details": {
                "base_directory": str(base_dir),
                "exists": True,
                "writable": True,
                "max_file_size": storage_service.max_file_size,
                "allowed_types": storage_service.allowed_types
            }
        }
        
    except Exception as e:
        logger.error(f"Storage health check failed: {e}")
        return {
            "status": "error",
            "message": f"Storage service error: {str(e)}",
            "details": {"error_type": type(e).__name__}
        }


async def _check_pdf_extractor_service() -> Dict[str, Any]:
    """
    Check PDF extractor service health.
    
    Returns:
        Dictionary with PDF extractor status
    """
    try:
        pdf_extractor = get_pdf_extractor()
        
        # Test basic functionality - check if pdfplumber is available
        try:
            import pdfplumber
            pdfplumber_available = True
        except ImportError:
            pdfplumber_available = False
        
        # Test pdfforms availability
        try:
            import pdfforms
            pdfforms_available = True
        except ImportError:
            pdfforms_available = False
        
        if not pdfplumber_available or not pdfforms_available:
            missing_deps = []
            if not pdfplumber_available:
                missing_deps.append("pdfplumber")
            if not pdfforms_available:
                missing_deps.append("pdfforms")
            
            return {
                "status": "error",
                "message": f"Missing required dependencies: {', '.join(missing_deps)}",
                "details": {
                    "pdfplumber_available": pdfplumber_available,
                    "pdfforms_available": pdfforms_available,
                    "confidence_levels": pdf_extractor.confidence_levels
                }
            }
        
        return {
            "status": "healthy",
            "message": "PDF extractor service operational",
            "details": {
                "pdfplumber_available": pdfplumber_available,
                "pdfforms_available": pdfforms_available,
                "max_pages_per_chunk": pdf_extractor.max_pages_per_chunk,
                "confidence_levels": pdf_extractor.confidence_levels
            }
        }
        
    except Exception as e:
        logger.error(f"PDF extractor health check failed: {e}")
        return {
            "status": "error",
            "message": f"PDF extractor service error: {str(e)}",
            "details": {"error_type": type(e).__name__}
        }


async def _check_mistral_service() -> Dict[str, Any]:
    """
    Check Mistral OCR service health with connection test.
    
    Returns:
        Dictionary with Mistral service status
    """
    try:
        mistral_service = get_mistral_service()
        
        # Get service status
        service_status = mistral_service.get_service_status()
        
        if not service_status["api_key_configured"]:
            return {
                "status": "error",
                "message": "Mistral API key not configured",
                "details": service_status
            }
        
        # Test actual connection with timeout
        try:
            connection_test = await asyncio.wait_for(
                _test_mistral_connection(mistral_service),
                timeout=10.0
            )
            
            if connection_test:
                return {
                    "status": "healthy",
                    "message": "Mistral OCR API connection successful",
                    "details": service_status
                }
            else:
                return {
                    "status": "error",
                    "message": "Mistral OCR API connection failed",
                    "details": service_status
                }
                
        except asyncio.TimeoutError:
            return {
                "status": "warning",
                "message": "Mistral OCR API connection timeout",
                "details": service_status
            }
        
    except Exception as e:
        logger.error(f"Mistral health check failed: {e}")
        return {
            "status": "error",
            "message": f"Mistral service error: {str(e)}",
            "details": {"error_type": type(e).__name__}
        }


async def _test_mistral_connection(mistral_service) -> bool:
    """
    Test Mistral API connection.
    
    Args:
        mistral_service: Mistral service instance
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Initialize client if not already done
        if not mistral_service.client:
            success = mistral_service.initialize_client()
            return success
        return True
    except Exception as e:
        logger.warning(f"Mistral connection test failed: {e}")
        return False


async def _check_gemini_service() -> Dict[str, Any]:
    """
    Check Gemini AI service health with connection test.
    
    Returns:
        Dictionary with Gemini service status
    """
    try:
        gemini_service = get_gemini_service()
        
        # Get service status
        service_status = gemini_service.get_service_status()
        
        if not service_status["api_key_configured"]:
            return {
                "status": "warning",
                "message": "Gemini API key not configured",
                "details": service_status
            }
        
        # Test actual connection with timeout
        try:
            connection_test = await asyncio.wait_for(
                _test_gemini_connection(gemini_service),
                timeout=10.0
            )
            
            if connection_test:
                return {
                    "status": "healthy",
                    "message": "Gemini API connection successful",
                    "details": service_status
                }
            else:
                return {
                    "status": "error",
                    "message": "Gemini API connection failed",
                    "details": service_status
                }
                
        except asyncio.TimeoutError:
            return {
                "status": "warning",
                "message": "Gemini API connection timeout",
                "details": service_status
            }
        
    except Exception as e:
        logger.error(f"Gemini health check failed: {e}")
        return {
            "status": "error",
            "message": f"Gemini service error: {str(e)}",
            "details": {"error_type": type(e).__name__}
        }


async def _test_gemini_connection(gemini_service) -> bool:
    """
    Test Gemini API connection.
    
    Args:
        gemini_service: Gemini service instance
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Initialize client if not already done
        if not gemini_service.client:
            success = gemini_service.initialize_client()
            return success
        return True
    except Exception as e:
        logger.warning(f"Gemini connection test failed: {e}")
        return False


def _check_configuration(settings) -> Dict[str, Any]:
    """
    Check application configuration.
    
    Args:
        settings: Application settings instance
        
    Returns:
        Dictionary with configuration status
    """
    try:
        config_issues = []
        config_warnings = []
        
        # Check required settings
        if not settings.PROJECT_NAME:
            config_issues.append("PROJECT_NAME not set")
        
        if not settings.UPLOAD_DIR:
            config_issues.append("UPLOAD_DIR not configured")
        
        if settings.MAX_FILE_SIZE <= 0:
            config_issues.append("MAX_FILE_SIZE must be positive")
        
        if not settings.ALLOWED_FILE_TYPES:
            config_issues.append("ALLOWED_FILE_TYPES not configured")
        
        # Check optional but recommended settings
        if not settings.MISTRAL_API_KEY:
            config_warnings.append("MISTRAL_API_KEY not configured - Primary OCR will be unavailable")
        
        if not settings.GEMINI_API_KEY:
            config_warnings.append("GEMINI_API_KEY not configured - Fallback AI vision will be unavailable")
        
        if not settings.REDIS_URL:
            config_warnings.append("REDIS_URL not configured - caching will be disabled")
        
        # Determine status
        if config_issues:
            status = "error"
            message = f"Configuration errors: {', '.join(config_issues)}"
        elif config_warnings:
            status = "warning"
            message = f"Configuration warnings: {', '.join(config_warnings)}"
        else:
            status = "healthy"
            message = "Configuration valid"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "project_name": settings.PROJECT_NAME,
                "version": settings.VERSION,
                "debug_mode": settings.DEBUG,
                "upload_dir": settings.UPLOAD_DIR,
                "max_file_size": settings.MAX_FILE_SIZE,
                "allowed_file_types": settings.ALLOWED_FILE_TYPES,
                "mistral_configured": bool(settings.MISTRAL_API_KEY),
                "gemini_configured": bool(settings.GEMINI_API_KEY),
                "redis_configured": bool(settings.REDIS_URL),
                "issues": config_issues,
                "warnings": config_warnings
            }
        }
        
    except Exception as e:
        logger.error(f"Configuration check failed: {e}")
        return {
            "status": "error",
            "message": f"Configuration check error: {str(e)}",
            "details": {"error_type": type(e).__name__}
        }


async def _check_redis_service(settings) -> Dict[str, Any]:
    """
    Check Redis service health if configured.
    
    Args:
        settings: Application settings instance
        
    Returns:
        Dictionary with Redis service status
    """
    if not settings.REDIS_URL:
        return {
            "status": "not_configured",
            "message": "Redis not configured - caching disabled",
            "details": {"redis_url": None}
        }
    
    try:
        # Try to import redis
        try:
            import redis
        except ImportError:
            return {
                "status": "error",
                "message": "Redis package not installed",
                "details": {"redis_url": settings.REDIS_URL}
            }
        
        # Test connection with timeout
        try:
            redis_client = redis.from_url(settings.REDIS_URL, socket_timeout=5)
            redis_client.ping()
            
            return {
                "status": "healthy",
                "message": "Redis connection successful",
                "details": {
                    "redis_url": settings.REDIS_URL,
                    "cache_ttl_hours": settings.CACHE_TTL_HOURS
                }
            }
            
        except redis.ConnectionError:
            return {
                "status": "error",
                "message": "Redis connection failed",
                "details": {"redis_url": settings.REDIS_URL}
            }
        except redis.TimeoutError:
            return {
                "status": "warning",
                "message": "Redis connection timeout",
                "details": {"redis_url": settings.REDIS_URL}
            }
        
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "status": "error",
            "message": f"Redis service error: {str(e)}",
            "details": {
                "redis_url": settings.REDIS_URL,
                "error_type": type(e).__name__
            }
        }