# backend/app/api/routes/download.py
"""
Download Routes for retrieving filled PA forms and missing fields reports.

This module provides secure file download endpoints with streaming support,
proper MIME type handling, and comprehensive error handling for completed
PA form processing workflows.
"""

import logging
import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Response
from fastapi.responses import StreamingResponse, FileResponse
from starlette.background import BackgroundTask

from core.deps import get_current_settings
from core.config import Settings
from services.storage import get_file_storage
from services.cache import get_cache_service
from utils.file_handler import validate_session_id, get_file_size

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/download", tags=["download"])


@router.get("/{session_id}/filled")
async def download_filled_form(
    session_id: str,
    settings: Settings = Depends(get_current_settings),
    file_storage=Depends(get_file_storage),
    cache_service=Depends(get_cache_service)
):
    """
    Download the filled PA form PDF for a completed processing session.
    
    Args:
        session_id: Processing session identifier
        
    Returns:
        StreamingResponse with the filled PDF file
        
    Raises:
        HTTPException: If session not found, processing incomplete, or file errors
    """
    try:
        # Validate session ID format
        if not validate_session_id(session_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid session ID format"
            )
        
        # Import processing sessions from process route
        from app.api.routes.process import processing_sessions
        
        # Check if processing is complete
        if session_id not in processing_sessions:
            raise HTTPException(
                status_code=404,
                detail="Session not found or processing not complete"
            )
        
        session_data = processing_sessions[session_id]
        
        # Verify processing status
        if session_data.get("status") != "completed":
            status = session_data.get("status", "unknown")
            raise HTTPException(
                status_code=409,
                detail=f"Processing not complete. Current status: {status}"
            )
        
        # Get filled form file path
        filled_form_path = _get_filled_form_path(session_id, file_storage)
        
        if not filled_form_path or not filled_form_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Filled form file not found. Processing may have failed."
            )
        
        # Validate file integrity
        file_size = get_file_size(filled_form_path)
        if file_size == 0:
            raise HTTPException(
                status_code=500,
                detail="Filled form file is corrupted (empty file)"
            )
        
        # Get original form name for download filename
        original_name = _get_original_form_name(session_id, cache_service)
        download_filename = f"filled_{original_name}" if original_name else f"filled_form_{session_id}.pdf"
        
        # Log download attempt
        logger.info(f"Download request for filled form - Session: {session_id}, File: {filled_form_path.name}")
        
        # Return streaming response
        return _create_file_stream_response(
            file_path=filled_form_path,
            filename=download_filename,
            media_type="application/pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download filled form for session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while downloading filled form"
        )


@router.get("/{session_id}/report")
async def download_missing_fields_report(
    session_id: str,
    format: str = "markdown",
    settings: Settings = Depends(get_current_settings),
    file_storage=Depends(get_file_storage),
    cache_service=Depends(get_cache_service)
):
    """
    Download the missing fields report for a completed processing session.
    
    Args:
        session_id: Processing session identifier
        format: Report format (markdown, html, txt) - default: markdown
        
    Returns:
        StreamingResponse with the report file
        
    Raises:
        HTTPException: If session not found, processing incomplete, or file errors
    """
    try:
        # Validate session ID format
        if not validate_session_id(session_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid session ID format"
            )
        
        # Validate format parameter
        allowed_formats = ["markdown", "md", "html", "txt", "text"]
        if format.lower() not in allowed_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format. Allowed formats: {', '.join(allowed_formats)}"
            )
        
        # Normalize format
        if format.lower() in ["md", "markdown"]:
            format = "markdown"
            file_extension = "md"
            media_type = "text/markdown"
        elif format.lower() == "html":
            format = "html"
            file_extension = "html"
            media_type = "text/html"
        else:
            format = "text"
            file_extension = "txt"
            media_type = "text/plain"
        
        # Import processing sessions from process route
        from app.api.routes.process import processing_sessions
        
        # Check if processing is complete
        if session_id not in processing_sessions:
            raise HTTPException(
                status_code=404,
                detail="Session not found or processing not complete"
            )
        
        # Get report file path
        report_path = _get_report_path(session_id, file_storage, file_extension)
        
        if not report_path or not report_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Report file not found in {format} format"
            )
        
        # Validate file integrity
        file_size = get_file_size(report_path)
        if file_size == 0:
            raise HTTPException(
                status_code=500,
                detail="Report file is corrupted (empty file)"
            )
        
        # Generate download filename
        download_filename = f"missing_fields_report_{session_id}.{file_extension}"
        
        # Log download attempt
        logger.info(f"Download request for report - Session: {session_id}, Format: {format}, File: {report_path.name}")
        
        # Return streaming response
        return _create_file_stream_response(
            file_path=report_path,
            filename=download_filename,
            media_type=media_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download report for session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while downloading report"
        )


@router.get("/{session_id}/status")
async def get_download_status(
    session_id: str,
    settings: Settings = Depends(get_current_settings),
    file_storage=Depends(get_file_storage),
    cache_service=Depends(get_cache_service)
):
    """
    Get download availability status for a processing session.
    
    Args:
        session_id: Processing session identifier
        
    Returns:
        Dictionary with download availability status
    """
    try:
        # Validate session ID format
        if not validate_session_id(session_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid session ID format"
            )
        
        # Import processing sessions from process route
        from app.api.routes.process import processing_sessions
        
        # Check processing status
        if session_id not in processing_sessions:
            return {
                "session_id": session_id,
                "processing_complete": False,
                "filled_form_available": False,
                "report_available": False,
                "status": "Session not found"
            }
        
        session_data = processing_sessions[session_id]
        processing_status = session_data.get("status", "unknown")
        processing_complete = processing_status == "completed"
        
        # Check file availability
        filled_form_available = False
        report_available = False
        
        if processing_complete:
            # Check filled form
            filled_form_path = _get_filled_form_path(session_id, file_storage)
            filled_form_available = (
                filled_form_path and 
                filled_form_path.exists() and 
                get_file_size(filled_form_path) > 0
            )
            
            # Check report
            report_path = _get_report_path(session_id, file_storage, "md")
            report_available = (
                report_path and 
                report_path.exists() and 
                get_file_size(report_path) > 0
            )
        
        return {
            "session_id": session_id,
            "processing_complete": processing_complete,
            "processing_status": processing_status,
            "filled_form_available": filled_form_available,
            "report_available": report_available,
            "available_formats": ["markdown", "html", "txt"] if report_available else [],
            "download_urls": {
                "filled_form": f"/api/download/{session_id}/filled" if filled_form_available else None,
                "report_markdown": f"/api/download/{session_id}/report?format=markdown" if report_available else None,
                "report_html": f"/api/download/{session_id}/report?format=html" if report_available else None,
                "report_text": f"/api/download/{session_id}/report?format=txt" if report_available else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get download status for session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while checking download status"
        )


def _get_filled_form_path(session_id: str, file_storage) -> Optional[Path]:
    """Get the path to the filled form PDF file."""
    try:
        session_dir = file_storage.get_session_directory(session_id)
        if not session_dir or not session_dir.exists():
            return None
        
        # Look for filled form files
        possible_names = [
            "filled_form.pdf",
            f"filled_{session_id}.pdf",
            "output.pdf"
        ]
        
        for name in possible_names:
            file_path = session_dir / name
            if file_path.exists():
                return file_path
        
        # Look for any PDF file in outputs directory
        outputs_dir = session_dir / "outputs"
        if outputs_dir.exists():
            pdf_files = list(outputs_dir.glob("*.pdf"))
            if pdf_files:
                return pdf_files[0]  # Return first PDF found
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get filled form path for session {session_id}: {e}")
        return None


def _get_report_path(session_id: str, file_storage, extension: str) -> Optional[Path]:
    """Get the path to the report file."""
    try:
        session_dir = file_storage.get_session_directory(session_id)
        if not session_dir or not session_dir.exists():
            return None
        
        # Look for report files
        possible_names = [
            f"missing_fields_report.{extension}",
            f"report.{extension}",
            f"missing_fields_{session_id}.{extension}"
        ]
        
        for name in possible_names:
            file_path = session_dir / name
            if file_path.exists():
                return file_path
        
        # Look in outputs directory
        outputs_dir = session_dir / "outputs"
        if outputs_dir.exists():
            for name in possible_names:
                file_path = outputs_dir / name
                if file_path.exists():
                    return file_path
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get report path for session {session_id}: {e}")
        return None


def _get_original_form_name(session_id: str, cache_service) -> Optional[str]:
    """Get the original PA form filename from cache."""
    try:
        processing_result = cache_service.get_cached_processing_result(session_id)
        if processing_result and "pa_form_filename" in processing_result:
            return processing_result["pa_form_filename"]
        return None
    except Exception:
        return None


def _create_file_stream_response(
    file_path: Path,
    filename: str,
    media_type: str
) -> StreamingResponse:
    """
    Create a streaming response for file download.
    
    Args:
        file_path: Path to the file to stream
        filename: Filename for download
        media_type: MIME type for the response
        
    Returns:
        StreamingResponse with proper headers
    """
    try:
        # Get file size for Content-Length header
        file_size = file_path.stat().st_size
        
        # Create file iterator for streaming
        def file_iterator():
            with open(file_path, "rb") as file:
                while chunk := file.read(8192):  # 8KB chunks
                    yield chunk
        
        # Create response headers
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(file_size),
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        return StreamingResponse(
            content=file_iterator(),
            media_type=media_type,
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Failed to create streaming response for {file_path}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to prepare file for download"
        )


def _cleanup_temp_file(file_path: Path):
    """Background task to clean up temporary files."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")