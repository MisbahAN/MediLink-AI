"""
Process endpoint for handling document processing requests and status tracking.

This module provides endpoints for initiating document processing, tracking progress,
and retrieving processing results with comprehensive error handling and background
task management.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse

from core.config import Settings
from core.deps import get_current_settings
from models.schemas import (
    ProcessingStatus, ProcessingResult, ProcessingStatusEnum, ErrorResponse
)
from services.processing_pipeline import get_processing_pipeline
from services.storage import get_file_storage
from utils.file_handler import validate_pdf

logger = logging.getLogger(__name__)
router = APIRouter()

# Global storage for tracking processing sessions
# In production, this should be replaced with Redis or database
processing_sessions: Dict[str, Dict[str, Any]] = {}


async def progress_callback(progress_data: Dict[str, Any]) -> None:
    """
    Callback function for processing progress updates.
    
    Args:
        progress_data: Progress information from processing pipeline
    """
    session_id = progress_data.get("session_id")
    if session_id and session_id in processing_sessions:
        # Update session progress
        processing_sessions[session_id].update({
            "status": progress_data.get("stage", "unknown"),
            "progress_percentage": progress_data.get("progress", 0),
            "current_step": progress_data.get("message", "Processing..."),
            "last_updated": datetime.now(timezone.utc),
            "stage_description": progress_data.get("message", "")
        })
        
        logger.info(f"Session {session_id} progress: {progress_data.get('progress', 0)}% - {progress_data.get('message', '')}")


async def process_documents_background(
    session_id: str,
    referral_file_path: str,
    pa_form_file_path: str,
    settings: Settings
) -> None:
    """
    Background task for processing documents.
    
    Args:
        session_id: Unique session identifier
        referral_file_path: Path to referral packet PDF
        pa_form_file_path: Path to PA form PDF
        settings: Application settings
    """
    try:
        logger.info(f"Starting background processing for session {session_id}")
        
        # Initialize session tracking
        processing_sessions[session_id] = {
            "status": ProcessingStatusEnum.EXTRACTING,
            "progress_percentage": 0,
            "current_step": "Initializing processing pipeline...",
            "total_steps": 6,
            "stage_description": "Starting document processing",
            "last_updated": datetime.now(timezone.utc),
            "started_at": datetime.now(timezone.utc),
            "error_message": None,
            "result": None
        }
        
        # Get processing pipeline
        pipeline = get_processing_pipeline()
        
        # Process documents with progress tracking
        result = await pipeline.process_documents(
            session_id=session_id,
            referral_pdf_path=referral_file_path,
            pa_form_pdf_path=pa_form_file_path,
            progress_callback=progress_callback
        )
        
        # Update session with final result
        processing_sessions[session_id].update({
            "status": result.processing_status,
            "progress_percentage": 100,
            "current_step": "Processing completed",
            "stage_description": "Document processing finished successfully",
            "last_updated": datetime.now(timezone.utc),
            "completed_at": datetime.now(timezone.utc),
            "result": result,
            "error_message": None
        })
        
        logger.info(f"Background processing completed successfully for session {session_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for session {session_id}: {e}")
        
        # Update session with error
        if session_id in processing_sessions:
            processing_sessions[session_id].update({
                "status": ProcessingStatusEnum.FAILED,
                "current_step": f"Processing failed: {str(e)}",
                "stage_description": "An error occurred during processing",
                "last_updated": datetime.now(timezone.utc),
                "error_message": str(e),
                "result": None
            })


@router.post("/process/{session_id}")
async def start_processing(
    session_id: str,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_current_settings)
) -> JSONResponse:
    """
    Start document processing for uploaded files.
    
    Args:
        session_id: Session ID from file upload
        background_tasks: FastAPI background tasks
        settings: Application settings
        
    Returns:
        Processing initiation response with status
        
    Raises:
        HTTPException: If session not found or files invalid
    """
    try:
        logger.info(f"Processing request received for session {session_id}")
        
        # Get file storage service
        file_storage = get_file_storage()
        
        # Verify session exists and get file paths
        session_info = await _get_session_files(session_id, file_storage)
        if not session_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found or files missing"
            )
        
        referral_path = session_info["referral_path"]
        pa_form_path = session_info["pa_form_path"]
        
        # Validate files exist and are accessible
        if not Path(referral_path).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Referral file not found: {referral_path}"
            )
        
        if not Path(pa_form_path).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PA form file not found: {pa_form_path}"
            )
        
        # Validate PDF files
        referral_validation = validate_pdf(referral_path)
        if not referral_validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid referral PDF: {referral_validation['error']}"
            )
        
        pa_form_validation = validate_pdf(pa_form_path)
        if not pa_form_validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid PA form PDF: {pa_form_validation['error']}"
            )
        
        # Check if processing is already in progress
        if session_id in processing_sessions:
            current_status = processing_sessions[session_id].get("status")
            if current_status not in [ProcessingStatusEnum.COMPLETED, ProcessingStatusEnum.FAILED]:
                return JSONResponse(
                    status_code=status.HTTP_409_CONFLICT,
                    content={
                        "message": "Processing already in progress for this session",
                        "session_id": session_id,
                        "current_status": current_status,
                        "progress": processing_sessions[session_id].get("progress_percentage", 0)
                    }
                )
        
        # Add background task for processing
        background_tasks.add_task(
            process_documents_background,
            session_id,
            referral_path,
            pa_form_path,
            settings
        )
        
        logger.info(f"Background processing task queued for session {session_id}")
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "message": "Document processing started",
                "session_id": session_id,
                "status": "processing",
                "estimated_time_minutes": 5,
                "status_endpoint": f"/api/process/{session_id}/status",
                "initiated_at": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start processing for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start processing: {str(e)}"
        )


@router.get("/process/{session_id}/status", response_model=ProcessingStatus)
async def get_processing_status(
    session_id: str,
    settings: Settings = Depends(get_current_settings)
) -> ProcessingStatus:
    """
    Get current processing status for a session.
    
    Args:
        session_id: Session ID to check status for
        settings: Application settings
        
    Returns:
        Current processing status and progress
        
    Raises:
        HTTPException: If session not found
    """
    try:
        logger.debug(f"Status check requested for session {session_id}")
        
        # Check if session exists in processing tracking
        if session_id not in processing_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processing session {session_id} not found"
            )
        
        session_data = processing_sessions[session_id]
        
        # Calculate estimated time remaining
        estimated_time_remaining = None
        if session_data.get("status") not in [ProcessingStatusEnum.COMPLETED, ProcessingStatusEnum.FAILED]:
            progress = session_data.get("progress_percentage", 0)
            if progress > 0:
                started_at = session_data.get("started_at")
                if started_at:
                    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
                    if progress < 100:
                        estimated_total = (elapsed / progress) * 100
                        estimated_time_remaining = max(0, int(estimated_total - elapsed))
        
        # Create processing status response
        processing_status = ProcessingStatus(
            session_id=session_id,
            status=session_data.get("status", ProcessingStatusEnum.PENDING),
            stage_description=session_data.get("stage_description", "Processing documents"),
            progress_percentage=session_data.get("progress_percentage", 0),
            current_step=session_data.get("current_step", "Initializing..."),
            total_steps=session_data.get("total_steps", 6),
            estimated_time_remaining=estimated_time_remaining,
            error_message=session_data.get("error_message"),
            last_updated=session_data.get("last_updated", datetime.now(timezone.utc))
        )
        
        return processing_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get processing status: {str(e)}"
        )


@router.get("/process/{session_id}/result")
async def get_processing_result(
    session_id: str,
    settings: Settings = Depends(get_current_settings)
) -> JSONResponse:
    """
    Get the final processing result for a completed session.
    
    Args:
        session_id: Session ID to get results for
        settings: Application settings
        
    Returns:
        Processing result with mapped fields and missing fields report
        
    Raises:
        HTTPException: If session not found or processing not completed
    """
    try:
        logger.info(f"Result requested for session {session_id}")
        
        # Check if session exists
        if session_id not in processing_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processing session {session_id} not found"
            )
        
        session_data = processing_sessions[session_id]
        
        # Check if processing is completed
        if session_data.get("status") != ProcessingStatusEnum.COMPLETED:
            current_status = session_data.get("status", "unknown")
            if current_status == ProcessingStatusEnum.FAILED:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Processing failed: {session_data.get('error_message', 'Unknown error')}"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_202_ACCEPTED,
                    detail=f"Processing not completed. Current status: {current_status}"
                )
        
        # Get processing result
        result = session_data.get("result")
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Processing result not available"
            )
        
        # Prepare response data
        response_data = {
            "session_id": session_id,
            "processing_status": result.processing_status,
            "completion_summary": {
                "total_pa_fields": len(result.pa_form_fields),
                "successfully_mapped": len([f for f in result.pa_form_fields.values() if f.mapped_value]),
                "missing_fields": len(result.missing_fields),
                "completion_rate": result.processing_summary.get("completion_rate", 0.0),
                "processing_duration": result.processing_duration
            },
            "field_mappings": {
                field_id: {
                    "field_name": field.field_name,
                    "mapped_value": field.mapped_value,
                    "confidence": field.confidence,
                    "confidence_level": field.confidence_level,
                    "required": field.required
                }
                for field_id, field in result.pa_form_fields.items()
                if field.mapped_value
            },
            "missing_fields": [
                {
                    "field_name": field.field_name,
                    "display_label": field.display_label,
                    "required": field.required,
                    "reason": field.reason,
                    "priority": field.priority,
                    "suggested_value": field.suggested_value
                }
                for field in result.missing_fields
            ],
            "extraction_summary": result.extracted_data.extraction_summary,
            "completed_at": result.completed_at.isoformat() if result.completed_at else None
        }
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get result for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get processing result: {str(e)}"
        )


@router.delete("/process/{session_id}")
async def cancel_processing(
    session_id: str,
    settings: Settings = Depends(get_current_settings)
) -> JSONResponse:
    """
    Cancel ongoing processing for a session.
    
    Args:
        session_id: Session ID to cancel
        settings: Application settings
        
    Returns:
        Cancellation confirmation
        
    Raises:
        HTTPException: If session not found
    """
    try:
        logger.info(f"Cancellation requested for session {session_id}")
        
        # Check if session exists
        if session_id not in processing_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processing session {session_id} not found"
            )
        
        session_data = processing_sessions[session_id]
        current_status = session_data.get("status")
        
        # Check if cancellation is possible
        if current_status in [ProcessingStatusEnum.COMPLETED, ProcessingStatusEnum.FAILED]:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "message": f"Session {session_id} already {current_status.lower()}",
                    "session_id": session_id,
                    "status": current_status
                }
            )
        
        # Mark session as cancelled
        processing_sessions[session_id].update({
            "status": ProcessingStatusEnum.FAILED,
            "current_step": "Processing cancelled by user",
            "stage_description": "Processing was cancelled",
            "error_message": "Processing cancelled by user request",
            "last_updated": datetime.now(timezone.utc)
        })
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": f"Processing cancelled for session {session_id}",
                "session_id": session_id,
                "cancelled_at": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel processing for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel processing: {str(e)}"
        )


async def _get_session_files(session_id: str, file_storage) -> Optional[Dict[str, str]]:
    """
    Get file paths for a session from storage.
    
    Args:
        session_id: Session identifier
        file_storage: File storage service
        
    Returns:
        Dictionary with file paths or None if not found
    """
    try:
        # This is a simplified implementation
        # In practice, you would check the file storage service for uploaded files
        session_dir = Path(file_storage.upload_dir) / session_id
        
        if not session_dir.exists():
            return None
        
        # Look for uploaded files in session directory
        referral_files = list(session_dir.glob("*referral*.pdf"))
        pa_form_files = list(session_dir.glob("*PA*.pdf")) or list(session_dir.glob("*pa*.pdf"))
        
        if not referral_files or not pa_form_files:
            # Try to find any two PDF files if naming convention not followed
            pdf_files = list(session_dir.glob("*.pdf"))
            if len(pdf_files) >= 2:
                return {
                    "referral_path": str(pdf_files[0]),
                    "pa_form_path": str(pdf_files[1])
                }
            return None
        
        return {
            "referral_path": str(referral_files[0]),
            "pa_form_path": str(pa_form_files[0])
        }
        
    except Exception as e:
        logger.error(f"Failed to get session files for {session_id}: {e}")
        return None


@router.get("/process/sessions/active")
async def get_active_sessions(
    settings: Settings = Depends(get_current_settings)
) -> JSONResponse:
    """
    Get list of currently active processing sessions.
    
    Args:
        settings: Application settings
        
    Returns:
        List of active sessions with status
    """
    try:
        active_sessions = []
        
        for session_id, session_data in processing_sessions.items():
            status_value = session_data.get("status")
            if status_value not in [ProcessingStatusEnum.COMPLETED, ProcessingStatusEnum.FAILED]:
                active_sessions.append({
                    "session_id": session_id,
                    "status": status_value,
                    "progress": session_data.get("progress_percentage", 0),
                    "started_at": session_data.get("started_at", "").isoformat() if session_data.get("started_at") else None,
                    "last_updated": session_data.get("last_updated", "").isoformat() if session_data.get("last_updated") else None
                })
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "active_sessions": active_sessions,
                "total_active": len(active_sessions),
                "checked_at": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get active sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active sessions: {str(e)}"
        )