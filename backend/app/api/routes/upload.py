"""
Upload endpoint for handling PA form and referral packet uploads.
"""
from typing import List
import uuid
import os
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.models.schemas import UploadResponse
from app.utils.file_handler import validate_pdf, get_file_size, generate_session_id
from app.services.storage import FileStorage

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    settings: Settings = Depends(get_settings)
) -> UploadResponse:
    """
    Upload PA form and referral packet files for processing.
    
    Args:
        files: List of exactly 2 PDF files (referral packet + PA form)
        settings: Application settings
        
    Returns:
        UploadResponse with session ID and file information
        
    Raises:
        HTTPException: For validation errors or file handling issues
    """
    # Validate file count
    if len(files) != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Exactly 2 files required: referral packet and PA form"
        )
    
    # Generate unique session ID
    session_id = generate_session_id()
    
    try:
        # Initialize file storage service
        storage = FileStorage(settings.upload_dir)
        
        # Create session directory
        session_dir = await storage.create_session_directory(session_id)
        
        uploaded_files = []
        total_size = 0
        
        for file in files:
            # Validate file is PDF
            if not validate_pdf(file):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {file.filename} is not a valid PDF"
                )
            
            # Check file size
            file_size = get_file_size(file.file)
            if file_size > settings.max_file_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File {file.filename} exceeds maximum size of {settings.max_file_size} bytes"
                )
            
            total_size += file_size
            
            # Reset file pointer after size check
            file.file.seek(0)
            
            # Save file to session directory
            file_path = await storage.save_file(file, session_id)
            
            uploaded_files.append({
                "filename": file.filename,
                "file_path": file_path,
                "size": file_size,
                "content_type": file.content_type
            })
        
        # Classify files based on naming patterns and size
        referral_file = None
        pa_form_file = None
        
        for file_info in uploaded_files:
            filename_lower = file_info["filename"].lower()
            
            # Classify based on filename patterns
            if any(keyword in filename_lower for keyword in ["referral", "package", "packet"]):
                referral_file = file_info
            elif any(keyword in filename_lower for keyword in ["pa", "prior", "auth", "form"]):
                pa_form_file = file_info
            else:
                # Fallback: classify by file size (referral packets are typically larger)
                if file_info["size"] > max(f["size"] for f in uploaded_files if f != file_info):
                    referral_file = file_info
                else:
                    pa_form_file = file_info
        
        # Ensure both files are classified
        if not referral_file or not pa_form_file:
            # If classification failed, assign by size
            sorted_files = sorted(uploaded_files, key=lambda x: x["size"], reverse=True)
            referral_file = sorted_files[0]  # Larger file assumed to be referral
            pa_form_file = sorted_files[1]   # Smaller file assumed to be PA form
        
        return UploadResponse(
            session_id=session_id,
            message="Files uploaded successfully",
            referral_file={
                "filename": referral_file["filename"],
                "size": referral_file["size"],
                "file_path": referral_file["file_path"]
            },
            pa_form_file={
                "filename": pa_form_file["filename"],
                "size": pa_form_file["size"],
                "file_path": pa_form_file["file_path"]
            },
            total_size=total_size,
            upload_timestamp=None  # Will be set by Pydantic default
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload files: {str(e)}"
        )


@router.get("/upload/status/{session_id}")
async def get_upload_status(
    session_id: str,
    settings: Settings = Depends(get_settings)
) -> JSONResponse:
    """
    Get upload status and file information for a session.
    
    Args:
        session_id: Session identifier
        settings: Application settings
        
    Returns:
        JSON response with session status
    """
    try:
        storage = FileStorage(settings.upload_dir)
        session_dir = os.path.join(settings.upload_dir, session_id)
        
        if not os.path.exists(session_dir):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        # List files in session directory
        files = []
        for filename in os.listdir(session_dir):
            file_path = os.path.join(session_dir, filename)
            if os.path.isfile(file_path):
                files.append({
                    "filename": filename,
                    "size": os.path.getsize(file_path),
                    "file_path": file_path
                })
        
        return JSONResponse(content={
            "session_id": session_id,
            "files": files,
            "file_count": len(files),
            "status": "uploaded"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get upload status: {str(e)}"
        )