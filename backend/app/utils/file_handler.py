# backend/app/utils/file_handler.py
"""
File handling utilities for PDF validation, session management, and file operations.

This module provides utility functions for validating PDF files, managing file sizes,
generating unique session identifiers, and supporting the upload workflow.
"""

import hashlib
import logging
import magic
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def validate_pdf(
    file: Union[UploadFile, Path, bytes], 
    check_content: bool = True,
    max_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Comprehensive PDF file validation.
    
    Validates file type, size, content integrity, and basic structure
    to ensure it's a valid PDF that can be processed by the system.
    
    Args:
        file: UploadFile instance, file path, or bytes content
        check_content: Whether to perform deep content validation
        max_size: Maximum file size in bytes (defaults to config setting)
        
    Returns:
        Dictionary containing validation results:
        - is_valid: Boolean indicating if file is valid
        - file_type: Detected MIME type
        - file_size: File size in bytes
        - validation_details: Additional validation information
        - errors: List of validation errors if any
        - warnings: List of non-critical issues
        
    Raises:
        HTTPException: If file cannot be read or processed
    """
    max_file_size = max_size or settings.MAX_FILE_SIZE
    validation_result = {
        "is_valid": True,
        "file_type": None,
        "file_size": 0,
        "validation_details": {},
        "errors": [],
        "warnings": []
    }
    
    try:
        # Get file content and basic info
        if isinstance(file, UploadFile):
            content = _read_upload_file_content(file)
            filename = file.filename or "unknown.pdf"
            content_type = file.content_type
        elif isinstance(file, Path):
            content = file.read_bytes()
            filename = file.name
            content_type = None
        elif isinstance(file, bytes):
            content = file
            filename = "content.pdf"
            content_type = None
        else:
            raise ValueError(f"Unsupported file type: {type(file)}")
        
        file_size = len(content)
        validation_result["file_size"] = file_size
        
        # 1. File size validation
        if file_size == 0:
            validation_result["errors"].append("File is empty")
            validation_result["is_valid"] = False
            return validation_result
        
        if file_size > max_file_size:
            validation_result["errors"].append(
                f"File size ({file_size} bytes) exceeds maximum allowed size ({max_file_size} bytes)"
            )
            validation_result["is_valid"] = False
        
        # 2. MIME type detection using python-magic
        try:
            detected_mime = magic.from_buffer(content, mime=True)
            validation_result["file_type"] = detected_mime
            
            # Check if detected type is PDF
            if detected_mime != "application/pdf":
                validation_result["errors"].append(
                    f"File is not a PDF. Detected type: {detected_mime}"
                )
                validation_result["is_valid"] = False
        except Exception as e:
            validation_result["warnings"].append(f"Could not detect MIME type: {str(e)}")
            # Fall back to content type from upload
            if content_type:
                validation_result["file_type"] = content_type
        
        # 3. Filename validation
        filename_lower = filename.lower()
        if not filename_lower.endswith('.pdf'):
            validation_result["warnings"].append("Filename does not end with .pdf extension")
        
        # 4. PDF content validation
        if check_content:
            pdf_validation = _validate_pdf_content(content)
            validation_result["validation_details"].update(pdf_validation)
            
            if not pdf_validation["has_pdf_header"]:
                validation_result["errors"].append("File does not have valid PDF header")
                validation_result["is_valid"] = False
            
            if not pdf_validation["has_pdf_trailer"]:
                validation_result["warnings"].append("File may be truncated or corrupted")
            
            if pdf_validation["page_count"] == 0:
                validation_result["warnings"].append("Could not determine page count")
            elif pdf_validation["page_count"] > 50:
                validation_result["warnings"].append(
                    f"Large document ({pdf_validation['page_count']} pages) may require chunking"
                )
        
        # 5. Generate file hash for integrity checking
        file_hash = hashlib.sha256(content).hexdigest()
        validation_result["validation_details"]["file_hash"] = file_hash
        validation_result["validation_details"]["filename"] = filename
        
        logger.info(
            f"PDF validation completed for {filename}: "
            f"valid={validation_result['is_valid']}, size={file_size}, "
            f"errors={len(validation_result['errors'])}, warnings={len(validation_result['warnings'])}"
        )
        
        return validation_result
        
    except Exception as e:
        logger.error(f"PDF validation failed: {e}")
        validation_result["is_valid"] = False
        validation_result["errors"].append(f"Validation error: {str(e)}")
        return validation_result


def _read_upload_file_content(upload_file: UploadFile) -> bytes:
    """
    Read content from UploadFile with proper position management.
    
    Args:
        upload_file: FastAPI UploadFile instance
        
    Returns:
        File content as bytes
    """
    # Save current position
    current_position = upload_file.file.tell()
    
    try:
        # Read content from beginning
        upload_file.file.seek(0)
        content = upload_file.file.read()
        return content
    finally:
        # Restore original position
        upload_file.file.seek(current_position)


def _validate_pdf_content(content: bytes) -> Dict[str, Any]:
    """
    Validate PDF content structure and extract basic information.
    
    Args:
        content: PDF file content as bytes
        
    Returns:
        Dictionary with PDF structure validation results
    """
    pdf_info = {
        "has_pdf_header": False,
        "has_pdf_trailer": False,
        "pdf_version": None,
        "page_count": 0,
        "is_encrypted": False,
        "content_length": len(content)
    }
    
    try:
        # Check PDF header
        if content.startswith(b'%PDF-'):
            pdf_info["has_pdf_header"] = True
            # Extract PDF version
            header_line = content[:20].decode('ascii', errors='ignore')
            if '%PDF-' in header_line:
                version_start = header_line.find('%PDF-') + 5
                version_end = version_start + 3
                pdf_info["pdf_version"] = header_line[version_start:version_end]
        
        # Check for PDF trailer
        if b'%%EOF' in content[-1024:]:  # Check last 1KB for EOF marker
            pdf_info["has_pdf_trailer"] = True
        
        # Check for encryption
        if b'/Encrypt' in content:
            pdf_info["is_encrypted"] = True
        
        # Attempt to count pages using pdfplumber
        try:
            import pdfplumber
            from io import BytesIO
            
            with pdfplumber.open(BytesIO(content)) as pdf:
                pdf_info["page_count"] = len(pdf.pages)
                
        except Exception as e:
            logger.debug(f"Could not count pages with pdfplumber: {e}")
            # Fall back to simple page count estimation
            pdf_info["page_count"] = content.count(b'/Type/Page')
        
    except Exception as e:
        logger.warning(f"PDF content validation failed: {e}")
    
    return pdf_info


def get_file_size(file: Union[UploadFile, Path, str]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file: UploadFile instance, Path object, or file path string
        
    Returns:
        File size in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
        OSError: If file cannot be accessed
    """
    try:
        if isinstance(file, UploadFile):
            # For UploadFile, we need to read the content to get accurate size
            current_position = file.file.tell()
            try:
                file.file.seek(0, 2)  # Seek to end
                size = file.file.tell()
                return size
            finally:
                file.file.seek(current_position)  # Restore position
                
        elif isinstance(file, (Path, str)):
            file_path = Path(file)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            return file_path.stat().st_size
        else:
            raise ValueError(f"Unsupported file type: {type(file)}")
            
    except Exception as e:
        logger.error(f"Failed to get file size: {e}")
        raise


def generate_session_id(prefix: str = "pa_session", include_timestamp: bool = True) -> str:
    """
    Generate a unique session identifier.
    
    Creates a unique session ID using UUID4 with optional timestamp
    and custom prefix for organizing processing sessions.
    
    Args:
        prefix: Prefix for the session ID (default: "pa_session")
        include_timestamp: Whether to include timestamp in ID
        
    Returns:
        Unique session identifier string
        
    Example:
        "pa_session_20240611_abc123def456"
    """
    # Generate base UUID
    session_uuid = uuid.uuid4().hex[:12]  # Use first 12 characters
    
    if include_timestamp:
        # Add timestamp for better organization and debugging
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_id = f"{prefix}_{timestamp}_{session_uuid}"
    else:
        session_id = f"{prefix}_{session_uuid}"
    
    logger.debug(f"Generated session ID: {session_id}")
    return session_id


def create_session_metadata(
    session_id: str,
    uploaded_files: list,
    user_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create metadata for a processing session.
    
    Args:
        session_id: Unique session identifier
        uploaded_files: List of uploaded file information
        user_info: Optional user information
        
    Returns:
        Session metadata dictionary
    """
    metadata = {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "initialized",
        "files": [],
        "total_files": len(uploaded_files),
        "total_size": 0,
        "user_info": user_info or {}
    }
    
    # Process file information
    for file_info in uploaded_files:
        file_meta = {
            "filename": file_info.get("filename", "unknown"),
            "size": file_info.get("size", 0),
            "content_type": file_info.get("content_type", "application/pdf"),
            "file_hash": file_info.get("file_hash"),
            "upload_timestamp": file_info.get("saved_at")
        }
        metadata["files"].append(file_meta)
        metadata["total_size"] += file_meta["size"]
    
    return metadata


def classify_uploaded_files(files_info: list) -> Dict[str, Any]:
    """
    Classify uploaded files into referral packets and PA forms.
    
    Analyzes uploaded files to determine which are likely referral packets
    vs PA forms based on filename patterns and characteristics.
    
    Args:
        files_info: List of file information dictionaries
        
    Returns:
        Dictionary with classified files:
        - referral_files: List of likely referral packet files
        - pa_form_files: List of likely PA form files
        - unclassified_files: Files that couldn't be classified
        - classification_confidence: Overall confidence in classification
    """
    classification = {
        "referral_files": [],
        "pa_form_files": [],
        "unclassified_files": [],
        "classification_confidence": 0.0
    }
    
    # Keywords that suggest referral packets
    referral_keywords = [
        "referral", "packet", "medical", "chart", "record", "history",
        "clinical", "notes", "consultation", "report", "summary"
    ]
    
    # Keywords that suggest PA forms
    pa_form_keywords = [
        "pa", "prior", "authorization", "auth", "form", "request",
        "aetna", "anthem", "cigna", "humana", "bcbs", "medicare"
    ]
    
    for file_info in files_info:
        filename = file_info.get("filename", "").lower()
        file_size = file_info.get("size", 0)
        
        referral_score = sum(1 for keyword in referral_keywords if keyword in filename)
        pa_form_score = sum(1 for keyword in pa_form_keywords if keyword in filename)
        
        # Size-based heuristics (referral packets are typically larger)
        if file_size > 5 * 1024 * 1024:  # > 5MB likely referral
            referral_score += 2
        elif file_size < 1 * 1024 * 1024:  # < 1MB likely PA form
            pa_form_score += 1
        
        # Classify based on scores
        if referral_score > pa_form_score:
            classification["referral_files"].append({
                **file_info,
                "classification_score": referral_score,
                "confidence": min(referral_score / 3.0, 1.0)
            })
        elif pa_form_score > referral_score:
            classification["pa_form_files"].append({
                **file_info,
                "classification_score": pa_form_score,
                "confidence": min(pa_form_score / 3.0, 1.0)
            })
        else:
            classification["unclassified_files"].append({
                **file_info,
                "classification_score": 0,
                "confidence": 0.0
            })
    
    # Calculate overall confidence
    total_files = len(files_info)
    classified_files = len(classification["referral_files"]) + len(classification["pa_form_files"])
    
    if total_files > 0:
        classification["classification_confidence"] = classified_files / total_files
    
    logger.info(
        f"File classification: {len(classification['referral_files'])} referral, "
        f"{len(classification['pa_form_files'])} PA forms, "
        f"{len(classification['unclassified_files'])} unclassified "
        f"(confidence: {classification['classification_confidence']:.2f})"
    )
    
    return classification


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe storage.
    
    Removes or replaces dangerous characters and ensures filename
    is safe for filesystem storage across different operating systems.
    
    Args:
        filename: Original filename
        max_length: Maximum allowed filename length
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return f"file_{uuid.uuid4().hex[:8]}.pdf"
    
    # Replace dangerous characters
    dangerous_chars = '<>:"/\\|?*'
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Ensure it's not empty after sanitization
    if not filename.strip():
        filename = f"file_{uuid.uuid4().hex[:8]}.pdf"
    
    # Truncate if too long
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_length = max_length - len(ext) - 1 if ext else max_length
        filename = f"{name[:max_name_length]}.{ext}" if ext else name[:max_length]
    
    return filename


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size string (e.g., "1.5 MB", "500 KB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def validate_session_id(session_id: str) -> bool:
    """
    Validate session ID format and security.
    
    Args:
        session_id: Session identifier to validate
        
    Returns:
        True if valid, False otherwise
    """
    import re
    
    # Basic format validation (alphanumeric, hyphens, underscores)
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        return False
    
    # Length validation (between 8 and 50 characters)
    if len(session_id) < 8 or len(session_id) > 50:
        return False
    
    # Security: prevent directory traversal attempts
    if '..' in session_id or '/' in session_id or '\\' in session_id:
        return False
    
    return True


def validate_file_type_by_extension(filename: str, allowed_extensions: list = None) -> bool:
    """
    Validate file type by extension.
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions (default: ['.pdf'])
        
    Returns:
        True if file extension is allowed, False otherwise
    """
    if not filename:
        return False
    
    allowed = allowed_extensions or ['.pdf']
    file_ext = Path(filename).suffix.lower()
    
    return file_ext in [ext.lower() for ext in allowed]