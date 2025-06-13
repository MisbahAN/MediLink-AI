"""
File storage service for managing uploaded PDFs and processing results.

This service handles secure file storage and retrieval for the Prior Authorization
automation system, including session-based file organization and cleanup.
"""

import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import aiofiles
import logging
from fastapi import UploadFile, HTTPException

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FileStorage:
    """
    Manages file storage operations for uploaded PDFs and processing results.
    
    Provides session-based file organization with automatic cleanup and
    secure file handling for the PA automation workflow.
    """
    
    def __init__(self, base_upload_dir: Optional[str] = None):
        """
        Initialize file storage service.
        
        Args:
            base_upload_dir: Base directory for file uploads (defaults to config setting)
        """
        self.base_upload_dir = Path(base_upload_dir or settings.UPLOAD_DIR)
        self.max_file_size = settings.MAX_FILE_SIZE
        self.allowed_types = settings.ALLOWED_FILE_TYPES
        
        # Ensure base upload directory exists
        self.base_upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"File storage initialized with base directory: {self.base_upload_dir}")
    
    def create_session_directory(self, session_id: str) -> Path:
        """
        Create a session-specific directory for file storage.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Path to created session directory
            
        Raises:
            OSError: If directory creation fails
        """
        session_dir = self.base_upload_dir / session_id
        
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created session directory: {session_dir}")
            return session_dir
        except OSError as e:
            logger.error(f"Failed to create session directory {session_dir}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create storage directory for session {session_id}"
            )
    
    def _validate_file(self, file: UploadFile) -> None:
        """
        Validate uploaded file against size and type constraints.
        
        Args:
            file: FastAPI UploadFile instance
            
        Raises:
            HTTPException: If file validation fails
        """
        # Check file type
        if file.content_type not in self.allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file.content_type} not allowed. Accepted types: {self.allowed_types}"
            )
        
        # Check file size (if available in headers)
        if hasattr(file, 'size') and file.size and file.size > self.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({file.size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)"
            )
        
        # Validate filename
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="File must have a valid PDF filename"
            )
    
    async def save_file(
        self, 
        file: UploadFile, 
        session_id: str, 
        custom_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save uploaded file to session directory.
        
        Args:
            file: FastAPI UploadFile instance
            session_id: Session identifier for file organization
            custom_filename: Optional custom filename (defaults to original)
            
        Returns:
            Dictionary containing file metadata:
            - file_path: Full path to saved file
            - filename: Final filename used
            - size: File size in bytes
            - content_type: MIME content type
            - saved_at: Timestamp of save operation
            
        Raises:
            HTTPException: If file validation or saving fails
        """
        # Validate file
        self._validate_file(file)
        
        # Create session directory
        session_dir = self.create_session_directory(session_id)
        
        # Determine filename
        filename = custom_filename or file.filename
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        
        # Sanitize filename to prevent path traversal
        filename = self._sanitize_filename(filename)
        file_path = session_dir / filename
        
        # Save file content
        try:
            file_size = 0
            async with aiofiles.open(file_path, 'wb') as dest_file:
                # Read and write file in chunks to handle large files
                while chunk := await file.read(8192):  # 8KB chunks
                    file_size += len(chunk)
                    
                    # Check size during reading if not available in headers
                    if file_size > self.max_file_size:
                        # Clean up partially written file
                        await dest_file.close()
                        file_path.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413,
                            detail=f"File size exceeds maximum allowed size ({self.max_file_size} bytes)"
                        )
                    
                    await dest_file.write(chunk)
            
            logger.info(f"Successfully saved file {filename} ({file_size} bytes) for session {session_id}")
            
            return {
                "file_path": str(file_path),
                "filename": filename,
                "size": file_size,
                "content_type": file.content_type,
                "saved_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            # Clean up file if save failed
            file_path.unlink(missing_ok=True)
            logger.error(f"Failed to save file {filename} for session {session_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save file: {str(e)}"
            )
        finally:
            # Reset file position for potential reuse
            await file.seek(0)
    
    def get_file(self, session_id: str, filename: str) -> Optional[Path]:
        """
        Get path to a file in session directory.
        
        Args:
            session_id: Session identifier
            filename: Name of file to retrieve
            
        Returns:
            Path object if file exists, None otherwise
        """
        session_dir = self.base_upload_dir / session_id
        file_path = session_dir / filename
        
        # Verify file exists and is within session directory (security check)
        if file_path.exists() and file_path.is_file():
            try:
                # Resolve any symbolic links and check if still within session directory
                resolved_path = file_path.resolve()
                resolved_session_dir = session_dir.resolve()
                
                if resolved_path.is_relative_to(resolved_session_dir):
                    return file_path
                else:
                    logger.warning(f"File access attempt outside session directory: {file_path}")
                    return None
            except (OSError, ValueError):
                return None
        
        return None
    
    def list_files(self, session_id: str) -> List[Dict[str, Any]]:
        """
        List all files in a session directory.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of dictionaries containing file metadata
        """
        session_dir = self.base_upload_dir / session_id
        
        if not session_dir.exists():
            return []
        
        files = []
        try:
            for file_path in session_dir.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        "filename": file_path.name,
                        "size": stat.st_size,
                        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        except OSError as e:
            logger.error(f"Failed to list files in session {session_id}: {e}")
        
        return files
    
    def delete_file(self, session_id: str, filename: str) -> bool:
        """
        Delete a specific file from session directory.
        
        Args:
            session_id: Session identifier
            filename: Name of file to delete
            
        Returns:
            True if file was deleted, False if file didn't exist
            
        Raises:
            HTTPException: If deletion fails
        """
        file_path = self.get_file(session_id, filename)
        
        if not file_path:
            return False
        
        try:
            file_path.unlink()
            logger.info(f"Deleted file {filename} from session {session_id}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete file {filename} from session {session_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete file {filename}"
            )
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete entire session directory and all contained files.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was deleted, False if session didn't exist
            
        Raises:
            HTTPException: If deletion fails
        """
        session_dir = self.base_upload_dir / session_id
        
        if not session_dir.exists():
            return False
        
        try:
            shutil.rmtree(session_dir)
            logger.info(f"Deleted session directory: {session_dir}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete session directory {session_dir}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete session {session_id}"
            )
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up session directories older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of sessions cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        try:
            for session_dir in self.base_upload_dir.iterdir():
                if session_dir.is_dir():
                    # Check directory modification time
                    dir_stat = session_dir.stat()
                    dir_modified = datetime.fromtimestamp(dir_stat.st_mtime)
                    
                    if dir_modified < cutoff_time:
                        try:
                            shutil.rmtree(session_dir)
                            cleaned_count += 1
                            logger.info(f"Cleaned up old session: {session_dir.name}")
                        except OSError as e:
                            logger.error(f"Failed to clean up session {session_dir.name}: {e}")
                            
        except OSError as e:
            logger.error(f"Failed to list sessions for cleanup: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old sessions")
        
        return cleaned_count
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a session directory.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session metadata or None if session doesn't exist
        """
        session_dir = self.base_upload_dir / session_id
        
        if not session_dir.exists():
            return None
        
        try:
            stat = session_dir.stat()
            files = self.list_files(session_id)
            total_size = sum(f['size'] for f in files)
            
            return {
                "session_id": session_id,
                "directory_path": str(session_dir),
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "file_count": len(files),
                "total_size": total_size,
                "files": files
            }
        except OSError as e:
            logger.error(f"Failed to get session info for {session_id}: {e}")
            return None
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and invalid characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path separators and other dangerous characters
        filename = filename.replace('/', '_').replace('\\', '_')
        filename = filename.replace('..', '_')
        
        # Remove any control characters
        filename = ''.join(char for char in filename if ord(char) >= 32)
        
        # Ensure filename isn't empty after sanitization
        if not filename or filename.isspace():
            filename = f"upload_{uuid.uuid4().hex[:8]}.pdf"
        
        return filename


# Global storage instance
file_storage = FileStorage()


def get_file_storage() -> FileStorage:
    """
    Get the global file storage instance.
    
    Returns:
        FileStorage instance for dependency injection
    """
    return file_storage