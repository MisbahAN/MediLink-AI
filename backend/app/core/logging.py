"""
Structured logging configuration for the MediLink-AI application.

This module provides comprehensive logging setup with structured formatting,
request ID tracking, log rotation, and HIPAA-compliant logging practices
for medical data processing applications.
"""

import json
import logging
import logging.config
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from .config import get_settings

settings = get_settings()


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging.
    
    Formats log records as JSON with consistent fields including
    timestamps, request IDs, and structured extra data for
    easy parsing and analysis.
    """
    
    def __init__(self):
        super().__init__()
        self.hostname = self._get_hostname()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Build base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "hostname": self.hostname,
            "service": "medilink-ai-backend"
        }
        
        # Add request ID if available
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        
        # Add session ID if available
        if hasattr(record, 'session_id'):
            log_entry["session_id"] = record.session_id
        
        # Add user ID if available
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add any extra fields from the log record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'message', 'request_id', 'session_id', 'user_id'
            }:
                extra_fields[key] = self._serialize_value(value)
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, ensure_ascii=False)
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for JSON output."""
        try:
            # Test if value is JSON serializable
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            # Convert non-serializable values to string
            return str(value)
    
    def _get_hostname(self) -> str:
        """Get hostname for logging."""
        import socket
        try:
            return socket.gethostname()
        except Exception:
            return "unknown"


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for console output.
    
    Provides clean, readable log output for development and debugging
    with color coding and structured layout.
    """
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human readability."""
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Get color for log level
        color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET']
        
        # Build basic log line
        log_parts = [
            f"{color}[{record.levelname}]{reset}",
            f"{timestamp}",
            f"{record.name}",
        ]
        
        # Add request ID if available
        if hasattr(record, 'request_id'):
            log_parts.append(f"[req:{record.request_id}]")
        
        # Add session ID if available
        if hasattr(record, 'session_id'):
            log_parts.append(f"[session:{record.session_id}]")
        
        # Add the main message
        log_parts.append(f"- {record.getMessage()}")
        
        log_line = " ".join(log_parts)
        
        # Add extra fields if present
        extra_fields = []
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'message', 'request_id', 'session_id', 'user_id'
            }:
                extra_fields.append(f"{key}={value}")
        
        if extra_fields:
            log_line += f" | {', '.join(extra_fields)}"
        
        # Add exception information if present
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"
        
        return log_line


class RequestIDFilter(logging.Filter):
    """
    Filter to add request ID to log records.
    
    Attempts to extract request ID from various sources including
    context variables and current request state.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add request ID to log record if available."""
        # Try to get request ID from context (this would need contextvars in a real implementation)
        request_id = getattr(record, 'request_id', None)
        
        if not request_id:
            # Try to get from thread-local storage or other sources
            request_id = self._get_request_id_from_context()
        
        if request_id:
            record.request_id = request_id
        
        return True
    
    def _get_request_id_from_context(self) -> Optional[str]:
        """Get request ID from context variables or thread-local storage."""
        # This is a placeholder - in a real implementation, you'd use
        # contextvars or similar to track request IDs across async contexts
        return None


class SecurityFilter(logging.Filter):
    """
    Filter to prevent logging of sensitive information.
    
    Removes or masks potentially sensitive data from log messages
    to maintain HIPAA compliance and security best practices.
    """
    
    SENSITIVE_PATTERNS = [
        # Common sensitive patterns
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b\d{16}\b',              # Credit card pattern
        r'password["\']?\s*[:=]\s*["\']?[^"\'\s]+',  # Password fields
        r'api[_-]?key["\']?\s*[:=]\s*["\']?[^"\'\s]+',  # API keys
        r'token["\']?\s*[:=]\s*["\']?[^"\'\s]+',  # Tokens
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Remove sensitive information from log messages."""
        import re
        
        # Check message for sensitive patterns
        message = record.getMessage()
        for pattern in self.SENSITIVE_PATTERNS:
            message = re.sub(pattern, '[REDACTED]', message, flags=re.IGNORECASE)
        
        # Update the record message
        record.msg = message
        record.args = ()
        
        # Check extra fields for sensitive data
        for key, value in record.__dict__.items():
            if isinstance(value, str):
                for pattern in self.SENSITIVE_PATTERNS:
                    if re.search(pattern, value, re.IGNORECASE):
                        setattr(record, key, '[REDACTED]')
        
        return True


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True,
    structured_format: bool = True
) -> None:
    """
    Set up comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        enable_console: Whether to enable console logging
        structured_format: Whether to use structured JSON format for files
    """
    # Create logs directory if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set root log level
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatters
    structured_formatter = StructuredFormatter()
    human_formatter = HumanReadableFormatter()
    
    # Create filters
    request_id_filter = RequestIDFilter()
    security_filter = SecurityFilter()
    
    # Console handler (human-readable for development)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(human_formatter)
        console_handler.addFilter(request_id_filter)
        console_handler.addFilter(security_filter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation (structured for production)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if structured_format:
            file_handler.setFormatter(structured_formatter)
        else:
            file_handler.setFormatter(human_formatter)
        
        file_handler.addFilter(request_id_filter)
        file_handler.addFilter(security_filter)
        root_logger.addHandler(file_handler)
    
    # Error file handler (separate file for errors)
    if log_file:
        error_file = log_path.parent / f"{log_path.stem}_errors{log_path.suffix}"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(structured_formatter)
        error_handler.addFilter(request_id_filter)
        error_handler.addFilter(security_filter)
        root_logger.addHandler(error_handler)
    
    # Configure specific loggers
    _configure_specific_loggers()
    
    # Log configuration completion
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configuration completed",
        extra={
            "log_level": log_level,
            "log_file": str(log_file) if log_file else None,
            "console_enabled": enable_console,
            "structured_format": structured_format
        }
    )


def _configure_specific_loggers() -> None:
    """Configure specific loggers with appropriate levels."""
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    # Application-specific loggers
    logging.getLogger("app.services").setLevel(logging.INFO)
    logging.getLogger("app.api").setLevel(logging.INFO)
    logging.getLogger("app.core").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with proper configuration.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log function call with parameters.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    logger = logging.getLogger("app.function_calls")
    logger.debug(
        f"Function call: {func_name}",
        extra={
            "function": func_name,
            "parameters": {k: str(v) for k, v in kwargs.items()}
        }
    )


def log_processing_step(step_name: str, session_id: str, **context) -> None:
    """
    Log processing pipeline steps.
    
    Args:
        step_name: Name of the processing step
        session_id: Session identifier
        **context: Additional context information
    """
    logger = logging.getLogger("app.processing")
    logger.info(
        f"Processing step: {step_name}",
        extra={
            "step": step_name,
            "session_id": session_id,
            **context
        }
    )


def log_api_request(method: str, path: str, status_code: int, duration: float, **context) -> None:
    """
    Log API request information.
    
    Args:
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration: Request duration in seconds
        **context: Additional context information
    """
    logger = logging.getLogger("app.api.requests")
    
    # Choose log level based on status code
    if status_code >= 500:
        log_level = logging.ERROR
    elif status_code >= 400:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    
    logger.log(
        log_level,
        f"{method} {path} - {status_code}",
        extra={
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_seconds": duration,
            **context
        }
    )


def log_security_event(event_type: str, severity: str, **context) -> None:
    """
    Log security-related events.
    
    Args:
        event_type: Type of security event
        severity: Event severity (low, medium, high, critical)
        **context: Additional context information
    """
    logger = logging.getLogger("app.security")
    
    # Map severity to log level
    severity_mapping = {
        "low": logging.INFO,
        "medium": logging.WARNING,
        "high": logging.ERROR,
        "critical": logging.CRITICAL
    }
    
    log_level = severity_mapping.get(severity.lower(), logging.WARNING)
    
    logger.log(
        log_level,
        f"Security event: {event_type}",
        extra={
            "event_type": event_type,
            "severity": severity,
            **context
        }
    )


# Initialize logging with settings
def init_logging() -> None:
    """Initialize logging configuration based on application settings."""
    log_file = getattr(settings, 'LOG_FILE', 'logs/app.log')
    log_level = getattr(settings, 'LOG_LEVEL', 'INFO')
    debug_mode = getattr(settings, 'DEBUG', False)
    
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        enable_console=True,
        structured_format=not debug_mode  # Use human-readable in debug mode
    )


# Context manager for adding request context to logs
class LogContext:
    """Context manager for adding context to log records."""
    
    def __init__(self, **context):
        self.context = context
        self.old_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)