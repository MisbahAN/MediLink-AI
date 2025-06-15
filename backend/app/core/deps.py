# backend/app/core/deps.py
"""
Common dependencies for FastAPI routes.

This module provides reusable dependencies for dependency injection
across the application routes.
"""

from functools import lru_cache
from core.config import Settings, get_settings


@lru_cache()
def get_settings_cached() -> Settings:
    """
    Get cached application settings instance.
    
    Returns:
        Application settings with environment variables loaded
    """
    return get_settings()


def get_current_settings() -> Settings:
    """
    Dependency to inject settings into route handlers.
    
    Returns:
        Current application settings instance
    """
    return get_settings_cached()