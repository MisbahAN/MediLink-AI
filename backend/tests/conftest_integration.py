"""
Integration test configuration with real API keys.
"""

import os
import pytest
from app.core.config import Settings


@pytest.fixture
def integration_settings() -> Settings:
    """Create integration test settings with real API keys from environment."""
    # Load real API keys from .env file or environment
    mistral_key = os.getenv("MISTRAL_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") 
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not all([mistral_key, gemini_key, openai_key]):
        pytest.skip("Integration tests require real API keys. Set MISTRAL_API_KEY, GEMINI_API_KEY, and OPENAI_API_KEY environment variables.")
    
    return Settings(
        MISTRAL_API_KEY=mistral_key,
        GEMINI_API_KEY=gemini_key,
        OPENAI_API_KEY=openai_key,
        REDIS_URL="redis://localhost:6379/1",  # Use test database
        UPLOAD_DIR="./test_uploads",
        MAX_FILE_SIZE=10485760,  # 10MB for tests
        ALLOWED_FILE_TYPES=["application/pdf"],
        CACHE_TTL_HOURS=1,
        PROCESSING_TIMEOUT_SECONDS=60
    )


def pytest_configure(config):
    """Configure pytest with custom marks."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring real API keys"
    )