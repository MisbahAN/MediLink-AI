# backend/tests/conftest.py
"""
Test configuration and fixtures for MediLink-AI backend tests.
"""

import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
import redis
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.core.config import Settings
from app.services.cache import CacheService
from app.services.storage import FileStorage


@pytest.fixture
def integration_settings() -> Settings:
    """Create integration test settings with real API keys from environment."""
    # Try to load environment variables from .env file if not already set
    from dotenv import load_dotenv
    load_dotenv()
    
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


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with real API keys for integration tests."""
    return Settings(
        MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY", "test_mistral_key"),
        GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", "test_gemini_key"),
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "test_openai_key"),
        REDIS_URL="redis://localhost:6379/1",  # Use test database
        UPLOAD_DIR="./test_uploads",
        MAX_FILE_SIZE=10485760,  # 10MB for tests
        ALLOWED_FILE_TYPES=["application/pdf"],
        CACHE_TTL_HOURS=1,
        PROCESSING_TIMEOUT_SECONDS=60
    )


@pytest.fixture
def test_client(test_settings: Settings) -> TestClient:
    """Create FastAPI test client."""
    app.dependency_overrides = {}
    from app.core.deps import get_current_settings
    app.dependency_overrides[get_current_settings] = lambda: test_settings
    
    client = TestClient(app)
    yield client
    
    # Cleanup
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(test_settings: Settings) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client for testing."""
    app.dependency_overrides = {}
    from app.core.deps import get_current_settings
    app.dependency_overrides[get_current_settings] = lambda: test_settings
    
    async with AsyncClient(base_url="http://test") as client:
        # Set up the ASGI app context
        import httpx
        from fastapi.testclient import TestClient
        
        # Use TestClient for simplicity with file uploads
        sync_client = TestClient(app)
        
        # Monkey patch the client to work with our async interface
        class AsyncTestClient:
            def __init__(self, sync_client):
                self._sync_client = sync_client
                
            async def post(self, url, **kwargs):
                return self._sync_client.post(url, **kwargs)
                
            async def get(self, url, **kwargs):
                return self._sync_client.get(url, **kwargs)
                
        yield AsyncTestClient(sync_client)
    
    # Cleanup
    app.dependency_overrides.clear()


@pytest.fixture
def temp_upload_dir() -> Generator[Path, None, None]:
    """Create temporary upload directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="medilink_test_")
    upload_path = Path(temp_dir)
    yield upload_path
    
    # Cleanup
    if upload_path.exists():
        shutil.rmtree(upload_path)


@pytest.fixture
def file_storage(temp_upload_dir: Path, test_settings: Settings) -> FileStorage:
    """Create FileStorage instance with test directory."""
    test_settings.upload_dir = str(temp_upload_dir)
    return FileStorage(test_settings)


@pytest.fixture
def test_pdf_path() -> Path:
    """Path to test PDF file."""
    return Path(__file__).parent / "test_data" / "test_1_PA.pdf"


@pytest.fixture
def test_referral_path() -> Path:
    """Path to test referral PDF file."""
    return Path(__file__).parent / "test_data" / "test_1_referral_package.pdf"


@pytest.fixture
def sample_pdf_bytes(test_pdf_path: Path) -> bytes:
    """Load sample PDF as bytes."""
    if test_pdf_path.exists():
        return test_pdf_path.read_bytes()
    else:
        # Create minimal PDF for testing if file doesn't exist
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n178\n%%EOF"


@pytest.fixture
def sample_extracted_data() -> dict:
    """Sample extracted data for testing."""
    return {
        "patient_info": {
            "name": {"value": "John Doe", "confidence": 0.95, "source_page": 1},
            "dob": {"value": "01/15/1980", "confidence": 0.92, "source_page": 1},
            "insurance_id": {"value": "123456789", "confidence": 0.88, "source_page": 1}
        },
        "clinical_data": {
            "diagnosis": {"value": "Rheumatoid Arthritis", "confidence": 0.91, "source_page": 2},
            "treatment_plan": {"value": "Rituximab infusion", "confidence": 0.85, "source_page": 3}
        },
        "raw_extracted": {
            "page_1": {"text": "Patient: John Doe DOB: 01/15/1980", "bbox": [100, 200, 300, 220]}
        }
    }


@pytest.fixture
def sample_pa_form_fields() -> dict:
    """Sample PA form fields for testing."""
    return {
        "patient_name": {
            "type": "text",
            "required": True,
            "coordinates": {"x": 100, "y": 200},
            "mapped_value": "John Doe",
            "confidence": 0.95
        },
        "patient_dob": {
            "type": "text",
            "required": True,
            "coordinates": {"x": 100, "y": 240},
            "mapped_value": "01/15/1980",
            "confidence": 0.92
        },
        "insurance_id": {
            "type": "text",
            "required": True,
            "coordinates": {"x": 100, "y": 280},
            "mapped_value": "123456789",
            "confidence": 0.88
        }
    }


@pytest.fixture
def mock_mistral_response() -> dict:
    """Mock Mistral API response."""
    return {
        "text": "Patient Name: John Doe\nDate of Birth: 01/15/1980\nInsurance ID: 123456789",
        "confidence": 0.92,
        "pages_processed": 1
    }


@pytest.fixture
def mock_gemini_response() -> dict:
    """Mock Gemini API response."""
    return {
        "text": "Patient: John Doe\nDOB: January 15, 1980\nMember ID: 123456789",
        "confidence": 0.85,
        "method": "vision_extraction"
    }


@pytest.fixture
def mock_openai_response() -> dict:
    """Mock OpenAI field mapping response."""
    return {
        "mapped_fields": {
            "patient_name": {"value": "John Doe", "confidence": 0.95, "source": "demographics"},
            "patient_dob": {"value": "01/15/1980", "confidence": 0.92, "source": "demographics"},
            "insurance_id": {"value": "123456789", "confidence": 0.88, "source": "insurance_info"}
        },
        "missing_fields": [
            {"field": "provider_npi", "reason": "Not found in referral"},
            {"field": "prior_auth_number", "reason": "Low confidence (0.45)"}
        ]
    }


@pytest.fixture(scope="session")
def redis_client():
    """Redis client for test database."""
    try:
        client = redis.Redis.from_url("redis://localhost:6379/1", decode_responses=True)
        client.ping()
        yield client
        # Cleanup test database
        client.flushdb()
    except redis.ConnectionError:
        pytest.skip("Redis not available for testing")


@pytest.fixture
def cache_service(redis_client, test_settings: Settings) -> CacheService:
    """Create CacheService instance for testing."""
    test_settings.redis_url = "redis://localhost:6379/1"
    service = CacheService(test_settings)
    service.redis_client = redis_client
    return service


@pytest.fixture
def session_id() -> str:
    """Generate test session ID."""
    return "test_session_12345"


@pytest.fixture
def mock_processing_status() -> dict:
    """Mock processing status response."""
    return {
        "session_id": "test_session_12345",
        "status": "completed",
        "stage": "form_filling",
        "progress": 100,
        "message": "Processing completed successfully",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:05:00Z",
        "error": None
    }


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test."""
    yield
    
    # Clean up any test upload directories
    test_dirs = Path(".").glob("test_uploads*")
    for dir_path in test_dirs:
        if dir_path.is_dir():
            shutil.rmtree(dir_path, ignore_errors=True)
    
    # Clean up any temporary test files
    temp_files = Path(".").glob("temp_test_*")
    for file_path in temp_files:
        if file_path.is_file():
            file_path.unlink(missing_ok=True)


@pytest.fixture
def mock_pdf_form_fields() -> list:
    """Mock PDF form fields from pdfforms."""
    return [
        {
            "field_name": "patient_name",
            "field_type": "text",
            "coordinates": {"x": 100, "y": 200, "width": 200, "height": 20},
            "required": True,
            "max_length": 50
        },
        {
            "field_name": "patient_dob",
            "field_type": "text",
            "coordinates": {"x": 100, "y": 240, "width": 100, "height": 20},
            "required": True,
            "format": "date"
        },
        {
            "field_name": "insurance_primary",
            "field_type": "checkbox",
            "coordinates": {"x": 150, "y": 300, "width": 15, "height": 15},
            "options": ["yes", "no"]
        }
    ]


@pytest.fixture
def test_file_upload_data():
    """Test file upload data structure."""
    return {
        "referral_file": "test_referral.pdf",
        "pa_form_file": "test_pa_form.pdf",
        "session_id": "test_session_12345"
    }


# Environment setup for testing
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("MISTRAL_API_KEY", "test_mistral_key")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")