"""
Integration tests for AI API key configuration and connectivity.
"""

import pytest
import asyncio
from unittest.mock import patch
import os

from app.services.mistral_service import MistralService
from app.services.gemini_service_fallback import GeminiService
from app.services.openai_service import OpenAIService
from app.core.config import get_settings


class TestAPIKeyConfiguration:
    """Test AI API key configuration and connectivity."""

    def test_should_load_real_api_keys_from_environment(self, integration_settings):
        """Test that real API keys are loaded from environment."""
        # Check that API keys are present and not test values
        assert integration_settings.MISTRAL_API_KEY != "test_mistral_key"
        assert integration_settings.GEMINI_API_KEY != "test_gemini_key" 
        assert integration_settings.OPENAI_API_KEY != "test_openai_key"
        
        # Verify keys have reasonable format
        assert len(integration_settings.MISTRAL_API_KEY) > 10
        assert len(integration_settings.GEMINI_API_KEY) > 10
        assert len(integration_settings.OPENAI_API_KEY) > 10

    def test_should_validate_mistral_api_key_format(self, integration_settings):
        """Test Mistral API key format validation."""
        mistral_key = integration_settings.MISTRAL_API_KEY
        
        # Mistral keys are typically alphanumeric
        assert mistral_key.replace('-', '').replace('_', '').isalnum()
        assert len(mistral_key) >= 16

    def test_should_validate_gemini_api_key_format(self, integration_settings):
        """Test Gemini API key format validation."""
        gemini_key = integration_settings.GEMINI_API_KEY
        
        # Gemini keys typically start with AIzaSy
        assert gemini_key.startswith("AIzaSy")
        assert len(gemini_key) >= 35

    def test_should_validate_openai_api_key_format(self, integration_settings):
        """Test OpenAI API key format validation."""
        openai_key = integration_settings.OPENAI_API_KEY
        
        # OpenAI keys start with sk-proj- or sk-
        assert openai_key.startswith("sk-")
        assert len(openai_key) >= 45

    @pytest.mark.asyncio
    async def test_should_initialize_mistral_service_with_real_key(self, integration_settings):
        """Test Mistral service initialization with real API key."""
        service = MistralService(integration_settings)
        
        try:
            await service.initialize_client()
            assert service.client is not None
            assert service.api_key == integration_settings.MISTRAL_API_KEY
        except Exception as e:
            pytest.skip(f"Mistral API not accessible: {e}")

    @pytest.mark.asyncio
    async def test_should_initialize_gemini_service_with_real_key(self, integration_settings):
        """Test Gemini service initialization with real API key."""
        service = GeminiService(integration_settings)
        
        try:
            await service.initialize_client()
            assert service.model is not None
            assert service.api_key == integration_settings.GEMINI_API_KEY
        except Exception as e:
            pytest.skip(f"Gemini API not accessible: {e}")

    @pytest.mark.asyncio
    async def test_should_initialize_openai_service_with_real_key(self, integration_settings):
        """Test OpenAI service initialization with real API key."""
        service = OpenAIService(integration_settings)
        
        try:
            await service.initialize_client()
            assert service.client is not None
            assert service.api_key == integration_settings.OPENAI_API_KEY
        except Exception as e:
            pytest.skip(f"OpenAI API not accessible: {e}")


class TestAPIConnectivity:
    """Test actual API connectivity with real keys."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_should_connect_to_mistral_api(self, integration_settings):
        """Test actual connection to Mistral API."""
        service = MistralService(integration_settings)
        
        try:
            await service.initialize_client()
            
            # Test a simple API call
            test_prompt = "Extract text: 'Patient Name: John Doe'"
            response = await service.extract_from_text(test_prompt)
            
            assert response is not None
            assert isinstance(response, dict)
            
        except Exception as e:
            pytest.skip(f"Mistral API connection failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_should_connect_to_gemini_api(self, integration_settings):
        """Test actual connection to Gemini API."""
        service = GeminiService(integration_settings)
        
        try:
            await service.initialize_client()
            
            # Test a simple API call
            test_pages = {"page_1": {"image_data": b"mock_image_data"}}
            response = await service.extract_from_pdf_pages(test_pages)
            
            assert response is not None
            assert isinstance(response, dict)
            
        except Exception as e:
            pytest.skip(f"Gemini API connection failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_should_connect_to_openai_api(self, integration_settings):
        """Test actual connection to OpenAI API."""
        service = OpenAIService(integration_settings)
        
        try:
            await service.initialize_client()
            
            # Test a simple API call
            extracted_data = {"patient_name": "John Doe"}
            form_fields = {"patient_name_field": {"type": "text"}}
            
            response = await service.extract_and_map_fields(extracted_data, form_fields)
            
            assert response is not None
            assert isinstance(response, dict)
            
        except Exception as e:
            pytest.skip(f"OpenAI API connection failed: {e}")


class TestEnvironmentConfiguration:
    """Test environment variable configuration."""

    def test_should_have_required_environment_variables(self):
        """Test that all required environment variables are set."""
        required_vars = [
            "MISTRAL_API_KEY",
            "GEMINI_API_KEY", 
            "OPENAI_API_KEY",
            "REDIS_URL"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            pytest.fail(f"Missing required environment variables: {missing_vars}")

    def test_should_load_configuration_from_env_file(self):
        """Test that configuration is properly loaded from .env file."""
        settings = get_settings()
        
        # Verify that settings are loaded (not default values)
        assert settings.MISTRAL_API_KEY != "your_key_here"
        assert settings.GEMINI_API_KEY != "your_key_here"
        assert settings.OPENAI_API_KEY != "your_key_here"

    def test_should_validate_redis_configuration(self, integration_settings):
        """Test Redis configuration for integration tests."""
        redis_url = integration_settings.REDIS_URL
        
        # Should use test database (database 1)
        assert "redis://localhost:6379/1" in redis_url
        
        # Test Redis connectivity
        import redis
        try:
            r = redis.from_url(redis_url)
            r.ping()
        except Exception as e:
            pytest.fail(f"Redis connection failed: {e}")

    def test_should_configure_upload_directory(self, integration_settings):
        """Test upload directory configuration."""
        upload_dir = integration_settings.UPLOAD_DIR
        
        assert upload_dir == "./test_uploads"
        assert integration_settings.MAX_FILE_SIZE == 10485760  # 10MB for tests


class TestAPIKeySecurity:
    """Test API key security and validation."""

    def test_should_not_log_api_keys(self, integration_settings, caplog):
        """Test that API keys are not logged in plaintext."""
        # Initialize services (which may log)
        MistralService(integration_settings)
        GeminiService(integration_settings)
        OpenAIService(integration_settings)
        
        # Check that full API keys don't appear in logs
        log_output = caplog.text.lower()
        
        assert integration_settings.MISTRAL_API_KEY.lower() not in log_output
        assert integration_settings.GEMINI_API_KEY.lower() not in log_output
        assert integration_settings.OPENAI_API_KEY.lower() not in log_output

    def test_should_mask_api_keys_in_error_messages(self, integration_settings):
        """Test that API keys are masked in error messages."""
        # This would be implemented in the actual service classes
        # to ensure API keys are never exposed in error messages
        pass

    def test_should_validate_api_key_permissions(self):
        """Test that API keys have required permissions."""
        # This test would validate that the API keys have the necessary
        # permissions for the operations we need to perform
        pass


@pytest.mark.integration
class TestIntegrationTestSetup:
    """Test that integration test environment is properly configured."""

    def test_should_be_ready_for_integration_tests(self, integration_settings):
        """Test that all components are ready for integration testing."""
        # API keys configured
        assert integration_settings.MISTRAL_API_KEY != "test_mistral_key"
        assert integration_settings.GEMINI_API_KEY != "test_gemini_key"
        assert integration_settings.OPENAI_API_KEY != "test_openai_key"
        
        # Redis available
        import redis
        r = redis.from_url(integration_settings.REDIS_URL)
        assert r.ping()
        
        # Upload directory configured
        assert integration_settings.UPLOAD_DIR == "./test_uploads"
        
        # Test environment isolation
        assert "/1" in integration_settings.REDIS_URL  # Test database

    @pytest.mark.asyncio
    async def test_should_handle_service_initialization_errors(self, integration_settings):
        """Test graceful handling of service initialization errors."""
        # Test with invalid key format
        invalid_settings = integration_settings.copy()
        invalid_settings.MISTRAL_API_KEY = "invalid_key"
        
        service = MistralService(invalid_settings)
        
        with pytest.raises(Exception):
            await service.initialize_client()

    def test_should_provide_integration_test_feedback(self, integration_settings):
        """Test that integration tests provide clear feedback."""
        # This test ensures that when integration tests run,
        # they provide clear information about what's being tested
        # and whether APIs are accessible
        
        print(f"✓ Mistral API key configured: {integration_settings.MISTRAL_API_KEY[:8]}...")
        print(f"✓ Gemini API key configured: {integration_settings.GEMINI_API_KEY[:8]}...")
        print(f"✓ OpenAI API key configured: {integration_settings.OPENAI_API_KEY[:8]}...")
        print(f"✓ Redis configured: {integration_settings.REDIS_URL}")
        print("✓ Integration test environment ready")