"""
Tests for API endpoints functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
from io import BytesIO

from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestUploadEndpoints:
    """Test file upload endpoints."""

    def test_should_upload_valid_pdf_files(self, test_client, sample_pdf_bytes):
        """Test uploading valid PDF files."""
        files = [
            ("files", ("test_referral.pdf", BytesIO(sample_pdf_bytes), "application/pdf")),
            ("files", ("test_pa_form.pdf", BytesIO(sample_pdf_bytes), "application/pdf"))
        ]
        
        response = test_client.post("/api/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "files_received" in data
        assert len(data["files_received"]) == 2
        assert "message" in data
        assert "Files uploaded successfully" in data["message"]

    def test_should_reject_non_pdf_files(self, test_client):
        """Test rejection of non-PDF files."""
        files = [
            ("files", ("test.txt", BytesIO(b"not a pdf"), "text/plain")),
            ("files", ("test_pa_form.pdf", BytesIO(b"%PDF-1.4"), "application/pdf"))
        ]
        
        response = test_client.post("/api/upload", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid file type" in data["detail"]

    def test_should_reject_oversized_files(self, test_client):
        """Test rejection of oversized files."""
        # Create a large file (simulate 100MB)
        large_content = b"x" * (100 * 1024 * 1024)
        
        files = [
            ("files", ("large.pdf", BytesIO(large_content), "application/pdf")),
            ("files", ("test_pa_form.pdf", BytesIO(b"%PDF-1.4"), "application/pdf"))
        ]
        
        response = test_client.post("/api/upload", files=files)
        
        assert response.status_code == 413
        data = response.json()
        assert "File too large" in data["detail"]

    def test_should_require_both_files(self, test_client, sample_pdf_bytes):
        """Test that both files are required."""
        files = [
            ("files", ("test_referral.pdf", BytesIO(sample_pdf_bytes), "application/pdf"))
            # Missing second file
        ]
        
        response = test_client.post("/api/upload", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "Exactly 2 files required" in data["detail"]

    def test_should_generate_unique_session_ids(self, test_client, sample_pdf_bytes):
        """Test that unique session IDs are generated."""
        files = [
            ("files", ("test_referral.pdf", BytesIO(sample_pdf_bytes), "application/pdf")),
            ("files", ("test_pa_form.pdf", BytesIO(sample_pdf_bytes), "application/pdf"))
        ]
        
        # Upload first file set
        response1 = test_client.post("/api/upload", files=files)
        session_id1 = response1.json()["session_id"]
        
        # Upload second file set
        response2 = test_client.post("/api/upload", files=files)
        session_id2 = response2.json()["session_id"]
        
        assert session_id1 != session_id2
        assert len(session_id1) >= 10
        assert len(session_id2) >= 10

    @patch('app.services.storage.FileStorage.save_file')
    def test_should_handle_storage_errors(self, mock_save, test_client, sample_pdf_bytes):
        """Test handling of file storage errors."""
        mock_save.side_effect = Exception("Storage error")
        
        files = {
            "referral_file": ("test_referral.pdf", BytesIO(sample_pdf_bytes), "application/pdf"),
            "pa_form_file": ("test_pa_form.pdf", BytesIO(sample_pdf_bytes), "application/pdf")
        }
        
        response = test_client.post("/api/upload", files=files)
        
        assert response.status_code == 500
        data = response.json()
        assert "upload" in data["detail"].lower()


class TestProcessingEndpoints:
    """Test document processing endpoints."""

    @pytest.mark.asyncio
    async def test_should_start_processing(self, async_client, session_id):
        """Test starting document processing."""
        with patch('app.services.processing_pipeline.ProcessingPipeline.process_documents') as mock_process:
            mock_process.return_value = AsyncMock()
            
            response = await async_client.post(f"/api/process/{session_id}")
            
            assert response.status_code == 202
            data = response.json()
            assert data["session_id"] == session_id
            assert data["status"] == "processing"
            assert "message" in data

    @pytest.mark.asyncio
    async def test_should_get_processing_status(self, async_client, session_id, mock_processing_status):
        """Test getting processing status."""
        with patch('app.services.cache.CacheService.get_cached_result') as mock_cache:
            mock_cache.return_value = mock_processing_status
            
            response = await async_client.get(f"/api/process/{session_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id
            assert "status" in data
            assert "progress" in data

    @pytest.mark.asyncio
    async def test_should_handle_nonexistent_session(self, async_client):
        """Test handling of nonexistent session ID."""
        fake_session_id = "nonexistent_session_123"
        
        response = await async_client.get(f"/api/process/{fake_session_id}/status")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_should_handle_processing_errors(self, async_client, session_id):
        """Test handling of processing pipeline errors."""
        with patch('app.services.processing_pipeline.ProcessingPipeline.process_documents') as mock_process:
            mock_process.side_effect = Exception("Processing error")
            
            response = await async_client.post(f"/api/process/{session_id}")
            
            assert response.status_code == 500
            data = response.json()
            assert "processing" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_should_prevent_duplicate_processing(self, async_client, session_id):
        """Test prevention of duplicate processing requests."""
        # Mock that processing is already in progress
        with patch('app.services.cache.CacheService.get_cached_result') as mock_cache:
            mock_cache.return_value = {"status": "processing", "session_id": session_id}
            
            response = await async_client.post(f"/api/process/{session_id}")
            
            assert response.status_code == 409
            data = response.json()
            assert "already processing" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_should_track_processing_progress(self, async_client, session_id):
        """Test processing progress tracking."""
        progress_states = [
            {"status": "extracting", "progress": 25},
            {"status": "mapping", "progress": 50},
            {"status": "filling", "progress": 75},
            {"status": "completed", "progress": 100}
        ]
        
        with patch('app.services.cache.CacheService.get_cached_result') as mock_cache:
            for state in progress_states:
                mock_cache.return_value = {**state, "session_id": session_id}
                
                response = await async_client.get(f"/api/process/{session_id}/status")
                
                assert response.status_code == 200
                data = response.json()
                assert data["progress"] == state["progress"]
                assert data["status"] == state["status"]


class TestDownloadEndpoints:
    """Test file download endpoints."""

    @pytest.mark.asyncio
    async def test_should_download_filled_form(self, async_client, session_id):
        """Test downloading filled PA form."""
        with patch('app.services.storage.FileStorage.get_file') as mock_get_file:
            mock_get_file.return_value = b"mock_filled_pdf_content"
            
            response = await async_client.get(f"/api/download/{session_id}/filled")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert "attachment" in response.headers["content-disposition"]
            assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_should_download_missing_fields_report(self, async_client, session_id):
        """Test downloading missing fields report."""
        with patch('app.services.storage.FileStorage.get_file') as mock_get_file:
            mock_get_file.return_value = b"# Missing Fields Report\n\n## High Priority\n- provider_npi"
            
            response = await async_client.get(f"/api/download/{session_id}/report")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/markdown"
            assert "attachment" in response.headers["content-disposition"]
            assert b"Missing Fields Report" in response.content

    @pytest.mark.asyncio
    async def test_should_handle_missing_files(self, async_client, session_id):
        """Test handling of missing download files."""
        with patch('app.services.storage.FileStorage.get_file') as mock_get_file:
            mock_get_file.side_effect = FileNotFoundError("File not found")
            
            response = await async_client.get(f"/api/download/{session_id}/filled")
            
            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_should_stream_large_files(self, async_client, session_id):
        """Test streaming of large download files."""
        large_content = b"x" * (10 * 1024 * 1024)  # 10MB file
        
        with patch('app.services.storage.FileStorage.get_file') as mock_get_file:
            mock_get_file.return_value = large_content
            
            response = await async_client.get(f"/api/download/{session_id}/filled")
            
            assert response.status_code == 200
            assert len(response.content) == len(large_content)

    @pytest.mark.asyncio
    async def test_should_require_completed_processing(self, async_client, session_id):
        """Test that downloads require completed processing."""
        with patch('app.services.cache.CacheService.get_cached_result') as mock_cache:
            mock_cache.return_value = {"status": "processing", "session_id": session_id}
            
            response = await async_client.get(f"/api/download/{session_id}/filled")
            
            assert response.status_code == 409
            data = response.json()
            assert "not completed" in data["detail"].lower()


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_should_return_basic_health_status(self, test_client):
        """Test basic health check endpoint."""
        response = test_client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data

    @patch('app.services.mistral_service.MistralService.health_check')
    @patch('app.services.gemini_service_fallback.GeminiService.health_check')
    def test_should_check_ai_services_health(self, mock_gemini_health, mock_mistral_health, test_client):
        """Test AI services health check."""
        mock_mistral_health.return_value = {"status": "healthy", "response_time": 0.5}
        mock_gemini_health.return_value = {"status": "healthy", "response_time": 0.3}
        
        response = test_client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["services"]["mistral"]["status"] == "healthy"
        assert data["services"]["gemini"]["status"] == "healthy"

    @patch('app.services.mistral_service.MistralService.health_check')
    def test_should_handle_service_degradation(self, mock_mistral_health, test_client):
        """Test handling of degraded service health."""
        mock_mistral_health.side_effect = Exception("Service unavailable")
        
        response = test_client.get("/api/health")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert "mistral" in data["services"]

    @patch('redis.Redis.ping')
    def test_should_check_redis_health(self, mock_redis_ping, test_client):
        """Test Redis health check."""
        mock_redis_ping.return_value = True
        
        response = test_client.get("/api/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert "redis" in data["services"]
        assert data["services"]["redis"]["status"] == "healthy"

    def test_should_return_service_versions(self, test_client):
        """Test that service versions are included in health check."""
        response = test_client.get("/api/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "build" in data


class TestErrorHandling:
    """Test API error handling."""

    def test_should_handle_404_errors(self, test_client):
        """Test handling of 404 errors."""
        response = test_client.get("/api/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_should_handle_validation_errors(self, test_client):
        """Test handling of request validation errors."""
        # Send malformed request
        response = test_client.post("/api/upload", data={"invalid": "data"})
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_should_include_request_id_in_errors(self, test_client):
        """Test that request ID is included in error responses."""
        response = test_client.get("/api/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert "request_id" in data or "request_id" in response.headers

    @patch('app.services.storage.FileStorage.save_file')
    def test_should_handle_internal_server_errors(self, mock_save, test_client, sample_pdf_bytes):
        """Test handling of internal server errors."""
        mock_save.side_effect = Exception("Unexpected error")
        
        files = {
            "referral_file": ("test.pdf", BytesIO(sample_pdf_bytes), "application/pdf"),
            "pa_form_file": ("test2.pdf", BytesIO(sample_pdf_bytes), "application/pdf")
        }
        
        response = test_client.post("/api/upload", files=files)
        
        assert response.status_code == 500
        data = response.json()
        assert "internal server error" in data["detail"].lower()

    def test_should_sanitize_error_messages(self, test_client):
        """Test that error messages don't leak sensitive information."""
        # This would typically test that stack traces, file paths, etc. are not exposed
        response = test_client.get("/api/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        # Ensure no sensitive paths or internal details are exposed
        assert "/app/" not in str(data)
        assert "traceback" not in str(data).lower()


class TestRateLimiting:
    """Test API rate limiting (if implemented)."""

    @pytest.mark.skip(reason="Rate limiting not yet implemented")
    def test_should_enforce_rate_limits(self, test_client):
        """Test that rate limits are enforced."""
        # Make many requests quickly
        responses = []
        for _ in range(100):
            response = test_client.get("/api/health")
            responses.append(response)
        
        # Check that some requests are rate limited
        rate_limited = [r for r in responses if r.status_code == 429]
        assert len(rate_limited) > 0

    @pytest.mark.skip(reason="Rate limiting not yet implemented")
    def test_should_include_rate_limit_headers(self, test_client):
        """Test that rate limit headers are included."""
        response = test_client.get("/api/health")
        
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers


class TestSecurityHeaders:
    """Test security headers."""

    def test_should_include_security_headers(self, test_client):
        """Test that security headers are included."""
        response = test_client.get("/api/health")
        
        # Check for common security headers
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security"
        ]
        
        for header in expected_headers:
            assert header in response.headers

    def test_should_set_cors_headers(self, test_client):
        """Test that CORS headers are properly set."""
        response = test_client.options("/api/health")
        
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers


class TestConcurrentRequests:
    """Test handling of concurrent requests."""

    @pytest.mark.asyncio
    async def test_should_handle_concurrent_uploads(self, async_client, sample_pdf_bytes):
        """Test handling of concurrent file uploads."""
        async def upload_files():
            files = {
                "referral_file": ("test.pdf", BytesIO(sample_pdf_bytes), "application/pdf"),
                "pa_form_file": ("test2.pdf", BytesIO(sample_pdf_bytes), "application/pdf")
            }
            return await async_client.post("/api/upload", files=files)
        
        # Start multiple concurrent uploads
        tasks = [upload_files() for _ in range(5)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed or fail gracefully
        for response in responses:
            if not isinstance(response, Exception):
                assert response.status_code in [200, 429, 503]  # Success or rate limited

    @pytest.mark.asyncio
    async def test_should_handle_concurrent_processing(self, async_client):
        """Test handling of concurrent processing requests."""
        session_ids = [f"test_session_{i}" for i in range(3)]
        
        async def start_processing(session_id):
            with patch('app.services.processing_pipeline.ProcessingPipeline.process_documents'):
                return await async_client.post(f"/api/process/{session_id}")
        
        tasks = [start_processing(sid) for sid in session_ids]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all are handled properly
        for response in responses:
            if not isinstance(response, Exception):
                assert response.status_code in [202, 409, 503]


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_should_serve_openapi_schema(self, test_client):
        """Test that OpenAPI schema is available."""
        response = test_client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_should_serve_swagger_docs(self, test_client):
        """Test that Swagger documentation is available."""
        response = test_client.get("/docs")
        
        assert response.status_code == 200
        assert "swagger" in response.text.lower()

    def test_should_serve_redoc_docs(self, test_client):
        """Test that ReDoc documentation is available."""
        response = test_client.get("/redoc")
        
        assert response.status_code == 200
        assert "redoc" in response.text.lower()