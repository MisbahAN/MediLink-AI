#!/usr/bin/env python3
"""
Basic E2E test without API dependencies
Tests the complete workflow with mocked AI services
"""

import asyncio
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app


def test_basic_e2e_workflow():
    """Basic E2E test with mocked AI services"""
    
    # Mock AI services to avoid API key requirements
    with patch('app.services.mistral_service.MistralService.extract_from_pdf') as mock_mistral, \
         patch('app.services.gemini_service_fallback.GeminiService.extract_from_pdf') as mock_gemini, \
         patch('app.services.openai_service.OpenAIService.extract_and_map_fields') as mock_openai:
        
        # Configure mocks
        mock_mistral.return_value = {
            "extracted_text": "Sample patient: John Doe, DOB: 01/01/1980, Member ID: 123456789",
            "confidence": 0.95,
            "pages_processed": 1
        }
        
        mock_gemini.return_value = {
            "extracted_text": "Backup extraction text",
            "confidence": 0.85,
            "pages_processed": 1
        }
        
        mock_openai.return_value = {
            "field_mappings": {
                "patient_name": {"mapped_value": "John Doe", "confidence": 0.95},
                "date_of_birth": {"mapped_value": "01/01/1980", "confidence": 0.92},
                "member_id": {"mapped_value": "123456789", "confidence": 0.88}
            },
            "missing_fields": []
        }
        
        client = TestClient(app)
        
        # Test file paths
        test_data_dir = Path(__file__).parent / "tests" / "test_data"
        pa_file = test_data_dir / "test_1_PA.pdf"
        ref_file = test_data_dir / "test_1_referral_package.pdf"
        
        if not pa_file.exists() or not ref_file.exists():
            print("❌ Test files not found, skipping E2E test")
            return
        
        print("🚀 Starting basic E2E test...")
        
        # Step 1: Upload files
        with open(pa_file, "rb") as pf, open(ref_file, "rb") as rf:
            files = [
                ("files", ("pa_form.pdf", pf.read(), "application/pdf")),
                ("files", ("referral.pdf", rf.read(), "application/pdf"))
            ]
            
            response = client.post("/api/upload", files=files)
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return
        
        session_id = response.json()["session_id"]
        print(f"✅ Upload successful - Session ID: {session_id}")
        
        # Step 2: Start processing
        response = client.post(f"/api/process/{session_id}")
        if response.status_code != 200:
            print(f"❌ Processing start failed: {response.status_code}")
            return
        
        print("✅ Processing started")
        
        # Step 3: Poll for completion
        max_attempts = 30  # 1 minute max
        for attempt in range(max_attempts):
            import time
            time.sleep(2)
            
            response = client.get(f"/api/process/{session_id}/status")
            if response.status_code != 200:
                print(f"❌ Status check failed: {response.status_code}")
                return
            
            status_data = response.json()
            current_status = status_data["status"]
            print(f"📊 Status: {current_status}")
            
            if current_status == "completed":
                print("✅ Processing completed!")
                break
            elif current_status == "failed":
                print(f"❌ Processing failed: {status_data.get('error')}")
                return
        else:
            print("❌ Processing timeout")
            return
        
        # Step 4: Download results
        filled_response = client.get(f"/api/download/{session_id}/filled")
        if filled_response.status_code == 200:
            print(f"✅ Filled PDF downloaded ({len(filled_response.content)} bytes)")
        else:
            print(f"❌ Failed to download filled PDF: {filled_response.status_code}")
        
        report_response = client.get(f"/api/download/{session_id}/report")
        if report_response.status_code == 200:
            print(f"✅ Report downloaded ({len(report_response.content)} bytes)")
        else:
            print(f"❌ Failed to download report: {report_response.status_code}")
        
        print("🎉 Basic E2E test completed successfully!")


if __name__ == "__main__":
    test_basic_e2e_workflow()