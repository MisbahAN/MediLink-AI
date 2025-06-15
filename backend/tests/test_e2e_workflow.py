"""
End-to-End Workflow Tests for MediLink-AI Prior Authorization System

Tests the complete document processing pipeline from upload to final outputs.
Validates real-world scenarios with actual test documents.
"""

import pytest
import asyncio
import os
import time
from pathlib import Path
from typing import Dict, Any, List
import json
import hashlib

from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.core.config import settings
from app.services.processing_pipeline import ProcessingPipeline
from app.services.storage import FileStorage
from app.models.schemas import ProcessingStatus

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "test_data"
EXPECTED_OUTPUTS_DIR = Path(__file__).parent / "expected_outputs"

# Create expected outputs directory if it doesn't exist
EXPECTED_OUTPUTS_DIR.mkdir(exist_ok=True)


class E2EWorkflowTest:
    """End-to-end workflow test manager"""
    
    def __init__(self):
        self.test_results = {}
        self.processing_times = {}
        self.confidence_scores = {}
        
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for comparison"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def validate_processing_result(self, result: Dict[str, Any], test_case: str) -> Dict[str, Any]:
        """Validate processing result meets quality thresholds"""
        validation_report = {
            "test_case": test_case,
            "status": "PASS",
            "issues": [],
            "metrics": {}
        }
        
        # Extract key metrics
        extracted_fields = result.get("pa_form_fields", {})
        missing_fields = result.get("missing_fields", [])
        referral_data = result.get("referral_data", {})
        
        # Critical field validation
        critical_fields = ["patient_name", "date_of_birth", "member_id", "prescriber_name"]
        missing_critical = []
        
        for field in critical_fields:
            if field not in extracted_fields or not extracted_fields[field].get("mapped_value"):
                missing_critical.append(field)
        
        if missing_critical:
            validation_report["status"] = "FAIL"
            validation_report["issues"].append(f"Missing critical fields: {missing_critical}")
        
        # Confidence score validation
        total_confidence = 0
        field_count = 0
        low_confidence_fields = []
        
        for field_name, field_data in extracted_fields.items():
            confidence = field_data.get("confidence", 0)
            total_confidence += confidence
            field_count += 1
            
            if confidence < 0.70:
                low_confidence_fields.append((field_name, confidence))
        
        avg_confidence = total_confidence / field_count if field_count > 0 else 0
        validation_report["metrics"]["average_confidence"] = avg_confidence
        validation_report["metrics"]["total_fields_extracted"] = field_count
        validation_report["metrics"]["missing_fields_count"] = len(missing_fields)
        validation_report["metrics"]["low_confidence_fields"] = len(low_confidence_fields)
        
        # Quality thresholds
        if avg_confidence < 0.75:
            validation_report["issues"].append(f"Average confidence too low: {avg_confidence:.2f}")
        
        if len(missing_fields) > 10:
            validation_report["issues"].append(f"Too many missing fields: {len(missing_fields)}")
        
        # Set final status
        if validation_report["issues"]:
            validation_report["status"] = "WARN" if validation_report["status"] == "PASS" else "FAIL"
        
        return validation_report


@pytest.fixture
def e2e_test_manager():
    """Create E2E test manager instance"""
    return E2EWorkflowTest()


@pytest.fixture
def test_files():
    """Get available test file pairs"""
    test_pairs = []
    
    # Find all PA forms and matching referral packages
    for i in range(1, 4):  # test_1, test_2, test_3
        pa_file = TEST_DATA_DIR / f"test_{i}_PA.pdf"
        referral_file = TEST_DATA_DIR / f"test_{i}_referral_package.pdf"
        
        if pa_file.exists() and referral_file.exists():
            test_pairs.append({
                "name": f"test_case_{i}",
                "pa_form": pa_file,
                "referral_package": referral_file
            })
    
    return test_pairs


class TestE2EWorkflow:
    """Complete end-to-end workflow tests"""
    
    async def test_complete_pipeline_single_document(self, async_client: AsyncClient, test_files, e2e_test_manager):
        """Test complete processing pipeline with single document pair"""
        if not test_files:
            pytest.skip("No test file pairs available")
        
        test_case = test_files[0]  # Use first available test case
        
        # Step 1: Upload files
        upload_start = time.time()
        
        with open(test_case["pa_form"], "rb") as pa_file, \
             open(test_case["referral_package"], "rb") as ref_file:
            
            files = [
                ("files", ("pa_form.pdf", pa_file.read(), "application/pdf")),
                ("files", ("referral_package.pdf", ref_file.read(), "application/pdf"))
            ]
            
            response = await async_client.post("/api/upload", files=files)
        
        assert response.status_code == 200
        upload_data = response.json()
        session_id = upload_data["session_id"]
        
        upload_time = time.time() - upload_start
        e2e_test_manager.processing_times["upload"] = upload_time
        
        # Step 2: Start processing
        process_start = time.time()
        
        response = await async_client.post(f"/api/process/{session_id}")
        assert response.status_code == 200
        
        # Step 3: Poll for completion
        max_wait_time = 300  # 5 minutes max
        poll_interval = 2  # Check every 2 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            await asyncio.sleep(poll_interval)
            elapsed_time += poll_interval
            
            response = await async_client.get(f"/api/process/{session_id}/status")
            assert response.status_code == 200
            
            status_data = response.json()
            current_status = status_data["status"]
            
            if current_status == ProcessingStatus.COMPLETED:
                break
            elif current_status == ProcessingStatus.FAILED:
                pytest.fail(f"Processing failed: {status_data.get('error', 'Unknown error')}")
        
        if elapsed_time >= max_wait_time:
            pytest.fail("Processing timed out after 5 minutes")
        
        process_time = time.time() - process_start
        e2e_test_manager.processing_times["processing"] = process_time
        
        # Step 4: Download filled form
        response = await async_client.get(f"/api/download/{session_id}/filled")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        
        filled_pdf_path = EXPECTED_OUTPUTS_DIR / f"{test_case['name']}_filled.pdf"
        with open(filled_pdf_path, "wb") as f:
            f.write(response.content)
        
        # Step 5: Download report
        response = await async_client.get(f"/api/download/{session_id}/report")
        assert response.status_code == 200
        assert "markdown" in response.headers["content-type"] or "text" in response.headers["content-type"]
        
        report_path = EXPECTED_OUTPUTS_DIR / f"{test_case['name']}_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(response.content.decode("utf-8"))
        
        # Step 6: Validate results
        # Get the processing result for validation
        storage = FileStorage()
        result_file = storage.get_session_directory(session_id) / "processing_result.json"
        
        if result_file.exists():
            with open(result_file, "r") as f:
                processing_result = json.load(f)
            
            validation_report = e2e_test_manager.validate_processing_result(
                processing_result, test_case["name"]
            )
            
            # Save validation report
            validation_path = EXPECTED_OUTPUTS_DIR / f"{test_case['name']}_validation.json"
            with open(validation_path, "w") as f:
                json.dump(validation_report, f, indent=2)
            
            # Assert quality thresholds
            if validation_report["status"] == "FAIL":
                pytest.fail(f"Quality validation failed: {validation_report['issues']}")
        
        # Record test results
        e2e_test_manager.test_results[test_case["name"]] = {
            "status": "COMPLETED",
            "upload_time": upload_time,
            "processing_time": process_time,
            "total_time": upload_time + process_time,
            "filled_pdf_size": len(response.content),
            "session_id": session_id
        }
    
    async def test_all_document_pairs(self, async_client: AsyncClient, test_files, e2e_test_manager):
        """Test processing for all available document pairs"""
        if len(test_files) < 2:
            pytest.skip("Need at least 2 test cases for comprehensive testing")
        
        results_summary = []
        
        for test_case in test_files:
            try:
                # Upload and process each test case
                with open(test_case["pa_form"], "rb") as pa_file, \
                     open(test_case["referral_package"], "rb") as ref_file:
                    
                    files = [
                        ("files", ("pa_form.pdf", pa_file.read(), "application/pdf")),
                        ("files", ("referral_package.pdf", ref_file.read(), "application/pdf"))
                    ]
                    
                    response = await async_client.post("/api/upload", files=files)
                
                assert response.status_code == 200
                upload_data = response.json()
                session_id = upload_data["session_id"]
                
                # Start processing
                start_time = time.time()
                response = await async_client.post(f"/api/process/{session_id}")
                assert response.status_code == 200
                
                # Poll for completion (shorter timeout for multiple tests)
                max_wait_time = 180  # 3 minutes per test
                poll_interval = 3
                elapsed_time = 0
                
                while elapsed_time < max_wait_time:
                    await asyncio.sleep(poll_interval)
                    elapsed_time += poll_interval
                    
                    response = await async_client.get(f"/api/process/{session_id}/status")
                    status_data = response.json()
                    current_status = status_data["status"]
                    
                    if current_status == ProcessingStatus.COMPLETED:
                        break
                    elif current_status == ProcessingStatus.FAILED:
                        raise Exception(f"Processing failed: {status_data.get('error')}")
                
                processing_time = time.time() - start_time
                
                # Download results
                filled_response = await async_client.get(f"/api/download/{session_id}/filled")
                report_response = await async_client.get(f"/api/download/{session_id}/report")
                
                # Save outputs
                filled_path = EXPECTED_OUTPUTS_DIR / f"{test_case['name']}_filled.pdf"
                report_path = EXPECTED_OUTPUTS_DIR / f"{test_case['name']}_report.md"
                
                with open(filled_path, "wb") as f:
                    f.write(filled_response.content)
                
                with open(report_path, "w") as f:
                    f.write(report_response.content.decode("utf-8"))
                
                results_summary.append({
                    "test_case": test_case["name"],
                    "status": "SUCCESS",
                    "processing_time": processing_time,
                    "filled_pdf_size": len(filled_response.content),
                    "session_id": session_id
                })
                
            except Exception as e:
                results_summary.append({
                    "test_case": test_case["name"],
                    "status": "FAILED",
                    "error": str(e),
                    "processing_time": None
                })
        
        # Save comprehensive results
        summary_path = EXPECTED_OUTPUTS_DIR / "e2e_test_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        
        # Assert all tests passed
        failed_tests = [r for r in results_summary if r["status"] == "FAILED"]
        if failed_tests:
            pytest.fail(f"Failed test cases: {[t['test_case'] for t in failed_tests]}")
    
    async def test_error_handling_scenarios(self, async_client: AsyncClient):
        """Test error handling in various failure scenarios"""
        
        # Test 1: Invalid file types
        invalid_files = [
            ("files", ("test.txt", b"not a pdf", "text/plain")),
            ("files", ("test2.txt", b"also not a pdf", "text/plain"))
        ]
        
        response = await async_client.post("/api/upload", files=invalid_files)
        assert response.status_code == 400
        
        # Test 2: Single file upload (should require 2 files)
        with open(TEST_DATA_DIR / "test_1_PA.pdf", "rb") as f:
            single_file = [("files", ("test.pdf", f.read(), "application/pdf"))]
            response = await async_client.post("/api/upload", files=single_file)
        
        assert response.status_code == 400
        
        # Test 3: Invalid session ID
        response = await async_client.post("/api/process/invalid-session-id")
        assert response.status_code == 404
        
        response = await async_client.get("/api/process/invalid-session-id/status")
        assert response.status_code == 404
        
        response = await async_client.get("/api/download/invalid-session-id/filled")
        assert response.status_code == 404
    
    async def test_performance_benchmarks(self, async_client: AsyncClient, test_files, e2e_test_manager):
        """Test performance against established benchmarks"""
        if not test_files:
            pytest.skip("No test files available for performance testing")
        
        test_case = test_files[0]
        
        # Get file sizes for context
        pa_size = test_case["pa_form"].stat().st_size
        ref_size = test_case["referral_package"].stat().st_size
        
        # Run performance test
        start_time = time.time()
        
        with open(test_case["pa_form"], "rb") as pa_file, \
             open(test_case["referral_package"], "rb") as ref_file:
            
            files = [
                ("files", ("pa_form.pdf", pa_file.read(), "application/pdf")),
                ("files", ("referral_package.pdf", ref_file.read(), "application/pdf"))
            ]
            
            upload_response = await async_client.post("/api/upload", files=files)
        
        session_id = upload_response.json()["session_id"]
        
        process_response = await async_client.post(f"/api/process/{session_id}")
        assert process_response.status_code == 200
        
        # Wait for completion and measure time
        max_wait = 300  # 5 minutes
        elapsed = 0
        
        while elapsed < max_wait:
            await asyncio.sleep(2)
            elapsed += 2
            
            status_response = await async_client.get(f"/api/process/{session_id}/status")
            status = status_response.json()["status"]
            
            if status == ProcessingStatus.COMPLETED:
                break
        
        total_time = time.time() - start_time
        
        # Performance assertions based on analysis
        # Expected: <120 seconds per page with real APIs
        # Assuming average 10 pages per document = 1200 seconds max
        assert total_time < 1200, f"Processing took too long: {total_time:.2f}s"
        
        # Save performance metrics
        performance_report = {
            "test_case": test_case["name"],
            "pa_form_size_mb": pa_size / (1024 * 1024),
            "referral_size_mb": ref_size / (1024 * 1024),
            "total_processing_time_seconds": total_time,
            "performance_rating": "PASS" if total_time < 600 else "SLOW"
        }
        
        perf_path = EXPECTED_OUTPUTS_DIR / f"{test_case['name']}_performance.json"
        with open(perf_path, "w") as f:
            json.dump(performance_report, f, indent=2)


@pytest.mark.asyncio
async def test_end_to_end_golden_outputs(async_client: AsyncClient, test_files):
    """Generate golden outputs for regression testing"""
    if not test_files:
        pytest.skip("No test files available")
    
    golden_outputs = {}
    
    for test_case in test_files[:2]:  # Limit to first 2 for golden outputs
        with open(test_case["pa_form"], "rb") as pa_file, \
             open(test_case["referral_package"], "rb") as ref_file:
            
            files = [
                ("files", ("pa_form.pdf", pa_file.read(), "application/pdf")),
                ("files", ("referral_package.pdf", ref_file.read(), "application/pdf"))
            ]
            
            response = await async_client.post("/api/upload", files=files)
        
        session_id = response.json()["session_id"]
        
        # Process and wait for completion
        await async_client.post(f"/api/process/{session_id}")
        
        # Poll until complete (with timeout)
        for _ in range(150):  # 5 minutes max
            await asyncio.sleep(2)
            status_response = await async_client.get(f"/api/process/{session_id}/status")
            if status_response.json()["status"] == ProcessingStatus.COMPLETED:
                break
        
        # Download and save golden outputs
        filled_response = await async_client.get(f"/api/download/{session_id}/filled")
        report_response = await async_client.get(f"/api/download/{session_id}/report")
        
        golden_filled_path = EXPECTED_OUTPUTS_DIR / f"golden_{test_case['name']}_filled.pdf"
        golden_report_path = EXPECTED_OUTPUTS_DIR / f"golden_{test_case['name']}_report.md"
        
        with open(golden_filled_path, "wb") as f:
            f.write(filled_response.content)
        
        with open(golden_report_path, "w") as f:
            f.write(report_response.content.decode("utf-8"))
        
        golden_outputs[test_case["name"]] = {
            "filled_pdf_hash": hashlib.md5(filled_response.content).hexdigest(),
            "report_length": len(report_response.content),
            "session_id": session_id
        }
    
    # Save golden output metadata
    with open(EXPECTED_OUTPUTS_DIR / "golden_outputs_metadata.json", "w") as f:
        json.dump(golden_outputs, f, indent=2)


if __name__ == "__main__":
    # Run specific test for debugging
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))