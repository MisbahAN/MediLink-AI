#!/usr/bin/env python3
"""
Complete End-to-End Workflow Test for MediLink-AI

This script provides a comprehensive test of the complete workflow:
1. Starts the backend server
2. Tests complete processing with all 3 test document pairs
3. Validates outputs and generates golden outputs
4. Tests error scenarios
5. Provides detailed reporting

This demonstrates the system working end-to-end with real documents and real AI services.
"""

import asyncio
import json
import os
import sys
import time
import subprocess
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import httpx

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings


class ComprehensiveE2ETest:
    """Comprehensive end-to-end test runner"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_data_dir = Path("tests/test_data")
        self.output_dir = Path("test_outputs/e2e_workflow")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.server_process = None
        
        # Test document pairs
        self.test_cases = [
            {
                "name": "test_1",
                "referral": "test_1_referral_package.pdf",
                "pa_form": "test_1_PA.pdf",
                "description": "Aetna Rituximab PA + 15-page referral package",
                "expected_complexity": "high"
            },
            {
                "name": "test_2", 
                "referral": "test_2_referral_package.pdf",
                "pa_form": "test_2_PA.pdf",
                "description": "Aetna Skyrizi PA + 10-page referral package",
                "expected_complexity": "medium"
            },
            {
                "name": "test_3",
                "referral": "test_3_referral_package.pdf", 
                "pa_form": "test_3_PA.pdf",
                "description": "Anthem Vyepti PA + 9-page referral package",
                "expected_complexity": "medium"
            }
        ]
        
        self.test_results = []
        
    def start_server(self) -> bool:
        """Start the FastAPI server"""
        print("🚀 Starting FastAPI server...")
        
        # Start uvicorn server as subprocess
        self.server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "127.0.0.1", 
            "--port", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        for _ in range(30):  # 30 second timeout
            try:
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(f"{self.base_url}/api/health")
                    if response.status_code == 200:
                        health_data = response.json()
                        print(f"✅ Server is ready! Health: {health_data}")
                        return True
            except Exception:
                pass
            time.sleep(1)
            
        print("❌ Server failed to start")
        return False
        
    def stop_server(self):
        """Stop the FastAPI server"""
        if self.server_process:
            print("🛑 Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
                
    async def test_single_document_workflow(self, test_case: Dict) -> Dict:
        """Test the complete workflow with one document pair"""
        
        print(f"\n📋 Testing: {test_case['name']}")
        print(f"   Description: {test_case['description']}")
        
        start_time = time.time()
        
        # Verify files exist
        referral_path = self.test_data_dir / test_case["referral"]
        pa_form_path = self.test_data_dir / test_case["pa_form"]
        
        for file_type, file_path in [("referral", referral_path), ("pa_form", pa_form_path)]:
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"{file_type} file not found: {file_path}",
                    "test_case": test_case["name"]
                }
                
        print(f"📄 Files:")
        print(f"  Referral: {referral_path.name} ({referral_path.stat().st_size:,} bytes)")
        print(f"  PA Form: {pa_form_path.name} ({pa_form_path.stat().st_size:,} bytes)")
        
        try:
            # Step 1: Upload files
            print("📤 Step 1: Uploading files...")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                with open(referral_path, "rb") as ref_file, \
                     open(pa_form_path, "rb") as pa_file:
                    
                    files = [
                        ("files", (test_case["referral"], ref_file.read(), "application/pdf")),
                        ("files", (test_case["pa_form"], pa_file.read(), "application/pdf"))
                    ]
                    
                    response = await client.post(f"{self.base_url}/api/upload", files=files)
                    
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Upload failed: {response.status_code} - {response.text}",
                    "test_case": test_case["name"]
                }
                
            upload_data = response.json()
            session_id = upload_data["session_id"]
            print(f"✅ Upload successful. Session ID: {session_id}")
            
            # Step 2: Start processing
            print("⚙️  Step 2: Starting processing...")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self.base_url}/api/process/{session_id}")
                
            if response.status_code not in [200, 202]:
                return {
                    "success": False,
                    "error": f"Processing start failed: {response.status_code} - {response.text}",
                    "test_case": test_case["name"]
                }
                
            print("✅ Processing started successfully")
            
            # Step 3: Monitor processing
            print("👀 Step 3: Monitoring processing...")
            
            max_wait_time = 600  # 10 minutes for real AI processing
            processing_start = time.time()
            
            while True:
                elapsed = time.time() - processing_start
                if elapsed > max_wait_time:
                    return {
                        "success": False,
                        "error": f"Processing timeout after {max_wait_time} seconds",
                        "test_case": test_case["name"]
                    }
                    
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(f"{self.base_url}/api/process/{session_id}/status")
                    
                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Status check failed: {response.status_code}",
                        "test_case": test_case["name"]
                    }
                    
                status_data = response.json()
                status = status_data.get("status", "unknown")
                stage = status_data.get("current_stage", "unknown")
                progress = status_data.get("progress", 0)
                
                print(f"  Status: {status} | Stage: {stage} | Progress: {progress}% | Elapsed: {elapsed:.1f}s")
                
                if status == "completed":
                    print("✅ Processing completed!")
                    break
                elif status == "failed":
                    error = status_data.get("error", "Unknown error")
                    return {
                        "success": False,
                        "error": f"Processing failed: {error}",
                        "test_case": test_case["name"]
                    }
                    
                await asyncio.sleep(10)  # Check every 10 seconds
                
            processing_time = time.time() - processing_start
            
            # Step 4: Download outputs
            print("📥 Step 4: Downloading outputs...")
            
            # Download filled PDF
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(f"{self.base_url}/api/download/{session_id}/filled")
                
            if response.status_code == 200:
                filled_pdf_path = self.output_dir / f"{test_case['name']}_filled.pdf"
                with open(filled_pdf_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ Downloaded filled PDF: {filled_pdf_path}")
                print(f"  Size: {len(response.content):,} bytes")
                filled_pdf_size = len(response.content)
            else:
                return {
                    "success": False,
                    "error": f"Failed to download filled PDF: {response.status_code}",
                    "test_case": test_case["name"]
                }
                
            # Download report
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/api/download/{session_id}/report")
                
            if response.status_code == 200:
                report_path = self.output_dir / f"{test_case['name']}_report.md"
                with open(report_path, "w") as f:
                    f.write(response.content.decode("utf-8"))
                print(f"✅ Downloaded report: {report_path}")
                print(f"  Size: {len(response.content):,} bytes")
                report_content = response.content.decode("utf-8")
            else:
                return {
                    "success": False,
                    "error": f"Failed to download report: {response.status_code}",
                    "test_case": test_case["name"]
                }
                
            # Step 5: Validate outputs
            print("✅ Step 5: Validating outputs...")
            
            validation_results = {
                "filled_pdf_size": filled_pdf_size,
                "report_length": len(report_content),
                "has_missing_fields": "missing" in report_content.lower(),
                "has_confidence_scores": "confidence" in report_content.lower(),
                "has_patient_info": "patient" in report_content.lower(),
                "has_field_mappings": "mapped" in report_content.lower() or "field" in report_content.lower()
            }
            
            # Basic validation checks
            validation_issues = []
            
            if filled_pdf_size < 1000:
                validation_issues.append("Filled PDF seems too small")
                
            if len(report_content) < 100:
                validation_issues.append("Report seems too small")
                
            if not validation_results["has_confidence_scores"]:
                validation_issues.append("Report missing confidence scores")
                
            total_time = time.time() - start_time
            
            result = {
                "success": True,
                "test_case": test_case["name"],
                "description": test_case["description"],
                "session_id": session_id,
                "processing_time": processing_time,
                "total_time": total_time,
                "validation": validation_results,
                "validation_issues": validation_issues,
                "outputs": {
                    "filled_pdf": str(filled_pdf_path),
                    "report": str(report_path)
                }
            }
            
            # Save validation results
            validation_path = self.output_dir / f"{test_case['name']}_validation.json"
            with open(validation_path, "w") as f:
                json.dump(result, f, indent=2)
                
            print(f"🎉 Test case {test_case['name']} completed successfully!")
            print(f"  Processing time: {processing_time:.1f}s")
            print(f"  Total time: {total_time:.1f}s")
            
            if validation_issues:
                print(f"⚠️  Validation issues: {validation_issues}")
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception during test: {str(e)}",
                "test_case": test_case["name"]
            }
            
    async def test_error_scenarios(self) -> Dict:
        """Test error handling scenarios"""
        print("\n🔧 Testing error scenarios...")
        
        error_tests = []
        
        try:
            # Test 1: Invalid file types
            print("  Test 1: Invalid file types...")
            async with httpx.AsyncClient(timeout=10.0) as client:
                files = [
                    ("files", ("test.txt", b"not a pdf", "text/plain")),
                    ("files", ("test2.txt", b"also not a pdf", "text/plain"))
                ]
                response = await client.post(f"{self.base_url}/api/upload", files=files)
                
            if response.status_code == 400:
                error_tests.append({"test": "invalid_file_types", "status": "pass"})
                print("    ✅ Correctly rejected invalid file types")
            else:
                error_tests.append({"test": "invalid_file_types", "status": "fail", "details": f"Expected 400, got {response.status_code}"})
                print(f"    ❌ Should have rejected invalid files (got {response.status_code})")
                
            # Test 2: Invalid session ID
            print("  Test 2: Invalid session ID...")
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(f"{self.base_url}/api/process/invalid-session-id")
                
            if response.status_code == 404:
                error_tests.append({"test": "invalid_session_id", "status": "pass"})
                print("    ✅ Correctly handled invalid session ID")
            else:
                error_tests.append({"test": "invalid_session_id", "status": "fail", "details": f"Expected 404, got {response.status_code}"})
                print(f"    ❌ Should have returned 404 for invalid session (got {response.status_code})")
                
            # Test 3: Health check
            print("  Test 3: Health check...")
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/health")
                
            if response.status_code == 200:
                health_data = response.json()
                if "status" in health_data:
                    error_tests.append({"test": "health_check", "status": "pass"})
                    print("    ✅ Health check working correctly")
                else:
                    error_tests.append({"test": "health_check", "status": "fail", "details": "Missing status in health response"})
                    print("    ❌ Health check missing status field")
            else:
                error_tests.append({"test": "health_check", "status": "fail", "details": f"Expected 200, got {response.status_code}"})
                print(f"    ❌ Health check failed (got {response.status_code})")
                
        except Exception as e:
            error_tests.append({"test": "error_scenario_testing", "status": "fail", "details": str(e)})
            print(f"    ❌ Error scenario testing failed: {str(e)}")
            
        return {
            "success": len([t for t in error_tests if t["status"] == "pass"]) == len(error_tests),
            "tests": error_tests,
            "passed": len([t for t in error_tests if t["status"] == "pass"]),
            "total": len(error_tests)
        }
        
    async def run_complete_test_suite(self) -> Dict:
        """Run the complete test suite"""
        print("🧪 MediLink-AI Comprehensive End-to-End Test Suite")
        print("=" * 60)
        
        suite_start_time = time.time()
        
        # Test all document pairs
        for test_case in self.test_cases:
            result = await self.test_single_document_workflow(test_case)
            self.test_results.append(result)
            
        # Test error scenarios
        error_result = await self.test_error_scenarios()
        
        # Generate summary
        successful_tests = [r for r in self.test_results if r["success"]]
        failed_tests = [r for r in self.test_results if not r["success"]]
        
        summary = {
            "total_tests": len(self.test_cases),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests) / len(self.test_cases) * 100 if self.test_cases else 0,
            "error_tests": error_result,
            "total_time": time.time() - suite_start_time,
            "test_results": self.test_results
        }
        
        # Calculate performance metrics
        if successful_tests:
            processing_times = [r["processing_time"] for r in successful_tests]
            summary["performance_metrics"] = {
                "average_processing_time": sum(processing_times) / len(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "total_processing_time": sum(processing_times)
            }
            
        # Save comprehensive summary
        summary_path = self.output_dir / "comprehensive_e2e_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n📊 Test Suite Summary:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Successful: {summary['successful_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Error Tests: {error_result['passed']}/{error_result['total']} passed")
        
        if "performance_metrics" in summary:
            print(f"  Average Processing Time: {summary['performance_metrics']['average_processing_time']:.1f}s")
            print(f"  Min Processing Time: {summary['performance_metrics']['min_processing_time']:.1f}s")
            print(f"  Max Processing Time: {summary['performance_metrics']['max_processing_time']:.1f}s")
            
        print(f"  Total Suite Time: {summary['total_time']:.1f}s")
        print(f"📋 Full summary saved to: {summary_path}")
        
        if failed_tests:
            print(f"\n❌ Failed tests:")
            for test in failed_tests:
                print(f"  - {test['test_case']}: {test.get('error', 'Unknown error')}")
                
        return summary


async def main():
    """Main test function"""
    test_runner = ComprehensiveE2ETest()
    
    try:
        # Start server
        if not test_runner.start_server():
            print("❌ Cannot start server. Exiting.")
            return False
            
        # Run the complete test suite
        summary = await test_runner.run_complete_test_suite()
        
        success_threshold = 66  # At least 2 out of 3 tests should pass
        overall_success = (summary["success_rate"] >= success_threshold and 
                          summary["error_tests"]["passed"] >= summary["error_tests"]["total"] * 0.75)
        
        print("\n" + "="*60)
        if overall_success:
            print("🎉 Complete End-to-End Test Suite PASSED!")
        else:
            print("❌ Complete End-to-End Test Suite FAILED!")
        print("="*60)
        
        return overall_success
        
    except Exception as e:
        print(f"\n💥 Test suite crashed: {str(e)}")
        return False
        
    finally:
        # Always stop the server
        test_runner.stop_server()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)