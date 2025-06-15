#!/usr/bin/env python3
"""
End-to-End Workflow Testing for MediLink-AI

This script tests the complete workflow from file upload through processing 
to final output generation, using the actual test documents.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import httpx
import aiofiles
from datetime import datetime

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import Settings
from app.services.storage import FileStorage


class E2EWorkflowTester:
    """End-to-end workflow testing suite"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.settings = Settings()
        self.storage = FileStorage(self.settings)
        self.test_data_dir = Path("tests/test_data")
        self.output_dir = Path("test_outputs/e2e_workflow")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test document pairs
        self.test_cases = [
            {
                "name": "test_1",
                "referral": "test_1_referral_package.pdf",
                "pa_form": "test_1_PA.pdf",
                "description": "Aetna Rituximab PA + 15-page referral package"
            },
            {
                "name": "test_2", 
                "referral": "test_2_referral_package.pdf",
                "pa_form": "test_2_PA.pdf",
                "description": "Aetna Skyrizi PA + 10-page referral package"
            },
            {
                "name": "test_3",
                "referral": "test_3_referral_package.pdf", 
                "pa_form": "test_3_PA.pdf",
                "description": "Anthem Vyepti PA + 9-page referral package"
            }
        ]
        
        self.results = []
        
    async def run_complete_workflow_test(self) -> Dict:
        """Run complete end-to-end workflow tests"""
        print("🚀 Starting End-to-End Workflow Testing...")
        print(f"Base URL: {self.base_url}")
        print(f"Test Data Directory: {self.test_data_dir.absolute()}")
        print(f"Output Directory: {self.output_dir.absolute()}")
        print("-" * 60)
        
        # Check if backend is running
        if not await self._check_backend_health():
            return {"error": "Backend is not running or not healthy"}
            
        # Run tests for each document pair
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n📋 Test Case {i}/3: {test_case['name']}")
            print(f"Description: {test_case['description']}")
            
            try:
                result = await self._test_document_pair(test_case)
                result["test_case"] = test_case["name"]
                result["description"] = test_case["description"]
                self.results.append(result)
                
                if result["success"]:
                    print(f"✅ Test case {test_case['name']} completed successfully")
                else:
                    print(f"❌ Test case {test_case['name']} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                error_result = {
                    "test_case": test_case["name"],
                    "description": test_case["description"],
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                self.results.append(error_result)
                print(f"💥 Test case {test_case['name']} crashed: {str(e)}")
                
        # Generate summary report
        await self._generate_summary_report()
        
        # Cleanup temporary files
        await self._cleanup_temp_files()
        
        return {
            "total_tests": len(self.test_cases),
            "successful": len([r for r in self.results if r["success"]]),
            "failed": len([r for r in self.results if not r["success"]]),
            "results": self.results
        }
        
    async def _check_backend_health(self) -> bool:
        """Check if backend is running and healthy"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/health")
                if response.status_code == 200:
                    health_data = response.json()
                    print(f"✅ Backend health check passed: {health_data}")
                    return True
                else:
                    print(f"❌ Backend health check failed: {response.status_code}")
                    return False
        except Exception as e:
            print(f"❌ Cannot connect to backend: {str(e)}")
            return False
            
    async def _test_document_pair(self, test_case: Dict) -> Dict:
        """Test processing of a referral + PA form pair"""
        start_time = time.time()
        
        # Step 1: Upload files
        print(f"  📤 Step 1: Uploading files...")
        upload_result = await self._upload_files(test_case)
        if not upload_result["success"]:
            return upload_result
            
        session_id = upload_result["session_id"]
        print(f"  ✅ Files uploaded successfully. Session ID: {session_id}")
        
        # Step 2: Start processing
        print(f"  ⚙️  Step 2: Starting processing...")
        process_result = await self._start_processing(session_id)
        if not process_result["success"]:
            return process_result
            
        # Step 3: Monitor processing status
        print(f"  👀 Step 3: Monitoring processing status...")
        status_result = await self._monitor_processing(session_id)
        if not status_result["success"]:
            return status_result
            
        # Step 4: Download outputs
        print(f"  📥 Step 4: Downloading outputs...")
        download_result = await self._download_outputs(session_id, test_case["name"])
        if not download_result["success"]:
            return download_result
            
        # Step 5: Validate outputs
        print(f"  ✅ Step 5: Validating outputs...")
        validation_result = await self._validate_outputs(test_case["name"])
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "session_id": session_id,
            "processing_time": total_time,
            "upload_details": upload_result,
            "processing_details": status_result,
            "download_details": download_result,
            "validation_details": validation_result,
            "timestamp": datetime.now().isoformat()
        }
        
    async def _upload_files(self, test_case: Dict) -> Dict:
        """Upload referral and PA form files"""
        try:
            referral_path = self.test_data_dir / test_case["referral"]
            pa_form_path = self.test_data_dir / test_case["pa_form"]
            
            if not referral_path.exists():
                return {"success": False, "error": f"Referral file not found: {referral_path}"}
            if not pa_form_path.exists():
                return {"success": False, "error": f"PA form file not found: {pa_form_path}"}
                
            # Prepare files for upload
            files = [
                ("files", (test_case["referral"], open(referral_path, "rb"), "application/pdf")),
                ("files", (test_case["pa_form"], open(pa_form_path, "rb"), "application/pdf"))
            ]
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self.base_url}/api/upload", files=files)
                
            # Close files
            for _, (_, file_obj, _) in files:
                file_obj.close()
                
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "session_id": data["session_id"],
                    "files_uploaded": data["files"],
                    "response": data
                }
            else:
                return {
                    "success": False,
                    "error": f"Upload failed with status {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {"success": False, "error": f"Upload exception: {str(e)}"}
            
    async def _start_processing(self, session_id: str) -> Dict:
        """Start document processing"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self.base_url}/api/process/{session_id}")
                
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "message": data.get("message", "Processing started"),
                    "response": data
                }
            else:
                return {
                    "success": False,
                    "error": f"Processing start failed with status {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {"success": False, "error": f"Processing start exception: {str(e)}"}
            
    async def _monitor_processing(self, session_id: str, max_wait_time: int = 300) -> Dict:
        """Monitor processing status until completion"""
        start_time = time.time()
        
        try:
            while True:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(f"{self.base_url}/api/process/{session_id}/status")
                    
                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Status check failed with status {response.status_code}: {response.text}"
                    }
                    
                data = response.json()
                status = data.get("status", "unknown")
                stage = data.get("current_stage", "unknown")
                progress = data.get("progress", 0)
                
                print(f"    Status: {status} | Stage: {stage} | Progress: {progress}%")
                
                if status == "completed":
                    return {
                        "success": True,
                        "status": status,
                        "total_processing_time": time.time() - start_time,
                        "final_response": data
                    }
                elif status == "failed":
                    return {
                        "success": False,
                        "error": f"Processing failed: {data.get('error', 'Unknown error')}",
                        "response": data
                    }
                elif time.time() - start_time > max_wait_time:
                    return {
                        "success": False,
                        "error": f"Processing timeout after {max_wait_time} seconds",
                        "last_status": data
                    }
                    
                await asyncio.sleep(5)  # Wait 5 seconds before next check
                
        except Exception as e:
            return {"success": False, "error": f"Status monitoring exception: {str(e)}"}
            
    async def _download_outputs(self, session_id: str, test_name: str) -> Dict:
        """Download filled form and report"""
        try:
            download_results = {}
            
            # Download filled PDF
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(f"{self.base_url}/api/download/{session_id}/filled")
                
            if response.status_code == 200:
                filled_pdf_path = self.output_dir / f"{test_name}_filled.pdf"
                async with aiofiles.open(filled_pdf_path, "wb") as f:
                    await f.write(response.content)
                download_results["filled_pdf"] = str(filled_pdf_path)
                print(f"    ✅ Downloaded filled PDF: {filled_pdf_path}")
            else:
                download_results["filled_pdf_error"] = f"Status {response.status_code}: {response.text}"
                
            # Download report
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/api/download/{session_id}/report")
                
            if response.status_code == 200:
                report_path = self.output_dir / f"{test_name}_report.md"
                async with aiofiles.open(report_path, "wb") as f:
                    await f.write(response.content)
                download_results["report"] = str(report_path)
                print(f"    ✅ Downloaded report: {report_path}")
            else:
                download_results["report_error"] = f"Status {response.status_code}: {response.text}"
                
            return {
                "success": True,
                "downloads": download_results
            }
            
        except Exception as e:
            return {"success": False, "error": f"Download exception: {str(e)}"}
            
    async def _validate_outputs(self, test_name: str) -> Dict:
        """Validate generated outputs"""
        validation_results = {}
        
        # Check filled PDF
        filled_pdf_path = self.output_dir / f"{test_name}_filled.pdf"
        if filled_pdf_path.exists():
            file_size = filled_pdf_path.stat().st_size
            validation_results["filled_pdf"] = {
                "exists": True,
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2)
            }
        else:
            validation_results["filled_pdf"] = {"exists": False}
            
        # Check report
        report_path = self.output_dir / f"{test_name}_report.md"
        if report_path.exists():
            file_size = report_path.stat().st_size
            # Read and analyze report content
            async with aiofiles.open(report_path, "r") as f:
                content = await f.read()
            
            validation_results["report"] = {
                "exists": True,
                "size_bytes": file_size,
                "content_length": len(content),
                "has_missing_fields": "missing fields" in content.lower(),
                "has_confidence_scores": "confidence" in content.lower()
            }
        else:
            validation_results["report"] = {"exists": False}
            
        return validation_results
        
    async def _generate_summary_report(self):
        """Generate comprehensive summary report"""
        summary = {
            "test_run_timestamp": datetime.now().isoformat(),
            "total_test_cases": len(self.test_cases),
            "successful_tests": len([r for r in self.results if r["success"]]),
            "failed_tests": len([r for r in self.results if not r["success"]]),
            "success_rate": len([r for r in self.results if r["success"]]) / len(self.test_cases) * 100 if self.test_cases else 0,
            "test_results": self.results
        }
        
        # Calculate performance metrics
        successful_results = [r for r in self.results if r["success"]]
        if successful_results:
            processing_times = [r["processing_time"] for r in successful_results]
            summary["performance_metrics"] = {
                "average_processing_time": sum(processing_times) / len(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "total_processing_time": sum(processing_times)
            }
            
        # Save summary to JSON
        summary_path = self.output_dir / "e2e_workflow_summary.json"
        async with aiofiles.open(summary_path, "w") as f:
            await f.write(json.dumps(summary, indent=2))
            
        print(f"\n📊 Test Summary:")
        print(f"  Total Tests: {summary['total_test_cases']}")
        print(f"  Successful: {summary['successful_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        
        if "performance_metrics" in summary:
            print(f"  Average Processing Time: {summary['performance_metrics']['average_processing_time']:.1f}s")
            print(f"  Min Processing Time: {summary['performance_metrics']['min_processing_time']:.1f}s")
            print(f"  Max Processing Time: {summary['performance_metrics']['max_processing_time']:.1f}s")
            
        print(f"  📋 Full summary saved to: {summary_path}")
        
    async def _cleanup_temp_files(self):
        """Clean up temporary files created during testing"""
        try:
            # Clean up session directories older than test run
            if hasattr(self.storage, 'uploads_dir'):
                uploads_dir = Path(self.storage.uploads_dir)
                if uploads_dir.exists():
                    session_dirs = [d for d in uploads_dir.iterdir() if d.is_dir()]
                    cleaned_count = 0
                    
                    for session_dir in session_dirs:
                        # Remove session directories that are more than 1 hour old
                        if time.time() - session_dir.stat().st_mtime > 3600:
                            import shutil
                            shutil.rmtree(session_dir)
                            cleaned_count += 1
                            
                    if cleaned_count > 0:
                        print(f"🧹 Cleaned up {cleaned_count} old session directories")
                        
        except Exception as e:
            print(f"⚠️  Cleanup warning: {str(e)}")


async def run_e2e_tests():
    """Main function to run end-to-end tests"""
    tester = E2EWorkflowTester()
    results = await tester.run_complete_workflow_test()
    
    print("\n" + "="*60)
    print("🏁 End-to-End Workflow Testing Complete!")
    print("="*60)
    
    if "error" in results:
        print(f"❌ Critical Error: {results['error']}")
        return False
        
    success_rate = (results["successful"] / results["total_tests"]) * 100 if results["total_tests"] > 0 else 0
    
    print(f"📈 Overall Results:")
    print(f"  Total Tests: {results['total_tests']}")
    print(f"  Successful: {results['successful']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 66:  # At least 2 out of 3 tests should pass
        print("✅ End-to-end workflow testing PASSED!")
        return True
    else:
        print("❌ End-to-end workflow testing FAILED!")
        return False


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(run_e2e_tests())
    sys.exit(0 if success else 1)