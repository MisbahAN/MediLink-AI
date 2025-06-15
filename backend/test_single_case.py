#!/usr/bin/env python3
"""
Single Document E2E Test for MediLink-AI

Tests just one document pair to validate the complete workflow works.
"""

import asyncio
import json
import os
import sys
import time
import subprocess
import signal
from pathlib import Path

import httpx

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))


class SingleCaseTest:
    """Single case E2E test"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_data_dir = Path("tests/test_data")
        self.output_dir = Path("test_outputs/e2e_workflow")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.server_process = None
        
    def start_server(self) -> bool:
        """Start the FastAPI server"""
        print("🚀 Starting FastAPI server...")
        
        self.server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "127.0.0.1", 
            "--port", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        for _ in range(30):
            try:
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(f"{self.base_url}/api/health")
                    if response.status_code == 200:
                        print("✅ Server is ready!")
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
                
    async def test_single_case(self):
        """Test processing with test_2 (medium complexity)"""
        
        print("📋 Testing: test_2 (Aetna Skyrizi PA + 10-page referral)")
        
        # Use test_2 files
        referral_path = self.test_data_dir / "test_2_referral_package.pdf"
        pa_form_path = self.test_data_dir / "test_2_PA.pdf"
        
        if not referral_path.exists() or not pa_form_path.exists():
            print("❌ Test files not found")
            return False
            
        print(f"📄 Files:")
        print(f"  Referral: {referral_path.name} ({referral_path.stat().st_size:,} bytes)")
        print(f"  PA Form: {pa_form_path.name} ({pa_form_path.stat().st_size:,} bytes)")
        
        try:
            # Step 1: Upload
            print("📤 Step 1: Uploading files...")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                with open(referral_path, "rb") as ref_file, \
                     open(pa_form_path, "rb") as pa_file:
                    
                    files = [
                        ("files", ("test_2_referral_package.pdf", ref_file.read(), "application/pdf")),
                        ("files", ("test_2_PA.pdf", pa_file.read(), "application/pdf"))
                    ]
                    
                    response = await client.post(f"{self.base_url}/api/upload", files=files)
                    
            if response.status_code != 200:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                return False
                
            upload_data = response.json()
            session_id = upload_data["session_id"]
            print(f"✅ Upload successful. Session ID: {session_id}")
            
            # Step 2: Start processing
            print("⚙️  Step 2: Starting processing...")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self.base_url}/api/process/{session_id}")
                
            if response.status_code not in [200, 202]:
                print(f"❌ Processing start failed: {response.status_code} - {response.text}")
                return False
                
            print("✅ Processing started successfully")
            
            # Step 3: Monitor processing (with shorter timeout for testing)
            print("👀 Step 3: Monitoring processing...")
            
            max_wait_time = 180  # 3 minutes for this test
            processing_start = time.time()
            last_status = "unknown"
            
            while True:
                elapsed = time.time() - processing_start
                if elapsed > max_wait_time:
                    print(f"❌ Processing timeout after {max_wait_time} seconds")
                    print(f"   Last status: {last_status}")
                    return False
                    
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(f"{self.base_url}/api/process/{session_id}/status")
                    
                if response.status_code != 200:
                    print(f"❌ Status check failed: {response.status_code}")
                    return False
                    
                status_data = response.json()
                status = status_data.get("status", "unknown")
                stage = status_data.get("current_stage", "unknown")
                progress = status_data.get("progress", 0)
                
                if status != last_status:
                    print(f"  Status: {status} | Stage: {stage} | Progress: {progress}% | Elapsed: {elapsed:.1f}s")
                    last_status = status
                
                if status == "completed":
                    print("✅ Processing completed!")
                    break
                elif status == "failed":
                    error = status_data.get("error", "Unknown error")
                    print(f"❌ Processing failed: {error}")
                    return False
                    
                await asyncio.sleep(5)
                
            processing_time = time.time() - processing_start
            
            # Step 4: Download outputs
            print("📥 Step 4: Downloading outputs...")
            
            # Download filled PDF
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(f"{self.base_url}/api/download/{session_id}/filled")
                
            if response.status_code == 200:
                filled_pdf_path = self.output_dir / "test_2_filled.pdf"
                with open(filled_pdf_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ Downloaded filled PDF: {filled_pdf_path} ({len(response.content):,} bytes)")
            else:
                print(f"❌ Failed to download filled PDF: {response.status_code}")
                return False
                
            # Download report
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/api/download/{session_id}/report")
                
            if response.status_code == 200:
                report_path = self.output_dir / "test_2_report.md"
                with open(report_path, "w") as f:
                    f.write(response.content.decode("utf-8"))
                print(f"✅ Downloaded report: {report_path} ({len(response.content):,} bytes)")
                
                # Show a preview of the report
                with open(report_path, "r") as f:
                    content = f.read()
                print(f"\n📋 Report Preview (first 500 chars):")
                print(content[:500] + "..." if len(content) > 500 else content)
                
            else:
                print(f"❌ Failed to download report: {response.status_code}")
                return False
                
            print(f"\n🎉 Single case test completed successfully!")
            print(f"  Processing time: {processing_time:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"💥 Test failed with exception: {str(e)}")
            return False


async def main():
    """Main test function"""
    print("🧪 MediLink-AI Single Case E2E Test")
    print("=" * 50)
    
    test_runner = SingleCaseTest()
    
    try:
        if not test_runner.start_server():
            return False
            
        success = await test_runner.test_single_case()
        
        if success:
            print("\n✅ Single case test PASSED!")
        else:
            print("\n❌ Single case test FAILED!")
            
        return success
        
    except Exception as e:
        print(f"\n💥 Test crashed: {str(e)}")
        return False
        
    finally:
        test_runner.stop_server()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)