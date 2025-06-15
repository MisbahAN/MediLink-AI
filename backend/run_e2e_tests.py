#!/usr/bin/env python3
"""
End-to-End Test Runner for MediLink-AI

Comprehensive test runner for validating the complete document processing workflow.
Generates detailed reports and golden outputs for regression testing.
"""

import os
import sys
import time
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any
import psutil

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.core.config import settings


class E2ETestRunner:
    """Comprehensive end-to-end test runner"""
    
    def __init__(self):
        self.start_time = time.time()
        self.test_results = {}
        self.system_metrics = {}
        self.output_dir = Path(__file__).parent / "test_outputs" / "e2e_reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check if all prerequisites are met for E2E testing"""
        checks = {}
        
        # Check test data exists
        test_data_dir = Path(__file__).parent / "tests" / "test_data"
        checks["test_data_available"] = test_data_dir.exists() and list(test_data_dir.glob("*.pdf"))
        
        # Check API keys configured
        checks["mistral_key"] = bool(os.getenv("MISTRAL_API_KEY"))
        checks["gemini_key"] = bool(os.getenv("GEMINI_API_KEY"))
        checks["openai_key"] = bool(os.getenv("OPENAI_API_KEY"))
        
        # Check Redis connection
        try:
            import redis
            r = redis.from_url(settings.REDIS_URL)
            r.ping()
            checks["redis_connection"] = True
        except Exception:
            checks["redis_connection"] = False
        
        # Check upload directory
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        checks["upload_directory"] = upload_dir.exists() and os.access(upload_dir, os.W_OK)
        
        # Check dependencies
        required_packages = [
            "fastapi", "uvicorn", "pdfplumber", "mistralai", 
            "google.generativeai", "openai"
        ]
        checks["dependencies"] = True
        
        for package in required_packages:
            try:
                if package == "google.generativeai":
                    import google.generativeai
                else:
                    __import__(package.replace("-", "_"))
            except ImportError:
                checks["dependencies"] = False
                break
        
        return checks
    
    def run_test_suite(self, test_mode: str = "comprehensive") -> Dict[str, Any]:
        """Run the E2E test suite"""
        
        print(f"🚀 Starting E2E Test Suite - Mode: {test_mode}")
        print(f"📊 Output directory: {self.output_dir}")
        
        # Monitor system resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        test_commands = {
            "basic": [
                "python", "-m", "pytest", 
                "tests/test_e2e_workflow.py::TestE2EWorkflow::test_complete_pipeline_single_document",
                "-v", "-s", "--tb=short"
            ],
            "comprehensive": [
                "python", "-m", "pytest", 
                "tests/test_e2e_workflow.py::TestE2EWorkflow",
                "-v", "-s", "--tb=short"
            ],
            "performance": [
                "python", "-m", "pytest", 
                "tests/test_e2e_workflow.py::TestE2EWorkflow::test_performance_benchmarks",
                "-v", "-s", "--tb=short"
            ],
            "golden_outputs": [
                "python", "-m", "pytest", 
                "tests/test_e2e_workflow.py::test_end_to_end_golden_outputs",
                "-v", "-s", "--tb=short"
            ],
            "error_handling": [
                "python", "-m", "pytest", 
                "tests/test_e2e_workflow.py::TestE2EWorkflow::test_error_handling_scenarios",
                "-v", "-s", "--tb=short"
            ]
        }
        
        if test_mode not in test_commands:
            raise ValueError(f"Invalid test mode: {test_mode}. Available: {list(test_commands.keys())}")
        
        # Run the test
        print(f"🧪 Executing test command: {' '.join(test_commands[test_mode])}")
        
        try:
            result = subprocess.run(
                test_commands[test_mode],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes max
            )
            
            # Collect final metrics
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            execution_time = time.time() - self.start_time
            
            test_result = {
                "mode": test_mode,
                "exit_code": result.returncode,
                "success": result.returncode == 0,
                "execution_time_seconds": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "system_metrics": {
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "memory_delta_mb": final_memory - initial_memory,
                    "cpu_count": psutil.cpu_count(),
                    "disk_usage": dict(psutil.disk_usage('/')._asdict())
                }
            }
            
            return test_result
            
        except subprocess.TimeoutExpired:
            return {
                "mode": test_mode,
                "exit_code": -1,
                "success": False,
                "error": "Test execution timed out after 30 minutes",
                "execution_time_seconds": 1800
            }
        except Exception as e:
            return {
                "mode": test_mode,
                "exit_code": -1,
                "success": False,
                "error": str(e),
                "execution_time_seconds": time.time() - self.start_time
            }
    
    def analyze_test_outputs(self) -> Dict[str, Any]:
        """Analyze generated test outputs and create summary"""
        
        expected_outputs_dir = Path(__file__).parent / "tests" / "expected_outputs"
        
        if not expected_outputs_dir.exists():
            return {"error": "No test outputs directory found"}
        
        analysis = {
            "files_generated": [],
            "validation_reports": [],
            "performance_metrics": [],
            "golden_outputs": []
        }
        
        # Analyze generated files
        for file_path in expected_outputs_dir.glob("*"):
            file_info = {
                "filename": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "type": file_path.suffix
            }
            
            if "validation" in file_path.name and file_path.suffix == ".json":
                try:
                    with open(file_path, 'r') as f:
                        validation_data = json.load(f)
                        analysis["validation_reports"].append({
                            "file": file_path.name,
                            "status": validation_data.get("status"),
                            "metrics": validation_data.get("metrics", {})
                        })
                except Exception as e:
                    file_info["error"] = str(e)
            
            elif "performance" in file_path.name and file_path.suffix == ".json":
                try:
                    with open(file_path, 'r') as f:
                        perf_data = json.load(f)
                        analysis["performance_metrics"].append(perf_data)
                except Exception as e:
                    file_info["error"] = str(e)
            
            elif "golden" in file_path.name:
                analysis["golden_outputs"].append(file_info)
            
            analysis["files_generated"].append(file_info)
        
        return analysis
    
    def generate_comprehensive_report(self, test_result: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_execution": test_result,
            "output_analysis": analysis,
            "summary": {
                "overall_status": "PASS" if test_result.get("success") else "FAIL",
                "execution_time": test_result.get("execution_time_seconds", 0),
                "files_generated": len(analysis.get("files_generated", [])),
                "validation_reports": len(analysis.get("validation_reports", [])),
                "performance_tests": len(analysis.get("performance_metrics", []))
            },
            "recommendations": []
        }
        
        # Add recommendations based on results
        if test_result.get("success"):
            report["recommendations"].append("✅ All E2E tests passed successfully")
        else:
            report["recommendations"].append("❌ Some E2E tests failed - review stderr output")
        
        if test_result.get("execution_time_seconds", 0) > 600:
            report["recommendations"].append("⚠️  Tests took longer than 10 minutes - consider performance optimization")
        
        memory_delta = test_result.get("system_metrics", {}).get("memory_delta_mb", 0)
        if memory_delta > 500:
            report["recommendations"].append(f"⚠️  High memory usage detected: {memory_delta:.1f}MB - check for memory leaks")
        
        validation_failures = [v for v in analysis.get("validation_reports", []) if v.get("status") == "FAIL"]
        if validation_failures:
            report["recommendations"].append(f"❌ {len(validation_failures)} validation failures detected")
        
        return report
    
    def save_report(self, report: Dict[str, Any], test_mode: str):
        """Save the comprehensive report"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"e2e_report_{test_mode}_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save a human-readable summary
        summary_file = self.output_dir / f"e2e_summary_{test_mode}_{timestamp}.md"
        
        with open(summary_file, 'w') as f:
            f.write(f"# MediLink-AI E2E Test Report\n\n")
            f.write(f"**Date:** {report['timestamp']}\n")
            f.write(f"**Test Mode:** {test_mode}\n")
            f.write(f"**Overall Status:** {report['summary']['overall_status']}\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- **Execution Time:** {report['summary']['execution_time']:.2f} seconds\n")
            f.write(f"- **Files Generated:** {report['summary']['files_generated']}\n")
            f.write(f"- **Validation Reports:** {report['summary']['validation_reports']}\n")
            f.write(f"- **Performance Tests:** {report['summary']['performance_tests']}\n\n")
            
            if report['recommendations']:
                f.write(f"## Recommendations\n\n")
                for rec in report['recommendations']:
                    f.write(f"- {rec}\n")
                f.write("\n")
            
            if report['test_execution'].get('stderr'):
                f.write(f"## Error Output\n\n")
                f.write(f"```\n{report['test_execution']['stderr']}\n```\n\n")
        
        print(f"📋 Reports saved:")
        print(f"   - {report_file}")
        print(f"   - {summary_file}")
        
        return report_file, summary_file


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description="MediLink-AI End-to-End Test Runner")
    parser.add_argument(
        "--mode", 
        choices=["basic", "comprehensive", "performance", "golden_outputs", "error_handling"],
        default="comprehensive",
        help="Test mode to run"
    )
    parser.add_argument(
        "--check-prereqs", 
        action="store_true",
        help="Only check prerequisites and exit"
    )
    
    args = parser.parse_args()
    
    runner = E2ETestRunner()
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    prereqs = runner.check_prerequisites()
    
    for check, status in prereqs.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {check}: {status}")
    
    failed_prereqs = [k for k, v in prereqs.items() if not v]
    
    if failed_prereqs:
        print(f"\n❌ Prerequisites failed: {failed_prereqs}")
        if not args.check_prereqs:
            print("   Fix these issues before running tests.")
            return 1
    else:
        print("\n✅ All prerequisites passed!")
    
    if args.check_prereqs:
        return 0 if not failed_prereqs else 1
    
    # Run tests
    print(f"\n🚀 Running E2E tests in {args.mode} mode...")
    test_result = runner.run_test_suite(args.mode)
    
    # Analyze outputs
    print("\n📊 Analyzing test outputs...")
    analysis = runner.analyze_test_outputs()
    
    # Generate report
    print("\n📋 Generating comprehensive report...")
    report = runner.generate_comprehensive_report(test_result, analysis)
    
    # Save reports
    report_file, summary_file = runner.save_report(report, args.mode)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"🎯 E2E TEST RESULTS - {args.mode.upper()} MODE")
    print(f"{'='*60}")
    print(f"Status: {report['summary']['overall_status']}")
    print(f"Execution Time: {report['summary']['execution_time']:.2f} seconds")
    print(f"Files Generated: {report['summary']['files_generated']}")
    
    if report['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations'][:3]:  # Show first 3
            print(f"   {rec}")
    
    print(f"\n📁 Full reports available at: {runner.output_dir}")
    
    return 0 if test_result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())