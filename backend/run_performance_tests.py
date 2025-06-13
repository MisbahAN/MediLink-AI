#!/usr/bin/env python3
"""
Performance Test Runner for MediLink-AI
=======================================

This script provides a convenient way to run performance tests with different configurations.

Usage:
    # Run basic performance tests (with mocked APIs)
    python run_performance_tests.py --basic

    # Run comprehensive performance suite
    python run_performance_tests.py --comprehensive

    # Run specific test categories
    python run_performance_tests.py --large-doc
    python run_performance_tests.py --concurrency
    python run_performance_tests.py --memory-leak

    # Run with real APIs (requires API keys)
    python run_performance_tests.py --real-apis

    # Generate performance report only
    python run_performance_tests.py --report-only
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Import our performance testing modules
from tests.test_performance import PerformanceTester, PerformanceMetrics, ConcurrencyResults


def setup_logging(verbose: bool = False):
    """Configure logging for the test runner."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


async def run_basic_tests(tester: PerformanceTester) -> list:
    """Run basic performance tests with mocked APIs."""
    print("🚀 Running basic performance tests...")
    
    metrics_list = []
    
    # Test each document size
    for size in ["small", "medium", "large"]:
        print(f"  Testing {size} document processing...")
        try:
            metrics = await tester.run_single_document_test(size, mock_apis=True)
            metrics_list.append(metrics)
            print(f"    ✅ Completed in {metrics.total_duration:.2f}s")
            print(f"    📊 {metrics.processing_time_per_page:.2f}s per page")
            print(f"    💾 {metrics.peak_memory_mb:.2f} MB peak memory")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
    
    return metrics_list


async def run_comprehensive_tests(tester: PerformanceTester) -> dict:
    """Run comprehensive performance test suite."""
    print("🔬 Running comprehensive performance test suite...")
    
    results = {}
    
    # Single document tests
    print("\n📄 Single Document Tests:")
    results["single_doc"] = await run_basic_tests(tester)
    
    # Concurrency test
    print("\n🔄 Concurrency Test:")
    try:
        concurrency_results = await tester.run_concurrent_processing_test(
            concurrent_requests=3, mock_apis=True
        )
        results["concurrency"] = concurrency_results
        print(f"  ✅ {concurrency_results.successful_completions}/{concurrency_results.concurrent_requests} requests completed")
        print(f"  📊 {concurrency_results.avg_response_time:.2f}s average response time")
        print(f"  🚀 {concurrency_results.throughput_per_second:.2f} requests/second")
    except Exception as e:
        print(f"  ❌ Concurrency test failed: {e}")
        results["concurrency"] = None
    
    # Memory leak test
    print("\n🔍 Memory Leak Test:")
    try:
        memory_results = await tester.run_memory_leak_test(iterations=5)
        results["memory_leak"] = memory_results
        if memory_results["potential_memory_leak"]:
            print(f"  ⚠️  Potential memory leak detected!")
            print(f"  📈 {memory_results['avg_memory_growth_per_iteration_mb']:.2f} MB average growth per iteration")
        else:
            print(f"  ✅ No memory leak detected")
            print(f"  📊 {memory_results['avg_memory_growth_per_iteration_mb']:.2f} MB average growth per iteration")
    except Exception as e:
        print(f"  ❌ Memory leak test failed: {e}")
        results["memory_leak"] = None
    
    return results


async def run_real_api_tests(tester: PerformanceTester) -> list:
    """Run tests with real API calls."""
    print("🌐 Running tests with real API calls...")
    print("⚠️  This requires valid API keys and will make actual API calls!")
    
    # Check for API keys
    required_keys = ["MISTRAL_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"❌ Missing required API keys: {', '.join(missing_keys)}")
        print("Set these environment variables before running real API tests.")
        return []
    
    metrics_list = []
    
    # Test with large document only (to avoid excessive API costs)
    print("  Testing large document with real APIs...")
    try:
        metrics = await tester.run_single_document_test("large", mock_apis=False)
        metrics_list.append(metrics)
        print(f"    ✅ Completed in {metrics.total_duration:.2f}s")
        print(f"    🎯 Extraction confidence: {metrics.extraction_confidence:.2f}")
        print(f"    📊 {metrics.processing_time_per_page:.2f}s per page")
    except Exception as e:
        print(f"    ❌ Failed: {e}")
    
    return metrics_list


def generate_and_save_report(tester: PerformanceTester, results: dict) -> Path:
    """Generate and save performance report."""
    print("\n📊 Generating performance report...")
    
    # Extract metrics from results
    metrics_list = results.get("single_doc", [])
    concurrency_results = results.get("concurrency")
    memory_leak_results = results.get("memory_leak")
    
    # Generate report
    report = tester.generate_performance_report(
        metrics_list, concurrency_results, memory_leak_results
    )
    
    # Save to file in organized test outputs directory
    timestamp = asyncio.get_event_loop().time()
    test_outputs_dir = Path("test_outputs/performance_reports")
    test_outputs_dir.mkdir(parents=True, exist_ok=True)
    report_file = test_outputs_dir / f"performance_report_{int(timestamp)}.json"
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Report saved to: {report_file}")
    
    # Print summary
    print("\n📋 Performance Summary:")
    print(f"  Tests run: {report['test_summary']['total_tests']}")
    print(f"  Success rate: {report['test_summary']['success_rate']:.2%}")
    
    if report['test_summary']['successful_tests'] > 0:
        print(f"  Avg processing time: {report['performance_metrics']['avg_processing_time_seconds']:.2f}s")
        print(f"  Avg memory usage: {report['performance_metrics']['avg_memory_usage_mb']:.2f} MB")
        print(f"  Avg extraction confidence: {report['performance_metrics']['avg_extraction_confidence']:.2f}")
    
    if report['performance_issues']:
        print("\n⚠️  Performance Issues Detected:")
        for issue in report['performance_issues']:
            print(f"    • {issue}")
    else:
        print("\n✅ No performance issues detected!")
    
    return report_file


async def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="MediLink-AI Performance Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--basic", action="store_true", 
                       help="Run basic performance tests (default)")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run comprehensive performance test suite")
    parser.add_argument("--large-doc", action="store_true", 
                       help="Run large document test only")
    parser.add_argument("--concurrency", action="store_true", 
                       help="Run concurrency test only")
    parser.add_argument("--memory-leak", action="store_true", 
                       help="Run memory leak test only")
    parser.add_argument("--real-apis", action="store_true", 
                       help="Run tests with real API calls (requires API keys)")
    parser.add_argument("--report-only", action="store_true", 
                       help="Generate performance report from previous results")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    parser.add_argument("--concurrent-requests", type=int, default=3, 
                       help="Number of concurrent requests for concurrency test")
    parser.add_argument("--memory-iterations", type=int, default=5, 
                       help="Number of iterations for memory leak test")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # If no specific test selected, run basic tests
    if not any([args.basic, args.comprehensive, args.large_doc, args.concurrency, 
               args.memory_leak, args.real_apis, args.report_only]):
        args.basic = True
    
    # Initialize tester
    tester = PerformanceTester()
    results = {}
    
    try:
        if args.report_only:
            print("📊 Report generation only mode - using empty results")
            results = {"single_doc": [], "concurrency": None, "memory_leak": None}
        
        elif args.basic:
            results["single_doc"] = await run_basic_tests(tester)
        
        elif args.comprehensive:
            results = await run_comprehensive_tests(tester)
        
        elif args.large_doc:
            print("📄 Running large document test...")
            metrics = await tester.run_single_document_test("large", mock_apis=True)
            results["single_doc"] = [metrics]
            print(f"✅ Large document test completed in {metrics.total_duration:.2f}s")
        
        elif args.concurrency:
            print("🔄 Running concurrency test...")
            concurrency_results = await tester.run_concurrent_processing_test(
                concurrent_requests=args.concurrent_requests, mock_apis=True
            )
            results["concurrency"] = concurrency_results
            print(f"✅ Concurrency test completed: {concurrency_results.successful_completions}/{concurrency_results.concurrent_requests}")
        
        elif args.memory_leak:
            print("🔍 Running memory leak test...")
            memory_results = await tester.run_memory_leak_test(iterations=args.memory_iterations)
            results["memory_leak"] = memory_results
            if memory_results["potential_memory_leak"]:
                print("⚠️  Potential memory leak detected!")
            else:
                print("✅ No memory leak detected")
        
        elif args.real_apis:
            results["single_doc"] = await run_real_api_tests(tester)
        
        # Generate report
        if results:
            report_file = generate_and_save_report(tester, results)
            print(f"\n🎉 Performance testing completed! Report: {report_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Performance testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Performance testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())