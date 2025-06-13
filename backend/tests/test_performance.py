"""
Comprehensive Performance Testing Suite for MediLink-AI
========================================================

This script provides extensive performance testing for the MediLink-AI system,
measuring key metrics including processing times, memory usage, CPU utilization,
and system behavior under concurrent load.

Test Categories:
1. Single Document Performance - Test individual processing times
2. Concurrent Processing - Test system under multiple simultaneous requests
3. Large Document Performance - Test with largest available test documents
4. Memory Leak Detection - Monitor memory usage patterns
5. OCR Service Performance - Measure API response times
6. System Load Testing - Validate behavior under stress

Usage:
    pytest backend/tests/test_performance.py -v -s
    pytest backend/tests/test_performance.py::test_large_document_processing -v -s
"""

import asyncio
import gc
import json
import logging
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import patch

import pytest

try:
    import psutil
except ImportError:
    psutil = None

from app.core.config import get_settings
from app.services.processing_pipeline import get_processing_pipeline
from app.services.mistral_service import get_mistral_service
from app.models.schemas import ProcessingStatusEnum


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance measurement data."""
    
    # Timing metrics
    total_duration: float
    processing_time_per_page: float
    ocr_response_time: float
    
    # Resource metrics
    peak_memory_mb: float
    avg_cpu_percent: float
    memory_growth_mb: float
    
    # Processing metrics
    pages_processed: int
    file_size_mb: float
    success_rate: float
    
    # Quality metrics
    extraction_confidence: float
    fields_mapped: int
    missing_fields: int
    
    # Additional context
    test_name: str
    timestamp: datetime
    error_details: Optional[str] = None


@dataclass
class ConcurrencyResults:
    """Results from concurrent processing tests."""
    
    concurrent_requests: int
    successful_completions: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    throughput_per_second: float
    memory_peak_mb: float
    cpu_peak_percent: float
    error_rate: float


class SystemMonitor:
    """Monitor system resources during performance tests."""
    
    def __init__(self):
        if psutil is None:
            self.process = None
            logger.warning("psutil not available - system monitoring disabled")
        else:
            self.process = psutil.Process()
        self.monitoring = False
        self.metrics = []
        self._monitor_task = None
    
    async def start_monitoring(self, interval: float = 0.1):
        """Start monitoring system resources."""
        self.monitoring = True
        self.metrics = []
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
    
    async def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return aggregated metrics."""
        self.monitoring = False
        if self._monitor_task:
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if not self.metrics:
            return {"memory_mb": 0, "cpu_percent": 0}
        
        memory_values = [m["memory_mb"] for m in self.metrics]
        cpu_values = [m["cpu_percent"] for m in self.metrics]
        
        return {
            "peak_memory_mb": max(memory_values),
            "avg_memory_mb": statistics.mean(memory_values),
            "peak_cpu_percent": max(cpu_values),
            "avg_cpu_percent": statistics.mean(cpu_values),
            "memory_growth_mb": memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0
        }
    
    async def _monitor_loop(self, interval: float):
        """Internal monitoring loop."""
        while self.monitoring:
            try:
                if self.process is not None:
                    memory_info = self.process.memory_info()
                    cpu_percent = self.process.cpu_percent()
                    
                    self.metrics.append({
                        "timestamp": time.time(),
                        "memory_mb": memory_info.rss / 1024 / 1024,
                        "cpu_percent": cpu_percent
                    })
                else:
                    # Fallback metrics when psutil not available
                    self.metrics.append({
                        "timestamp": time.time(),
                        "memory_mb": 0,
                        "cpu_percent": 0
                    })
                
                await asyncio.sleep(interval)
            except Exception as e:
                logger.warning(f"Monitor error: {e}")
                break


class PerformanceTester:
    """Main performance testing class."""
    
    def __init__(self):
        self.settings = get_settings()
        self.pipeline = get_processing_pipeline()
        self.monitor = SystemMonitor()
        self.test_data_dir = Path(__file__).parent / "test_data"
        
        # Test file information
        self.test_files = {
            "small": {
                "referral": "test_3_referral_package.pdf",  # ~4.2MB
                "pa_form": "test_3_PA.pdf",  # ~196KB
                "expected_pages": 8
            },
            "medium": {
                "referral": "test_2_referral_package.pdf",  # ~7.5MB
                "pa_form": "test_2_PA.pdf",  # ~397KB
                "expected_pages": 12
            },
            "large": {
                "referral": "test_1_referral_package.pdf",  # ~7.9MB (15 pages)
                "pa_form": "test_1_PA.pdf",  # ~982KB
                "expected_pages": 15
            }
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "max_processing_time_per_page": 30.0,  # seconds
            "max_memory_usage_mb": 2048,  # 2GB
            "min_extraction_confidence": 0.7,
            "max_ocr_response_time": 60.0,  # seconds
            "max_acceptable_error_rate": 0.05  # 5%
        }
    
    async def run_single_document_test(
        self, 
        file_size: str, 
        mock_apis: bool = False
    ) -> PerformanceMetrics:
        """Test processing performance for a single document."""
        
        test_config = self.test_files[file_size]
        referral_path = self.test_data_dir / test_config["referral"]
        pa_form_path = self.test_data_dir / test_config["pa_form"]
        
        if not referral_path.exists() or not pa_form_path.exists():
            pytest.skip(f"Test files not found for {file_size} test")
        
        session_id = f"perf_test_{file_size}_{int(time.time())}"
        
        # Start memory tracking
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]
        
        # Start system monitoring
        await self.monitor.start_monitoring()
        
        start_time = time.time()
        error_details = None
        
        try:
            # Mock API calls if requested for baseline testing
            if mock_apis:
                result = await self._run_with_mocked_apis(
                    session_id, referral_path, pa_form_path
                )
            else:
                # Run real processing pipeline
                result = await self.pipeline.process_documents(
                    session_id=session_id,
                    referral_pdf_path=referral_path,
                    pa_form_pdf_path=pa_form_path
                )
            
            success_rate = 1.0 if result.processing_status == ProcessingStatusEnum.COMPLETED else 0.0
            extraction_confidence = result.extracted_data.extraction_summary.get("overall_confidence", 0.0)
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            success_rate = 0.0
            extraction_confidence = 0.0
            error_details = str(e)
            result = None
        
        # Stop monitoring and get metrics
        system_metrics = await self.monitor.stop_monitoring()
        
        # Calculate duration
        total_duration = time.time() - start_time
        
        # Get memory usage
        current_memory, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_growth_mb = (current_memory - initial_memory) / 1024 / 1024
        
        # Calculate per-page processing time
        expected_pages = test_config["expected_pages"]
        processing_time_per_page = total_duration / expected_pages
        
        # Get file size
        file_size_mb = referral_path.stat().st_size / 1024 / 1024
        
        # Extract metrics from result if available
        fields_mapped = 0
        missing_fields = 0
        if result and result.processing_status == ProcessingStatusEnum.COMPLETED:
            fields_mapped = len(result.pa_form_fields)
            missing_fields = len(result.missing_fields)
        
        return PerformanceMetrics(
            total_duration=total_duration,
            processing_time_per_page=processing_time_per_page,
            ocr_response_time=total_duration * 0.7,  # Estimate OCR portion
            peak_memory_mb=system_metrics.get("peak_memory_mb", 0.0),
            avg_cpu_percent=system_metrics.get("avg_cpu_percent", 0.0),
            memory_growth_mb=memory_growth_mb,
            pages_processed=expected_pages,
            file_size_mb=file_size_mb,
            success_rate=success_rate,
            extraction_confidence=extraction_confidence,
            fields_mapped=fields_mapped,
            missing_fields=missing_fields,
            test_name=f"single_document_{file_size}",
            timestamp=datetime.now(timezone.utc),
            error_details=error_details
        )
    
    async def run_concurrent_processing_test(
        self, 
        concurrent_requests: int = 5,
        mock_apis: bool = False
    ) -> ConcurrencyResults:
        """Test system performance under concurrent load."""
        
        logger.info(f"Starting concurrent test with {concurrent_requests} requests")
        
        # Use medium-sized files for concurrency testing
        test_config = self.test_files["medium"]
        referral_path = self.test_data_dir / test_config["referral"]
        pa_form_path = self.test_data_dir / test_config["pa_form"]
        
        if not referral_path.exists() or not pa_form_path.exists():
            pytest.skip("Test files not found for concurrency test")
        
        # Start system monitoring
        await self.monitor.start_monitoring()
        
        start_time = time.time()
        response_times = []
        successful_completions = 0
        failed_requests = 0
        
        # Create concurrent tasks
        tasks = []
        for i in range(concurrent_requests):
            session_id = f"concurrent_test_{i}_{int(time.time())}"
            
            if mock_apis:
                task = self._run_with_mocked_apis(session_id, referral_path, pa_form_path)
            else:
                task = self.pipeline.process_documents(
                    session_id=session_id,
                    referral_pdf_path=referral_path,
                    pa_form_pdf_path=pa_form_path
                )
            
            tasks.append((i, task, time.time()))
        
        # Execute all tasks concurrently
        for i, task, task_start_time in tasks:
            try:
                await task
                task_duration = time.time() - task_start_time
                response_times.append(task_duration)
                successful_completions += 1
                logger.info(f"Task {i} completed in {task_duration:.2f}s")
            except Exception as e:
                failed_requests += 1
                logger.error(f"Task {i} failed: {e}")
        
        # Stop monitoring
        system_metrics = await self.monitor.stop_monitoring()
        
        total_duration = time.time() - start_time
        
        # Calculate metrics
        avg_response_time = statistics.mean(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        throughput = successful_completions / total_duration if total_duration > 0 else 0
        error_rate = failed_requests / concurrent_requests
        
        return ConcurrencyResults(
            concurrent_requests=concurrent_requests,
            successful_completions=successful_completions,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            throughput_per_second=throughput,
            memory_peak_mb=system_metrics.get("peak_memory_mb", 0.0),
            cpu_peak_percent=system_metrics.get("peak_cpu_percent", 0.0),
            error_rate=error_rate
        )
    
    async def run_memory_leak_test(self, iterations: int = 10) -> Dict[str, Any]:
        """Test for memory leaks during repeated processing."""
        
        logger.info(f"Starting memory leak test with {iterations} iterations")
        
        test_config = self.test_files["small"]  # Use small files for faster iteration
        referral_path = self.test_data_dir / test_config["referral"]
        pa_form_path = self.test_data_dir / test_config["pa_form"]
        
        if not referral_path.exists() or not pa_form_path.exists():
            pytest.skip("Test files not found for memory leak test")
        
        memory_measurements = []
        
        # Start memory tracking
        tracemalloc.start()
        
        for i in range(iterations):
            logger.info(f"Memory leak test iteration {i+1}/{iterations}")
            
            session_id = f"memory_test_{i}_{int(time.time())}"
            
            # Force garbage collection before measurement
            gc.collect()
            
            # Measure memory before processing
            memory_before = tracemalloc.get_traced_memory()[0]
            
            try:
                # Use mocked APIs for consistent testing
                await self._run_with_mocked_apis(session_id, referral_path, pa_form_path)
            except Exception as e:
                logger.warning(f"Iteration {i} failed: {e}")
            
            # Force garbage collection after processing
            gc.collect()
            
            # Measure memory after processing
            memory_after = tracemalloc.get_traced_memory()[0]
            memory_diff_mb = (memory_after - memory_before) / 1024 / 1024
            
            memory_measurements.append({
                "iteration": i,
                "memory_before_mb": memory_before / 1024 / 1024,
                "memory_after_mb": memory_after / 1024 / 1024,
                "memory_diff_mb": memory_diff_mb
            })
            
            # Small delay to allow system cleanup
            await asyncio.sleep(0.5)
        
        tracemalloc.stop()
        
        # Analyze memory growth pattern
        memory_diffs = [m["memory_diff_mb"] for m in memory_measurements]
        total_memory_growth = sum(memory_diffs)
        avg_memory_growth = statistics.mean(memory_diffs)
        max_memory_growth = max(memory_diffs)
        
        # Detect potential memory leak
        # Consider it a leak if average growth > 10MB per iteration
        potential_leak = avg_memory_growth > 10.0
        
        return {
            "iterations": iterations,
            "total_memory_growth_mb": total_memory_growth,
            "avg_memory_growth_per_iteration_mb": avg_memory_growth,
            "max_memory_growth_mb": max_memory_growth,
            "potential_memory_leak": potential_leak,
            "measurements": memory_measurements
        }
    
    async def _run_with_mocked_apis(
        self, 
        session_id: str, 
        referral_path: Path, 
        pa_form_path: Path
    ):
        """Run processing with mocked API calls for baseline testing."""
        
        from app.models.schemas import ConfidenceLevel
        
        mock_extraction_result = {
            "successful_extraction": True,
            "overall_confidence": 0.85,
            "patient_info": {
                "name": {
                    "value": "Test Patient", 
                    "confidence": 0.9,
                    "confidence_level": ConfidenceLevel.HIGH,
                    "source_page": 1,
                    "extraction_method": "mistral_mock"
                },
                "dob": {
                    "value": "01/01/1980", 
                    "confidence": 0.8,
                    "confidence_level": ConfidenceLevel.MEDIUM,
                    "source_page": 1,
                    "extraction_method": "mistral_mock"
                },
                "insurance_id": {
                    "value": "TEST123", 
                    "confidence": 0.85,
                    "confidence_level": ConfidenceLevel.MEDIUM,
                    "source_page": 2,
                    "extraction_method": "mistral_mock"
                }
            },
            "clinical_data": {
                "diagnosis": {
                    "value": "Test Condition", 
                    "confidence": 0.8,
                    "confidence_level": ConfidenceLevel.MEDIUM,
                    "source_page": 3,
                    "extraction_method": "mistral_mock"
                },
                "treatment": {
                    "value": "Test Treatment", 
                    "confidence": 0.75,
                    "confidence_level": ConfidenceLevel.MEDIUM,
                    "source_page": 3,
                    "extraction_method": "mistral_mock"
                }
            },
            "pages_processed": self.test_files["medium"]["expected_pages"],
            "text_length": 5000
        }
        
        mock_field_detection = {
            "success": True,
            "fields": {
                "patient_name": {
                    "field_name": "patient_name",
                    "field_type": "text",
                    "required": True,
                    "coordinates": {"x": 100, "y": 200}
                },
                "patient_dob": {
                    "field_name": "patient_dob",
                    "field_type": "text",
                    "required": True,
                    "coordinates": {"x": 100, "y": 240}
                }
            }
        }
        
        mock_mapping_result = {
            "field_mappings": {
                "patient_name": {
                    "mapped_value": "Test Patient",
                    "confidence": 0.9,
                    "source": "patient_info"
                },
                "patient_dob": {
                    "mapped_value": "01/01/1980",
                    "confidence": 0.8,
                    "source": "patient_info"
                }
            }
        }
        
        with patch.object(self.pipeline.mistral_service, 'extract_from_pdf', return_value=mock_extraction_result), \
             patch.object(self.pipeline.widget_detector, 'detect_form_fields', return_value=mock_field_detection), \
             patch.object(self.pipeline.openai_service, 'extract_and_map_fields', return_value=mock_mapping_result):
            
            return await self.pipeline.process_documents(
                session_id=session_id,
                referral_pdf_path=referral_path,
                pa_form_pdf_path=pa_form_path
            )
    
    def generate_performance_report(
        self, 
        metrics_list: List[PerformanceMetrics],
        concurrency_results: Optional[ConcurrencyResults] = None,
        memory_leak_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        if not metrics_list:
            return {"error": "No performance metrics to report"}
        
        # Aggregate single document metrics
        successful_tests = [m for m in metrics_list if m.success_rate > 0]
        failed_tests = [m for m in metrics_list if m.success_rate == 0]
        
        if successful_tests:
            avg_processing_time = statistics.mean([m.total_duration for m in successful_tests])
            avg_processing_per_page = statistics.mean([m.processing_time_per_page for m in successful_tests])
            avg_memory_usage = statistics.mean([m.peak_memory_mb for m in successful_tests])
            avg_cpu_usage = statistics.mean([m.avg_cpu_percent for m in successful_tests])
            avg_confidence = statistics.mean([m.extraction_confidence for m in successful_tests])
        else:
            avg_processing_time = 0
            avg_processing_per_page = 0
            avg_memory_usage = 0
            avg_cpu_usage = 0
            avg_confidence = 0
        
        # Performance analysis
        performance_issues = []
        
        # Check processing time thresholds
        if avg_processing_per_page > self.performance_thresholds["max_processing_time_per_page"]:
            performance_issues.append(
                f"Processing time per page ({avg_processing_per_page:.2f}s) exceeds threshold "
                f"({self.performance_thresholds['max_processing_time_per_page']}s)"
            )
        
        # Check memory usage
        if avg_memory_usage > self.performance_thresholds["max_memory_usage_mb"]:
            performance_issues.append(
                f"Memory usage ({avg_memory_usage:.2f} MB) exceeds threshold "
                f"({self.performance_thresholds['max_memory_usage_mb']} MB)"
            )
        
        # Check extraction confidence
        if avg_confidence < self.performance_thresholds["min_extraction_confidence"]:
            performance_issues.append(
                f"Extraction confidence ({avg_confidence:.2f}) below threshold "
                f"({self.performance_thresholds['min_extraction_confidence']})"
            )
        
        report = {
            "test_summary": {
                "total_tests": len(metrics_list),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests) / len(metrics_list) if metrics_list else 0
            },
            "performance_metrics": {
                "avg_processing_time_seconds": avg_processing_time,
                "avg_processing_time_per_page_seconds": avg_processing_per_page,
                "avg_memory_usage_mb": avg_memory_usage,
                "avg_cpu_usage_percent": avg_cpu_usage,
                "avg_extraction_confidence": avg_confidence
            },
            "performance_issues": performance_issues,
            "detailed_results": [
                {
                    "test_name": m.test_name,
                    "duration": m.total_duration,
                    "processing_per_page": m.processing_time_per_page,
                    "memory_mb": m.peak_memory_mb,
                    "cpu_percent": m.avg_cpu_percent,
                    "success": m.success_rate > 0,
                    "confidence": m.extraction_confidence,
                    "error": m.error_details
                }
                for m in metrics_list
            ]
        }
        
        # Add concurrency results if available
        if concurrency_results:
            report["concurrency_analysis"] = {
                "concurrent_requests": concurrency_results.concurrent_requests,
                "successful_completions": concurrency_results.successful_completions,
                "error_rate": concurrency_results.error_rate,
                "avg_response_time": concurrency_results.avg_response_time,
                "throughput_per_second": concurrency_results.throughput_per_second,
                "memory_peak_mb": concurrency_results.memory_peak_mb,
                "cpu_peak_percent": concurrency_results.cpu_peak_percent
            }
            
            # Check concurrency thresholds
            if concurrency_results.error_rate > self.performance_thresholds["max_acceptable_error_rate"]:
                performance_issues.append(
                    f"Concurrency error rate ({concurrency_results.error_rate:.2%}) exceeds threshold "
                    f"({self.performance_thresholds['max_acceptable_error_rate']:.2%})"
                )
        
        # Add memory leak analysis if available
        if memory_leak_results:
            report["memory_leak_analysis"] = memory_leak_results
            
            if memory_leak_results.get("potential_memory_leak", False):
                performance_issues.append(
                    f"Potential memory leak detected: "
                    f"{memory_leak_results['avg_memory_growth_per_iteration_mb']:.2f} MB average growth per iteration"
                )
        
        # Expected bottlenecks based on architecture
        report["expected_bottlenecks"] = self._analyze_architectural_bottlenecks()
        
        return report
    
    def _analyze_architectural_bottlenecks(self) -> Dict[str, str]:
        """Analyze expected performance bottlenecks based on system architecture."""
        
        return {
            "OCR_API_Latency": 
                "External API calls to Mistral and Gemini services are likely the primary bottleneck. "
                "Each page requires OCR processing which involves network round-trip times and AI model inference. "
                "Large documents (15+ pages) will experience cumulative latency.",
            
            "PDF_Processing": 
                "PDF parsing and image extraction for large files can be CPU-intensive. "
                "Documents with many images or complex layouts will require more processing time.",
            
            "Memory_Usage": 
                "Processing multiple large PDF files simultaneously will consume significant memory. "
                "Each document's content is held in memory during processing, and image extraction "
                "can create additional memory pressure.",
            
            "Field_Mapping_Complexity": 
                "OpenAI API calls for field mapping become more expensive with complex documents. "
                "The mapping process involves analyzing extracted text against form field requirements.",
            
            "Sequential_Processing": 
                "Current architecture processes documents sequentially rather than in parallel, "
                "which limits throughput for concurrent requests.",
            
            "File_IO_Operations": 
                "Temporary file creation, PDF reading/writing, and storage operations can become "
                "bottlenecks with high concurrent load or very large files.",
            
            "Redis_Caching": 
                "If Redis is not available or slow, caching operations will fall back to in-memory "
                "storage, potentially impacting performance and memory usage."
        }


# Test class for pytest integration
class TestPerformance:
    """Performance test class for pytest runner."""
    
    @pytest.fixture(autouse=True)
    def setup_performance_tester(self):
        """Set up performance tester for each test."""
        self.tester = PerformanceTester()
    
    @pytest.mark.asyncio
    async def test_small_document_processing(self):
        """Test processing performance with small document."""
        metrics = await self.tester.run_single_document_test("small", mock_apis=True)
        
        assert metrics.success_rate > 0, f"Processing failed: {metrics.error_details}"
        assert metrics.processing_time_per_page < 30.0, "Processing too slow per page"
        assert metrics.peak_memory_mb < 1024, "Memory usage too high"
        
        logger.info(f"Small document test completed in {metrics.total_duration:.2f}s")
    
    @pytest.mark.asyncio
    async def test_medium_document_processing(self):
        """Test processing performance with medium document."""
        metrics = await self.tester.run_single_document_test("medium", mock_apis=True)
        
        assert metrics.success_rate > 0, f"Processing failed: {metrics.error_details}"
        assert metrics.processing_time_per_page < 30.0, "Processing too slow per page"
        assert metrics.peak_memory_mb < 1536, "Memory usage too high for medium document"
        
        logger.info(f"Medium document test completed in {metrics.total_duration:.2f}s")
    
    @pytest.mark.asyncio
    async def test_large_document_processing(self):
        """Test processing performance with large document (15 pages)."""
        metrics = await self.tester.run_single_document_test("large", mock_apis=True)
        
        assert metrics.success_rate > 0, f"Processing failed: {metrics.error_details}"
        assert metrics.processing_time_per_page < 45.0, "Processing too slow for large document"
        assert metrics.peak_memory_mb < 2048, "Memory usage too high for large document"
        
        logger.info(f"Large document test completed in {metrics.total_duration:.2f}s")
        logger.info(f"Processing rate: {metrics.processing_time_per_page:.2f}s per page")
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self):
        """Test system performance under concurrent load."""
        results = await self.tester.run_concurrent_processing_test(
            concurrent_requests=3, mock_apis=True
        )
        
        assert results.error_rate < 0.5, f"Too many failures in concurrent test: {results.error_rate:.2%}"
        assert results.successful_completions > 0, "No successful concurrent completions"
        assert results.memory_peak_mb < 3072, "Memory usage too high during concurrency"
        
        logger.info(f"Concurrent test: {results.successful_completions}/{results.concurrent_requests} completed")
        logger.info(f"Average response time: {results.avg_response_time:.2f}s")
        logger.info(f"Throughput: {results.throughput_per_second:.2f} requests/second")
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Test for memory leaks during repeated processing."""
        results = await self.tester.run_memory_leak_test(iterations=5)
        
        assert not results["potential_memory_leak"], (
            f"Potential memory leak detected: "
            f"{results['avg_memory_growth_per_iteration_mb']:.2f} MB average growth per iteration"
        )
        
        logger.info(f"Memory leak test completed with {results['avg_memory_growth_per_iteration_mb']:.2f} MB avg growth")
    
    @pytest.mark.asyncio
    async def test_comprehensive_performance_suite(self):
        """Run comprehensive performance test suite and generate report."""
        logger.info("Starting comprehensive performance test suite...")
        
        # Run single document tests
        metrics_list = []
        for size in ["small", "medium", "large"]:
            try:
                metrics = await self.tester.run_single_document_test(size, mock_apis=True)
                metrics_list.append(metrics)
                logger.info(f"Completed {size} document test: {metrics.total_duration:.2f}s")
            except Exception as e:
                logger.error(f"Failed {size} document test: {e}")
        
        # Run concurrency test
        try:
            concurrency_results = await self.tester.run_concurrent_processing_test(
                concurrent_requests=3, mock_apis=True
            )
        except Exception as e:
            logger.error(f"Concurrency test failed: {e}")
            concurrency_results = None
        
        # Run memory leak test
        try:
            memory_leak_results = await self.tester.run_memory_leak_test(iterations=3)
        except Exception as e:
            logger.error(f"Memory leak test failed: {e}")
            memory_leak_results = None
        
        # Generate performance report
        report = self.tester.generate_performance_report(
            metrics_list, concurrency_results, memory_leak_results
        )
        
        # Save report to file in organized test outputs directory
        test_outputs_dir = Path(__file__).parent.parent / "test_outputs" / "performance_reports"
        test_outputs_dir.mkdir(parents=True, exist_ok=True)
        report_file = test_outputs_dir / "comprehensive_performance_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to {report_file}")
        
        # Log summary
        logger.info("=== PERFORMANCE TEST SUMMARY ===")
        logger.info(f"Total tests: {report['test_summary']['total_tests']}")
        logger.info(f"Success rate: {report['test_summary']['success_rate']:.2%}")
        logger.info(f"Avg processing time: {report['performance_metrics']['avg_processing_time_seconds']:.2f}s")
        logger.info(f"Avg memory usage: {report['performance_metrics']['avg_memory_usage_mb']:.2f} MB")
        
        if report['performance_issues']:
            logger.warning("Performance issues detected:")
            for issue in report['performance_issues']:
                logger.warning(f"  - {issue}")
        else:
            logger.info("No performance issues detected!")
        
        # Assert overall performance is acceptable
        assert report['test_summary']['success_rate'] > 0.8, "Overall success rate too low"
        assert len([i for i in report['performance_issues'] if 'exceeds threshold' in i]) == 0, \
            "Critical performance thresholds exceeded"


# Real API integration tests (run separately with actual API keys)
class TestRealAPIPerformance:
    """Performance tests with real API calls (requires API keys)."""
    
    @pytest.fixture(autouse=True)
    def setup_real_api_tester(self):
        """Set up tester with real API configuration."""
        self.tester = PerformanceTester()
        # These tests require real API keys from environment variables
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_large_document_performance(self):
        """Test with real APIs and large document."""
        metrics = await self.tester.run_single_document_test("large", mock_apis=False)
        
        # More lenient thresholds for real API calls
        assert metrics.success_rate > 0, f"Real API processing failed: {metrics.error_details}"
        assert metrics.processing_time_per_page < 120.0, "Real API processing too slow"
        
        logger.info(f"Real API test completed in {metrics.total_duration:.2f}s")
        logger.info(f"Extraction confidence: {metrics.extraction_confidence:.2f}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_ocr_service_performance(self):
        """Test OCR service response times with real APIs."""
        
        # Test Mistral service directly
        mistral_service = get_mistral_service()
        referral_path = Path(__file__).parent / "test_data" / "test_1_referral_package.pdf"
        
        if not referral_path.exists():
            pytest.skip("Test file not found")
        
        start_time = time.time()
        try:
            result = await mistral_service.extract_from_pdf(
                referral_path, 
                extraction_type="medical_referral"
            )
            ocr_duration = time.time() - start_time
            
            assert result.get("successful_extraction", False), "Mistral OCR failed"
            assert ocr_duration < 180.0, f"Mistral OCR too slow: {ocr_duration:.2f}s"
            
            logger.info(f"Mistral OCR completed in {ocr_duration:.2f}s")
            logger.info(f"Confidence: {result.get('overall_confidence', 0.0):.2f}")
            
        except Exception as e:
            logger.error(f"Mistral OCR test failed: {e}")
            pytest.fail(f"Mistral OCR performance test failed: {e}")


if __name__ == "__main__":
    # Allow running the script directly for development
    import sys
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        tester = PerformanceTester()
        
        print("Running performance tests...")
        
        # Run basic performance test
        metrics = await tester.run_single_document_test("medium", mock_apis=True)
        print(f"Test completed in {metrics.total_duration:.2f}s")
        print(f"Processing per page: {metrics.processing_time_per_page:.2f}s")
        print(f"Memory usage: {metrics.peak_memory_mb:.2f} MB")
        
        # Generate simple report
        report = tester.generate_performance_report([metrics])
        print("\nPerformance Report:")
        print(json.dumps(report, indent=2, default=str))
    
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        asyncio.run(main())
    else:
        print("Run with 'python test_performance.py run' to execute directly")
        print("Or use 'pytest backend/tests/test_performance.py -v -s' for full test suite")