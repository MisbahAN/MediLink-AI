# Performance Testing Guide for MediLink-AI

This document provides comprehensive information about the performance testing system for the MediLink-AI prior authorization automation project.

## Overview

The performance testing suite is designed to measure and validate system performance under various conditions, helping identify bottlenecks and ensure the system meets production requirements.

## Test Categories

### 1. Single Document Performance Tests
- **Purpose**: Measure processing time, memory usage, and CPU utilization for individual documents
- **Test Files**: 
  - Small: `test_3_referral_package.pdf` (~4.2MB, 8 pages)
  - Medium: `test_2_referral_package.pdf` (~7.5MB, 12 pages) 
  - Large: `test_1_referral_package.pdf` (~7.9MB, 15 pages)
- **Metrics Measured**:
  - Total processing duration
  - Processing time per page
  - Peak memory usage (MB)
  - Average CPU utilization (%)
  - Extraction confidence scores
  - Number of fields mapped vs missing

### 2. Concurrent Processing Tests
- **Purpose**: Validate system behavior under multiple simultaneous requests
- **Default Configuration**: 3 concurrent requests (configurable)
- **Metrics Measured**:
  - Success rate under load
  - Average response time
  - Throughput (requests/second)
  - Memory and CPU peaks during concurrency
  - Error rate

### 3. Memory Leak Detection
- **Purpose**: Identify potential memory leaks during repeated processing
- **Method**: Process documents multiple times and monitor memory growth
- **Default**: 5 iterations (configurable)
- **Detection Criteria**: Average growth > 10MB per iteration indicates potential leak

### 4. Large Document Stress Testing
- **Purpose**: Test system limits with the largest available test documents
- **Focus**: Processing the 15-page referral package efficiently
- **Validates**: System can handle production-sized documents

### 5. OCR Service Performance
- **Purpose**: Measure external API response times
- **Services Tested**: Mistral AI (primary), Gemini (fallback)
- **Metrics**: API response time, extraction quality, confidence scores

## Performance Thresholds

The system uses the following performance thresholds to identify issues:

```python
performance_thresholds = {
    "max_processing_time_per_page": 30.0,    # seconds
    "max_memory_usage_mb": 2048,             # 2GB
    "min_extraction_confidence": 0.7,        # 70%
    "max_ocr_response_time": 60.0,           # seconds  
    "max_acceptable_error_rate": 0.05        # 5%
}
```

## Running Performance Tests

### Prerequisites

1. **Install Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Test Data**: Ensure test PDF files are in `backend/tests/test_data/`

3. **Optional - API Keys**: For real API testing, set environment variables:
   ```bash
   export MISTRAL_API_KEY="your_mistral_key"
   export GEMINI_API_KEY="your_gemini_key" 
   export OPENAI_API_KEY="your_openai_key"
   ```

### Using pytest (Recommended)

```bash
# Run all performance tests with mocked APIs
pytest backend/tests/test_performance.py -v -s

# Run specific test categories
pytest backend/tests/test_performance.py::test_large_document_processing -v -s
pytest backend/tests/test_performance.py::test_concurrent_processing_performance -v -s
pytest backend/tests/test_performance.py::test_memory_leak_detection -v -s

# Run comprehensive test suite
pytest backend/tests/test_performance.py::test_comprehensive_performance_suite -v -s

# Run with real APIs (requires API keys)
pytest backend/tests/test_performance.py -m integration -v -s
```

### Using the Performance Test Runner

```bash
cd backend

# Basic performance tests (default)
python run_performance_tests.py --basic

# Comprehensive test suite
python run_performance_tests.py --comprehensive

# Specific test categories
python run_performance_tests.py --large-doc
python run_performance_tests.py --concurrency --concurrent-requests 5
python run_performance_tests.py --memory-leak --memory-iterations 10

# Real API testing (costs money!)
python run_performance_tests.py --real-apis

# Generate report only
python run_performance_tests.py --report-only

# Verbose output
python run_performance_tests.py --comprehensive --verbose
```

## Test Output and Reports

### Console Output
Performance tests provide real-time feedback with emojis and formatted output:
```
🚀 Running basic performance tests...
  Testing small document processing...
    ✅ Completed in 2.45s
    📊 0.31s per page
    💾 245.67 MB peak memory
```

### Performance Reports
Detailed JSON reports are generated with comprehensive metrics:

```json
{
  "test_summary": {
    "total_tests": 3,
    "successful_tests": 3,
    "failed_tests": 0,
    "success_rate": 1.0
  },
  "performance_metrics": {
    "avg_processing_time_seconds": 4.23,
    "avg_processing_time_per_page_seconds": 0.35,
    "avg_memory_usage_mb": 387.45,
    "avg_cpu_usage_percent": 45.2,
    "avg_extraction_confidence": 0.85
  },
  "performance_issues": [],
  "expected_bottlenecks": {
    "OCR_API_Latency": "External API calls are the primary bottleneck...",
    "PDF_Processing": "PDF parsing for large files is CPU-intensive...",
    "Memory_Usage": "Multiple large PDFs consume significant memory..."
  }
}
```

## Expected Performance Bottlenecks

Based on the system architecture, the following bottlenecks are anticipated:

### 1. OCR API Latency (Primary Bottleneck)
- **Issue**: External API calls to Mistral and Gemini services
- **Impact**: Each page requires network round-trip and AI inference
- **Mitigation**: Parallel page processing, caching, API optimization

### 2. PDF Processing 
- **Issue**: CPU-intensive PDF parsing and image extraction
- **Impact**: Complex layouts and many images increase processing time
- **Mitigation**: Optimize PDF libraries, consider preprocessing

### 3. Memory Usage
- **Issue**: Large documents held in memory during processing
- **Impact**: Multiple concurrent requests can exhaust memory
- **Mitigation**: Streaming processing, memory pooling, garbage collection

### 4. Sequential Processing
- **Issue**: Documents processed sequentially rather than in parallel
- **Impact**: Limits throughput for concurrent requests
- **Mitigation**: Implement parallel processing pipelines

### 5. Field Mapping Complexity
- **Issue**: OpenAI API calls for complex field mapping
- **Impact**: Cost and latency increase with document complexity
- **Mitigation**: Optimize prompts, cache common mappings

### 6. File I/O Operations
- **Issue**: Temporary file creation and storage operations
- **Impact**: Bottleneck with high concurrent load
- **Mitigation**: In-memory processing, SSD storage, I/O optimization

## Optimization Recommendations

### Short-term Improvements
1. **Enable Redis Caching**: Implement caching for processed documents
2. **Optimize PDF Processing**: Use more efficient PDF libraries
3. **Memory Management**: Implement explicit garbage collection
4. **API Optimization**: Batch API requests where possible

### Medium-term Improvements  
1. **Parallel Processing**: Implement concurrent page processing
2. **Streaming Architecture**: Process documents in streams
3. **Load Balancing**: Distribute load across multiple workers
4. **Database Optimization**: Optimize data storage and retrieval

### Long-term Improvements
1. **Microservices Architecture**: Separate OCR, mapping, and form filling
2. **Caching Layer**: Implement comprehensive caching strategy
3. **Auto-scaling**: Dynamic resource allocation based on load
4. **Performance Monitoring**: Real-time performance dashboards

## Continuous Performance Monitoring

### Integration with CI/CD
Add performance tests to your CI/CD pipeline:

```yaml
# .github/workflows/performance.yml
name: Performance Tests
on: [push, pull_request]
jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Performance Tests
        run: |
          cd backend
          pip install -r requirements.txt
          pytest tests/test_performance.py::test_comprehensive_performance_suite -v
```

### Performance Monitoring Checklist
- [ ] Run performance tests before each release
- [ ] Monitor memory usage trends over time
- [ ] Track processing time improvements/regressions
- [ ] Validate system performance under expected load
- [ ] Test with realistic document sizes and types
- [ ] Monitor API response times and costs

## Troubleshooting

### Common Issues

1. **Tests Skip Due to Missing Files**
   - Ensure test PDF files are in `backend/tests/test_data/`
   - Check file permissions

2. **Memory Errors During Testing**
   - Reduce concurrent requests (`--concurrent-requests 2`)
   - Reduce memory leak iterations (`--memory-iterations 3`)
   - Close other applications to free memory

3. **API Key Errors for Real Testing**
   - Verify environment variables are set correctly
   - Check API key validity and quotas
   - Use `--basic` mode for testing without real APIs

4. **Performance Test Failures**
   - Check performance thresholds in code
   - Review system resources (CPU, memory availability)
   - Run tests in isolated environment

### Debug Mode
Run tests with verbose logging to debug issues:
```bash
python run_performance_tests.py --comprehensive --verbose
```

## Performance Test Maintenance

### Updating Test Thresholds
Modify thresholds in `tests/test_performance.py`:
```python
self.performance_thresholds = {
    "max_processing_time_per_page": 30.0,  # Adjust as needed
    "max_memory_usage_mb": 2048,           # Adjust based on hardware
    # ...
}
```

### Adding New Test Cases
1. Create new test methods in `TestPerformance` class
2. Add corresponding CLI options in `run_performance_tests.py`
3. Update this documentation

### Test Data Management
- Keep test PDFs under 10MB for CI/CD compatibility
- Use representative document types and sizes
- Regularly update test data to reflect real usage patterns

## Getting Help

- **Performance Issues**: Review the bottleneck analysis in generated reports
- **Test Failures**: Check logs and error messages for specific issues
- **Custom Testing**: Modify test parameters in the PerformanceTester class
- **CI/CD Integration**: Follow the pipeline examples in this guide

For additional support, consult the main project documentation and logs in `backend/logs/`.