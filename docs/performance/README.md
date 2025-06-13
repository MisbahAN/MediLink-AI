# Performance Testing Documentation

This folder contains comprehensive documentation for the MediLink-AI performance testing suite.

## 📁 Files Overview

### `performance_analysis.md`
- **Bottleneck identification** and analysis
- **Performance projections** under various load conditions
- **Optimization roadmap** for production readiness
- **Architecture assessment** and scaling challenges

### `performance_testing.md`
- **Implementation guide** for the performance testing suite
- **Test categories** and methodologies
- **Usage instructions** for running performance tests
- **Expected results** and validation criteria

## 🚀 Quick Start

To run performance tests:

```bash
# Basic performance testing
cd backend
python run_performance_tests.py --basic

# Large document testing
python run_performance_tests.py --large-doc

# Comprehensive test suite
python run_performance_tests.py --comprehensive

# Using pytest
pytest tests/test_performance.py -v -s
```

## 📊 Test Outputs

Performance test results are automatically saved to:
- `backend/test_outputs/performance_reports/` - JSON reports with detailed metrics
- Console output - Real-time progress and summary statistics

## 🔍 Key Metrics Tracked

- **Processing time** per document and per page
- **Memory usage** patterns and leak detection
- **CPU utilization** during processing
- **Concurrency performance** and throughput
- **Extraction confidence** and success rates
- **System resource** monitoring and bottleneck analysis

These tests are designed to validate system performance with large documents (15+ pages) and ensure production readiness.