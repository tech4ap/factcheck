# Testing Guide for Deepfake Detection Project

This document provides comprehensive guidance for running tests, understanding test coverage, and generating test reports for the deepfake detection project.

## Table of Contents
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Test Reports](#test-reports)
- [Adding New Tests](#adding-new-tests)
- [Troubleshooting](#troubleshooting)

## Test Structure

The project includes the following test files:

```
dfd/
├── test_data_loading.py      # Data loading functionality tests
├── test_evaluation.py        # Evaluation metrics and visualization tests
├── test_edge_cases.py        # Edge cases and error handling tests
└── tests/                    # Additional test modules (if any)
```

### Test Categories

1. **Data Loading Tests** (`test_data_loading.py`)
   - Image data loading from directories
   - Video data loading and frame extraction
   - Audio data loading and spectrogram conversion
   - CSV-based data loading (legacy support)

2. **Evaluation Tests** (`test_evaluation.py`)
   - Comprehensive metrics calculation
   - Visualization capabilities
   - Model comparison functionality
   - Classification report generation

3. **Edge Case Tests** (`test_edge_cases.py`)
   - Empty directory handling
   - Single-class data scenarios
   - Extreme metric values
   - Error handling

## Running Tests

### Prerequisites

Ensure you have the required testing dependencies installed:

```bash
pip install pytest pytest-cov coverage
```

### Basic Test Execution

Run all tests:
```bash
python -m pytest -v
```

Run specific test files:
```bash
# Run data loading tests only
python -m pytest test_data_loading.py -v

# Run evaluation tests only
python -m pytest test_evaluation.py -v

# Run edge case tests only
python -m pytest test_edge_cases.py -v
```

Run specific test functions:
```bash
# Run a specific test function
python -m pytest test_data_loading.py::test_image_loading -v

# Run tests matching a pattern
python -m pytest -k "image" -v
```

### Test Execution with Coverage

Run tests with coverage reporting:
```bash
# Basic coverage report
python -m pytest --cov=src --cov-report=term-missing

# Detailed coverage with HTML report
python -m pytest --cov=src --cov-report=html --cov-report=term-missing

# XML coverage report for CI/CD
python -m pytest --cov=src --cov-report=xml --cov-report=term-missing
```

### Test Execution Examples

```bash
# Quick test run
python -m pytest -v

# Run with coverage and generate HTML report
python -m pytest --cov=src --cov-report=html --cov-report=term-missing -v

# Run only fast tests (exclude slow ones)
python -m pytest -m "not slow" -v

# Run tests in parallel (if pytest-xdist is installed)
python -m pytest -n auto -v

# Run tests with detailed output
python -m pytest -v -s --tb=long
```

## Test Coverage

### Coverage Reports

The project generates several types of coverage reports:

1. **Terminal Report**: Shows coverage summary in the terminal
2. **HTML Report**: Detailed coverage report in `htmlcov/` directory
3. **XML Report**: Coverage data in XML format for CI/CD integration

### Coverage Targets

- **Overall Coverage**: Aim for >80% code coverage
- **Critical Modules**: >90% coverage for core functionality
- **New Features**: >95% coverage for new code

### Viewing Coverage Reports

```bash
# Generate HTML coverage report
python -m pytest --cov=src --cov-report=html

# Open HTML report in browser
open htmlcov/index.html  # macOS
# or
firefox htmlcov/index.html  # Linux
# or
start htmlcov/index.html  # Windows
```

## Test Reports

### Automated Test Report Generation

The project includes automated test report generation that creates comprehensive reports including:

1. **Test Results Summary**
2. **Coverage Analysis**
3. **Performance Metrics**
4. **Visualization Examples**

### Generating Test Reports

```bash
# Generate comprehensive test report
python -m pytest --cov=src --cov-report=html --cov-report=term-missing \
    --html=reports/test_report.html --self-contained-html \
    --junitxml=reports/test_results.xml
```

### Report Structure

Generated reports are stored in the `reports/` directory:

```
reports/
├── test_report.html          # HTML test report
├── test_results.xml          # JUnit XML results
├── coverage_report.html      # Coverage report
└── test_summary.json         # JSON summary
```

## Adding New Tests

### Test File Structure

When adding new tests, follow this structure:

```python
import pytest
import numpy as np
from pathlib import Path

def test_functionality_name():
    """Test description."""
    # Arrange
    # Act
    # Assert
    pass

def test_edge_case_name():
    """Test edge case description."""
    # Test edge cases
    pass

class TestClass:
    """Test class for related functionality."""
    
    def test_method(self):
        """Test method description."""
        pass
```

### Test Naming Conventions

- Test functions: `test_<functionality>_<scenario>`
- Test classes: `Test<ClassName>`
- Test files: `test_<module_name>.py`

### Example Test

```python
def test_image_preprocessing():
    """Test image preprocessing functionality."""
    # Arrange
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Act
    processed_image = preprocess_image(test_image)
    
    # Assert
    assert processed_image.shape == (256, 256, 3)
    assert processed_image.dtype == np.float32
    assert np.all(processed_image >= 0) and np.all(processed_image <= 1)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd /path/to/dfd
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Coverage Not Working**
   ```bash
   # Install coverage package
   pip install pytest-cov coverage
   
   # Clear coverage cache
   coverage erase
   ```

3. **Test Failures**
   ```bash
   # Run with verbose output
   python -m pytest -v -s
   
   # Run specific failing test
   python -m pytest test_file.py::test_function -v -s
   ```

### Debug Mode

Run tests in debug mode for detailed output:

```bash
# Debug mode with full traceback
python -m pytest -v -s --tb=long

# Debug specific test
python -m pytest test_file.py::test_function -v -s --pdb
```

### Performance Testing

For performance-critical tests:

```bash
# Run with timing
python -m pytest --durations=10

# Profile slow tests
python -m pytest --profile
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: |
        python -m pytest --cov=src --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Best Practices

1. **Write Tests First**: Follow TDD principles when possible
2. **Test Edge Cases**: Include tests for boundary conditions
3. **Keep Tests Fast**: Avoid slow operations in unit tests
4. **Use Descriptive Names**: Make test names self-documenting
5. **Isolate Tests**: Each test should be independent
6. **Mock External Dependencies**: Use mocks for external services
7. **Maintain Coverage**: Keep coverage above target thresholds

## Support

For testing-related issues:

1. Check the troubleshooting section above
2. Review test logs and error messages
3. Consult the pytest documentation
4. Create an issue in the project repository

---

**Last Updated**: December 2024
**Version**: 1.0.0 