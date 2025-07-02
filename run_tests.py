#!/usr/bin/env python3
"""
Simple test runner script for the deepfake detection project.

This script provides easy-to-use commands for running tests and generating reports.
"""

import sys
import subprocess
from pathlib import Path

def run_basic_tests():
    """Run basic tests without coverage."""
    print("🧪 Running basic tests...")
    cmd = ["python", "-m", "pytest", "-v"]
    subprocess.run(cmd)

def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    print("📊 Running tests with coverage...")
    cmd = [
        "python", "-m", "pytest",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html",
        "-v"
    ]
    subprocess.run(cmd)

def generate_comprehensive_report():
    """Generate comprehensive test report."""
    print("📋 Generating comprehensive test report...")
    cmd = ["python", "generate_test_report.py"]
    subprocess.run(cmd)

def run_specific_tests(test_files):
    """Run specific test files."""
    print(f"🎯 Running specific tests: {test_files}")
    cmd = ["python", "-m", "pytest", "-v"] + test_files
    subprocess.run(cmd)

def main():
    """Main function with command-line interface."""
    if len(sys.argv) < 2:
        print("""
🧪 Deepfake Detection Test Runner

Usage:
    python run_tests.py [command] [options]

Commands:
    basic              Run basic tests without coverage
    coverage           Run tests with coverage reporting
    report             Generate comprehensive test report
    specific <files>   Run specific test files
    all                Run all tests and generate comprehensive report

Examples:
    python run_tests.py basic
    python run_tests.py coverage
    python run_tests.py report
    python run_tests.py specific test_data_loading.py test_evaluation.py
    python run_tests.py all
        """)
        return

    command = sys.argv[1]

    if command == "basic":
        run_basic_tests()
    elif command == "coverage":
        run_tests_with_coverage()
    elif command == "report":
        generate_comprehensive_report()
    elif command == "specific":
        if len(sys.argv) < 3:
            print("❌ Please specify test files to run")
            return
        test_files = sys.argv[2:]
        run_specific_tests(test_files)
    elif command == "all":
        print("🚀 Running all tests and generating comprehensive report...")
        run_tests_with_coverage()
        generate_comprehensive_report()
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python run_tests.py' for help")

if __name__ == "__main__":
    main() 