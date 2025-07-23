#!/usr/bin/env python3
"""
Test Report Generator for Deepfake Detection Project

This script runs comprehensive tests and generates detailed reports including:
- Test results summary
- Coverage analysis
- Performance metrics
- Visualization examples
- Recommendations for improvement
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

class TestReportGenerator:
    """Generate comprehensive test reports for the deepfake detection project."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the test report generator.
        
        Args:
            output_dir: Directory to store generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.test_results = {}
        self.coverage_data = {}
        self.start_time = time.time()
        
    def run_tests(self, test_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run pytest with coverage and collect results.
        
        Args:
            test_files: Specific test files to run (None for all)
            
        Returns:
            Dictionary containing test results
        """
        print("Running tests with coverage...")
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=src",
            "--cov-report=json:coverage.json",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--junitxml=test-results.xml",
            "-v"
        ]
        
        if test_files:
            cmd.extend(test_files)
        
        # Run tests
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            
            # Parse results
            self.test_results = {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }
            
            print(f"Tests completed with return code: {result.returncode}")
            return self.test_results
            
        except Exception as e:
            print(f"Error running tests: {e}")
            return {"error": str(e)}
    
    def parse_coverage_data(self) -> Dict[str, Any]:
        """Parse coverage data from XML report."""
        try:
            import xml.etree.ElementTree as ET
            
            coverage_file = Path("coverage.xml")
            if not coverage_file.exists():
                return {}
            
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            
            coverage_data = {
                "overall": {},
                "packages": []
            }
            
            # Parse overall coverage
            for package in root.findall(".//package"):
                package_name = package.get("name", "unknown")
                line_rate = float(package.get("line-rate", 0))
                branch_rate = float(package.get("branch-rate", 0))
                
                coverage_data["packages"].append({
                    "name": package_name,
                    "line_rate": line_rate,
                    "branch_rate": branch_rate,
                    "lines_covered": int(package.get("lines-covered", 0)),
                    "lines_valid": int(package.get("lines-valid", 0))
                })
            
            # Calculate overall coverage
            if coverage_data["packages"]:
                total_lines_covered = sum(p["lines_covered"] for p in coverage_data["packages"])
                total_lines_valid = sum(p["lines_valid"] for p in coverage_data["packages"])
                
                coverage_data["overall"] = {
                    "line_rate": total_lines_covered / total_lines_valid if total_lines_valid > 0 else 0,
                    "lines_covered": total_lines_covered,
                    "lines_valid": total_lines_valid
                }
            
            self.coverage_data = coverage_data
            return coverage_data
            
        except Exception as e:
            print(f"Warning: Could not parse coverage data: {e}")
            return {}
    
    def generate_coverage_chart(self) -> str:
        """Generate coverage visualization chart."""
        if not self.coverage_data.get("packages"):
            return ""
        
        try:
            packages = self.coverage_data["packages"]
            names = [p["name"] for p in packages]
            line_rates = [p["line_rate"] * 100 for p in packages]
            branch_rates = [p["branch_rate"] * 100 for p in packages]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Line coverage
            bars1 = ax1.bar(names, line_rates, color='skyblue', alpha=0.7)
            ax1.set_title('Line Coverage by Package', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Coverage (%)')
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, rate in zip(bars1, line_rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom')
            
            # Branch coverage
            bars2 = ax2.bar(names, branch_rates, color='lightcoral', alpha=0.7)
            ax2.set_title('Branch Coverage by Package', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Coverage (%)')
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, rate in zip(bars2, branch_rates):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / "coverage_chart.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Warning: Could not generate coverage chart: {e}")
            return ""
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate test summary statistics."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "duration": time.time() - self.start_time,
            "test_status": "PASSED" if self.test_results.get("return_code") == 0 else "FAILED",
            "coverage": self.coverage_data.get("overall", {}),
            "packages": len(self.coverage_data.get("packages", [])),
            "recommendations": []
        }
        
        # Add recommendations based on coverage
        overall_coverage = self.coverage_data.get("overall", {}).get("line_rate", 0)
        
        if overall_coverage < 0.7:
            summary["recommendations"].append("Increase overall test coverage to at least 70%")
        elif overall_coverage < 0.8:
            summary["recommendations"].append("Consider increasing test coverage to 80% for better quality")
        else:
            summary["recommendations"].append("Excellent test coverage! Maintain this level")
        
        # Check for packages with low coverage
        low_coverage_packages = [
            p for p in self.coverage_data.get("packages", [])
            if p.get("line_rate", 0) < 0.6
        ]
        
        if low_coverage_packages:
            summary["recommendations"].append(
                f"Focus on improving coverage for: {', '.join(p['name'] for p in low_coverage_packages)}"
            )
        
        return summary
    
    def create_html_report(self) -> str:
        """Create comprehensive HTML test report."""
        summary = self.generate_test_summary()
        coverage_chart = self.generate_coverage_chart()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deepfake Detection - Test Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #007bff;
            margin: 0;
            font-size: 2.5em;
        }}
        .status {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .status.passed {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .status.failed {{
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            background-color: #f8f9fa;
        }}
        .section h2 {{
            color: #495057;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-width: 150px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .coverage-chart {{
            text-align: center;
            margin: 20px 0;
        }}
        .coverage-chart img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .recommendations {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .recommendations h3 {{
            color: #856404;
            margin-top: 0;
        }}
        .recommendations ul {{
            color: #856404;
        }}
        .test-output {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Test Report</h1>
            <p>Deepfake Detection Project</p>
            <div class="status {'passed' if summary['test_status'] == 'PASSED' else 'failed'}">
                {summary['test_status']}
            </div>
            <p>Generated on {summary['timestamp']}</p>
        </div>
        
        <div class="section">
            <h2>Test Summary</h2>
            <div class="metric">
                <div class="metric-value">{summary['test_status']}</div>
                <div class="metric-label">Test Status</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['duration']:.2f}s</div>
                <div class="metric-label">Duration</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['packages']}</div>
                <div class="metric-label">Packages Tested</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Coverage Analysis</h2>
            <div class="metric">
                <div class="metric-value">{summary['coverage'].get('line_rate', 0)*100:.1f}%</div>
                <div class="metric-label">Line Coverage</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['coverage'].get('lines_covered', 0)}</div>
                <div class="metric-label">Lines Covered</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['coverage'].get('lines_valid', 0)}</div>
                <div class="metric-label">Total Lines</div>
            </div>
        </div>
        
        {f'<div class="coverage-chart"><img src="{coverage_chart}" alt="Coverage Chart"></div>' if coverage_chart else ''}
        
        <div class="section">
            <h2>Recommendations</h2>
            <div class="recommendations">
                <h3>Improvement Suggestions:</h3>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in summary['recommendations'])}
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>Test Output</h2>
            <div class="test-output">{self.test_results.get('stdout', 'No output available')}</div>
        </div>
        
        <div class="footer">
            <p>Generated by Test Report Generator v1.0</p>
            <p>For more information, see <a href="TESTING_README.md">TESTING_README.md</a></p>
        </div>
    </div>
</body>
</html>
        """
        
        # Save HTML report
        report_path = self.output_dir / "test_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def save_summary_json(self) -> str:
        """Save test summary as JSON file."""
        summary = self.generate_test_summary()
        summary_path = self.output_dir / "test_summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(summary_path)
    
    def generate_report(self, test_files: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Generate comprehensive test report.
        
        Args:
            test_files: Specific test files to run (None for all)
            
        Returns:
            Dictionary with paths to generated reports
        """
        print("Starting comprehensive test report generation...")
        
        # Run tests
        self.run_tests(test_files)
        
        # Parse coverage data
        self.parse_coverage_data()
        
        # Generate reports
        reports = {}
        
        # HTML report
        html_report = self.create_html_report()
        reports["html_report"] = html_report
        print(f"HTML report generated: {html_report}")
        
        # JSON summary
        json_summary = self.save_summary_json()
        reports["json_summary"] = json_summary
        print(f"JSON summary generated: {json_summary}")
        
        # Coverage chart
        if self.coverage_data.get("packages"):
            coverage_chart = self.generate_coverage_chart()
            if coverage_chart:
                reports["coverage_chart"] = coverage_chart
                print(f"📈 Coverage chart generated: {coverage_chart}")
        
        print("Test report generation completed!")
        return reports


def main():
    """Main function to run test report generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive test reports")
    parser.add_argument("--test-files", nargs="+", help="Specific test files to run")
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Create report generator
    generator = TestReportGenerator(args.output_dir)
    
    # Generate reports
    reports = generator.generate_report(args.test_files)
    
    # Print summary
    print("\n" + "="*60)
    print("REPORT SUMMARY")
    print("="*60)
    print(f"HTML Report: {reports.get('html_report', 'Not generated')}")
    print(f"JSON Summary: {reports.get('json_summary', 'Not generated')}")
    print(f"Coverage Chart: {reports.get('coverage_chart', 'Not generated')}")
    print("="*60)
    
    # Open HTML report if available
    if reports.get("html_report"):
        try:
            import webbrowser
            webbrowser.open(f"file://{Path(reports['html_report']).absolute()}")
            print("Opening HTML report in browser...")
        except Exception as e:
            print(f"Could not open browser: {e}")


if __name__ == "__main__":
    main() 