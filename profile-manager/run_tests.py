#!/usr/bin/env python3
"""Test runner script for GraphDB integration tests."""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ {description} completed successfully")
        return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run GraphDB integration tests")
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="Run unit tests only (mock-based)"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="Run integration tests (requires database)"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all tests"
    )
    parser.add_argument(
        "--slow", 
        action="store_true", 
        help="Include slow tests"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run with coverage report"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--file", 
        type=str, 
        help="Run specific test file"
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent
    import os
    os.chdir(project_root)
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        cmd.append("-v")
    
    if args.coverage:
        cmd.extend(["--cov=mcp_server", "--cov-report=html", "--cov-report=term"])
    
    if args.file:
        cmd.append(f"tests/{args.file}")
    else:
        if args.unit:
            cmd.extend(["-m", "unit"])
        elif args.integration:
            cmd.extend(["--run-integration", "-m", "integration"])
        elif args.all:
            cmd.extend(["--run-integration"])
        else:
            # Default: run unit tests
            cmd.extend(["-m", "unit"])
    
    if args.slow:
        cmd.append("--run-slow")
    
    # Add test discovery
    cmd.append("tests/")
    
    # Run the tests
    success = run_command(cmd, "GraphDB Integration Tests")
    
    if args.coverage and success:
        print(f"\n📊 Coverage report generated in htmlcov/index.html")
    
    if not success:
        sys.exit(1)
    
    print(f"\n🎉 All tests completed successfully!")


if __name__ == "__main__":
    main()
