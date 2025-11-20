"""
Quick test script to verify parallel training system is working
Runs a minimal test with just 2 experiments
"""

import subprocess
import sys
import os

def test_imports():
    """Test that all required packages are available"""
    print("Testing imports...")
    try:
        import torch
        import numpy as np
        import pandas as pd
        import matplotlib
        import seaborn as sns
        print("  ✓ All required packages available")
        return True
    except ImportError as e:
        print(f"  ✗ Missing package: {e}")
        return False


def test_parallel_runner():
    """Test parallel experiment runner with minimal configuration"""
    print("\nTesting parallel experiment runner...")

    # Create minimal test configuration
    cmd = [
        "python", "run_parallel_experiments.py",
        "--batch-sizes", "64",
        "--learning-rates", "0.001",
        "--lambda-params", "0.75",
        "--nasa-weights", "0.1",
        "--dropout-rates", "0.1",
        "--datasets", "turbofan",
        "--num-epochs", "2",  # Very short for testing
        "--output-dir", "parallel_test",
        "--num-workers", "1"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            print("  ✓ Parallel runner executed successfully")
            return True
        else:
            print(f"  ✗ Parallel runner failed with return code {result.returncode}")
            print(f"  Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("  ✗ Test timed out (>10 minutes)")
        return False
    except Exception as e:
        print(f"  ✗ Exception: {str(e)}")
        return False


def test_analyzer():
    """Test results analysis script"""
    print("\nTesting results analyzer...")

    # Check if test results exist
    if not os.path.exists("parallel_test/all_results.json"):
        print("  ⚠ No test results to analyze (run parallel runner test first)")
        return False

    cmd = [
        "python", "analyze_parallel_results.py",
        "--output-dir", "parallel_test"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print("  ✓ Analyzer executed successfully")

            # Check if outputs were created
            expected_files = [
                "parallel_test/summary_statistics.csv",
                "parallel_test/best_configurations.csv",
                "parallel_test/analysis_plots"
            ]

            all_exist = True
            for filepath in expected_files:
                if os.path.exists(filepath):
                    print(f"  ✓ Created: {filepath}")
                else:
                    print(f"  ✗ Missing: {filepath}")
                    all_exist = False

            return all_exist

        else:
            print(f"  ✗ Analyzer failed with return code {result.returncode}")
            print(f"  Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("  ✗ Test timed out (>2 minutes)")
        return False
    except Exception as e:
        print(f"  ✗ Exception: {str(e)}")
        return False


def main():
    print("="*60)
    print("PARALLEL TRAINING SYSTEM - VERIFICATION TEST")
    print("="*60)
    print("\nThis will run a minimal test to verify the system works.")
    print("Expected duration: 5-10 minutes")
    print("")

    # Test 1: Imports
    if not test_imports():
        print("\n✗ FAILED: Missing required packages")
        print("  Install with: pip install torch numpy pandas matplotlib seaborn")
        sys.exit(1)

    # Test 2: Parallel Runner
    print("\nRunning minimal parallel experiment (2 epochs)...")
    print("This may take 5-10 minutes...")

    if not test_parallel_runner():
        print("\n✗ FAILED: Parallel runner test failed")
        print("  Check the error messages above")
        sys.exit(1)

    # Test 3: Analyzer
    if not test_analyzer():
        print("\n✗ FAILED: Analyzer test failed")
        print("  Check the error messages above")
        sys.exit(1)

    # Success!
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    print("\nYour parallel training system is working correctly!")
    print("\nNext steps:")
    print("  1. Run: ./quick_experiments.sh")
    print("  2. Select option 5 (Quick Model Comparison)")
    print("  3. Check results in quick_comparison/analysis_plots/")
    print("")
    print("Test results saved to: parallel_test/")
    print("You can delete this with: rm -rf parallel_test/")
    print("="*60)


if __name__ == '__main__':
    main()
