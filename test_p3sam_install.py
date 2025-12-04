#!/usr/bin/env python3
"""
Test script to verify P3-SAM package installation.

Run this from any directory to test if P3-SAM is properly installed.
"""

import sys
import os

def test_imports():
    """Test if all necessary imports work."""
    print("=" * 70)
    print("Testing P3-SAM Package Installation")
    print("=" * 70)
    print()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Import AutoMask from p3sam.demo
    print("Test 1: Import AutoMask from p3sam.demo...")
    try:
        from p3sam.demo import AutoMask
        print("  ✓ SUCCESS: from p3sam.demo import AutoMask")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
    print()
    
    # Test 2: Import other components from p3sam.demo
    print("Test 2: Import other components from p3sam.demo...")
    try:
        from p3sam.demo import mesh_sam, P3SAM
        print("  ✓ SUCCESS: from p3sam.demo import mesh_sam, P3SAM")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
    print()
    
    # Test 3: Import model components
    print("Test 3: Import from p3sam.model...")
    try:
        from p3sam.model import build_P3SAM, load_state_dict
        print("  ✓ SUCCESS: from p3sam.model import build_P3SAM, load_state_dict")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
    print()
    
    # Test 4: Check if AutoMask can be instantiated (without checkpoint)
    print("Test 4: Check AutoMask class structure...")
    try:
        from p3sam.demo import AutoMask
        # Check if class has expected methods
        assert hasattr(AutoMask, '__init__')
        assert hasattr(AutoMask, 'predict_aabb')
        print("  ✓ SUCCESS: AutoMask has expected methods")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
    print()
    
    # Test 5: Check package version
    print("Test 5: Check package metadata...")
    try:
        import importlib.metadata
        version = importlib.metadata.version('p3sam')
        print(f"  ✓ SUCCESS: p3sam version {version} installed")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
    print()
    
    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests Passed: {tests_passed}/5")
    print(f"Tests Failed: {tests_failed}/5")
    print()
    
    if tests_failed == 0:
        print("🎉 All tests passed! P3-SAM is properly installed.")
        print()
        print("You can now use P3-SAM from any directory:")
        print()
        print("  from p3sam.demo import AutoMask")
        print("  from p3sam.model import build_P3SAM, load_state_dict")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the installation:")
        print()
        print("To reinstall:")
        print("  cd /home/nranawakaara/Projects/Hunyuan3D-Part")
        print("  ./reinstall_p3sam.sh")
        return 1


if __name__ == "__main__":
    exit_code = test_imports()
    sys.exit(exit_code)

