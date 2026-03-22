# -*- coding: utf-8 -*-
"""Test script for Serp search integration."""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print("Testing Serp search integration...")

# Test 1: Import configuration
try:
    from open_deep_research.configuration import SearchAPI
    print("✓ Configuration imported")
    print(f"  Available search APIs: {[api.value for api in SearchAPI]}")
    print(f"  Default: {SearchAPI.SERP.value}")
except Exception as e:
    print(f"✗ Configuration import failed: {e}")
    sys.exit(1)

# Test 2: Import search function
try:
    from open_deep_research.utils import serp_search
    print("✓ serp_search function imported")
except Exception as e:
    print(f"✗ serp_search import failed: {e}")
    sys.exit(1)

# Test 3: Check SERP_API_KEY
if os.getenv("SERP_API_KEY"):
    print("✓ SERP_API_KEY found in environment")
else:
    print("⚠ SERP_API_KEY not found in environment")

# Test 4: Test search (if key available)
if os.getenv("SERP_API_KEY"):
    try:
        result = serp_search("Type 1 diabetes", max_results=2)
        print(f"✓ Search successful, result length: {len(result)} chars")
        print("  First 200 chars:", result[:200])
    except Exception as e:
        print(f"✗ Search test failed: {e}")
else:
    print("⊘ Skipping search test (no API key)")

print("\nAll import tests passed!")

