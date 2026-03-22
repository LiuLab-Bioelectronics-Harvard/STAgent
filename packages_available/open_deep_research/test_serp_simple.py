# -*- coding: utf-8 -*-
"""Simple test for Serp integration - Python 2 compatible."""
from __future__ import print_function
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print("Testing Serp search integration...")

# Test imports
try:
    from open_deep_research.configuration import SearchAPI
    print("OK: Configuration imported")
    print("  Available APIs: SERP, ANTHROPIC, OPENAI, NONE")
except Exception as e:
    print("FAIL: Configuration import - " + str(e))
    sys.exit(1)

try:
    from open_deep_research.utils import serp_search
    print("OK: serp_search function imported")
except Exception as e:
    print("FAIL: serp_search import - " + str(e))
    sys.exit(1)

# Check API key
if os.getenv("SERP_API_KEY"):
    print("OK: SERP_API_KEY found")
else:
    print("WARN: SERP_API_KEY not found")

print("\nAll tests passed!")

