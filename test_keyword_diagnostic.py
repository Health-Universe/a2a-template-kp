#!/usr/bin/env python3
"""
Test script for keyword agent diagnostic mode
"""

import json
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from examples.pipeline.keyword.agent import KeywordAgent

async def test_diagnostic_mode():
    """Test the diagnostic mode"""
    print("=" * 60)
    print("Testing Keyword Agent Diagnostic Mode")
    print("=" * 60)
    
    agent = KeywordAgent()
    
    # Test 1: Run diagnostics
    print("\n1. Running diagnostics...")
    diagnostic_input = json.dumps({"diagnostic_mode": True})
    result = await agent.process_message(diagnostic_input)
    
    print("\nDiagnostic Results:")
    print(json.dumps(result, indent=2))
    
    # Test 2: Normal mode with forced error (no API key scenario)
    print("\n" + "=" * 60)
    print("2. Testing normal mode (will show error details if LLM fails)...")
    normal_input = json.dumps({
        "document_preview": "Patient diagnosed with diabetes type 2 in 2023."
    })
    result = await agent.process_message(normal_input)
    
    print("\nPattern Generation Results:")
    print(f"Source: {result.get('source', 'unknown')}")
    print(f"Status: {result.get('status', 'unknown')}")
    
    if "error_details" in result:
        print("\nError Details Found:")
        print(json.dumps(result["error_details"], indent=2))
    elif "error_info" in result:
        print("\nError Info Found:")
        print(json.dumps(result["error_info"], indent=2))
    
    # Show pattern count
    patterns = result.get("patterns", [])
    print(f"\nGenerated {len(patterns)} patterns")
    
    # Show first few patterns
    if patterns:
        print("Sample patterns:")
        for p in patterns[:3]:
            print(f"  - {p}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_diagnostic_mode())