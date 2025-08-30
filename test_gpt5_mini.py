#!/usr/bin/env python3
"""
Test script to verify GPT-5-mini model is working correctly
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.llm_utils import generate_text, _auto_model

async def test_model_selection():
    """Test that the model selection is working"""
    print("Testing model selection...")
    
    # Check what model would be auto-selected
    try:
        provider, model = _auto_model()
        print(f"✓ Auto-detected provider: {provider}")
        print(f"✓ Auto-detected model: {model}")
        
        if provider == "openai" and model == "gpt-5-mini":
            print("✓ GPT-5-mini is correctly set as default for OpenAI")
        elif provider == "openai":
            print(f"⚠ OpenAI detected but model is {model}, not gpt-5-mini")
        else:
            print(f"ℹ Using {provider} provider with {model}")
            
    except RuntimeError as e:
        print(f"✗ No API key found: {e}")
        return False
    
    return True

async def test_simple_generation():
    """Test a simple text generation"""
    print("\nTesting simple text generation...")
    
    try:
        # Simple test prompt
        result = await generate_text(
            prompt="Say 'Hello, GPT-5-mini is working!' and nothing else.",
            max_tokens=20,
            temperature=0
        )
        print(f"✓ Generation successful: {result}")
        
        if "GPT-5-mini" in result or "working" in result.lower():
            print("✓ Response looks correct")
        else:
            print("⚠ Response might be from fallback or different model")
            
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        print(f"  Error type: {type(e).__name__}")
        return False
    
    return True

async def test_json_generation():
    """Test JSON generation (used by keyword agent)"""
    print("\nTesting JSON generation...")
    
    try:
        result = await generate_json(
            prompt="Generate a JSON object with a single field 'status' set to 'success'",
            max_tokens=50,
            temperature=0
        )
        print(f"✓ JSON generation successful: {result}")
        
        if result.get("status") == "success":
            print("✓ JSON response is correct")
        else:
            print("⚠ JSON response unexpected")
            
    except Exception as e:
        print(f"✗ JSON generation failed: {e}")
        return False
    
    return True

async def main():
    """Run all tests"""
    print("=" * 60)
    print("GPT-5-mini Model Test")
    print("=" * 60)
    
    # Check environment
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    
    print("API Keys detected:")
    print(f"  OpenAI: {'✓' if has_openai else '✗'}")
    print(f"  Anthropic: {'✓' if has_anthropic else '✗'}")
    print(f"  Google: {'✓' if has_google else '✗'}")
    print()
    
    # Run tests
    tests_passed = []
    
    tests_passed.append(await test_model_selection())
    
    if has_openai or has_anthropic or has_google:
        tests_passed.append(await test_simple_generation())
        
        # Import here to avoid import errors if no API keys
        from utils.llm_utils import generate_json
        tests_passed.append(await test_json_generation())
    else:
        print("\n✗ No API keys found - skipping generation tests")
        print("  Set OPENAI_API_KEY to test GPT-5-mini")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Passed: {sum(tests_passed)}/{len(tests_passed)}")
    
    if all(tests_passed):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())