#!/usr/bin/env python3
"""
Test script for the Template Agent.
Verifies the agent starts correctly and can process messages.
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def print_test(name: str, passed: bool, details: str = ""):
    """Print colored test result."""
    if passed:
        print(f"{GREEN}✓{RESET} {name}")
        if details:
            print(f"  {details}")
    else:
        print(f"{RED}✗{RESET} {name}")
        if details:
            print(f"  {RED}{details}{RESET}")
    return passed

async def test_template_agent():
    """Test the template agent functionality."""
    print("\n" + "="*60)
    print("Testing Template Agent")
    print("="*60 + "\n")
    
    all_passed = True
    
    # Test 1: Import and instantiation
    print(f"{YELLOW}1. Agent Creation{RESET}")
    try:
        from examples.template_agent.agent import TemplateAgent
        agent = TemplateAgent()
        
        all_passed &= print_test(
            "Agent instantiated",
            True,
            f"Name: {agent.get_agent_name()}"
        )
        
        all_passed &= print_test(
            "Version correct",
            agent.get_agent_version() == "1.0.0"
        )
        
        all_passed &= print_test(
            "Has description",
            len(agent.get_agent_description()) > 0
        )
        
    except Exception as e:
        all_passed &= print_test("Agent creation", False, str(e))
        return False
    
    print()
    
    # Test 2: Agent Card creation
    print(f"{YELLOW}2. Agent Card{RESET}")
    try:
        card = agent.create_agent_card()
        
        # Check required fields
        checks = [
            ("Protocol version", hasattr(card, 'protocolVersion') or hasattr(card, 'protocol_version')),
            ("Name present", bool(card.name)),
            ("URL present", bool(card.url)),
            ("Transport is HTTP", getattr(card, 'preferredTransport', getattr(card, 'preferred_transport', None)) == "HTTP"),
            ("Has capabilities", hasattr(card, 'capabilities')),
            ("Has skills", hasattr(card, 'skills') and len(card.skills) > 0)
        ]
        
        for check_name, passed in checks:
            all_passed &= print_test(check_name, passed)
            
    except Exception as e:
        all_passed &= print_test("Agent card creation", False, str(e))
    
    print()
    
    # Test 3: Message processing (mock LLM)
    print(f"{YELLOW}3. Message Processing{RESET}")
    try:
        # Test with plain text
        text_response = await agent.process_message("Test message")
        all_passed &= print_test(
            "Plain text processing",
            isinstance(text_response, str),
            f"Response type: {type(text_response).__name__}"
        )
        
        # Test with JSON
        json_data = json.dumps({"key": "value", "number": 42})
        json_response = await agent.process_message(json_data)
        all_passed &= print_test(
            "JSON processing",
            isinstance(json_response, str),
            f"Response type: {type(json_response).__name__}"
        )
        
    except Exception as e:
        # Expected if no LLM key is set
        if "No API key found" in str(e) or "LLM generation failed" in str(e):
            print_test(
                "Message processing (no LLM key)",
                True,
                "Correctly requires API key"
            )
        else:
            all_passed &= print_test("Message processing", False, str(e))
    
    print()
    
    # Test 4: Tools availability
    print(f"{YELLOW}4. Tools Configuration{RESET}")
    try:
        tools = agent.get_tools()
        all_passed &= print_test(
            "get_tools() returns list",
            isinstance(tools, list),
            f"Tool count: {len(tools)}"
        )
        
        # Test importing example tools
        from examples.template_agent.tools.example_tools import EXAMPLE_TOOLS
        all_passed &= print_test(
            "Example tools importable",
            len(EXAMPLE_TOOLS) > 0,
            f"Found {len(EXAMPLE_TOOLS)} example tools"
        )
        
        # Test a tool function
        from examples.template_agent.tools.example_tools import ping, calculate
        
        result = ping("test")
        all_passed &= print_test(
            "Ping tool works",
            result == "pong: test",
            f"Result: {result}"
        )
        
        result = calculate("2 + 2")
        all_passed &= print_test(
            "Calculate tool works",
            result == "4",
            f"Result: {result}"
        )
        
    except Exception as e:
        all_passed &= print_test("Tools configuration", False, str(e))
    
    print()
    
    # Test 5: App creation
    print(f"{YELLOW}5. Server Application{RESET}")
    try:
        from examples.template_agent.main import create_app
        app, agent_instance = create_app()
        
        all_passed &= print_test(
            "App created successfully",
            app is not None
        )
        
        all_passed &= print_test(
            "Agent instance available",
            agent_instance is not None
        )
        
        # Check if app has required routes
        # Note: Starlette apps have a routes attribute
        all_passed &= print_test(
            "App has routes",
            hasattr(app, 'routes') or hasattr(app, 'router')
        )
        
    except Exception as e:
        all_passed &= print_test("Server application", False, str(e))
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print(f"{GREEN}✓ ALL TESTS PASSED!{RESET}")
        print("\nThe Template Agent is ready to use!")
        print(f"\nTo run it: {YELLOW}python examples/template_agent/main.py{RESET}")
        if not any([
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
            os.getenv("GOOGLE_API_KEY")
        ]):
            print(f"\n{YELLOW}Note: Set an API key to enable LLM responses:{RESET}")
            print("  export OPENAI_API_KEY=sk-...")
            print("  export ANTHROPIC_API_KEY=sk-ant-...")
            print("  export GOOGLE_API_KEY=...")
    else:
        print(f"{RED}✗ Some tests failed{RESET}")
    print("="*60 + "\n")
    
    return all_passed

if __name__ == "__main__":
    # Skip startup checks during test
    os.environ["A2A_SKIP_STARTUP"] = "1"
    
    success = asyncio.run(test_template_agent())
    sys.exit(0 if success else 1)