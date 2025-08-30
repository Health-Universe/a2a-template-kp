"""
Example tools for the Template Agent.
Shows how to add custom tools that the LLM can use.
"""

from datetime import datetime
from typing import Dict, Any
import json

# Note: We'll use langchain-style tools for compatibility with A2A framework
# The Google ADK FunctionTool would also work if using ADK-based agents

def ping(value: str) -> str:
    """
    Simple echo function for testing.
    
    Args:
        value: String to echo back
        
    Returns:
        Echoed string with 'pong:' prefix
    """
    return f"pong: {value}"


def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get the current time in the specified format.
    
    Args:
        format: strftime format string (default: YYYY-MM-DD HH:MM:SS)
        
    Returns:
        Current time as formatted string
    """
    return datetime.now().strftime(format)


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2")
        
    Returns:
        Result of the calculation as a string
    """
    try:
        # Only allow safe math operations
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow, 'len': len
        }
        # Restrict to basic math operations
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def json_query(data_json: str, query_path: str) -> str:
    """
    Query a JSON object using a simple path notation.
    
    Args:
        data_json: JSON string to query
        query_path: Dot-notation path (e.g., "user.name" or "items.0.price")
        
    Returns:
        Value at the path or error message
    """
    try:
        data = json.loads(data_json)
        
        # Navigate the path
        parts = query_path.split('.')
        current = data
        
        for part in parts:
            if part.isdigit():
                # Array index
                current = current[int(part)]
            else:
                # Object key
                current = current[part]
        
        return json.dumps(current) if isinstance(current, (dict, list)) else str(current)
    except Exception as e:
        return f"Query error: {str(e)}"


def word_count(text: str) -> str:
    """
    Count words in the provided text.
    
    Args:
        text: Text to count words in
        
    Returns:
        Number of words as string
    """
    words = text.split()
    return f"{len(words)} words"


# If using Google ADK FunctionTool (uncomment if needed):
# from google.adk.tools import FunctionTool
# EXAMPLE_TOOLS = [
#     FunctionTool(ping),
#     FunctionTool(get_current_time),
#     FunctionTool(calculate),
#     FunctionTool(json_query),
#     FunctionTool(word_count)
# ]

# For langchain compatibility (if using langchain tools):
def create_langchain_tools():
    """Create langchain-compatible tools from our functions."""
    from langchain_core.tools import Tool
    
    return [
        Tool(
            name="ping",
            description="Echo back a value with 'pong:' prefix",
            func=ping
        ),
        Tool(
            name="get_current_time",
            description="Get the current date and time",
            func=get_current_time
        ),
        Tool(
            name="calculate",
            description="Evaluate a mathematical expression",
            func=calculate
        ),
        Tool(
            name="json_query",
            description="Query a JSON object using dot notation path",
            func=json_query
        ),
        Tool(
            name="word_count",
            description="Count the number of words in text",
            func=word_count
        )
    ]

# Export a simple list for easy import
EXAMPLE_TOOLS = [
    ping,
    get_current_time,
    calculate,
    json_query,
    word_count
]