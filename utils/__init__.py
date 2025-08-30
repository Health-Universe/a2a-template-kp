"""
Production-ready utility modules for A2A agent development.
Phase 1 refactoring - reduced from 2600+ to ~420 LOC with enhanced features.
"""

from .registry import load_registry, resolve_agent_url, clear_cache
from .a2a_client import A2AClient, call_agent
from .llm_utils import LLMProvider, generate_text, generate_json, create_llm_agent
from .logging import setup_logging, get_logger, reset_logging

# Legacy aliases for backward compatibility
A2AAgentClient = A2AClient  # Old name -> new name
AgentRegistry = None  # No longer needed with JSON-RPC

__all__ = [
    # Registry functions
    "load_registry",
    "resolve_agent_url", 
    "clear_cache",
    
    # A2A client
    "A2AClient",
    "A2AAgentClient",  # Legacy name
    "AgentRegistry",   # Legacy compatibility
    "call_agent",
    
    # LLM utilities
    "LLMProvider",
    "generate_text",
    "generate_json",
    "create_llm_agent",
    
    # Logging
    "setup_logging",
    "get_logger",
    "reset_logging"
]

__version__ = "2.0.0"