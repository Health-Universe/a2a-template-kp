"""
Simple agent configuration loader for A2A agents.
~45 LOC - Focused on loading and validating agent registries.
"""

import os
import json
from typing import Dict, Any
from functools import lru_cache


_registry_cache = {}
REG_DEFAULT = os.getenv("AGENT_REGISTRY_PATH", "config/agents.json")


def load_registry(registry_path: str = None) -> Dict[str, Any]:
    """
    Load agent registry from file.
    
    Args:
        registry_path: Path to registry JSON file (default: config/agents.json)
        
    Returns:
        Dictionary of agent configurations
        
    Raises:
        ValueError: If registry is invalid or missing
    """
    registry_path = registry_path or REG_DEFAULT
    
    if registry_path in _registry_cache:
        return _registry_cache[registry_path]
    
    if not os.path.exists(registry_path):
        raise ValueError(f"Registry not found: {registry_path}")
    
    with open(registry_path, 'r') as f:
        data = json.load(f)
    
    agents = data.get("agents")
    
    # Allow optional empty registry for local development
    allow_empty = os.getenv("ALLOW_EMPTY_REGISTRY") == "1"
    if not isinstance(agents, dict):
        raise ValueError(f"{registry_path} must include an 'agents' object (dict)")
    if not agents and not allow_empty:
        raise ValueError(f"{registry_path} has empty 'agents' object. Set ALLOW_EMPTY_REGISTRY=1 to allow.")
    
    # Normalize URLs (strip trailing slashes)
    for agent_name, agent_config in agents.items():
        if "url" in agent_config and isinstance(agent_config["url"], str):
            agent_config["url"] = agent_config["url"].rstrip("/")
    
    _registry_cache[registry_path] = agents
    return agents


@lru_cache(maxsize=128)
def resolve_agent_url(agent_name_or_url: str, registry_path: str = None) -> str:
    """
    Resolve agent URL from registry or pass through raw URLs.
    
    Args:
        agent_name_or_url: Agent name to resolve or direct URL
        registry_path: Optional path to registry file
        
    Returns:
        Agent URL (normalized without trailing slash)
        
    Raises:
        ValueError: If agent not found or missing URL
    """
    # Pass through raw URLs directly
    if agent_name_or_url.startswith(("http://", "https://")):
        return agent_name_or_url.rstrip("/")
    
    # Otherwise resolve from registry
    agents = load_registry(registry_path)
    
    if agent_name_or_url not in agents:
        raise ValueError(f"Agent '{agent_name_or_url}' not found in registry")
    
    if "url" not in agents[agent_name_or_url]:
        raise ValueError(f"Agent '{agent_name_or_url}' missing 'url' in registry")
    
    return agents[agent_name_or_url]["url"]


def clear_cache():
    """Clear all cached registry data."""
    global _registry_cache
    _registry_cache.clear()
    resolve_agent_url.cache_clear()