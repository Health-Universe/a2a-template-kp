"""
Startup utilities for A2A agents.
Provides self-checks, debug output, and health verification.
"""

import os
import json
import asyncio
import aiohttp
import httpx
import uuid
from typing import Optional
from utils.logging import get_logger

logger = get_logger(__name__)


async def check_http_endpoint(base_url: str, timeout: float = 5.0) -> bool:
    """
    Self-check that the advertised JSON-RPC endpoint is reachable.
    
    Args:
        base_url: Base URL to check (no /a2a/v1 for JSON-RPC)
        timeout: Timeout in seconds
        
    Returns:
        True if endpoint is reachable, False otherwise
    """
    endpoint = base_url.rstrip('/') + '/'  # JSON-RPC posts to root
    
    # Minimal valid JSON-RPC message for testing with correct method
    test_message = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "health_check"}],
                "messageId": "health-check-001",
                "kind": "message"
            },
            "metadata": {}
        },
        "id": 1
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json=test_message,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers={"Content-Type": "application/json"}
            ) as response:
                # For JSON-RPC, only 200 with valid response is healthy
                if response.status == 200:
                    try:
                        data = await response.json()
                        # Check for valid JSON-RPC response structure
                        if "jsonrpc" in data and data["jsonrpc"] == "2.0":
                            logger.info(f"✓ JSON-RPC endpoint reachable: {endpoint}")
                            return True
                        else:
                            logger.error(f"✗ Invalid JSON-RPC response from {endpoint}")
                            return False
                    except:
                        logger.error(f"✗ Non-JSON response from {endpoint}")
                        return False
                else:
                    logger.error(f"✗ JSON-RPC endpoint returned {response.status}: {endpoint}")
                    return False
    except asyncio.TimeoutError:
        logger.error(f"✗ JSON-RPC endpoint timeout after {timeout}s: {endpoint}")
        return False
    except Exception as e:
        logger.error(f"✗ JSON-RPC endpoint unreachable: {endpoint} - {e}")
        return False


def debug_agent_card(agent_card: dict, agent_name: str = "Agent") -> None:
    """
    Debug output for AgentCard at startup.
    
    Args:
        agent_card: AgentCard as dictionary
        agent_name: Name of the agent for logging
    """
    if os.getenv("A2A_DEBUG_CARD", "").lower() not in ("true", "1"):
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"AgentCard Debug for {agent_name}")
    logger.info(f"{'='*60}")
    
    # Pretty print the card
    card_json = json.dumps(agent_card, indent=2)
    for line in card_json.split("\n"):
        logger.info(f"  {line}")
    
    logger.info(f"{'='*60}\n")
    
    # Validate critical fields
    validations = []
    
    # Check protocol version (handle both camelCase and snake_case)
    if agent_card.get("protocolVersion") == "0.3.0" or agent_card.get("protocol_version") == "0.3.0":
        validations.append(("Protocol version", "✓", "0.3.0"))
    else:
        protocol_ver = agent_card.get("protocolVersion") or agent_card.get("protocol_version")
        validations.append(("Protocol version", "✗", protocol_ver or "missing"))
    
    # Check transport
    transport = agent_card.get("preferredTransport")
    if transport == "JSONRPC":
        validations.append(("Preferred transport", "✓", "JSONRPC"))
    elif transport == "HTTP":
        validations.append(("Preferred transport", "⚠", "HTTP (should be JSONRPC)"))
    else:
        validations.append(("Preferred transport", "?", transport or "missing"))
    
    # Check URL
    url = agent_card.get("url")
    if url and not url.endswith("/a2a"):
        validations.append(("Base URL", "✓", url))
    else:
        validations.append(("Base URL", "✗", f"{url} (should be root, not endpoint)"))
    
    # Check capabilities
    caps = agent_card.get("capabilities", {})
    if isinstance(caps, dict):
        validations.append(("Capabilities", "✓", f"streaming={caps.get('streaming')}, push={caps.get('push_notifications')}"))
    else:
        validations.append(("Capabilities", "✗", "missing or invalid"))
    
    # Print validation summary
    logger.info("Validation Summary:")
    for field, status, value in validations:
        logger.info(f"  {status} {field}: {value}")
    logger.info("")


async def startup_checks(agent) -> None:
    """
    Run all startup checks for an agent.
    
    Args:
        agent: The A2A agent instance
    """
    # Debug card if enabled
    if hasattr(agent, 'create_agent_card'):
        try:
            card = agent.create_agent_card()
            # Convert to dict for debug output
            card_dict = card.dict() if hasattr(card, 'dict') else card.__dict__
            debug_agent_card(card_dict, agent.get_agent_name())
            
            # Self-check HTTP endpoint if enabled
            if os.getenv("A2A_STARTUP_CHECK", "true").lower() == "true":
                base_url = card_dict.get("url")
                if base_url:
                    await check_http_endpoint(base_url)
        except Exception as e:
            logger.error(f"Startup checks failed: {e}")


def probe_jsonrpc(base_url: str, timeout: float = 2.0) -> bool:
    """
    Synchronous JSON-RPC probe for health checks.
    
    Args:
        base_url: Base URL to probe
        timeout: Timeout in seconds
        
    Returns:
        True if endpoint responds correctly, False otherwise
    """
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "ping"}],
                "kind": "message",
                "messageId": str(uuid.uuid4()),
            },
            "metadata": {},
        },
        "id": 1,
    }
    try:
        r = httpx.post(base_url.rstrip('/') + '/', json=payload, timeout=timeout)
        ok = r.status_code == 200 and r.json().get("result") is not None
        if ok:
            logger.info(f"✓ JSON-RPC endpoint reachable: {base_url}")
        else:
            logger.error(f"✗ JSON-RPC endpoint check failed: {base_url}")
        return ok
    except Exception as e:
        logger.error(f"✗ JSON-RPC endpoint unreachable: {base_url} - {e}")
        return False


def run_startup_checks(agent) -> None:
    """
    Synchronous wrapper for startup checks.
    
    Args:
        agent: The A2A agent instance
    """
    # Debug card if enabled
    if hasattr(agent, 'create_agent_card'):
        try:
            card = agent.create_agent_card()
            # Convert to dict for debug output
            card_dict = card.dict() if hasattr(card, 'dict') else card.__dict__
            debug_agent_card(card_dict, agent.get_agent_name())
            
            # Sync probe if enabled
            if os.getenv("A2A_STARTUP_CHECK", "true").lower() == "true":
                base_url = card_dict.get("url")
                if base_url:
                    probe_jsonrpc(base_url)
        except Exception as e:
            logger.error(f"Startup checks failed: {e}")