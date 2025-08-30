"""
A2A Client for Inter-Agent Communication
Health Universe compatible client for A2A protocol communication.
"""

import json
import uuid
import os
import asyncio
from typing import Dict, Any, Optional, Union, List
import logging

import httpx
from a2a.types import Message, TextPart, DataPart


logger = logging.getLogger(__name__)


class A2AClient:
    """
    A2A-compliant client for inter-agent communication.
    Optimized for Health Universe deployment patterns.
    """
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize A2A client.
        
        Args:
            base_url: Base URL of the target agent
            timeout: Default timeout for requests
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        
        # Detect Health Universe endpoints
        self.is_health_universe = "healthuniverse.com" in base_url or "apps.healthuniverse.com" in base_url
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "A2A-Client/1.0"
            }
            
            # Add Health Universe specific headers if needed
            if self.is_health_universe:
                headers["Accept-Encoding"] = "gzip, deflate"
                
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=headers,
                follow_redirects=True
            )
            
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def send_message(self, message: str, timeout_sec: Optional[float] = None) -> Any:
        """
        Send a text message to the agent.
        
        Args:
            message: Text message to send
            timeout_sec: Optional timeout override
            
        Returns:
            Agent response
        """
        # Create proper A2A message structure
        a2a_message = Message(
            role="user",
            parts=[TextPart(kind="text", text=message)],
            messageId=str(uuid.uuid4()),
            kind="message"
        )
        
        return await self._send_a2a_request(a2a_message, timeout_sec)
    
    async def send_data(self, data: Union[Dict, List, Any], timeout_sec: Optional[float] = None) -> Any:
        """
        Send structured data to the agent.
        
        Args:
            data: Structured data to send
            timeout_sec: Optional timeout override
            
        Returns:
            Agent response
        """
        # Create proper A2A message structure with DataPart
        a2a_message = Message(
            role="user",
            parts=[DataPart(kind="data", data=data)],
            messageId=str(uuid.uuid4()),
            kind="message"
        )
        
        return await self._send_a2a_request(a2a_message, timeout_sec)
    
    async def send_a2a_message(self, message: Message, timeout_sec: Optional[float] = None) -> Any:
        """
        Send a complete A2A Message object.
        
        Args:
            message: A2A Message object
            timeout_sec: Optional timeout override
            
        Returns:
            Agent response
        """
        return await self._send_a2a_request(message, timeout_sec)
    
    async def _send_a2a_request(self, message: Message, timeout_sec: Optional[float] = None) -> Any:
        """
        Send A2A request using JSON-RPC protocol.
        
        Args:
            message: A2A Message object
            timeout_sec: Optional timeout override
            
        Returns:
            Parsed response
        """
        client = await self._get_client()
        
        # Update client timeout if specified
        if timeout_sec:
            client.timeout = httpx.Timeout(timeout_sec)
        
        # Create JSON-RPC request envelope
        request_payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": self._message_to_dict(message)
            },
            "id": str(uuid.uuid4())
        }
        
        try:
            logger.debug(f"Sending A2A request to {self.base_url}")
            
            # Health Universe typically uses root endpoint with JSON-RPC
            endpoint = self.base_url
            
            response = await client.post(
                endpoint,
                json=request_payload
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Handle JSON-RPC response
            if "error" in response_data:
                error = response_data["error"]
                raise Exception(f"A2A Error {error.get('code', 'unknown')}: {error.get('message', 'Unknown error')}")
            
            if "result" not in response_data:
                raise Exception("Invalid JSON-RPC response: missing result")
            
            result = response_data["result"]
            return self._parse_response(result)
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling {self.base_url}: {e}")
            if e.response.status_code == 403:
                raise Exception("Access forbidden - check if agent is public or authentication is required")
            elif e.response.status_code == 404:
                raise Exception("Agent endpoint not found - check URL and transport protocol")
            else:
                raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Request error calling {self.base_url}: {e}")
            raise Exception(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise Exception("Invalid JSON response from agent")
        except Exception as e:
            logger.error(f"Unexpected error calling {self.base_url}: {e}")
            raise
    
    def _message_to_dict(self, message: Message) -> Dict[str, Any]:
        """Convert Message object to dictionary."""
        # Handle both Pydantic models and regular objects
        if hasattr(message, 'model_dump'):
            return message.model_dump()
        elif hasattr(message, 'dict'):
            return message.dict()
        else:
            # Fallback for dictionary-like objects
            return {
                "role": getattr(message, 'role', 'user'),
                "parts": self._parts_to_dict(getattr(message, 'parts', [])),
                "messageId": getattr(message, 'messageId', str(uuid.uuid4())),
                "kind": "message"
            }
    
    def _parts_to_dict(self, parts: List[Any]) -> List[Dict[str, Any]]:
        """Convert Parts to dictionary format."""
        result = []
        
        for part in parts:
            if hasattr(part, 'model_dump'):
                result.append(part.model_dump())
            elif hasattr(part, 'dict'):
                result.append(part.dict())
            elif isinstance(part, dict):
                result.append(part)
            else:
                # Fallback - try to extract common attributes
                part_dict = {"kind": getattr(part, 'kind', 'text')}
                
                if hasattr(part, 'text'):
                    part_dict["text"] = part.text
                elif hasattr(part, 'data'):
                    part_dict["data"] = part.data
                elif hasattr(part, 'file'):
                    part_dict["file"] = part.file
                
                result.append(part_dict)
                
        return result
    
    def _parse_response(self, result: Any) -> Any:
        """
        Parse A2A response and extract relevant content.
        
        Args:
            result: Raw result from JSON-RPC response
            
        Returns:
            Parsed content
        """
        # Handle different response formats
        
        # Case 1: Direct Message response
        if isinstance(result, dict) and result.get("kind") == "message":
            return self._extract_message_content(result)
        
        # Case 2: Task response
        elif isinstance(result, dict) and result.get("kind") == "task":
            return self._extract_task_content(result)
        
        # Case 3: Direct string response
        elif isinstance(result, str):
            return result
        
        # Case 4: Raw dictionary - try to extract meaningful content
        elif isinstance(result, dict):
            # Look for message in status
            if "status" in result and isinstance(result["status"], dict):
                status_msg = result["status"].get("message")
                if status_msg:
                    return self._extract_message_content(status_msg)
            
            # Look for direct message field
            if "message" in result:
                return self._extract_message_content(result["message"])
            
            # Look for text field (simplified response)
            if "text" in result:
                return result["text"]
            
            # Return the whole thing if nothing else matches
            return result
        
        # Default: return as-is
        else:
            return result
    
    def _extract_message_content(self, message: Dict[str, Any]) -> Union[str, Dict, List]:
        """
        Extract content from A2A Message structure.
        
        Args:
            message: Message dictionary
            
        Returns:
            Extracted content
        """
        if not isinstance(message, dict):
            return str(message)
        
        parts = message.get("parts", [])
        if not parts:
            return ""
        
        # Extract content from all parts
        texts = []
        data_parts = []
        
        for part in parts:
            if not isinstance(part, dict):
                continue
                
            kind = part.get("kind")
            if kind == "text":
                text = part.get("text")
                if text:
                    texts.append(str(text))
            elif kind == "data":
                data = part.get("data")
                if data is not None:
                    data_parts.append(data)
        
        # Return structured data if present, otherwise text
        if data_parts:
            return data_parts[0] if len(data_parts) == 1 else data_parts
        elif texts:
            return "\n".join(texts)
        else:
            return ""
    
    def _extract_task_content(self, task: Dict[str, Any]) -> Union[str, Dict, List]:
        """
        Extract content from A2A Task structure.
        
        Args:
            task: Task dictionary
            
        Returns:
            Extracted content
        """
        # Try to get content from task status message
        status = task.get("status", {})
        if isinstance(status, dict):
            status_message = status.get("message")
            if status_message:
                return self._extract_message_content(status_message)
        
        # Try to get content from artifacts
        artifacts = task.get("artifacts", [])
        if artifacts:
            # Use first artifact
            artifact = artifacts[0]
            if isinstance(artifact, dict):
                parts = artifact.get("parts", [])
                if parts:
                    return self._extract_content_from_parts(parts)
        
        # Fallback to task itself
        return task
    
    def _extract_content_from_parts(self, parts: List[Dict[str, Any]]) -> Union[str, Dict, List]:
        """Extract content from parts list."""
        texts = []
        data_parts = []
        
        for part in parts:
            if not isinstance(part, dict):
                continue
                
            kind = part.get("kind")
            if kind == "text":
                text = part.get("text")
                if text:
                    texts.append(str(text))
            elif kind == "data":
                data = part.get("data")
                if data is not None:
                    data_parts.append(data)
        
        if data_parts:
            return data_parts[0] if len(data_parts) == 1 else data_parts
        elif texts:
            return "\n".join(texts)
        else:
            return ""
    
    @classmethod
    def from_registry(cls, agent_name: str, registry_path: str = "config/agents.json") -> "A2AClient":
        """
        Create A2A client from agent registry.
        
        Args:
            agent_name: Name of agent in registry
            registry_path: Path to agent registry file
            
        Returns:
            Configured A2AClient
            
        Raises:
            ValueError: If agent not found in registry
        """
        try:
            # Try to load registry
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
            else:
                raise ValueError(f"Agent registry not found: {registry_path}")
            
            # Look up agent
            agents = registry.get("agents", {})
            if agent_name not in agents:
                raise ValueError(f"Agent '{agent_name}' not found in registry")
            
            agent_config = agents[agent_name]
            agent_url = agent_config.get("url")
            
            if not agent_url:
                raise ValueError(f"No URL configured for agent '{agent_name}'")
            
            return cls(agent_url)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in registry file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load agent registry: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on target agent.
        
        Returns:
            Health status information
        """
        try:
            # Try to get agent card first
            client = await self._get_client()
            
            # Check well-known agent card endpoint
            card_url = f"{self.base_url}/.well-known/agentcard.json"
            response = await client.get(card_url)
            
            if response.status_code == 200:
                agent_card = response.json()
                return {
                    "status": "healthy",
                    "agent_name": agent_card.get("name", "Unknown"),
                    "protocol_version": agent_card.get("protocolVersion", "Unknown"),
                    "capabilities": agent_card.get("capabilities", {}),
                    "agent_card_accessible": True
                }
            else:
                return {
                    "status": "degraded",
                    "error": f"Agent card not accessible (HTTP {response.status_code})",
                    "agent_card_accessible": False
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "agent_card_accessible": False
            }