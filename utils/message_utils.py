"""
A2A Message Utilities
Enhanced utilities for creating and handling A2A protocol messages.
Ensures proper Part type usage and Health Universe compatibility.
"""

import json
import uuid
from typing import Dict, Any, List, Union, Optional

from a2a.types import Message, TextPart, DataPart, FilePart, Part


def create_text_part(text: str, metadata: Optional[Dict[str, Any]] = None) -> TextPart:
    """
    Create a TextPart with proper A2A compliance.
    
    Args:
        text: Text content
        metadata: Optional metadata for the part
        
    Returns:
        TextPart with kind discriminator
    """
    part_data = {
        "kind": "text",
        "text": text
    }
    
    if metadata:
        part_data["metadata"] = metadata
        
    return TextPart(**part_data)


def create_data_part(data: Union[Dict, List, Any], metadata: Optional[Dict[str, Any]] = None) -> DataPart:
    """
    Create a DataPart with proper A2A compliance.
    
    Args:
        data: Structured data (dict, list, or JSON-serializable object)
        metadata: Optional metadata for the part
        
    Returns:
        DataPart with kind discriminator
    """
    part_data = {
        "kind": "data",
        "data": data
    }
    
    if metadata:
        part_data["metadata"] = metadata
        
    return DataPart(**part_data)


def create_file_part(
    file_uri: Optional[str] = None,
    file_bytes: Optional[str] = None,
    name: Optional[str] = None,
    mime_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> FilePart:
    """
    Create a FilePart with proper A2A compliance.
    
    Args:
        file_uri: URI pointing to file content
        file_bytes: Base64-encoded file content
        name: Optional file name
        mime_type: Optional MIME type
        metadata: Optional metadata for the part
        
    Returns:
        FilePart with kind discriminator
        
    Raises:
        ValueError: If neither file_uri nor file_bytes provided
    """
    if not file_uri and not file_bytes:
        raise ValueError("Either file_uri or file_bytes must be provided")
    
    if file_uri and file_bytes:
        raise ValueError("Cannot provide both file_uri and file_bytes")
    
    file_data = {}
    if name:
        file_data["name"] = name
    if mime_type:
        file_data["mimeType"] = mime_type
    
    if file_uri:
        file_data["uri"] = file_uri
    else:
        file_data["bytes"] = file_bytes
    
    part_data = {
        "kind": "file",
        "file": file_data
    }
    
    if metadata:
        part_data["metadata"] = metadata
        
    return FilePart(**part_data)


def create_message_parts(content: Union[str, Dict, List, Any]) -> List[Part]:
    """
    Create appropriate Parts based on content type.
    
    Args:
        content: Content to convert to Parts
        
    Returns:
        List of Parts with proper discriminators
    """
    if isinstance(content, str):
        return [create_text_part(content)]
    elif isinstance(content, (dict, list)):
        return [create_data_part(content)]
    else:
        # Convert to string as fallback
        return [create_text_part(str(content))]


def create_agent_message(
    content: Union[str, Dict, List, Any],
    role: str = "agent",
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
    message_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    extensions: Optional[List[str]] = None,
    reference_task_ids: Optional[List[str]] = None
) -> Message:
    """
    Create a complete A2A Message with proper Part types.
    
    Args:
        content: Message content (auto-converted to appropriate Parts)
        role: Message role ("user" or "agent")
        task_id: Optional task ID
        context_id: Optional context ID
        message_id: Optional message ID (auto-generated if not provided)
        metadata: Optional message metadata
        extensions: Optional extension URIs
        reference_task_ids: Optional referenced task IDs
        
    Returns:
        Complete Message object with proper A2A compliance
    """
    if not message_id:
        message_id = str(uuid.uuid4())
    
    parts = create_message_parts(content)
    
    message_data = {
        "role": role,
        "parts": parts,
        "messageId": message_id,
        "kind": "message"
    }
    
    if task_id:
        message_data["taskId"] = task_id
    if context_id:
        message_data["contextId"] = context_id
    if metadata:
        message_data["metadata"] = metadata
    if extensions:
        message_data["extensions"] = extensions
    if reference_task_ids:
        message_data["referenceTaskIds"] = reference_task_ids
    
    return Message(**message_data)


def new_agent_text_message(
    text: str,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
    message_id: Optional[str] = None
) -> Message:
    """
    Create a simple agent text message.
    
    Args:
        text: Text content
        task_id: Optional task ID
        context_id: Optional context ID
        message_id: Optional message ID (auto-generated if not provided)
        
    Returns:
        Message with TextPart
    """
    return create_agent_message(
        content=text,
        role="agent",
        task_id=task_id,
        context_id=context_id,
        message_id=message_id
    )


def new_user_text_message(
    text: str,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
    message_id: Optional[str] = None
) -> Message:
    """
    Create a simple user text message.
    
    Args:
        text: Text content
        task_id: Optional task ID
        context_id: Optional context ID
        message_id: Optional message ID (auto-generated if not provided)
        
    Returns:
        Message with TextPart
    """
    return create_agent_message(
        content=text,
        role="user",
        task_id=task_id,
        context_id=context_id,
        message_id=message_id
    )


def extract_content_from_parts(parts: List[Union[Dict, Part]]) -> Union[str, Dict, List]:
    """
    Extract content from A2A Parts with proper type handling.
    
    Args:
        parts: List of Parts (dict or Part objects)
        
    Returns:
        Extracted content - string for text, dict/list for data
    """
    texts = []
    data_parts = []
    file_refs = []
    
    for part in parts:
        # Handle both dict and object formats
        if isinstance(part, dict):
            kind = part.get("kind")
            if kind == "text":
                text = part.get("text")
                if text:
                    texts.append(str(text))
            elif kind == "data":
                data = part.get("data")
                if data is not None:
                    data_parts.append(data)
            elif kind == "file":
                file_obj = part.get("file", {})
                name = file_obj.get("name", "file")
                if "uri" in file_obj:
                    file_refs.append(f"[{name}] {file_obj['uri']}")
                elif "bytes" in file_obj:
                    file_refs.append(f"[{name}] (binary data)")
        else:
            # Handle Part objects
            if hasattr(part, "kind"):
                if part.kind == "text" and hasattr(part, "text"):
                    texts.append(str(part.text))
                elif part.kind == "data" and hasattr(part, "data"):
                    data_parts.append(part.data)
                elif part.kind == "file" and hasattr(part, "file"):
                    file_obj = part.file
                    name = getattr(file_obj, "name", "file")
                    if hasattr(file_obj, "uri"):
                        file_refs.append(f"[{name}] {file_obj.uri}")
                    elif hasattr(file_obj, "bytes"):
                        file_refs.append(f"[{name}] (binary data)")
    
    # Return based on what we found
    if data_parts:
        # If there's structured data, return it (prefer first if multiple)
        return data_parts[0] if len(data_parts) == 1 else data_parts
    elif texts or file_refs:
        # Return text content
        all_text = texts + file_refs
        return "\n".join(all_text)
    else:
        # Nothing found
        return ""


def extract_text_from_parts(parts: List[Union[Dict, Part]]) -> str:
    """
    Extract only text content from Parts.
    
    Args:
        parts: List of Parts
        
    Returns:
        Concatenated text content
    """
    texts = []
    
    for part in parts:
        if isinstance(part, dict):
            if part.get("kind") == "text":
                text = part.get("text")
                if text:
                    texts.append(str(text))
        else:
            if hasattr(part, "kind") and part.kind == "text" and hasattr(part, "text"):
                texts.append(str(part.text))
    
    return "\n".join(texts)


def extract_data_from_parts(parts: List[Union[Dict, Part]]) -> List[Any]:
    """
    Extract only data content from Parts.
    
    Args:
        parts: List of Parts
        
    Returns:
        List of data objects
    """
    data_parts = []
    
    for part in parts:
        if isinstance(part, dict):
            if part.get("kind") == "data":
                data = part.get("data")
                if data is not None:
                    data_parts.append(data)
        else:
            if hasattr(part, "kind") and part.kind == "data" and hasattr(part, "data"):
                data_parts.append(part.data)
    
    return data_parts


def is_structured_response(content: Any) -> bool:
    """
    Check if content should be sent as DataPart.
    
    Args:
        content: Content to check
        
    Returns:
        True if content should use DataPart, False for TextPart
    """
    return isinstance(content, (dict, list))


def format_for_agent_response(content: Any) -> Union[str, Dict, List]:
    """
    Format content appropriately for agent responses.
    
    Args:
        content: Raw content
        
    Returns:
        Properly formatted content for A2A response
    """
    if isinstance(content, (dict, list)):
        return content  # Will be wrapped in DataPart
    else:
        return str(content)  # Will be wrapped in TextPart


def create_error_message(
    error: str,
    error_code: Optional[str] = None,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None
) -> Message:
    """
    Create an error message in A2A format.
    
    Args:
        error: Error description
        error_code: Optional error code
        task_id: Optional task ID
        context_id: Optional context ID
        
    Returns:
        Error message with appropriate structure
    """
    error_data = {"error": error}
    if error_code:
        error_data["code"] = error_code
    
    return create_agent_message(
        content=error_data,
        role="agent",
        task_id=task_id,
        context_id=context_id
    )


def validate_message_structure(message: Dict[str, Any]) -> bool:
    """
    Validate A2A message structure.
    
    Args:
        message: Message to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check required fields
    required_fields = ["role", "parts", "messageId", "kind"]
    for field in required_fields:
        if field not in message:
            return False
    
    # Validate role
    if message["role"] not in ["user", "agent"]:
        return False
    
    # Validate kind
    if message["kind"] != "message":
        return False
    
    # Validate parts
    parts = message.get("parts", [])
    if not isinstance(parts, list) or not parts:
        return False
    
    # Validate each part has kind discriminator
    for part in parts:
        if not isinstance(part, dict) or "kind" not in part:
            return False
        if part["kind"] not in ["text", "data", "file"]:
            return False
    
    return True