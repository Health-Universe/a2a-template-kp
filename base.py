"""
Full-featured A2A-compliant base class for agents.
Enhanced implementation following A2A protocol specification v0.3.0.
Optimized for Health Universe deployment with production-ready features.
"""

import os
import json
import logging
import time
from typing import Optional, List, Dict, Any, Union
from abc import ABC, abstractmethod
from collections import defaultdict

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCard,
    AgentProvider,
    AgentCapabilities,
    AgentSkill,
    TextPart,
    Part,
    TaskState,
    DataPart,
    Task,
    TaskStatus,
    Message
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError, InvalidParamsError

# Rate limiting for legacy warnings
_legacy_warnings = defaultdict(lambda: {"count": 0, "last_warn": 0})
_MAX_LEGACY_WARNINGS = 10  # Max warnings per type
_LEGACY_WARN_INTERVAL = 60  # Seconds between warnings


class A2AAgent(AgentExecutor, ABC):
    """
    A2A-compliant base class for all agents.
    
    This enhanced implementation follows the A2A specification v0.3.0:
    - Full protocol compliance with Health Universe optimization
    - Agents declare capabilities (name, description, tools)
    - The A2A framework handles execution, LLM integration, and task management
    - Enhanced error handling and streaming support
    
    Subclasses must implement:
    - get_agent_name()
    - get_agent_description()
    - process_message()
    
    Optional implementations:
    - get_tools() - for tool-based agents
    - get_system_instruction() - for custom LLM instructions
    - get_agent_skills() - for detailed capability declaration
    """
    
    def __init__(self):
        """Initialize the agent with logging and optional startup checks."""
        self.logger = self._setup_logging()
        self._current_task_id = None
        
        # Run startup checks if not disabled
        if os.getenv("A2A_SKIP_STARTUP", "").lower() not in ("true", "1"):
            self._run_startup_checks()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for the agent."""
        logger_name = self.__class__.__name__
        logger = logging.getLogger(logger_name)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _run_startup_checks(self):
        """Run basic startup validation."""
        try:
            # Validate agent metadata
            name = self.get_agent_name()
            description = self.get_agent_description()
            
            if not name or not description:
                self.logger.warning("Agent missing required metadata")
            
            # Check for LLM availability if tools are provided
            if self.get_tools():
                api_keys = [
                    os.getenv("GOOGLE_API_KEY"),
                    os.getenv("OPENAI_API_KEY"), 
                    os.getenv("ANTHROPIC_API_KEY")
                ]
                if not any(api_keys):
                    self.logger.warning("No LLM API keys found - tool agents may fail")
                    
        except Exception as e:
            self.logger.warning(f"Startup check failed: {e}")
    
    def create_agent_card(self) -> AgentCard:
        """
        Create the AgentCard for agent discovery.
        Fully compliant with A2A specification v0.3.0 and Health Universe.
        
        Returns:
            AgentCard with agent metadata and capabilities
        """
        # Get base URL - Health Universe specific detection
        base_url = self._get_base_url()
        
        return AgentCard(
            # Required fields per A2A spec
            protocolVersion="0.3.0",
            name=self.get_agent_name(),
            description=self.get_agent_description(),
            version=self.get_agent_version(),
            url=base_url,
            
            # Transport configuration - Health Universe uses HTTP with JSON-RPC
            preferredTransport="HTTP",
            additionalInterfaces=[
                {
                    "url": base_url,
                    "transport": "HTTP"
                },
                {
                    "url": base_url,
                    "transport": "JSONRPC"
                }
            ],
            
            # Provider information
            provider=AgentProvider(
                organization="Health Universe",
                url="https://healthuniverse.com"
            ),
            
            # Capabilities
            capabilities=AgentCapabilities(
                streaming=self.supports_streaming(),
                pushNotifications=self.supports_push_notifications(),
                stateTransitionHistory=True
            ),
            
            # Skills (required for proper agent discovery)
            skills=self.get_agent_skills(),
            
            # Security (can be extended as needed)
            securitySchemes={},
            security=[],
            
            # Input/output modes
            defaultInputModes=["text/plain", "application/json"],
            defaultOutputModes=["text/plain", "application/json"]
        )
    
    def _get_base_url(self) -> str:
        """Get base URL with Health Universe detection."""
        # Health Universe URL pattern
        hu_url = os.getenv("HU_APP_URL")
        if hu_url:
            return hu_url
            
        # Alternative environment variables
        base_url = os.getenv("A2A_BASE_URL")
        if base_url:
            return base_url
            
        # Default for local development
        port = os.getenv("PORT", "8000")
        return f"http://localhost:{port}"
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute agent logic - A2A compliant implementation with Health Universe optimization.
        
        The A2A framework handles:
        - Task management
        - LLM integration
        - Tool orchestration
        - Streaming and artifacts
        
        We extract the message, process it, and return the response with proper Part types.
        
        Args:
            context: Request context with message and metadata
            event_queue: Queue for sending events/responses
        """
        task = None
        try:
            # Extract message from A2A protocol format
            message = self._extract_message(context)
            
            if not message:
                raise InvalidParamsError("No message provided in request")
            
            # Get or create task for proper lifecycle management
            task = context.current_task
            if not task:
                task = new_task(context.message or new_agent_text_message("Processing..."))
                await event_queue.enqueue_event(task)
            
            self._current_task_id = task.id
            
            # Create TaskUpdater for status updates (spec-compliant)
            context_id = getattr(task, "contextId", None) or getattr(task, "context_id", None) or task.id
            updater = TaskUpdater(event_queue, task.id, context_id)
            
            # Signal task is being worked on (spec state: "working")
            if self.supports_streaming():
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"🔨 {self.get_agent_name()}: Processing your request...")
                )
            
            # Log if debug mode is enabled
            if os.getenv("SHOW_AGENT_CALLS", "false").lower() == "true":
                self.logger.info(f"🔨 {self.get_agent_name()}: Processing message ({len(message)} chars)")
            
            # Process message through agent's business logic
            response = await self.process_message(message)
            
            # Create appropriate message type based on response
            # CRITICAL: Proper DataPart vs TextPart handling for Health Universe
            if isinstance(response, (dict, list)):
                # Structured data should use DataPart
                response_message = Message(
                    role="agent",
                    parts=[DataPart(kind="data", data=response)],
                    messageId=f"response-{task.id}",
                    taskId=task.id,
                    contextId=context_id,
                    kind="message"
                )
            else:
                # String responses use TextPart
                response_message = new_agent_text_message(str(response))
            
            # Mark task as completed with the response attached
            await updater.update_status(
                TaskState.completed,
                response_message
            )
            
        except ServerError as e:
            # A2A SDK errors are already properly formatted
            if task:
                context_id = getattr(task, "contextId", None) or getattr(task, "context_id", None) or task.id
                updater = TaskUpdater(event_queue, task.id, context_id)
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(f"Task failed: {str(e)}")
                )
            raise
        except Exception as e:
            # Handle unexpected errors
            self.logger.error(f"Error processing message: {str(e)}")
            if task:
                context_id = getattr(task, "contextId", None) or getattr(task, "context_id", None) or task.id
                updater = TaskUpdater(event_queue, task.id, context_id)
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(f"Task failed: {str(e)}")
                )
            raise ServerError(error=e)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Handle cancellation requests per A2A specification.
        Attempts cooperative cancellation and returns Task with canceled state.
        """
        # Extract task ID from context
        task_id = self._extract_task_id(context)
        
        if not task_id:
            from a2a.utils.errors import InvalidParamsError
            raise ServerError(error=InvalidParamsError("No task ID provided for cancellation"))
        
        try:
            self.logger.info(f"Attempting to cancel task {task_id}")
            
            # Get consistent context ID for all operations
            ctx_id = getattr(context, 'contextId', getattr(context, 'context_id', task_id))
            
            # Create updater for the task
            updater = TaskUpdater(event_queue, task_id, ctx_id)
            
            # Signal task is being canceled
            await updater.update_status(
                TaskState.canceled,
                new_agent_text_message(f"Task {task_id} has been canceled")
            )
            
            # Emit a Task event with canceled state
            canceled_task = Task(
                id=task_id,
                contextId=ctx_id,
                status=TaskStatus(
                    state=TaskState.canceled,
                    message=Message(
                        role="agent",
                        parts=[TextPart(kind="text", text="Task canceled by user request")],
                        messageId=f"cancel-{task_id}",
                        taskId=task_id,
                        contextId=ctx_id,
                        kind="message"
                    )
                ),
                kind="task"
            )
            await event_queue.enqueue_event(canceled_task)
            
            self.logger.info(f"Task {task_id} canceled successfully")
            
        except Exception as e:
            self.logger.error(f"Error canceling task {task_id}: {str(e)}")
            ctx_id = getattr(context, 'contextId', getattr(context, 'context_id', task_id))
            failed_task = Task(
                id=task_id,
                contextId=ctx_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    message=Message(
                        role="agent",
                        parts=[TextPart(kind="text", text=f"Cancellation failed: {str(e)}")],
                        messageId=f"cancel-failed-{task_id}",
                        taskId=task_id,
                        contextId=ctx_id,
                        kind="message"
                    )
                ),
                kind="task"
            )
            await event_queue.enqueue_event(failed_task)
            raise ServerError(error=e)
    
    def _extract_task_id(self, context: RequestContext) -> Optional[str]:
        """Extract task ID from context (multiple sources)."""
        # First check current_task if present
        if hasattr(context, 'current_task') and context.current_task:
            return context.current_task.id
        # Then check task_id attribute
        elif hasattr(context, 'task_id'):
            return context.task_id
        # Check metadata
        elif context.metadata and 'task_id' in context.metadata:
            return context.metadata['task_id']
        # Fall back to stored task ID
        elif self._current_task_id:
            return self._current_task_id
        
        return None
    
    def _log_legacy_warning(self, legacy_type: str) -> None:
        """Log rate-limited warnings for legacy Part formats."""
        global _legacy_warnings
        
        warning_info = _legacy_warnings[legacy_type]
        current_time = time.time()
        
        # Check if we should log this warning
        should_warn = (
            warning_info["count"] < _MAX_LEGACY_WARNINGS and
            current_time - warning_info["last_warn"] > _LEGACY_WARN_INTERVAL
        )
        
        if should_warn:
            warning_info["count"] += 1
            warning_info["last_warn"] = current_time
            
            self.logger.warning(
                f"Legacy Part format detected: {legacy_type} "
                f"({warning_info['count']}/{_MAX_LEGACY_WARNINGS} warnings)"
            )
    
    def _extract_message(self, context: RequestContext) -> Optional[str]:
        """
        Extract message from A2A protocol format.
        Spec-compliant parsing that handles Part as discriminated union by 'kind'.
        
        Per A2A spec, Part is a union with kind: "text" | "file" | "data"
        We must branch on part["kind"] and handle TextPart, DataPart, and FilePart.
        
        Args:
            context: Request context containing message
            
        Returns:
            Extracted message as string or None
        """
        if not context.message or not context.message.parts:
            return None
        
        extracted = []
        
        for part in context.message.parts:
            # Handle both dict-like and object-like parts defensively
            # The spec defines Part as discriminated union by 'kind'
            
            # Try to get kind from part (handles both dict and object)
            kind = None
            if isinstance(part, dict):
                kind = part.get("kind")
            elif hasattr(part, "kind"):
                kind = part.kind
            
            if kind == "text":
                # TextPart: extract text field
                text = None
                if isinstance(part, dict):
                    text = part.get("text")
                elif hasattr(part, "text"):
                    text = part.text
                if text is not None:
                    extracted.append(str(text))
                    
            elif kind == "data":
                # DataPart: serialize data as JSON
                data = None
                if isinstance(part, dict):
                    data = part.get("data")
                elif hasattr(part, "data"):
                    data = part.data
                if data is not None:
                    if isinstance(data, (dict, list)):
                        extracted.append(json.dumps(data))
                    else:
                        extracted.append(str(data))
                        
            elif kind == "file":
                # FilePart: handle file with uri or bytes
                file_obj = None
                if isinstance(part, dict):
                    file_obj = part.get("file")
                elif hasattr(part, "file"):
                    file_obj = part.file
                    
                if file_obj:
                    # Prefer URI if present
                    if isinstance(file_obj, dict):
                        name = file_obj.get("name", "unnamed")
                        if "uri" in file_obj:
                            extracted.append(f"[file:{name}] {file_obj['uri']}")
                        elif "bytes" in file_obj:
                            extracted.append(f"[file-bytes:{name}] (binary data)")
                    elif hasattr(file_obj, "uri"):
                        name = getattr(file_obj, "name", "unnamed")
                        extracted.append(f"[file:{name}] {file_obj.uri}")
                    elif hasattr(file_obj, "bytes"):
                        name = getattr(file_obj, "name", "unnamed")
                        extracted.append(f"[file-bytes:{name}] (binary data)")
                        
            else:
                # Fallback for legacy/malformed parts with rate-limited warning
                handled = False
                legacy_type = None
                
                # Check for root.text pattern (legacy)
                if hasattr(part, 'root'):
                    if isinstance(part.root, TextPart):
                        extracted.append(part.root.text)
                        handled = True
                        legacy_type = "root.text"
                    elif isinstance(part.root, DataPart):
                        data = part.root.data
                        if isinstance(data, (dict, list)):
                            extracted.append(json.dumps(data))
                        else:
                            extracted.append(str(data))
                        handled = True
                        legacy_type = "root.data"
                
                # Direct text attribute (legacy)
                elif hasattr(part, "text"):
                    extracted.append(str(part.text))
                    handled = True
                    legacy_type = "direct.text"
                    
                # Direct data attribute (legacy)
                elif hasattr(part, "data"):
                    data = part.data
                    if isinstance(data, (dict, list)):
                        extracted.append(json.dumps(data))
                    else:
                        extracted.append(str(data))
                    handled = True
                    legacy_type = "direct.data"
                
                if handled and legacy_type:
                    # Rate-limited warning only if enabled
                    if os.getenv("A2A_WARN_LEGACY_PARTS", "true").lower() == "true":
                        self._log_legacy_warning(legacy_type)
        
        if not extracted:
            return None
        
        # Join all parts with newline
        return "\n".join(extracted)
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    def get_agent_name(self) -> str:
        """Return the agent's name for the AgentCard."""
        pass
    
    @abstractmethod
    def get_agent_description(self) -> str:
        """Return the agent's description for the AgentCard."""
        pass
    
    @abstractmethod
    async def process_message(self, message: str) -> Union[str, Dict[str, Any], List[Any]]:
        """
        Process an incoming message and return a response.
        
        Args:
            message: The extracted text message to process
            
        Returns:
            Response as string (TextPart), dict/list (DataPart), or other structured data
        """
        pass
    
    # Optional methods that subclasses can override
    
    def get_agent_version(self) -> str:
        """Return agent version. Override for custom versioning."""
        return "1.0.0"
    
    def get_system_instruction(self) -> str:
        """
        Return system instruction for LLM.
        Override to provide custom instructions.
        """
        return "You are a helpful AI assistant."
    
    def get_tools(self) -> List:
        """
        Return list of tools for the agent.
        Override to provide tool-based functionality.
        
        Tools should be compatible with the LLM provider being used.
        The A2A framework will handle tool execution.
        
        Returns:
            List of tools or empty list for no tools
        """
        return []
    
    def get_agent_skills(self) -> List[AgentSkill]:
        """
        Return list of skills for the AgentCard.
        Override to declare specific capabilities.
        
        Each skill should define its own inputModes/outputModes if they
        differ from the agent's defaults.
        
        Returns:
            List of AgentSkill objects or empty list
        """
        return []
    
    def supports_streaming(self) -> bool:
        """
        Whether this agent supports streaming responses.
        Override to enable streaming support.
        
        If you return True here, you MUST implement streaming in execute()
        and send TaskStatusUpdateEvent/TaskArtifactUpdateEvent payloads.
        """
        return False
    
    def supports_push_notifications(self) -> bool:
        """
        Whether this agent supports push notifications.
        Override to enable push notification support.
        """
        return False
    
    # Utility methods for inter-agent communication
    
    async def call_other_agent(self, agent_name_or_url: str, message: str, timeout: float = 30.0) -> str:
        """
        Call another A2A-compliant agent.
        
        This utility method helps agents communicate with each other
        following the A2A protocol.
        
        Args:
            agent_name_or_url: Agent name (from registry) or direct URL
            message: Message to send to the other agent
            timeout: Request timeout in seconds
            
        Returns:
            Response from the other agent as string
            
        Raises:
            Exception: If agent communication fails
        """
        try:
            # Import here to avoid circular imports
            from utils.a2a_client import A2AClient
            
            # Create client based on input type
            if agent_name_or_url.startswith(('http://', 'https://')):
                # Direct URL provided
                client = A2AClient(agent_name_or_url)
                self.logger.info(f"Calling agent at URL: {agent_name_or_url}")
            else:
                # Agent name provided - resolve from registry
                try:
                    client = A2AClient.from_registry(agent_name_or_url)
                    self.logger.info(f"Calling agent '{agent_name_or_url}' from registry")
                except ValueError as e:
                    raise ValueError(f"Failed to resolve agent '{agent_name_or_url}': {e}")
            
            try:
                # Call the agent with timeout propagation
                result = await client.send_message(message, timeout_sec=timeout)
                
                # Enhanced response handling for Health Universe compatibility
                if isinstance(result, str):
                    return result
                
                if isinstance(result, dict):
                    # Check for direct text field (from tolerant parsing)
                    if "text" in result and isinstance(result["text"], str):
                        return result["text"]
                    
                    # Check if response has a message field
                    msg = result.get("message")
                    if isinstance(msg, dict):
                        parts = msg.get("parts") or []
                        texts = []
                        for p in parts:
                            if p.get("kind") == "text":
                                texts.append(p.get("text", ""))
                            elif p.get("kind") == "data":
                                # For data parts, return JSON string
                                data = p.get("data")
                                if data is not None:
                                    texts.append(json.dumps(data))
                        if texts:
                            return "\n".join(texts)
                    
                    # Check status.message for completed tasks
                    status = result.get("status")
                    if isinstance(status, dict):
                        status_msg = status.get("message")
                        if isinstance(status_msg, dict):
                            parts = status_msg.get("parts", [])
                            texts = []
                            for p in parts:
                                if p.get("kind") == "text":
                                    texts.append(p.get("text", ""))
                                elif p.get("kind") == "data":
                                    data = p.get("data")
                                    if data is not None:
                                        texts.append(json.dumps(data))
                            if texts:
                                return "\n".join(texts)
                    
                    # Fallback to JSON representation
                    return json.dumps(result)
                
                # Final fallback
                return str(result)
                
            finally:
                # Clean up client connection
                await client.close()
                
        except ImportError:
            # Fallback if a2a_client is not available
            self.logger.error("A2A client not available - cannot call other agents")
            raise Exception("Inter-agent communication not available")
    
    async def call_other_agent_with_data(
        self,
        agent_name_or_url: str,
        data: Any,
        timeout: float = 30.0
    ) -> Any:
        """
        Call another A2A-compliant agent with structured data.
        
        This method properly formats structured data using DataPart
        for inter-agent communication, ensuring compatibility with
        agents expecting structured data in the 'data' field.
        
        Args:
            agent_name_or_url: Agent name (from registry) or direct URL
            data: Structured data (dict, list, or string) to send
            timeout: Request timeout in seconds
            
        Returns:
            Response from the other agent (structured or string)
            
        Raises:
            Exception: If agent communication fails
        """
        try:
            from utils.a2a_client import A2AClient
            
            # Create client based on input type
            if agent_name_or_url.startswith(('http://', 'https://')):
                client = A2AClient(agent_name_or_url)
                self.logger.info(f"Calling agent at URL: {agent_name_or_url}")
            else:
                try:
                    client = A2AClient.from_registry(agent_name_or_url)
                    self.logger.info(f"Calling agent '{agent_name_or_url}' from registry")
                except ValueError as e:
                    raise ValueError(f"Failed to resolve agent '{agent_name_or_url}': {e}")
            
            try:
                # Use proper Part construction for structured data
                if isinstance(data, str):
                    # For strings, use regular message sending
                    result = await client.send_message(data, timeout_sec=timeout)
                else:
                    # For structured data, send with proper DataPart formatting
                    result = await client.send_data(data, timeout_sec=timeout)
                
                # Extract content from response parts if needed
                if isinstance(result, dict) and "message" in result:
                    msg = result["message"]
                    if isinstance(msg, dict) and "parts" in msg:
                        # Use message utils if available
                        try:
                            from utils.message_utils import extract_content_from_parts
                            return extract_content_from_parts(msg["parts"])
                        except ImportError:
                            # Fallback extraction
                            return self._extract_content_from_parts(msg["parts"])
                
                return result
                
            finally:
                # Clean up client connection
                await client.close()
                
        except ImportError:
            # Fallback if a2a_client is not available
            self.logger.error("A2A client not available - cannot call other agents")
            raise Exception("Inter-agent communication not available")
    
    def _extract_content_from_parts(self, parts: List[Dict[str, Any]]) -> Any:
        """Fallback content extraction from parts."""
        texts = []
        data_parts = []
        
        for part in parts:
            kind = part.get("kind")
            if kind == "text":
                texts.append(part.get("text", ""))
            elif kind == "data":
                data_parts.append(part.get("data"))
        
        if data_parts:
            return data_parts[0] if len(data_parts) == 1 else data_parts
        elif texts:
            return "\n".join(texts)
        else:
            return ""