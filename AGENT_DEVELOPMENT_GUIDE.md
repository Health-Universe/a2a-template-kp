# Agent Development Guide

**A2A Protocol Version:** dev  
**Specification:** [A2A_SPECIFICATION.md](./A2A_SPECIFICATION.md)

> ⚠️ **Compliance Notice**: This guide follows the Agent-to-Agent (A2A) Protocol Specification. All implementations MUST comply with the requirements defined in [A2A Spec §11](./A2A_SPECIFICATION.md#11-a2a-compliance-requirements).

A step-by-step guide for creating A2A-compliant agents using this template.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Creating Your First Agent](#creating-your-first-agent)
4. [Migration Guide](#migration-guide-converting-existing-code)
5. [Testing Your Agent](#testing-your-agent)
6. [Deployment](#deployment)

## Overview

This template provides a simplified base class (`A2AAgent`) that handles all the complex A2A protocol requirements, letting you focus on your agent's business logic. The template is fully compliant with the A2A (Agent-to-Agent) protocol specification and includes utilities for proper message handling.

Agent types:
- **Simple agents** - Process messages and return responses (text or structured data)
- **Tool-based agents** - Use LLM with function tools for complex operations
- **Orchestrator agents** - Coordinate multiple agents to complete tasks

## Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up LLM API Key
You need at least one LLM provider API key:
```bash
# Choose one:
export GOOGLE_API_KEY="your-gemini-api-key"      # Preferred for Google ADK
export OPENAI_API_KEY="your-openai-api-key"      # Alternative
export ANTHROPIC_API_KEY="your-claude-api-key"   # Alternative
```

### 3. Understanding Part Types (Critical for A2A)
**[A2A Spec §6.5]** The A2A protocol uses discriminated union types for message parts. A Part MUST be one of:

- **TextPart** (`kind: "text"`) - For human-readable strings **[§6.5.1]**
  - MUST include `kind: "text"` discriminator
  - Contains `text: string` field
  
- **DataPart** (`kind: "data"`) - For structured data (JSON-serializable) **[§6.5.3]**
  - MUST include `kind: "data"` discriminator
  - Contains `data: { [key: string]: any }` field
  
- **FilePart** (`kind: "file"`) - For file references **[§6.5.2]**
  - MUST include `kind: "file"` discriminator
  - Contains `file: FileWithBytes | FileWithUri`

⚠️ **Compliance Alert [§6.5]**: The `kind` field is a discriminator and MUST be present. Using TextPart for JSON data violates the specification and will cause parsing issues!

## Creating Your First Agent

### Option 1: Simple Agent (No Tools)

For basic message processing without LLM tools:

```python
#!/usr/bin/env python3
"""
My Simple Agent - Description of what it does.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base import A2AAgent
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore


class MySimpleAgent(A2AAgent):
    """Your agent description."""
    
    def get_agent_name(self) -> str:
        """[A2A Spec §5.5] Required: Human-readable agent name"""
        return "My Simple Agent"
    
    def get_agent_description(self) -> str:
        """[A2A Spec §5.5] Required: Agent purpose description"""
        return "Detailed description of what your agent does"
    
    async def process_message(self, message: str) -> Union[str, Dict, List]:
        """Process the message and return a response.
        Returns:
        - str: Will be wrapped in TextPart
        - dict/list: Will be wrapped in DataPart (for structured data)
        """
        # Your logic here
        return f"Processed: {message}"  # TextPart
        # OR for structured data:
        # return {"result": "data", "count": 42}  # DataPart


# Module-level app creation (required for deployment)
agent = MySimpleAgent()
agent_card = agent.create_agent_card()
task_store = InMemoryTaskStore()
request_handler = DefaultRequestHandler(
    agent_executor=agent,
    task_store=task_store
)

app = A2AStarletteApplication(
    agent_card=agent_card,  # [A2A Spec §5.1] MUST make AgentCard available
    http_handler=request_handler  # [§7] Handles RPC methods
).build()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting {agent.get_agent_name()}")
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### Option 2: Tool-Based Agent (With LLM and Functions)

For agents that need to use tools/functions with an LLM:

```python
#!/usr/bin/env python3
"""
My Tool Agent - Uses LLM with function tools.
"""

import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from base import A2AAgent
from google.adk.tools import FunctionTool
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore


# Define your tool functions
def search_database(query: str, limit: int = 10) -> str:
    """Search the database for information."""
    # Your implementation
    return f"Found {limit} results for: {query}"

def calculate_something(value1: float, value2: float) -> str:
    """Perform a calculation."""
    result = value1 + value2
    return f"Result: {result}"


class MyToolAgent(A2AAgent):
    """Agent that uses tools via LLM."""
    
    def get_agent_name(self) -> str:
        return "My Tool Agent"
    
    def get_agent_description(self) -> str:
        return "Agent that uses tools to process requests"
    
    def get_system_instruction(self) -> str:
        """Instructions for the LLM on how to use tools."""
        return """You are a helpful assistant with access to tools.
        
        Use the search_database tool to find information.
        Use the calculate_something tool for calculations.
        
        Always explain what you're doing."""
    
    def get_tools(self) -> List:
        """Provide tools for the LLM to use."""
        return [
            FunctionTool(func=search_database),
            FunctionTool(func=calculate_something)
        ]
    
    async def process_message(self, message: str) -> str:
        # This won't be called when tools are provided
        # The base handles everything via Google ADK
        return "Handled by tool execution"


# Module-level app creation
agent = MyToolAgent()
agent_card = agent.create_agent_card()
task_store = InMemoryTaskStore()
request_handler = DefaultRequestHandler(
    agent_executor=agent,
    task_store=task_store
)

app = A2AStarletteApplication(
    agent_card=agent_card,  # [A2A Spec §5.1] MUST make AgentCard available
    http_handler=request_handler  # [§7] Handles RPC methods
).build()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting {agent.get_agent_name()}")
    uvicorn.run(app, host="0.0.0.0", port=port)
```

## Message Handling Best Practices

### Message Object Structure
**[A2A Spec §6.4]** A Message object represents communication between client and agent. Required fields:
- `role: "user" | "agent"` - Identifies the sender
- `parts: Part[]` - Array of content parts (MUST contain at least one Part)
- `messageId: string` - Unique identifier (typically UUID)
- `kind: "message"` - Type discriminator

### Using Message Utils (Recommended)
The template includes `utils/message_utils.py` for spec-compliant Part creation:

```python
from utils.message_utils import (
    create_text_part,      # Create TextPart
    create_data_part,      # Create DataPart  
    create_agent_message,  # Create full Message with Parts
    new_agent_text_message # Shorthand for text messages
)

# Examples:
text_part = create_text_part("Hello, world!")  
data_part = create_data_part({"key": "value"})

# Create a message with structured data
message = create_agent_message(
    {"patterns": ["pattern1", "pattern2"], "data": "value"},
    role="agent"
)
```

### DataPart vs TextPart Rules
**[A2A Spec §6.5.1 & §6.5.3]** Proper Part type selection is critical for compliance:

1. **Use DataPart for** **[§6.5.3]**:
   - JSON objects/arrays (structured data)
   - Machine-readable information
   - Inter-agent data transfer
   - Forms, parameters, or any structured content
   - API responses with structured data

2. **Use TextPart for** **[§6.5.1]**:
   - Plain textual content
   - Human-readable messages
   - Markdown formatted text
   - Error messages for human consumption
   - Status updates in natural language

### Returning Structured Data
```python
async def process_message(self, message: str) -> Union[str, Dict, List]:
    # Return dict/list for DataPart (automatic)
    return {
        "matches": ["item1", "item2"],
        "count": 2,
        "metadata": {"source": "agent"}
    }
    
    # Return string for TextPart (automatic)
    # return "This is a text response"
```

## Migration Guide: Converting Existing Code

If you have existing code (like our ADK demo pipeline), here's how to migrate it:

### Step 1: Analyze Your Existing Code

Identify:
- **Functions/tools** your code uses
- **System prompts** or instructions
- **Input/output formats**
- **Dependencies**

### Step 2: Extract Tool Functions

If your code has functions that should be LLM tools:

1. **Create a tools file** (`tools/my_tools.py`):
```python
from google.adk.tools import FunctionTool

def my_function(param1: str, param2: int) -> str:
    """Function description for LLM."""
    # Your existing function code
    return result

# Wrap functions as tools
my_tool = FunctionTool(func=my_function)

# Export tools list
MY_TOOLS = [my_tool]
```

2. **Keep functions unchanged** - Just wrap them with `FunctionTool`

### Step 3: Create Agent Class

Create your agent in `examples/` or `examples/pipeline/`:

```python
#!/usr/bin/env python3
"""
Agent name and description.
"""

import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))  # Adjust path as needed

from base import A2AAgent
from tools.my_tools import MY_TOOLS  # Import your tools
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill


class MyMigratedAgent(A2AAgent):
    
    def get_agent_name(self) -> str:
        return "Name from original agent"
    
    def get_agent_description(self) -> str:
        return "Description from original agent"
    
    def get_agent_version(self) -> str:
        return "2.0.0"  # Version after migration
    
    def get_system_instruction(self) -> str:
        """Copy system prompt from original agent."""
        return """Original system instructions here..."""
    
    def get_tools(self) -> List:
        """Return imported tools."""
        return MY_TOOLS
    
    def get_agent_skills(self) -> List[AgentSkill]:
        """[A2A Spec §5.5.4] Define agent capabilities/skills"""
        return [
            AgentSkill(
                # §5.5.4 - Required fields:
                id="skill_id",  # Unique identifier
                name="Skill Name",  # Human-readable name
                description="What this skill does",  # Detailed description
                tags=["tag1", "tag2"],  # Keywords for discovery
                # Optional fields:
                examples=["Example usage 1", "Example usage 2"],
                inputModes=["text/plain", "application/json"],  # Override defaults
                outputModes=["text/plain"]  # Override defaults
            )
        ]
    
    def supports_streaming(self) -> bool:
        """[A2A Spec §5.5.2] Declare streaming capability"""
        return True  # Enables message/stream [§7.2] and tasks/resubscribe [§7.9]
    
    async def process_message(self, message: str) -> str:
        # Not called when tools are provided
        return "Handled by tool execution"


# Standard app creation pattern
agent = MyMigratedAgent()
agent_card = agent.create_agent_card()  # [A2A Spec §5.5] Generate AgentCard
task_store = InMemoryTaskStore()  # [§6.1] Store Task objects
request_handler = DefaultRequestHandler(
    agent_executor=agent,
    task_store=task_store
)

app = A2AStarletteApplication(
    agent_card=agent_card,  # [A2A Spec §5.1] MUST make AgentCard available
    http_handler=request_handler  # [§7] Handles RPC methods
).build()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### Step 4: Key Migration Points

| Original Pattern | New Pattern |
|-----------------|-------------|
| Complex base class with streaming | Simplified `A2AAgent` base |
| Manual LLM setup | Automatic via `get_tools()` |
| Session management code | Handled by Google ADK |
| Heartbeat/streaming logic | Built-in via TaskUpdater |
| Manual error handling | Automatic A2A compliance |

### Step 5: What Gets Simplified

✅ **Remove**:
- Session management code
- Streaming/heartbeat logic
- LLM initialization code
- Manual task management
- Complex error handling

✅ **Keep**:
- Tool functions (unchanged)
- System instructions
- Agent metadata
- Business logic

## Multi-Part Message Handling

### Problem: Agents Only Processing First Part
A common issue is agents only reading `parts[0]`, missing data in subsequent parts.

### Solution: Iterate All Parts
```python
def extract_all_data(message: Dict) -> Dict[str, Any]:
    """Extract and merge data from ALL parts."""
    merged_data = {}
    
    for part in message.get("parts", []):
        if part.get("kind") == "data":
            data = part.get("data")
            if isinstance(data, dict):
                merged_data.update(data)
        elif part.get("kind") == "text":
            # Handle text parts
            text = part.get("text", "")
            # Process text...
    
    return merged_data
```

### Orchestrator Pattern for Multi-Part
See `examples/pipeline/simple_orchestrator/agent.py` for complete implementation:
```python
def _iter_messages(self, envelope: Any) -> List[Dict[str, Any]]:
    """Extract all messages from Task/Message structures."""
    # Implementation handles both Task and Message envelopes
    
def _iter_dataparts(self, message: Dict) -> List[Any]:
    """Extract all DataParts from a message."""
    # Iterates ALL parts, not just parts[0]
```

## LLM Integration Updates

### Critical Fix: Content Object Creation
**Wrong** (causes error):
```python
content = types.Content(role="user", parts=[types.Part(text=prompt)])
```

**Correct**:
```python
from google.genai import types
content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
```

This fix is implemented in `utils/llm_utils.py:217`.

### Using LLM Utils
```python
from utils.llm_utils import generate_completion

# Automatic provider selection based on available API keys
result = await generate_completion(
    prompt="Your prompt here",
    system_prompt="Optional system instructions"
)
```

## Testing Your Agent

### Universal Test Suite

**[A2A Spec §11.3]** Validate compliance through testing:

Test any agent with our universal tester:

```bash
# Test your agent
python test_any_agent.py examples.my_agent MyAgentClass

# Examples
python test_any_agent.py examples.pipeline.grep_agent GrepAgent
python test_any_agent.py examples.simple_echo_agent EchoAgent
```

The tester validates:
- Agent properties (name, description, version) **[§5.5]**
- Agent card generation **[§5.5]**
- Tool configuration (vendor-specific)
- Skills definition **[§5.5.4]**
- Message processing **[§6.4, §7.1]**
- A2A protocol compliance **[§11]**

### Manual Testing

1. **Start your agent**:
```bash
python examples/my_agent.py
```

2. **Check agent card** **[A2A Spec §5.3]**:
```bash
curl http://localhost:8000/.well-known/agentcard.json  # Primary well-known URI
# OR
curl http://localhost:8000/.well-known/agent-card.json  # Alternate
```

3. **Send a test message** **[A2A Spec §7.1]**:
```bash
# JSON-RPC transport [§3.2.1]
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Test message"}],
        "messageId": "uuid-here"
      }
    }
  }'
```

## Deployment

### For HealthUniverse

#### Agent Discovery
**[A2A Spec §5.2 & §5.3]** Agents MUST make their AgentCard available:

1. **Ensure module-level `app` variable**:
```python
# This MUST be at module level for HealthUniverse
app = A2AStarletteApplication(
    agent_card=agent_card,  # [A2A Spec §5.1] MUST make AgentCard available
    http_handler=request_handler  # [§7] Handles RPC methods
).build()
```

2. **Agent Discovery URLs**:
**[A2A Spec §5.3]** The recommended well-known URI location:
- `/.well-known/agentcard.json` - **Primary (A2A standard per RFC 8615)**
- `/.well-known/agent-card.json` - Alternate format
- `/.well-known/agent.json` - HealthUniverse compatible

To add these routes manually in your main.py (if not using the template):
```python
from fastapi.responses import JSONResponse

# After creating your app
card_payload = agent_card.model_dump() if hasattr(agent_card, "model_dump") else agent_card.dict()

@app.get("/.well-known/agentcard.json")
@app.get("/.well-known/agent-card.json")
@app.get("/.well-known/agent.json")
async def serve_agent_card():
    return JSONResponse(content=card_payload)
```

3. **Environment Variables**:
```bash
# Required
GOOGLE_API_KEY=xxx  # or OPENAI_API_KEY or ANTHROPIC_API_KEY

# Optional
PORT=8000
HU_APP_URL=https://apps.healthuniverse.com/xxx-xxx-xxx
```

### For Inter-Agent Communication

#### Text-Based Communication
**[A2A Spec §7.1]** Using the message/send method:

```python
# Simple text message via message/send [§7.1]
response = await self.call_other_agent(
    "agent-name",  # From config/agents.json [§5.2 Discovery]
    "Message to send"  # Will be wrapped in TextPart [§6.5.1]
)
```

#### Structured Data Communication (Recommended)
**[A2A Spec §7.1 & §6.5.3]** Send structured data using DataPart:

```python
from utils.message_utils import create_agent_message

# Create message with DataPart [§6.5.3] for structured data
message = create_agent_message({
    "patterns": ["pattern1", "pattern2"],
    "document": "content",
    "metadata": {"source": "orchestrator"}
}, role="user")  # [§6.4] role field is required

# Send using A2A client
from utils.a2a_client import A2AClient
client = A2AClient.from_registry("agent-name")
response = await client.send_message(message, timeout_sec=30)
await client.close()
```

#### Using A2A Client for Structured Data
```python
# Send structured data directly
response = await client.send_data({
    "key": "value",
    "nested": {"data": "here"}
}, timeout_sec=30)
```

Configure known agents in `config/agents.json`:
```json
{
  "agents": {
    "agent-name": {
      "url": "https://apps.healthuniverse.com/xxx-xxx-xxx",
      "name": "Agent Display Name",
      "description": "What this agent does"
    }
  }
}
```

## Best Practices

### 1. Tool Design
- Keep tool functions focused and single-purpose
- Include clear docstrings (LLM reads these)
- Handle errors gracefully
- Support optional parameters with defaults

### 2. System Instructions
- Be specific about tool usage
- Include examples in the prompt
- Explain expected input/output formats
- Guide the LLM's decision-making

### 3. File Organization
```
examples/
  pipeline/           # Complex multi-agent systems
    grep_agent.py
    chunk_agent.py
  simple_agent.py     # Standalone agents
  
tools/
  grep_tools.py       # Reusable tool functions
  common_tools.py
```

### 4. Error Handling
**[A2A Spec §8]** Error handling follows JSON-RPC 2.0 standards:

- The base handles A2A protocol errors **[§8.1 & §8.2]**
- Standard JSON-RPC error codes (-32700 to -32099) **[§8.1]**
- A2A-specific error codes (-32000 to -32099) **[§8.2]**
- Focus on your business logic errors
- Return helpful error messages
- Log appropriately
- For structured errors in DataPart, use error field:
  ```python
  # Return as DataPart with error info
  return {"error": "Invalid input", "code": -32602, "details": "..."}
  ```

### 5. Testing
- Always run the universal tester
- Test with actual LLM calls
- Verify agent card generation
- Test tool execution paths
- Verify Part type handling (DataPart vs TextPart)
- Test multi-part message scenarios

## Common Patterns

### Pattern 1: Search and Process
```python
def search_content(query: str, content: str) -> str:
    """Search content without file access."""
    # Process content directly
    return results
```

### Pattern 2: Validation
```python
def validate_input(data: str) -> str:
    """Validate and potentially fix input."""
    # Validation logic
    return validation_results
```

### Pattern 3: Analysis
```python
def analyze_data(data: str, criteria: List[str]) -> str:
    """Analyze data against criteria."""
    # Analysis logic
    return analysis_results
```

## Streaming Support

### Enabling Streaming
**[A2A Spec §5.5.2 & §3.3]** Streaming capability must be declared in AgentCapabilities:

```python
def supports_streaming(self) -> bool:
    """[A2A Spec §5.5.2] Declares SSE streaming support"""
    return True  # Enables message/stream [§7.2] and tasks/resubscribe [§7.9]
```

### Implementing Streaming Execute
**[A2A Spec §7.2]** Streaming uses Server-Sent Events (SSE) for real-time updates:

```python
async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
    """[A2A Spec §7.2] Execute with SSE streaming updates"""
    task = context.current_task  # [§6.1] Task object
    if not task:
        await super().execute(context, event_queue)
        return
        
    updater = TaskUpdater(event_queue, task.id, task.context_id)
    
    try:
        # [A2A Spec §7.2.2] Send TaskStatusUpdateEvent during processing
        await updater.update_status(
            TaskState.working,  # [§6.3] TaskState enum
            new_agent_text_message("Starting processing...")  # [§6.4] Message object
        )
        
        # Do work...
        result = await self.process_work()
        
        # Send final result
        await updater.update_status(
            TaskState.working,
            create_agent_message(result)  # Can be text or data
        )
        await updater.complete()
        
    except Exception as e:
        await updater.update_status(
            TaskState.failed,
            new_agent_text_message(f"Error: {str(e)}")
        )
        raise
```

## Production Deployment Best Practices

### 1. Configuration Management
- Keep `config/agents.json` with localhost URLs for development
- Use environment variables for production URLs:
  ```python
  agent_url = os.getenv("KEYWORD_AGENT_URL", "http://localhost:8101")
  ```

### 2. Debugging Inter-Agent Communication
- Log message types and content:
  ```python
  logger.info(f"Sending {type(message)} to {agent_name}")
  ```
- Save debug output for analysis:
  ```python
  with open(f"/tmp/debug_{timestamp}.json", "w") as f:
      json.dump(response, f, indent=2)
  ```

### 3. Handling Large Documents
- For single-line documents, implement intelligent splitting:
  ```python
  if len(lines) == 1 and len(document) > 500:
      # Split on sentence boundaries
      sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', document)
  ```

### 4. Memory Management
- Limit matches/results to prevent OOM:
  ```python
  MAX_MATCHES_PER_PATTERN = 100
  MAX_TOTAL_MATCHES = 1000
  ```

## Advanced Topics

### Inter-Agent Communication

Agents can call other agents using the `call_other_agent` method:

```python
async def process_message(self, message: str) -> str:
    # Call another agent
    response = await self.call_other_agent("agent_name", "message")
    return response
```

**Important**: Configure agent URLs in `config/agents.json`:
```json
{
  "agents": {
    "agent_name": {
      "url": "http://localhost:8002"
    }
  }
}
```

### Async Tool Functions

Google ADK's `FunctionTool` supports async functions natively (vendor-specific, not A2A):

```python
# Async functions work directly with FunctionTool
async def my_async_tool(param: str) -> str:
    """Async tool function."""
    result = await some_async_operation()
    return result

# No special handling needed
my_tool = FunctionTool(func=my_async_tool)
```

**Important**: Do NOT use `asyncio.run()` in tool functions - the agent's `execute()` method already runs in an async context.

### Handling Single-Line Documents

For agents processing documents (like medical records that may be single paragraphs):

1. **Detect single-line documents**:
```python
lines = document.split('\n')
is_single_line = len(lines) == 1
```

2. **Deduplicate processing**:
```python
# Don't process the same line multiple times
unique_lines = {}
for match in matches:
    line_num = match.get("line_number", 1)
    if line_num not in unique_lines:
        unique_lines[line_num] = match
```

3. **Smart chunking strategies**:
- For short single-line docs (<2000 chars): Process entire document once
- For long single-line docs: Extract segments around key matches
- Limit chunk extraction to avoid redundant processing

### Orchestrator Patterns

Two orchestration approaches:

1. **Dynamic Orchestrator**: Uses LLM with tools to decide agent calling sequence
   - More flexible, adapts to different queries
   - Higher latency due to LLM decision-making
   - Better for complex, varied requests

2. **Simple/Fixed Orchestrator**: Direct sequential pipeline execution
   - Faster execution (no LLM decisions)
   - Predictable behavior
   - Better for well-defined workflows

## Common Issues and Solutions

### Issue: "Orchestrator only receiving 1 match when grep finds 8"
**Cause**: Only reading `parts[0]` instead of all parts **[A2A Spec §6.4]**
**Solution**: Messages can have multiple Parts - iterate all:
```python
# [A2A Spec §6.4] parts is an array that may contain multiple Parts
for part in message.get("parts", []):
    if part.get("kind") == "data":  # [§6.5] Check discriminator
        # Process ALL data parts
```

### Issue: "Input should be a valid dictionary" (LLM error)
**Cause**: Incorrect Content object creation (Google ADK specific)
**Solution**: Use proper Part construction per vendor requirements:
```python
# Google ADK requirement (not A2A spec)
types.Part.from_text(text=prompt)  # Correct
# NOT: types.Part(text=prompt)  # Wrong
```

### Issue: "Agent does not support streaming"
**Cause**: Streaming not declared **[A2A Spec §5.5.2]**
**Solution**: 
1. Set `capabilities.streaming: true` in AgentCard **[§5.5.2]**
2. Return True from `supports_streaming()`
3. Implement `execute()` with SSE updates **[§7.2]**

### Issue: "No keywords detected" despite generation
**Cause**: Using TextPart for JSON data violates **[A2A Spec §6.5.3]**
**Solution**: Return dict/list from `process_message()` for automatic DataPart wrapping:
```python
# Correct: Returns DataPart [§6.5.3]
return {"keywords": [...]}  
# Wrong: Returns TextPart [§6.5.1]
return json.dumps({"keywords": [...]})  
```

## Troubleshooting

### Issue: "No LLM API key found"
**Authentication Required [A2A Spec §4]**
**Solution**: Set API key environment variables (provider-specific, not A2A):
- `GOOGLE_API_KEY`
- `OPENAI_API_KEY`  
- `ANTHROPIC_API_KEY`

**Note**: A2A authentication is handled via HTTP headers **[§4.3]**

### Issue: "asyncio.run() cannot be called from a running event loop"
**Solution**: Your tool functions are trying to create a new event loop. Use async functions directly without `asyncio.run()`.

### Issue: "Module not found"
**Solution**: Ensure proper path setup:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Issue: Tools not executing
**Solution**: 
- Verify `get_tools()` returns FunctionTool objects
- Check system instruction mentions the tools
- Ensure LLM API key is set

### Issue: Agent card not accessible  
**[A2A Spec §5.1]** Agents MUST make AgentCard available
**Solution**:
- Verify well-known URI: `/.well-known/agentcard.json` **[§5.3]**
- Check module-level `app` variable exists
- Verify A2AStarletteApplication is built correctly
- Ensure AgentCard has all required fields **[§5.5]**

## Next Steps

1. Start with a simple agent to understand the pattern
2. Ensure compliance with required methods **[A2A Spec §11.1.2]**
3. Add tools as needed for complex operations (vendor-specific)
4. Test thoroughly with the universal tester **[§11.3]**
5. Deploy with proper AgentCard discovery **[§5.2, §5.3]**
6. Build orchestrators to coordinate multiple agents **[§7.1]**

## Support

- Check existing examples in `examples/` directory
- Review the [A2A Specification](A2A_SPECIFICATION.md)
- See [Migration Guide Issue #18](https://github.com/jmjlacosta/a2a-template/issues/18) for complex migrations
- Test with `test_any_agent.py` for validation

## A2A Compliance Requirements

**[A2A Spec §11]** To be A2A-compliant, your agent MUST meet these requirements:

### Required Methods [§11.1.2]
Every A2A agent MUST implement:
- [ ] `message/send` - Send messages and initiate tasks [§7.1]
- [ ] `tasks/get` - Retrieve task status and results [§7.3]
- [ ] `tasks/cancel` - Request task cancellation [§7.4]

### Optional Methods [§11.1.3]
Agents MAY implement (must declare capability if supported):
- [ ] `message/stream` - SSE streaming (requires `capabilities.streaming: true`) [§7.2]
- [ ] `tasks/resubscribe` - Resume streaming (requires streaming capability) [§7.9]
- [ ] Push notification methods (requires `capabilities.pushNotifications: true`) [§7.5-7.8]
- [ ] `agent/getAuthenticatedExtendedCard` (requires `supportsAuthenticatedExtendedCard: true`) [§7.10]

### Transport Requirements [§11.1.1]
- [ ] Support at least ONE transport protocol [§3.2]:
  - JSON-RPC 2.0 over HTTP [§3.2.1]
  - gRPC over HTTP/2 [§3.2.2]
  - REST-style HTTP+JSON [§3.2.3]
- [ ] Declare supported transports in AgentCard [§5.6]

### AgentCard Requirements [§5.5]
Required fields in your AgentCard:
- [ ] `protocolVersion` - A2A protocol version (e.g., "dev")
- [ ] `name` - Human-readable agent name
- [ ] `description` - Agent purpose description
- [ ] `url` - Primary endpoint URL
- [ ] `preferredTransport` - Transport at main URL [§5.6.1]
- [ ] `version` - Your agent version
- [ ] `capabilities` - Supported optional features [§5.5.2]
- [ ] `defaultInputModes` - Default MIME types accepted
- [ ] `defaultOutputModes` - Default MIME types produced
- [ ] `skills` - Array of AgentSkill objects [§5.5.4]

### Data Format Compliance [§11.1.5]
- [ ] Use valid JSON-RPC 2.0 request/response format [§6.11]
- [ ] Use proper Part discriminators (`kind` field) [§6.5]
- [ ] Include all required Message fields [§6.4]
- [ ] Use standard error codes [§8.1, §8.2]

### Multi-Transport Compliance [§11.1.4]
If supporting multiple transports:
- [ ] Provide identical functionality across all transports
- [ ] Use standard method mappings [§3.5]
- [ ] Return semantically equivalent results

---

## Quick Reference

### Part Type Decision Tree
```
Is the data structured (dict/list)?
├─ Yes → DataPart (return dict/list)
└─ No → Is it for human reading?
    ├─ Yes → TextPart (return string)
    └─ No → DataPart (structure it)
```

### Message Utils Cheat Sheet
```python
from utils.message_utils import *

# Text message
msg = new_agent_text_message("Hello")

# Data message  
msg = create_agent_message({"data": "here"})

# Mixed message
msg = Message(
    role="agent",
    parts=[
        create_text_part("Status: Complete"),
        create_data_part({"results": [...]})
    ]
)
```

### A2A Client Usage
```python
from utils.a2a_client import A2AClient

# Create client
client = A2AClient.from_registry("agent-name")

# Send text
await client.send_text("message")

# Send data
await client.send_data({"key": "value"})

# Send message
await client.send_message(message)

# Clean up
await client.close()
```

## A2A Specification Reference Table

| Guide Topic | A2A Spec Section | Description |
|-------------|------------------|-------------|
| **Part Types** | §6.5 | Discriminated union (TextPart, DataPart, FilePart) |
| TextPart | §6.5.1 | Human-readable text content |
| DataPart | §6.5.3 | Structured JSON data |
| FilePart | §6.5.2 | File references (bytes or URI) |
| **Messages** | §6.4 | Communication between client/agent |
| Message Structure | §6.4 | role, parts[], messageId, kind fields |
| **Tasks** | §6.1 | Stateful unit of work |
| Task Status | §6.2 | Current state and context |
| Task States | §6.3 | Lifecycle states enum |
| **Agent Card** | §5.5 | Agent metadata and capabilities |
| Discovery | §5.2, §5.3 | Well-known URI and mechanisms |
| Skills | §5.5.4 | AgentSkill object definition |
| Capabilities | §5.5.2 | Optional features declaration |
| **Transport** | §3.2 | Protocol options (JSON-RPC, gRPC, REST) |
| JSON-RPC | §3.2.1 | Primary transport protocol |
| Streaming | §3.3 | Server-Sent Events (SSE) |
| **Methods** | §7 | RPC method definitions |
| message/send | §7.1 | Send message to agent |
| message/stream | §7.2 | Send with SSE streaming |
| tasks/get | §7.3 | Retrieve task status |
| tasks/cancel | §7.4 | Cancel ongoing task |
| **Errors** | §8 | Error handling standards |
| JSON-RPC Errors | §8.1 | Standard error codes |
| A2A Errors | §8.2 | Protocol-specific errors |
| **Authentication** | §4 | Security and auth |
| **Compliance** | §11 | Requirements for compliance |
| Agent Requirements | §11.1 | What agents must implement |
| Client Requirements | §11.2 | What clients must support |

## Transport Method Mapping [§3.5.6]

| JSON-RPC Method | gRPC Method | REST Endpoint | Required |
|-----------------|-------------|---------------|----------|
| message/send | SendMessage | POST /v1/message:send | ✅ Yes |
| message/stream | SendStreamingMessage | POST /v1/message:stream | Optional |
| tasks/get | GetTask | GET /v1/tasks/{id} | ✅ Yes |
| tasks/cancel | CancelTask | POST /v1/tasks/{id}:cancel | ✅ Yes |
| tasks/list | ListTask | GET /v1/tasks | Optional |
| tasks/resubscribe | TaskSubscription | POST /v1/tasks/{id}:subscribe | Optional |

---

**Remember**: The base handles all A2A complexity. Focus on your agent's unique functionality! Always use the correct Part types for A2A compliance. Refer to the [A2A Specification](./A2A_SPECIFICATION.md) for authoritative requirements.