# A2A Agent Template

A production-ready template for building Agent-to-Agent (A2A) protocol compliant agents that can be deployed on HealthUniverse or run locally.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a simple echo agent
python test_simple_agent.py

# Run an LLM-powered agent
python test_llm_agent.py

# Test compliance
python test_compliance.py
```

## 📚 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Building Your First Agent](#building-your-first-agent)
- [LLM Integration](#llm-integration)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Architecture Overview

The A2A (Agent-to-Agent) protocol enables standardized communication between AI agents. This template provides:

1. **Protocol Compliance**: Full A2A v0.3.0 protocol implementation
2. **Base Classes**: Reusable components for rapid agent development
3. **LLM Integration**: Support for Google ADK, OpenAI, and Anthropic
4. **HealthUniverse Ready**: Auto-detection and configuration for deployment

```
base/
├── base_agent.py        # LLM-powered agent base class
├── simple_agent.py      # Non-LLM agent base class
├── compliance.py        # A2A protocol compliance validation
└── app_builder.py       # Starlette app with A2A endpoints

Your Agent
    ↓
Inherits from BaseAgentExecutor or BaseLLMAgentExecutor
    ↓
Automatic A2A Compliance (AgentCard, endpoints, validation)
    ↓
Deploy to HealthUniverse or run locally
```

## Building Your First Agent

### Simple Non-LLM Agent

```python
from base import BaseAgentExecutor

class EchoAgent(BaseAgentExecutor):
    """Simple agent that echoes messages."""
    
    def get_agent_name(self) -> str:
        return "Echo Agent"
    
    def get_agent_description(self) -> str:
        return "Echoes back any message sent to it"
    
    async def process_message(self, message: str) -> str:
        """Process incoming message and return response."""
        return f"Echo: {message}"

# Run the agent
if __name__ == "__main__":
    agent = EchoAgent()
    agent.run(port=8001)
```

### LLM-Powered Agent

```python
from base import BaseLLMAgentExecutor
from google.adk.tools import FunctionTool
from typing import List

class AssistantAgent(BaseLLMAgentExecutor):
    """LLM agent with tools."""
    
    def get_agent_name(self) -> str:
        return "Assistant"
    
    def get_agent_description(self) -> str:
        return "AI assistant with tool capabilities"
    
    def get_system_instructions(self) -> str:
        return "You are a helpful AI assistant."
    
    def get_tools(self) -> List[FunctionTool]:
        """Return list of available tools."""
        return [
            FunctionTool(self._calculate),
            FunctionTool(self._get_time)
        ]
    
    def _calculate(self, expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    
    def _get_time(self) -> str:
        """Get current time."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Run the agent
if __name__ == "__main__":
    agent = AssistantAgent()
    agent.run(port=8002)
```

## LLM Integration

### Automatic LLM Detection

The template includes **automatic LLM provider detection**! Just set any of these environment variables and your agent will automatically use the right provider:

```bash
# Option 1: Use Claude (Anthropic)
export ANTHROPIC_API_KEY="your-anthropic-key"

# Option 2: Use GPT-3.5/GPT-4 (OpenAI)
export OPENAI_API_KEY="your-openai-key"

# Option 3: Use Gemini (Google)
export GOOGLE_API_KEY="your-google-key"
```

### Multi-Provider Support

The template automatically detects and configures LLM providers based on available API keys:

| Provider | Default Model | Streaming | Tools | Vision | Max Context | 
|----------|---------------|-----------|-------|--------|-------------|
| **Google Gemini** | `gemini-2.0-flash-001` | ✅ | ✅ | ✅ | 1M tokens |
| **OpenAI GPT** | `gpt-4o-mini` | ✅ | ✅ | ✅ | 128K tokens |
| **Anthropic Claude** | `claude-3-5-haiku-20241022` | ✅ | ✅ | ✅ | 200K tokens |

### Configuration

```bash
# LLM API Keys (at least one required)
export GOOGLE_API_KEY="your-key"      # For Gemini
export OPENAI_API_KEY="your-key"      # For GPT models  
export ANTHROPIC_API_KEY="your-key"   # For Claude

# LLM Provider Configuration
export LLM_PROVIDER="google"           # Override auto-detection
export GEMINI_MODEL="gemini-2.0-flash-001"
export OPENAI_MODEL="gpt-4o-mini"
export ANTHROPIC_MODEL="claude-3-5-haiku-20241022"
```

### Google ADK Integration Lessons

#### Key Discovery: Runner Method Usage

The Google ADK `Runner` class does NOT have a `stream()` method. Use `run_async()` instead:

```python
# CORRECT - Use run_async with proper Content structure
from google.adk.runners import types

message = types.Content(
    parts=[types.Part(text=prompt)],
    role="user"
)

async for event in self._runner.run_async(
    user_id="user_id",
    session_id="session_id", 
    new_message=message
):
    # Process events
```

#### Content and Part Types

```python
# Content expects 'parts' and 'role', not 'text'
message = types.Content(
    parts=[types.Part(text="your text here")],
    role="user"  # or "agent"
)
```

#### Base Class Selection

- **BaseAgentExecutor**: For non-LLM agents
- **BaseLLMAgentExecutor**: For LLM-powered agents with Google ADK integration

#### Required Abstract Methods

When extending `BaseLLMAgentExecutor`, you MUST implement:

```python
def get_tools(self) -> List[Any]:
    """Return list of tools for the agent."""
    # Even if no tools needed, must return empty list
    return []
```

## Example Agents

### Pipeline Examples

The `examples/pipeline/` directory contains a complete medical document analysis pipeline:

| Agent | Port | Description |
|-------|------|-------------|
| **Keyword Agent** | 8002 | Generates regex patterns for medical terms |
| **Grep Agent** | 8003/8013 | Searches documents using patterns |
| **Chunk Agent** | 8004 | Extracts context around matches |
| **Summarize Agent** | 8005 | Analyzes and summarizes chunks |
| **Orchestrator Agent** | 8006 | Coordinates the pipeline (LLM-driven) |
| **Simple Orchestrator** | 8008 | Fixed sequence pipeline (no LLM) |

### Running the Pipeline

```bash
# Start all agents
python examples/pipeline/keyword_agent.py &
PORT=8013 python examples/pipeline/grep_agent.py &
python examples/pipeline/chunk_agent.py &
python examples/pipeline/summarize_agent.py &
python examples/pipeline/orchestrator_agent.py &

# Test the pipeline
python tests/test_orchestrator.py
```

## Deployment

### Local Development

```bash
python your_agent.py
# Agent runs on http://localhost:8000
```

### HealthUniverse Deployment

#### Understanding Container Isolation

Each agent deployed to HealthUniverse runs in its **own isolated Kubernetes container**:

1. **No shared filesystem** - Agents cannot read/write to each other's files
2. **No shared memory** - Agents cannot share in-memory data structures
3. **No auto-discovery** - Agents must be configured with explicit URLs
4. **No runtime updates** - Configuration changes require redeployment

#### Single Agent Deployment

1. Push your code to GitHub
2. Connect repository to HealthUniverse
3. Deploy through HealthUniverse dashboard
4. Agent URL: `https://your-agent-id.agent.healthuniverse.com`

#### Multi-Agent System Deployment

##### Step 1: Deploy Individual Agents

```bash
# Deploy each agent separately
git push origin main
# HealthUniverse assigns ID: abc-def-ghi
# Agent URL: https://abc-def-ghi.agent.healthuniverse.com
```

##### Step 2: Record Agent IDs

| Agent | HealthUniverse ID | URL |
|-------|------------------|-----|
| grep-agent | abc-def-ghi | https://abc-def-ghi.agent.healthuniverse.com |
| chunk-agent | xyz-uvw-rst | https://xyz-uvw-rst.agent.healthuniverse.com |

##### Step 3: Configure Orchestrator

Update `config/agents.json`:

```json
{
  "agents": {
    "grep-agent": {
      "url": "https://abc-def-ghi.agent.healthuniverse.com",
      "name": "Grep Search Agent"
    },
    "chunk-agent": {
      "url": "https://xyz-uvw-rst.agent.healthuniverse.com",
      "name": "Document Chunk Agent"
    }
  }
}
```

##### Step 4: Deploy Orchestrator

```bash
git add config/agents.json
git commit -m "Configure agent URLs"
git push origin main
```

#### Deployment Checklist

- [ ] Deploy all individual agents first
- [ ] Record each agent's xxx-xxx-xxx ID
- [ ] Update config/agents.json with actual URLs
- [ ] Commit and push configuration
- [ ] Deploy orchestrator last
- [ ] Test inter-agent communication

## Troubleshooting

### Common Issues and Solutions

#### Import Errors

```python
# ❌ Wrong
from a2a.server.apps.starlette_app import A2AStarletteApplication
from base.llm_agent import LLMProvider
from google.adk import types

# ✅ Correct
from a2a.server.apps import A2AStarletteApplication
from base import BaseLLMAgentExecutor
from google.adk.runners import types
```

#### Google ADK Runner Issues

**Problem**: `'Runner' object has no attribute 'stream'`

**Solution**: Use `run_async()` instead:

```python
# ❌ Wrong
async for chunk in self._runner.stream(prompt=prompt):
    ...

# ✅ Correct
message = types.Content(
    parts=[types.Part(text=prompt)],
    role="user"
)

async for event in self._runner.run_async(
    user_id="user_id",
    session_id="session_id",
    new_message=message
):
    # Process events
```

#### Content Type Errors

**Problem**: `1 validation error for Content: text - Extra inputs are not permitted`

**Solution**: Use correct Content structure:

```python
# ❌ Wrong
message = types.Content(text="hello")

# ✅ Correct
message = types.Content(
    parts=[types.Part(text="hello")],
    role="user"
)
```

#### Attribute Access Errors

**Problem**: `AttributeError: 'AgentCard' object has no attribute 'protocolVersion'`

**Solution**: Use snake_case for attribute access:

```python
# ❌ Wrong
print(agent_card.protocolVersion)

# ✅ Correct
print(agent_card.protocol_version)
```

#### FunctionTool Errors

**Problem**: `TypeError: FunctionTool.__init__() got an unexpected keyword argument 'name'`

**Solution**: Pass only the function:

```python
# ❌ Wrong
tool = FunctionTool(name="my_tool", function=my_function)

# ✅ Correct
tool = FunctionTool(my_function)  # Name from function.__name__
```

#### Missing Abstract Methods

**Problem**: `TypeError: Can't instantiate abstract class ... with abstract method get_tools`

**Solution**: Implement all required methods:

```python
class MyAgent(BaseLLMAgentExecutor):
    def get_tools(self) -> List[Any]:
        return []  # Even if empty, must be implemented
```

#### LLM Not Configured

**Problem**: `No LLM API key found!`

**Solution**: Set at least one API key:

```bash
export GOOGLE_API_KEY="your-key"
# OR
export OPENAI_API_KEY="your-key"  
# OR
export ANTHROPIC_API_KEY="your-key"
```

#### HealthUniverse Deployment Issues

**"Agent not found" errors**:
- Verify config/agents.json has correct URLs
- Ensure agents are deployed and running
- Deploy orchestrator AFTER config update

**Configuration not updating**:
- Commit and push changes to Git
- Redeploy the agent (config loads at startup)
- Check you're on the right branch

**Agent URLs incorrect**:
- Format: `https://{xxx-xxx-xxx}.agent.healthuniverse.com`
- No trailing slashes
- Use HTTPS for production

### Quick Diagnostic Commands

```python
# Check imports
import a2a
print(dir(a2a.server.apps))

from google.adk import runners
print(dir(runners))

# Check API configuration
import os
print("Google API:", "✓" if os.getenv("GOOGLE_API_KEY") else "✗")
print("OpenAI API:", "✓" if os.getenv("OPENAI_API_KEY") else "✗")
print("Anthropic API:", "✓" if os.getenv("ANTHROPIC_API_KEY") else "✗")

# Test agent initialization
from your_agent import YourAgent
agent = YourAgent()
print(f"✅ Agent created: {agent.get_agent_name()}")
```

```bash
# Check A2A endpoints
curl http://localhost:8000/.well-known/agentcard.json
curl http://localhost:8000/health
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Advanced Topics

### A2A Protocol Compliance

The template ensures full A2A v0.3.0 compliance:

1. **A2A spec endpoint**: `/.well-known/agent-card.json`
2. **HealthUniverse endpoint**: `/.well-known/agent.json`
3. **Health endpoint**: `/health`
4. **JSONRPC endpoint**: `/` (POST)
5. **Protocol version**: "0.3.0"
6. **Message format**: Proper Task/Message/Part structure

### Required AgentCard Fields

```python
agent_card = AgentCard(
    # Required core fields
    protocol_version="0.3.0",
    name="Agent Name",
    description="Agent description",
    url="http://localhost:8000",
    preferred_transport="JSONRPC",
    version="1.0.0",
    
    # Required arrays (can be empty)
    skills=[],
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    
    # Required capabilities
    capabilities=AgentCapabilities(
        streaming=False,
        push_notifications=False
    )
)
```

### Platform Detection

The template automatically detects deployment environment:

```python
# Local development
http://localhost:8000

# HealthUniverse deployment
https://agent-id.agent.healthuniverse.com
```

### Testing Patterns

#### Minimal LLM Test

```python
import asyncio
from your_agent import YourLLMAgent

async def test():
    agent = YourLLMAgent()
    result = await agent.process_message("test input")
    print(result)

asyncio.run(test())
```

#### Testing in Jupyter

```python
# In Jupyter notebooks
result = await agent.process_message("test")

# In regular Python
import asyncio
result = asyncio.run(agent.process_message("test"))
```

### Best Practices

1. **Start Simple**: Test with BaseAgentExecutor first, then add LLM
2. **Check Imports**: Verify exact import paths with `dir()` and `help()`
3. **Read Error Messages**: Pydantic errors are descriptive
4. **Use Type Hints**: Helps catch integration issues early
5. **Test Incrementally**: Test each component separately
6. **Implement Fallbacks**: Handle cases when LLM is unavailable

```python
if self._runner and self._agent:
    # LLM processing
else:
    # Fallback logic
    return "LLM not available"
```

## 📁 Project Structure

```
a2a-template/
├── base/                    # Core implementation
│   ├── __init__.py
│   ├── base_agent.py       # LLM agent base class
│   ├── simple_agent.py     # Non-LLM agent base class
│   ├── compliance.py       # A2A compliance validation
│   └── app_builder.py      # Starlette app builder
├── examples/               # Example agents
│   ├── simple_echo_agent.py
│   ├── llm_assistant_agent.py
│   ├── regulatory_compliance_agent.py
│   └── orchestrator_agent.py
├── config/                 # Configuration files
│   └── agents.json.example
├── requirements.txt        # Python dependencies
├── test_*.py              # Test files
└── README.md              # This file
```

## 📋 Implementation Checklist

When implementing ANY new feature, verify:

- [ ] All A2A imports use `from a2a.server.apps import` (NOT `.starlette_app`)
- [ ] Access AgentCard attributes via snake_case (e.g., `card.protocol_version`)
- [ ] Access Message/Part attributes via snake_case  
- [ ] FunctionTool only receives the function itself
- [ ] AgentCard includes ALL required fields
- [ ] Protocol version is "0.3.0"
- [ ] Dependencies match requirements.txt versions

## 🔧 Environment Variables

### Minimal Setup - Just Set One API Key!

The utilities now auto-detect your LLM provider based on which API key is set. No other configuration needed:

```bash
# Option 1: Use Anthropic Claude (auto-selects claude-3-5-sonnet)
export ANTHROPIC_API_KEY="sk-ant-..."

# Option 2: Use OpenAI GPT (auto-selects gpt-4o-mini)  
export OPENAI_API_KEY="sk-..."

# Option 3: Use Google Gemini (auto-selects gemini-2.0-flash-exp)
export GOOGLE_API_KEY="..."
# OR
export GEMINI_API_KEY="..."
```

That's it! The agent will automatically:
- Detect which provider to use
- Select the appropriate default model
- Configure all settings correctly

### Optional Configuration

```bash
# Override default models (optional)
export ANTHROPIC_MODEL="anthropic/claude-3-5-sonnet-20241022"
export OPENAI_MODEL="openai/gpt-4o-mini"
export GOOGLE_MODEL="gemini-2.0-flash-exp"

# Agent Configuration
export HU_APP_URL="https://your-agent-url"          # Root URL (clients append /a2a/v1)
export PORT="8000"                                  # Server port
export LOG_LEVEL="INFO"                             # Logging level

# Debugging
export DEBUG_PAYLOADS="1"                           # Log request/response payloads

# Authentication
export AGENT_TOKEN="your-bearer-token"              # For inter-agent auth

# Agent Registry
export AGENT_REGISTRY_PATH="config/agents.json"     # Agent registry location

# Agent Metadata (optional)
export AGENT_NAME="My Agent"
export AGENT_VERSION="1.0.0"
export AGENT_ORG="My Organization"
export AGENT_ORG_URL="https://myorg.com"
```

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please ensure all agents maintain A2A v0.3.0 compliance.

## 📚 Resources

- [A2A Protocol Specification](./A2A_SPECIFICATION.md)
- [HealthUniverse Documentation](https://docs.healthuniverse.com)
- [Google ADK Documentation](https://cloud.google.com/agent-builder/docs)

## 🐛 Known Issues

1. **Streaming responses**: Currently not fully implemented
2. **Push notifications**: Not yet supported
3. **Multiple transport protocols**: Only JSONRPC currently supported

## 💡 Tips

1. Always validate compliance before deployment using `test_compliance.py`
2. Use environment variables for API keys and configuration
3. Test locally before deploying to HealthUniverse
4. Keep `requirements.txt` in sync with deployment
5. Use the base classes to ensure A2A compliance automatically

---

Built with ❤️ for the A2A ecosystem