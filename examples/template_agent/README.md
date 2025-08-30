# Template Agent

A minimal A2A-compliant agent using the production-ready base class and utilities. This template demonstrates the simplest possible agent that works with Anthropic, OpenAI, or Google LLMs.

## 🚀 Quick Start

```bash
# Set just ONE provider key - the utils auto-detect in this order:
# ANTHROPIC_API_KEY → OPENAI_API_KEY → GOOGLE_API_KEY

export OPENAI_API_KEY=sk-...
# OR
export ANTHROPIC_API_KEY=sk-ant-...
# OR
export GOOGLE_API_KEY=...

# Run the agent
cd examples/template_agent
python main.py
```

That's it! The agent auto-detects your LLM provider and starts on port 8010.

## 📋 Features

- **Zero configuration**: Just set one API key and run
- **Multi-provider support**: Works with Anthropic Claude, OpenAI GPT, or Google Gemini
- **A2A compliant**: Full v0.3.0 protocol support with proper Task lifecycle
- **Production ready**: Includes retry logic, error handling, and debug options
- **Extensible**: Easy to add tools, skills, or inter-agent communication

## 🔧 Configuration

### Required: LLM Provider (choose one)

```bash
# Option 1: Anthropic Claude (default: claude-3-5-sonnet)
export ANTHROPIC_API_KEY=sk-ant-...

# Option 2: OpenAI GPT (default: gpt-4o-mini)
export OPENAI_API_KEY=sk-...

# Option 3: Google Gemini (default: gemini-2.0-flash-exp)
export GOOGLE_API_KEY=...
```

### Optional: Override Models

```bash
export ANTHROPIC_MODEL=claude-3-5-haiku-20241022
export OPENAI_MODEL=gpt-4o
export GOOGLE_MODEL=gemini-1.5-pro
```

### Optional: Server Configuration

```bash
export PORT=8010                    # Server port (default: 8010)
export HOST=0.0.0.0                # Host to bind (default: 0.0.0.0)
export HU_APP_URL=http://localhost:8010  # Base URL for AgentCard

# Debug options
export LOG_LEVEL=DEBUG              # Logging level
export A2A_DEBUG_CARD=true         # Show AgentCard at startup
export DEBUG_PAYLOADS=1            # Log request/response payloads
```

## 📡 Endpoints

Once running, your agent exposes these endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /.well-known/agent-card.json` | A2A Agent Card (discovery) |
| `GET /.well-known/agentcard.json` | Alias for agent card |
| `GET /.well-known/agent.json` | HealthUniverse alias |
| `POST /a2a/v1/message/sync` | Synchronous message processing |
| `GET /health` | Health check endpoint |

## 💬 Testing the Agent

### Simple Text Query

```bash
curl -X POST http://localhost:8010/a2a/v1/message/sync \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "What is the capital of France?"}],
      "messageId": "test-001",
      "kind": "message"
    }
  }'
```

### JSON Data Processing

```bash
curl -X POST http://localhost:8010/a2a/v1/message/sync \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "role": "user",
      "parts": [
        {"kind": "text", "text": "Analyze this data:"},
        {"kind": "data", "data": {"sales": [100, 150, 200], "region": "North"}}
      ],
      "messageId": "test-002",
      "kind": "message"
    }
  }'
```

### Check Agent Card

```bash
curl http://localhost:8010/.well-known/agent-card.json | jq .
```

## 🛠️ Customization

### Adding Tools

Edit `agent.py` and uncomment the tools section:

```python
def get_tools(self) -> List:
    from examples.template_agent.tools.example_tools import EXAMPLE_TOOLS
    return EXAMPLE_TOOLS
```

### Calling Other Agents

The template includes a helper method for inter-agent communication:

```python
# In your process_message method:
response = await self.call_downstream_agent("other-agent", "Hello")
```

Configure other agents in `config/agents.json`:

```json
{
  "agents": {
    "other-agent": {
      "url": "http://localhost:8011",
      "name": "Another Agent"
    }
  }
}
```

### Custom System Instructions

Override `get_system_instruction()` in `agent.py`:

```python
def get_system_instruction(self) -> str:
    return "You are a specialized assistant for medical questions..."
```

## 📁 Project Structure

```
template_agent/
├── agent.py          # Core agent implementation
├── main.py          # Server entry point
├── README.md        # This file
└── tools/
    └── example_tools.py  # Optional tool definitions
```

## 🚢 Deployment

### Local Development

```bash
python main.py
```

### Production with Gunicorn

```bash
gunicorn examples.template_agent.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8010
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "examples/template_agent/main.py"]
```

### HealthUniverse

1. Push to GitHub
2. Connect repository to HealthUniverse
3. Deploy through the dashboard
4. Your agent URL: `https://xxx-xxx-xxx.agent.healthuniverse.com`

## 🐛 Troubleshooting

### "No LLM API key detected!"

Set one of the API keys:
```bash
export OPENAI_API_KEY=your-key-here
```

### "Import error: No module named 'base'"

Run from the repository root:
```bash
cd /path/to/a2a-template
python examples/template_agent/main.py
```

### "Task failed: LLM generation failed"

Check your API key is valid and has credits/quota remaining.

## 📚 Learn More

- [A2A Protocol Specification](../../A2A_SPECIFICATION.md)
- [Base Class Documentation](../../base.py)
- [Utils Documentation](../../utils/)
- [Main README](../../README.md)

## 💡 Tips

1. **Start simple**: Get the basic agent working before adding tools
2. **Use debug mode**: Set `LOG_LEVEL=DEBUG` to see detailed logs
3. **Test locally**: Use curl or Postman before deploying
4. **Monitor costs**: LLM API calls can add up quickly
5. **Cache responses**: Consider caching for repeated queries

---

Built with the A2A Template - Production-ready agents in minutes! 🚀