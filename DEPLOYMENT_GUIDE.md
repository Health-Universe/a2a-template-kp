# A2A Template Deployment Guide

This guide explains how to deploy A2A agents without encountering import errors.

## Quick Start

### Option 1: Using the Universal Runner (Recommended)

```bash
# Start any agent using the universal runner
python run_agent.py keyword      # Keyword generation agent (port 8002)
python run_agent.py grep         # Document search agent (port 8003)  
python run_agent.py chunk        # Content chunk agent (port 8004)
python run_agent.py summarize    # Document summarization agent (port 8005)
python run_agent.py orchestrator # Pipeline orchestrator agent (port 8006)

# Custom port
python run_agent.py keyword --port 9000
```

### Option 2: Using Individual Start Scripts

```bash
# Run each agent from its start script (avoids import conflicts)
python keyword/start.py
python grep/start.py
python chunk/start.py
python summarize/start.py
python orchestrator/start.py
```

### Option 3: Running from Agent Directories

```bash
# Change to agent directory first, then run main.py
cd keyword && python main.py
cd grep && python main.py  
cd chunk && python main.py
cd summarize && python main.py
cd orchestrator && python main.py
```

## Common Issues and Solutions

### ASGI Import Error: "Could not import module 'keyword.main'"

**Problem:** Python's built-in `keyword` module conflicts with the `keyword` agent directory.

**Solutions:**

1. **Use the universal runner** (recommended):
   ```bash
   python run_agent.py keyword
   ```

2. **Use agent-specific start scripts**:
   ```bash
   python keyword/start.py
   ```

3. **Run from agent directory**:
   ```bash
   cd keyword && python main.py
   ```

4. **For Docker/Production deployments**, use the start scripts in Dockerfile:
   ```dockerfile
   CMD ["python", "/app/keyword/start.py"]
   ```

### Import Errors in Utils Modules

**Problem:** Missing functions like `call_agent`, `LLMProvider`, `setup_logging`.

**Solution:** This template includes all required utility functions. If you see import errors:

1. Ensure you've installed dependencies: `pip install -r requirements.txt`
2. Check that you're using the latest version of this template
3. Verify Python path is set correctly when running agents

## Docker Deployment

### Individual Agent Containers

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Use start script to avoid import conflicts
CMD ["python", "/app/keyword/start.py"]
```

### Multi-Agent Container

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Use universal runner
CMD ["python", "run_agent.py", "orchestrator"]
```

## Health Universe Deployment

For Health Universe deployments, ensure your entry point uses one of the conflict-free methods:

```yaml
# health-universe.yml
entry_point: "run_agent.py orchestrator"
# OR
entry_point: "orchestrator/start.py"
```

## Environment Variables

Each agent supports these environment variables:

```bash
# Common variables
PORT=8002                    # Port to run on
LOG_LEVEL=INFO              # Logging level
HU_APP_URL=...              # Health Universe app URL (auto-set in HU)

# LLM API Keys (for agents that need them)
GOOGLE_API_KEY=...          # For Google Gemini
OPENAI_API_KEY=...          # For OpenAI
ANTHROPIC_API_KEY=...       # For Claude

# Agent-specific variables
MAX_GREP_MATCHES=100        # Max search results (grep agent)
CONTEXT_LINES_BEFORE=3      # Context lines (grep agent)
CONTEXT_LINES_AFTER=3       # Context lines (grep agent)
ORCH_AGENT_TIMEOUT=30       # Agent call timeout (orchestrator)
```

## Testing Your Deployment

After starting an agent, test it works:

```bash
# Test agent card endpoint
curl http://localhost:8002/.well-known/agent-card.json

# Test health endpoint (if available)
curl http://localhost:8002/health

# Test A2A endpoint
curl -X POST http://localhost:8002/a2a/v1/message/sync \
  -H "Content-Type: application/json" \
  -d '{"message": {"role": "user", "parts": [{"kind": "text", "text": "Hello"}], "messageId": "test-123"}, "kind": "message"}'
```

## Troubleshooting

### Agent Won't Start

1. **Check Python version**: Requires Python 3.11+
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Check port conflicts**: Each agent uses a different default port
4. **Use start scripts**: Avoid module path conflicts

### Import Errors

1. **Use provided start scripts**: They handle Python path correctly
2. **Run from agent directory**: `cd agent_name && python main.py`
3. **Check utils imports**: This template includes all required functions

### Agent Communication Issues

1. **Check ports**: Each agent runs on a different port by default
2. **Update agent registry**: See `config/agents.json`
3. **Test individually**: Start agents one by one to isolate issues

## Production Checklist

- [ ] Use start scripts or universal runner (not direct uvicorn commands)
- [ ] Set appropriate environment variables
- [ ] Configure logging level for production
- [ ] Test all agent endpoints
- [ ] Verify inter-agent communication
- [ ] Monitor resource usage
- [ ] Set up health checks

## Getting Help

If you encounter issues not covered here:

1. Check the main [README.md](README.md) for general information
2. Review agent-specific code in each agent directory
3. Test agents individually to isolate problems
4. Check logs for specific error messages