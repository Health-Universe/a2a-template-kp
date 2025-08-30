#!/usr/bin/env python3
"""
Main entry point for the Keyword Generator Agent.
Starts an A2A-compliant HTTP server with all required endpoints.
"""

import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from utils.logging import get_logger
from .agent import KeywordGeneratorAgent

logger = get_logger(__name__)

def create_app():
    """Create the Starlette application with A2A endpoints."""
    # Instantiate the agent
    agent = KeywordGeneratorAgent()
    logger.info(f"Initializing {agent.get_agent_name()} v{agent.get_agent_version()}")
    
    # Build Agent Card + handler
    agent_card = agent.create_agent_card()
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=agent,
        task_store=task_store
    )
    
    # Build Starlette app (includes A2A endpoints and well-known card routes)
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    ).build()
    
    return app, agent

# Create the app instance for uvicorn
app, agent = create_app()

if __name__ == "__main__":
    # Configuration from environment
    port = int(os.getenv("PORT", "8002"))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    # Show startup information
    logger.info("=" * 60)
    logger.info(f"🚀 Starting {agent.get_agent_name()} on http://{host}:{port}")
    logger.info("=" * 60)
    logger.info("📋 Endpoints:")
    logger.info(f"   Agent Card: http://localhost:{port}/.well-known/agentcard.json")
    logger.info(f"   JSON-RPC:   POST http://localhost:{port}/ (method: \"message/send\")")
    logger.info("=" * 60)
    logger.info("🔬 Capabilities:")
    logger.info("   - Generate search keywords from documents")
    logger.info("   - Extract medical terminology and patterns")
    logger.info("   - Create regex patterns for document search")
    logger.info("   - Provide structured keyword output")
    logger.info("=" * 60)
    
    # Check for LLM configuration
    if not any([
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("GOOGLE_API_KEY")
    ]):
        logger.warning("⚠️  No LLM API key detected!")
        logger.warning("   Will use fallback keyword extraction")
    else:
        provider = "Unknown"
        if os.getenv("GOOGLE_API_KEY"):
            provider = "Google Gemini"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "OpenAI GPT"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "Anthropic Claude"
        logger.info(f"✅ LLM Provider: {provider}")
    
    logger.info("=" * 60)
    logger.info("Example usage:")
    logger.info(f'  curl -X POST http://localhost:{port}/ \\')
    logger.info('    -H "Content-Type: application/json" \\')
    logger.info('    -d \'{"jsonrpc": "2.0", "method": "message/send", "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": "Patient has diabetes type 2 with elevated glucose levels."}], "messageId": "test-123"}}, "id": 1}\'')
    logger.info("=" * 60)
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )