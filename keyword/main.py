#!/usr/bin/env python3
"""
Main entry point for the Keyword Pattern Generator Agent.
Starts an A2A-compliant HTTP server with all required endpoints.
"""

import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from utils.logging import get_logger, setup_logging
from examples.pipeline.keyword.agent import KeywordAgent

# Setup logging first
setup_logging()
logger = get_logger(__name__)

def create_app():
    """Create the Starlette application with A2A endpoints."""
    # Instantiate the agent
    agent = KeywordAgent()
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
    port = int(os.getenv("PORT", os.getenv("AGENT_PORT", "8101")))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    # Show startup information
    logger.info("=" * 60)
    logger.info(f"🚀 Starting {agent.get_agent_name()} on http://{host}:{port}")
    logger.info("=" * 60)
    logger.info("📋 Endpoints:")
    logger.info(f"   Agent Card: http://localhost:{port}/.well-known/agent-card.json")
    logger.info(f"   JSON-RPC:   POST http://localhost:{port}/ (method: \"message/send\")")
    logger.info(f"   Health:     http://localhost:{port}/health")
    logger.info("=" * 60)
    logger.info("🔬 Capabilities:")
    logger.info("   - Generate regex patterns from document previews")
    logger.info("   - Identify medical terminology and abbreviations")
    logger.info("   - Create ripgrep-compatible patterns")
    logger.info("   - Categorize patterns by type")
    logger.info("   - Provide fallback patterns if LLM unavailable")
    logger.info("=" * 60)
    
    # Check for LLM configuration
    if not any([
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("GEMINI_API_KEY")
    ]):
        logger.warning("⚠️  No LLM API key detected!")
        logger.warning("   Will use fallback patterns only")
    else:
        provider = "Unknown"
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "Anthropic Claude"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "OpenAI GPT"
        elif os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            provider = "Google Gemini"
        logger.info(f"✅ LLM Provider: {provider}")
    
    logger.info("=" * 60)
    logger.info("Example JSON-RPC usage:")
    logger.info(f'  curl -X POST http://localhost:{port}/ \\')
    logger.info('    -H "Content-Type: application/json" \\')
    logger.info('    -d \'{"jsonrpc": "2.0", "method": "message/send", "params": {"message": {"role": "user", "parts": [{"kind": "data", "data": {"document_preview": "Patient has diabetes type 2"}}], "messageId": "test-123"}}, "id": 1}\'')
    logger.info("=" * 60)
    
    # Run the server
    uvicorn.run(
        "examples.pipeline.keyword.main:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )