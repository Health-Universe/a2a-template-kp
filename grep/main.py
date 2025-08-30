#!/usr/bin/env python3
"""
Main entry point for the Pattern Search (Grep) Agent.
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
from examples.pipeline.grep.agent import GrepAgent

# Setup logging first
setup_logging()
logger = get_logger(__name__)

def create_app():
    """Create the Starlette application with A2A endpoints."""
    # Instantiate the agent
    agent = GrepAgent()
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
    port = int(os.getenv("PORT", os.getenv("AGENT_PORT", "8102")))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    # Show startup information
    logger.info("=" * 60)
    logger.info(f"🚀 Starting {agent.get_agent_name()} on http://{host}:{port}")
    logger.info("=" * 60)
    logger.info("📋 Endpoints:")
    logger.info(f"   Agent Card: http://localhost:{port}/.well-known/agent-card.json")
    logger.info(f"   A2A Sync:   http://localhost:{port}/a2a/v1/message/sync")
    logger.info(f"   Health:     http://localhost:{port}/health")
    logger.info("=" * 60)
    logger.info("🔍 Capabilities:")
    logger.info("   - High-performance regex pattern searching")
    logger.info("   - Context extraction around matches")
    logger.info("   - Pattern validation and error handling")
    logger.info("   - Intelligent error recovery suggestions")
    logger.info("   - No LLM required - pure algorithmic")
    logger.info("=" * 60)
    logger.info("⚙️  Configuration:")
    logger.info(f"   Max matches/pattern: {agent.MAX_MATCHES_PER_PATTERN}")
    logger.info(f"   Context lines:       {agent.CONTEXT_LINES_BEFORE} before, {agent.CONTEXT_LINES_AFTER} after")
    logger.info("=" * 60)
    logger.info("Example usage:")
    logger.info(f'  curl -X POST http://localhost:{port}/a2a/v1/message/sync \\')
    logger.info('    -H "Content-Type: application/json" \\')
    logger.info('    -d \'{"message": {"role": "user", "parts": [{"kind": "data", "data": {')
    logger.info('      "patterns": ["diabetes", "\\\\b\\\\d+\\\\s*mg\\\\b"],')
    logger.info('      "document_content": "Patient has diabetes type 2. Metformin 500mg daily.",')
    logger.info('      "case_sensitive": false')
    logger.info('    }}], "kind": "message"}}\'')
    logger.info("=" * 60)
    
    # Run the server
    uvicorn.run(
        "examples.pipeline.grep.main:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )