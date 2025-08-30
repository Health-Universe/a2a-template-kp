#!/usr/bin/env python3
"""
Unified Extractor Agent - Performs comprehensive extraction of all medical entities and relationships from documents.
Migrated from KP pipeline to simplified A2A template.
"""

import os
import sys
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from base import A2AAgent
from google.adk.tools import FunctionTool
from tools.unified_extractor_tools import UNIFIED_EXTRACTOR_TOOLS
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill


class UnifiedExtractorAgent(A2AAgent):
    """Performs comprehensive extraction of all medical entities and relationships from documents."""
    
    def get_agent_name(self) -> str:
        return "Unified Extractor Agent"
    
    def get_agent_description(self) -> str:
        return "Performs comprehensive extraction of all medical entities and relationships from documents."
    
    def get_agent_version(self) -> str:
        return "1.0.0"
    
    def get_system_instruction(self) -> str:
        return """You are a comprehensive medical information extraction specialist. Your role is to extract all relevant medical entities and relationships.

When extracting information:
1. Extract all medical entities (conditions, medications, procedures)
2. Identify relationships between entities
3. Capture contextual information
4. Note certainty and negations
5. Structure data for integration

Use the provided tools to perform comprehensive medical extraction."""
    
    def get_tools(self) -> List:
        """Return the agent's tools."""
        return UNIFIED_EXTRACTOR_TOOLS
    
    def supports_streaming(self) -> bool:
        """Enable streaming for real-time processing."""
        return True
    
    async def process_message(self, message: str) -> str:
        # This won't be called when tools are provided
        return "Processing..."


# Module-level app creation (required for deployment)
agent = UnifiedExtractorAgent()
agent_card = agent.create_agent_card()
task_store = InMemoryTaskStore()
request_handler = DefaultRequestHandler(
    agent_executor=agent,
    task_store=task_store
)

app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler
).build()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8016))
    print(f"🚀 Starting {agent.get_agent_name()}")
    print(f"📍 Server: http://localhost:{port}")
    print(f"📋 Agent Card: http://localhost:{port}/.well-known/agent-card.json")
    uvicorn.run(app, host="0.0.0.0", port=port)