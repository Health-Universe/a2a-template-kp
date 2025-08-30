#!/usr/bin/env python3
"""
Checker Agent - Verifies the accuracy and consistency of extracted medical information and timelines.
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
from tools.checker_tools import CHECKER_TOOLS
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill


class CheckerAgent(A2AAgent):
    """Verifies the accuracy and consistency of extracted medical information and timelines."""
    
    def get_agent_name(self) -> str:
        return "Checker Agent"
    
    def get_agent_description(self) -> str:
        return "Verifies the accuracy and consistency of extracted medical information and timelines."
    
    def get_agent_version(self) -> str:
        return "1.0.0"
    
    def get_system_instruction(self) -> str:
        return """You are a medical information verification specialist. Your role is to check and validate extracted medical data.

When checking information:
1. Verify factual accuracy
2. Check for logical consistency
3. Validate temporal sequences
4. Identify missing information
5. Flag potential errors

Use the provided tools to verify and validate medical information."""
    
    def get_tools(self) -> List:
        """Return the agent's tools."""
        return CHECKER_TOOLS
    
    def supports_streaming(self) -> bool:
        """Enable streaming for real-time processing."""
        return True
    
    async def process_message(self, message: str) -> str:
        # This won't be called when tools are provided
        return "Processing..."


# Module-level app creation (required for deployment)
agent = CheckerAgent()
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
    port = int(os.getenv("PORT", 8015))
    print(f"🚀 Starting {agent.get_agent_name()}")
    print(f"📍 Server: http://localhost:{port}")
    print(f"📋 Agent Card: http://localhost:{port}/.well-known/agent-card.json")
    uvicorn.run(app, host="0.0.0.0", port=port)