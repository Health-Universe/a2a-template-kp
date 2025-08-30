#!/usr/bin/env python3
"""
Encounter Grouping Agent - Groups clinical content by encounter dates, distinguishing actual visits from referenced dates.
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
from tools.encounter_grouping_tools import ENCOUNTER_TOOLS
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill


class EncounterGroupingAgent(A2AAgent):
    """Groups clinical content by encounter dates, distinguishing actual visits from referenced dates."""
    
    def get_agent_name(self) -> str:
        return "Encounter Grouping Agent"
    
    def get_agent_description(self) -> str:
        return "Groups clinical content by encounter dates, distinguishing actual visits from referenced dates."
    
    def get_agent_version(self) -> str:
        return "1.0.0"
    
    def get_system_instruction(self) -> str:
        return """You are a medical encounter grouping specialist. Your role is to analyze temporal clinical data and group content by actual encounter dates.

When processing temporal data:
1. Identify true encounter dates vs. referenced dates
2. Group content that belongs to the same clinical encounter
3. Handle content with unknown dates properly
4. Identify relationships between encounters (follow-ups, test results)
5. Distinguish between primary content and carry-forward references

Use the provided tools to analyze and group encounter data."""
    
    def get_tools(self) -> List:
        """Return the agent's tools."""
        return ENCOUNTER_TOOLS
    
    def supports_streaming(self) -> bool:
        """Enable streaming for real-time processing."""
        return True
    
    async def process_message(self, message: str) -> str:
        # This won't be called when tools are provided
        return "Processing..."


# Module-level app creation (required for deployment)
agent = EncounterGroupingAgent()
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
    port = int(os.getenv("PORT", 8011))
    print(f"🚀 Starting {agent.get_agent_name()}")
    print(f"📍 Server: http://localhost:{port}")
    print(f"📋 Agent Card: http://localhost:{port}/.well-known/agent-card.json")
    uvicorn.run(app, host="0.0.0.0", port=port)