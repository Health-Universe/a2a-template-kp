#!/usr/bin/env python3
"""
Summary Extractor Agent - Extracts structured summaries from medical text, focusing on key clinical findings and outcomes.
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
from tools.summary_extractor_tools import SUMMARY_EXTRACTOR_TOOLS
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill


class SummaryExtractorAgent(A2AAgent):
    """Extracts structured summaries from medical text, focusing on key clinical findings and outcomes."""
    
    def get_agent_name(self) -> str:
        return "Summary Extractor Agent"
    
    def get_agent_description(self) -> str:
        return "Extracts structured summaries from medical text, focusing on key clinical findings and outcomes."
    
    def get_agent_version(self) -> str:
        return "1.0.0"
    
    def get_system_instruction(self) -> str:
        return """You are a medical summary extraction specialist. Your role is to extract structured summaries from clinical text.

When extracting summaries:
1. Identify key clinical findings
2. Extract diagnoses and conditions
3. Note treatments and interventions
4. Capture outcomes and prognosis
5. Structure information for easy retrieval

Use the provided tools to extract and structure medical summaries."""
    
    def get_tools(self) -> List:
        """Return the agent's tools."""
        return SUMMARY_EXTRACTOR_TOOLS
    
    def supports_streaming(self) -> bool:
        """Enable streaming for real-time processing."""
        return True
    
    async def process_message(self, message: str) -> str:
        # This won't be called when tools are provided
        return "Processing..."


# Module-level app creation (required for deployment)
agent = SummaryExtractorAgent()
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
    port = int(os.getenv("PORT", 8013))
    print(f"🚀 Starting {agent.get_agent_name()}")
    print(f"📍 Server: http://localhost:{port}")
    print(f"📋 Agent Card: http://localhost:{port}/.well-known/agent-card.json")
    uvicorn.run(app, host="0.0.0.0", port=port)