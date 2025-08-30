#!/usr/bin/env python3
"""
Reconciliation Agent - Reconciles conflicting temporal information and resolves inconsistencies in medical timelines.
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
from tools.reconciliation_tools import RECONCILIATION_TOOLS
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill


class ReconciliationAgent(A2AAgent):
    """Reconciles conflicting temporal information and resolves inconsistencies in medical timelines."""
    
    def get_agent_name(self) -> str:
        return "Reconciliation Agent"
    
    def get_agent_description(self) -> str:
        return "Reconciles conflicting temporal information and resolves inconsistencies in medical timelines."
    
    def get_agent_version(self) -> str:
        return "1.0.0"
    
    def get_system_instruction(self) -> str:
        return """You are a medical data reconciliation specialist. Your role is to resolve conflicts and inconsistencies in temporal medical data.

When reconciling data:
1. Identify conflicting dates or information
2. Apply medical logic to resolve conflicts
3. Maintain data integrity and traceability
4. Document reconciliation decisions
5. Flag irreconcilable conflicts for review

Use the provided tools to analyze and reconcile medical data."""
    
    def get_tools(self) -> List:
        """Return the agent's tools."""
        return RECONCILIATION_TOOLS
    
    def supports_streaming(self) -> bool:
        """Enable streaming for real-time processing."""
        return True
    
    async def process_message(self, message: str) -> str:
        # This won't be called when tools are provided
        return "Processing..."


# Module-level app creation (required for deployment)
agent = ReconciliationAgent()
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
    port = int(os.getenv("PORT", 8012))
    print(f"🚀 Starting {agent.get_agent_name()}")
    print(f"📍 Server: http://localhost:{port}")
    print(f"📋 Agent Card: http://localhost:{port}/.well-known/agent-card.json")
    uvicorn.run(app, host="0.0.0.0", port=port)