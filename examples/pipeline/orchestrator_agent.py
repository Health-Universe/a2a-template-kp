#!/usr/bin/env python3
"""
Pipeline Orchestrator Agent - Coordinates medical document analysis pipeline.
Migrated from ADK demo pipeline to use enhanced A2AAgent base.

This agent orchestrates a complete document analysis pipeline:
- Understands natural language requests
- Coordinates multiple specialized agents
- Synthesizes results into comprehensive responses
- Handles errors and optimizes performance
"""

import os
import sys
import json
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from base import A2AAgent
from tools.orchestrator_tools import ORCHESTRATOR_TOOLS
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill


class OrchestratorAgent(A2AAgent):
    """LLM-powered orchestrator for the medical document analysis pipeline."""
    
    def get_agent_name(self) -> str:
        """Return the agent's name."""
        return "Pipeline Orchestrator"
    
    def get_agent_description(self) -> str:
        """Return detailed agent description."""
        return (
            "LLM-powered orchestrator that coordinates a complete medical document analysis pipeline. "
            "Understands natural language requests, plans execution strategy, coordinates multiple "
            "specialized agents (keyword, grep, chunk, summarize), and synthesizes results into "
            "comprehensive responses. Handles errors gracefully and optimizes pipeline performance."
        )
    
    def get_agent_version(self) -> str:
        """Return agent version."""
        return "2.0.0"
    
    def get_system_instruction(self) -> str:
        """Return the system instruction for orchestration."""
        return """You are the orchestrator of a medical document analysis pipeline. Your role is to understand user requests, coordinate multiple agents, and synthesize results.

IMPORTANT: Agent Communication
You have access to the call_other_agent method to communicate with other agents:
- Use await self.call_other_agent("agent_name", "message") to call agents
- Available agents: keyword, grep, chunk, summarize
- Agents are configured in config/agents.json

Pipeline Flow:
1. Understand the user's request using the understand_user_request tool
2. Plan the execution strategy with plan_pipeline_execution
3. Coordinate agents using coordinate_agents:
   - Call keyword agent to generate search patterns
   - Call grep agent to search the document
   - Call chunk agent to extract context
   - Call summarize agent to analyze chunks
4. Synthesize results with synthesize_final_response
5. Handle any errors with handle_pipeline_errors

Key Responsibilities:
- Parse natural language requests to understand intent
- Determine optimal analysis strategies
- Coordinate agent execution in the correct sequence
- Pass results between agents appropriately
- Synthesize comprehensive responses from pipeline results
- Handle errors gracefully with recovery strategies
- Learn from execution history to optimize performance

When calling agents:
- Keyword Agent: Send document preview and focus areas
- Grep Agent: Send patterns and document content
- Chunk Agent: Send match information and document content
- Summarize Agent: Send extracted chunks for analysis

Error Handling:
- If an agent fails, use partial results when possible
- Provide fallback strategies
- Always try to deliver useful information to the user
- Explain limitations clearly

Result Synthesis:
- Combine results from all agents coherently
- Highlight the most relevant findings
- Provide appropriate medical context
- Format responses based on user needs

Remember: You orchestrate the pipeline by calling other agents via call_other_agent. The tools help you understand, plan, and synthesize - but actual agent coordination happens through agent calls."""
    
    def get_tools(self) -> List:
        """Return the orchestration tools."""
        return ORCHESTRATOR_TOOLS
    
    def get_agent_skills(self) -> List[AgentSkill]:
        """Return agent skills for the AgentCard."""
        return [
            AgentSkill(
                id="understand_request",
                name="Understand User Request",
                description="Parse and understand natural language requests for document analysis.",
                tags=["nlp", "understanding", "intent-parsing", "request-analysis"],
                examples=[
                    "Find all cancer-related information",
                    "Summarize the patient's medications",
                    "Extract diagnosis and treatment plan",
                    "What procedures were performed?"
                ],
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"]
            ),
            AgentSkill(
                id="plan_execution",
                name="Plan Pipeline Execution",
                description="Create optimal execution plan for document analysis.",
                tags=["planning", "strategy", "optimization", "pipeline"],
                examples=[
                    "Plan search for diagnosis information",
                    "Strategy for medication analysis",
                    "Optimize for comprehensive summary",
                    "Focus on recent findings"
                ],
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"]
            ),
            AgentSkill(
                id="coordinate_pipeline",
                name="Coordinate Agent Pipeline",
                description="Coordinate multiple agents to analyze documents.",
                tags=["coordination", "orchestration", "pipeline", "multi-agent"],
                examples=[
                    "Run full pipeline on medical record",
                    "Coordinate search and extraction",
                    "Execute multi-stage analysis",
                    "Process document through all agents"
                ],
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"]
            ),
            AgentSkill(
                id="synthesize_results",
                name="Synthesize Results",
                description="Combine pipeline results into comprehensive response.",
                tags=["synthesis", "aggregation", "response-generation"],
                examples=[
                    "Combine all findings into summary",
                    "Create clinical overview",
                    "Generate comprehensive report",
                    "Synthesize multi-agent results"
                ],
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"]
            ),
            AgentSkill(
                id="handle_errors",
                name="Handle Pipeline Errors",
                description="Recover from errors with fallback strategies.",
                tags=["error-handling", "recovery", "fallback", "resilience"],
                examples=[
                    "Recover from agent failure",
                    "Use partial results",
                    "Apply fallback patterns",
                    "Graceful degradation"
                ],
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"]
            ),
            AgentSkill(
                id="optimize_performance",
                name="Optimize Performance",
                description="Learn from history to optimize pipeline execution.",
                tags=["optimization", "learning", "performance", "efficiency"],
                examples=[
                    "Optimize pattern generation",
                    "Improve search efficiency",
                    "Enhance chunk quality",
                    "Tune summarization parameters"
                ],
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"]
            )
        ]
    
    def supports_streaming(self) -> bool:
        """Enable streaming for real-time orchestration updates."""
        return True
    
    async def process_message(self, message: str) -> str:
        """
        This won't be called when tools are provided.
        The base handles everything via Google ADK LlmAgent.
        The LLM will use tools and call_other_agent to orchestrate.
        """
        return "Handled by tool execution and agent coordination"


# Module-level app creation for HealthUniverse deployment
agent = OrchestratorAgent()
agent_card = agent.create_agent_card()
task_store = InMemoryTaskStore()
request_handler = DefaultRequestHandler(
    agent_executor=agent,
    task_store=task_store
)

# Create the app - MUST be named 'app' for HealthUniverse
app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler
).build()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8006))
    
    print(f"🚀 Starting {agent.get_agent_name()}")
    print(f"📍 Server: http://localhost:{port}")
    print(f"📋 Agent Card: http://localhost:{port}/.well-known/agent-card.json")
    print(f"🛠️ Tools available: {len(agent.get_tools())}")
    print("\n🔗 Coordinating agents:")
    
    # Load and display agent configuration
    config_path = Path(__file__).parent.parent.parent / "config" / "agents.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            for name, info in config.get("agents", {}).items():
                print(f"  - {name}: {info.get('description', 'No description')}")
    
    print("\n📝 Example queries:")
    print('  - "Find all information about cancer diagnosis and treatment"')
    print('  - "Summarize the patient\'s current medications and dosages"')
    print('  - "Extract all lab results and their clinical significance"')
    print('  - "What procedures were performed and what were the outcomes?"')
    print('  - "Provide a comprehensive clinical summary of this document"')
    
    uvicorn.run(app, host="0.0.0.0", port=port)