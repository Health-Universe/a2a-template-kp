#!/usr/bin/env python3
"""
Keyword Pattern Generator Agent - Generates regex patterns from document previews.
Migrated from ADK demo pipeline to use enhanced A2AAgent base.

This agent analyzes document structure and generates patterns for:
- Section headers and boundaries
- Medical terminology and abbreviations
- Clinical findings and events
- Temporal markers and dates
"""

import os
import sys
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from base import A2AAgent
from tools.keyword_tools import KEYWORD_TOOLS
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill


class KeywordAgent(A2AAgent):
    """LLM-powered keyword pattern generator for medical documents."""
    
    def get_agent_name(self) -> str:
        """Return the agent's name."""
        return "Keyword Pattern Generator"
    
    def get_agent_description(self) -> str:
        """Return detailed agent description."""
        return (
            "LLM-powered agent that analyzes medical document previews and generates "
            "regex patterns for identifying key information such as diagnoses, medications, "
            "procedures, and temporal markers."
        )
    
    def get_agent_version(self) -> str:
        """Return agent version."""
        return "2.0.0"
    
    def get_system_instruction(self) -> str:
        """Return the system instruction for pattern generation."""
        return """You are a medical document pattern generator. Your role is to analyze medical document previews and generate regex patterns that can identify important information.

When given a document preview, you should:
1. Identify the document structure and format
2. Recognize medical terminology and abbreviations
3. Generate regex patterns that match:
   - Section headers and boundaries
   - Clinical findings and events
   - Medical terms and their variations
   - Temporal markers and dates

Guidelines for pattern generation:
- Use ripgrep-compatible regex syntax
- Include case-insensitive flags (?i) where appropriate
- Create patterns specific enough to avoid false positives
- Include both abbreviated and full forms of medical terms
- Consider different formatting styles (structured vs narrative)

Return patterns organized by category:
- section_patterns: Headers and document structure
- clinical_patterns: Medical findings and events
- term_patterns: Medical terminology
- temporal_patterns: Dates and time references

Each pattern should include:
- pattern: The regex pattern string
- priority: high/medium/low based on importance
- description: What the pattern matches

Always validate patterns for regex syntax correctness before returning them."""
    
    def get_tools(self) -> List:
        """Return the keyword generation tools."""
        return KEYWORD_TOOLS
    
    def get_agent_skills(self) -> List[AgentSkill]:
        """Return agent skills for the AgentCard."""
        return [
            AgentSkill(
                id="generate_document_patterns",
                name="Generate Document Patterns",
                description="Analyze document preview and generate regex patterns for key information.",
                tags=["pattern-generation", "regex", "document-analysis", "medical"],
                examples=[
                    "Generate patterns from first 20 lines of document",
                    "Identify section headers and boundaries",
                    "Create patterns for medical terminology",
                    "Find temporal markers and dates"
                ],
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"]
            ),
            AgentSkill(
                id="generate_focused_patterns",
                name="Generate Focused Patterns",
                description="Generate patterns for specific medical concepts or sections.",
                tags=["focused-search", "medical-concepts", "targeted-patterns"],
                examples=[
                    "Generate patterns for diagnosis sections only",
                    "Find medication-related patterns",
                    "Create patterns for lab results",
                    "Identify procedure patterns"
                ],
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"]
            ),
            AgentSkill(
                id="analyze_preview_structure",
                name="Analyze Document Structure",
                description="Analyze document preview to understand structure and format.",
                tags=["structure-analysis", "document-format", "preview-analysis"],
                examples=[
                    "Identify document format (structured vs narrative)",
                    "Detect section organization",
                    "Analyze formatting patterns",
                    "Determine document type"
                ],
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"]
            ),
            AgentSkill(
                id="validate_patterns",
                name="Validate Patterns",
                description="Validate generated regex patterns for correctness.",
                tags=["validation", "regex", "quality-check"],
                examples=[
                    "Validate regex syntax",
                    "Check pattern efficiency",
                    "Verify pattern specificity",
                    "Test for false positives"
                ],
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"]
            )
        ]
    
    def supports_streaming(self) -> bool:
        """Enable streaming for real-time pattern generation."""
        return True
    
    async def process_message(self, message: str) -> str:
        """
        This won't be called when tools are provided.
        The base handles everything via Google ADK LlmAgent.
        """
        return "Handled by tool execution"


# Module-level app creation for HealthUniverse deployment
agent = KeywordAgent()
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
    port = int(os.getenv("PORT", 8002))
    
    print(f"🚀 Starting {agent.get_agent_name()}")
    print(f"📍 Server: http://localhost:{port}")
    print(f"📋 Agent Card: http://localhost:{port}/.well-known/agent-card.json")
    print(f"🛠️ Tools available: {len(agent.get_tools())}")
    print("\n📝 Example queries:")
    print('  - "Analyze this document preview and generate search patterns"')
    print('  - "Generate patterns for finding diagnosis sections"')
    print('  - "Create regex patterns for dates and times"')
    print('  - "Validate these regex patterns: [...]"')
    
    uvicorn.run(app, host="0.0.0.0", port=port)