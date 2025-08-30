#!/usr/bin/env python3
"""
Pattern Search Agent - Searches documents with regex patterns.
Migrated from ADK demo pipeline to use enhanced A2AAgent base.

This agent provides intelligent pattern searching with:
- Regex pattern validation and error recovery
- Context lines around matches
- Performance analysis
- Support for external content (S3, etc.)
"""

import os
import sys
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from base import A2AAgent
from tools.grep_tools import GREP_TOOLS
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill


class GrepAgent(A2AAgent):
    """LLM-powered grep agent for intelligent pattern searching."""
    
    def get_agent_name(self) -> str:
        """Return the agent's name."""
        return "Pattern Search Agent"
    
    def get_agent_description(self) -> str:
        """Return detailed agent description."""
        return (
            "LLM-powered agent that searches medical documents using regex patterns "
            "with intelligent error handling and performance optimization."
        )
    
    def get_agent_version(self) -> str:
        """Return agent version."""
        return "2.0.0"
    
    def get_system_instruction(self) -> str:
        """Return the system instruction for pattern searching."""
        return """You are a medical document search specialist. Your role is to search medical documents using regex patterns and handle any errors intelligently.

When given search requests, you should:
1. Execute pattern searches efficiently across documents
2. Handle regex errors by suggesting pattern fixes
3. Use fallback patterns when primary patterns fail
4. Analyze search performance and suggest improvements
5. Provide clear summaries of search results

IMPORTANT: File Content Handling
When you receive a request with both 'file_path' and 'file_content':
- Always pass the 'file_content' parameter to the search_medical_patterns function
- The function will use the provided content instead of reading from the file system
- This allows searching content from S3 or other external sources

Example: If you receive {"patterns": [...], "file_path": "doc.txt", "file_content": "...content..."}, 
call search_medical_patterns(file_path="doc.txt", patterns=[...], file_content="...content...")

Key responsibilities:
- Validate patterns before searching to catch errors early
- If a pattern fails, try to fix it automatically
- Use context lines to provide meaningful search results
- Handle large result sets by prioritizing most relevant matches
- Detect and report performance issues

Search guidelines:
- Case-insensitive search by default for medical terms
- Include sufficient context (3-5 lines) around matches
- Limit results to avoid overwhelming responses
- Group similar matches when appropriate
- Handle file errors gracefully

For error handling:
- Invalid regex: Suggest corrected patterns
- No matches: Recommend broader patterns
- Too many matches: Suggest more specific patterns
- File errors: Provide clear error messages

Always aim to return useful results even when encountering errors."""
    
    def get_tools(self) -> List:
        """Return the grep search tools."""
        return GREP_TOOLS
    
    def get_agent_skills(self) -> List[AgentSkill]:
        """Return agent skills for the AgentCard."""
        return [
            AgentSkill(
                id="search_medical_patterns",
                name="Search Medical Patterns",
                description="Search documents for regex patterns with context and error handling.",
                tags=["search", "regex", "pattern-matching", "medical", "text-processing"],
                examples=[
                    "Find all mentions of diagnosis in the document",
                    "Search for date patterns like MM/DD/YYYY",
                    "Locate section headers ending with colon",
                    "Find medication names and dosages",
                    "Search for diagnostic codes (ICD-10)"
                ],
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"]
            ),
            AgentSkill(
                id="validate_and_fix_patterns",
                name="Validate and Fix Patterns",
                description="Validate regex patterns and suggest fixes for common errors.",
                tags=["validation", "regex", "error-handling"],
                examples=[
                    "Fix unbalanced parentheses in patterns",
                    "Escape special characters properly",
                    "Validate pattern syntax before searching"
                ],
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"]
            ),
            AgentSkill(
                id="search_with_error_recovery",
                name="Search with Error Recovery",
                description="Search with automatic fallback to alternative patterns on failure.",
                tags=["search", "error-recovery", "fallback"],
                examples=[
                    "Try alternative patterns if primary fails",
                    "Recover from regex compilation errors",
                    "Use broader patterns as fallback"
                ],
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"]
            ),
            AgentSkill(
                id="analyze_search_performance",
                name="Analyze Search Performance",
                description="Analyze search results to identify issues and optimization opportunities.",
                tags=["analysis", "performance", "optimization"],
                examples=[
                    "Identify patterns with too many matches",
                    "Detect patterns with no matches",
                    "Suggest pattern improvements"
                ],
                input_modes=["application/json", "text/plain"],
                output_modes=["application/json", "text/plain"]
            )
        ]
    
    def supports_streaming(self) -> bool:
        """Enable streaming for real-time updates."""
        return True
    
    async def process_message(self, message: str) -> str:
        """
        This won't be called when tools are provided.
        The base handles everything via Google ADK LlmAgent.
        """
        return "Handled by tool execution"


# Module-level app creation for HealthUniverse deployment
agent = GrepAgent()
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
    port = int(os.getenv("PORT", 8003))
    
    print(f"🚀 Starting {agent.get_agent_name()}")
    print(f"📍 Server: http://localhost:{port}")
    print(f"📋 Agent Card: http://localhost:{port}/.well-known/agent-card.json")
    print(f"🛠️ Tools available: {len(agent.get_tools())}")
    print("\n📝 Example queries:")
    print('  - "Search for all mentions of diagnosis in the document"')
    print('  - "Find date patterns like MM/DD/YYYY"')
    print('  - "Validate the regex pattern [0-9]+"')
    print('  - "Search for medication names with dosages"')
    
    uvicorn.run(app, host="0.0.0.0", port=port)