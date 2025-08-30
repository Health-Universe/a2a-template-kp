# template_agent/agent.py
"""
A minimal A2A-compliant agent template.
Demonstrates the simplest possible agent using the base class and utilities.
"""

import json
from typing import List, Optional
from a2a.types import AgentSkill
from base import A2AAgent
from utils.logging import get_logger
from utils.llm_utils import generate_text

logger = get_logger(__name__)


class TemplateAgent(A2AAgent):
    """
    A minimal agent template.
    - Reads text or JSON (DataPart) content from the request
    - Calls the LLM once and returns the answer as text
    - Everything else (Task lifecycle, events, errors) handled by base.py
    """

    # --- Required metadata ---
    def get_agent_name(self) -> str:
        return "Template Agent"

    def get_agent_description(self) -> str:
        return "A minimal A2A-compliant agent that answers questions using your configured LLM."

    # --- Optional metadata ---
    def get_agent_version(self) -> str:
        return "1.0.0"

    def get_system_instruction(self) -> str:
        """Provide a custom system instruction for this agent."""
        return (
            "You are a helpful AI assistant integrated into an A2A-compliant agent. "
            "Answer questions clearly and concisely. If you receive JSON data, "
            "understand it as context for your response."
        )

    def supports_streaming(self) -> bool:
        # Flip to True only if you implement streaming status/artifact updates
        return False

    def get_agent_skills(self) -> List[AgentSkill]:
        """Declare the agent's capabilities."""
        return [
            AgentSkill(
                id="qa",
                name="Question Answering",
                description="Answers a single question using the default LLM.",
                tags=["qa", "llm", "simple"],
                inputModes=["text/plain", "application/json"],
                outputModes=["text/plain"],
            ),
            AgentSkill(
                id="json-processing",
                name="JSON Data Processing",
                description="Process structured JSON data and provide insights.",
                tags=["json", "data", "analysis"],
                inputModes=["application/json"],
                outputModes=["text/plain"],
            )
        ]

    # --- Core logic ---
    async def process_message(self, message: str) -> str:
        """
        The base class has already extracted and concatenated all message parts.
        `message` can be plain text or a JSON string (if the caller used a DataPart).
        
        Args:
            message: Extracted message content (text or JSON string)
            
        Returns:
            LLM-generated response
        """
        logger.debug(f"TemplateAgent.process_message() input length={len(message)}")
        
        # Try to detect if the message is JSON and format accordingly
        context = ""
        try:
            # Attempt to parse as JSON
            data = json.loads(message)
            context = f"The user provided this structured data:\n{json.dumps(data, indent=2)}\n\n"
            prompt = f"{context}Please analyze or respond to this data appropriately."
        except (json.JSONDecodeError, TypeError):
            # It's plain text
            prompt = f"Answer the user clearly and concisely:\n\nUser message:\n{message}"
        
        # One-shot LLM call with auto-provider detection
        try:
            answer = await generate_text(
                prompt=prompt,
                system_instruction=self.get_system_instruction(),
                temperature=0.7,
                max_tokens=1000
            )
            return answer or "I couldn't generate a response. Please try again."
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"I encountered an error while processing your request: {str(e)}"

    # --- Optional: Override to add tools ---
    def get_tools(self) -> List:
        """
        Return tools available to this agent.
        Uncomment to enable example tools.
        """
        # from examples.template_agent.tools.example_tools import EXAMPLE_TOOLS
        # return EXAMPLE_TOOLS
        return []  # No tools by default

    # --- Optional: Inter-agent communication example ---
    async def call_downstream_agent(self, agent_name: str, message: str) -> str:
        """
        Example of calling another agent.
        This uses the base class's call_other_agent method.
        """
        try:
            response = await self.call_other_agent(agent_name, message)
            return response
        except Exception as e:
            logger.error(f"Failed to call agent {agent_name}: {e}")
            return f"Could not reach agent {agent_name}"