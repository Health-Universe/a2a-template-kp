# summarize_agent.py
from __future__ import annotations

import json
import logging
from typing import Dict, Any, Optional, List, Callable

from base import A2AAgent
from a2a.types import AgentSkill
from google.adk.tools import FunctionTool


class SummarizeAgent(A2AAgent):
    """
    LLM-powered summarization agent that creates intelligent summaries from medical documents.
    Follows the simple_orchestrator pattern with tools for LLM interaction.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__()
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Bind the tool to this instance
        def _tool(chunk_content: str, chunk_metadata: Optional[Dict[str, Any]] = None, 
                 summary_style: str = "clinical") -> str:
            return self._summarize_with_llm(chunk_content, chunk_metadata, summary_style)
        
        self._summarize_tool: Callable[[str, Optional[Dict[str, Any]], str], str] = _tool

    # --------------- A2A metadata --------------- #
    def get_agent_name(self) -> str:
        return "CS Pipeline - Summarization"

    def get_agent_description(self) -> str:
        return (
            "LLM-powered agent that creates intelligent summaries from medical document chunks. "
            "Extracts key clinical information and generates structured summaries."
        )

    def get_agent_version(self) -> str:
        return "1.0.0"

    def get_agent_skills(self) -> List[AgentSkill]:
        return [
            AgentSkill(
                id="summarize_chunks",
                name="Summarize Document Chunks",
                description="Create intelligent summary from extracted text chunks using LLM",
                tags=["summarize", "extract", "medical", "analysis", "llm"],
                input_modes=["text/plain", "application/json"],
                output_modes=["text/plain", "text/markdown"],
            )
        ]

    def supports_streaming(self) -> bool:
        return False

    def get_system_instruction(self) -> str:
        return """You are a medical document summarization expert. Your role is to create concise, accurate summaries of medical document chunks.

When using the summarize_chunks tool, you should:
1. Extract key medical information from the provided chunks
2. Identify diagnoses, medications, procedures, and vital signs
3. Create structured summaries appropriate for clinical use
4. Preserve critical medical details while removing redundancy
5. Organize information in a logical, clinically relevant manner

Summarization guidelines:
- Focus on medically significant information
- Preserve exact medication names and dosages
- Include all diagnoses and conditions mentioned
- Note important dates and temporal relationships
- Highlight abnormal findings or critical values
- Maintain clinical accuracy and precision

Summary styles:
- clinical: Structured format for healthcare providers
- patient: Simplified language for patient understanding
- research: Detailed with emphasis on data and findings
- administrative: Focus on procedures and billing codes

Output format for clinical style:
1. Chief Findings
2. Diagnoses/Conditions
3. Medications
4. Procedures/Tests
5. Vital Signs/Labs
6. Follow-up/Recommendations

Always ensure medical accuracy and never omit critical information."""

    def get_tools(self) -> List:
        """Expose the summarization tool for LLM use."""
        return [FunctionTool(func=self._summarize_tool)]

    # --------------- Core Processing --------------- #
    async def process_message(self, message: str) -> str:
        """
        Process incoming message. With tools, the LLM will handle this.
        For direct calls, we process the message ourselves.
        """
        try:
            # Try to parse as JSON first
            data = json.loads(message)
            
            # Extract content and metadata
            content = data.get("chunk_content", data.get("content", ""))
            metadata = data.get("chunk_metadata", data.get("metadata", {}))
            summary_style = data.get("summary_style", "clinical")
            
            # Create summary using LLM
            summary = self._summarize_with_llm(content, metadata, summary_style)
            
            return summary
            
        except json.JSONDecodeError:
            # Plain text - create basic summary
            summary = self._summarize_with_llm(message, {}, "clinical")
            return summary

    def _summarize_with_llm(self, chunk_content: str, 
                           chunk_metadata: Optional[Dict[str, Any]] = None, 
                           summary_style: str = "clinical") -> str:
        """
        Tool function that creates an intelligent summary using LLM guidance.
        Returns formatted summary based on the specified style.
        """
        if not chunk_content or not chunk_content.strip():
            return "No content available for summarization."
        
        # Build the summarization prompt
        prompt = self._build_summary_prompt(chunk_content, chunk_metadata, summary_style)
        
        # For the tool-based approach, we return the prompt that guides the LLM
        # The actual LLM processing happens in the A2A framework
        return prompt

    def _build_summary_prompt(self, content: str, metadata: Dict[str, Any], style: str) -> str:
        """
        Build a detailed prompt for the LLM to create the summary.
        """
        prompt = f"""Create a {style} summary of the following medical document chunks.

"""
        
        # Add metadata context if available
        if metadata:
            prompt += "Document Context:\n"
            if "source" in metadata:
                prompt += f"- Source: {metadata['source']}\n"
            if "total_matches" in metadata:
                prompt += f"- Total pattern matches: {metadata['total_matches']}\n"
            if "chunks_extracted" in metadata:
                prompt += f"- Chunks analyzed: {metadata['chunks_extracted']}\n"
            prompt += "\n"
        
        prompt += f"Content to summarize:\n{content}\n\n"
        
        # Add style-specific instructions
        if style == "clinical":
            prompt += """Create a clinical summary with the following structure:

## Clinical Summary

### Key Findings
- List the most important medical findings

### Diagnoses/Conditions
- List all mentioned diagnoses and medical conditions

### Medications
- List all medications with dosages and frequencies

### Procedures/Tests
- List any procedures, surgeries, or diagnostic tests

### Vital Signs/Lab Values
- Include any vital signs or laboratory results

### Recommendations/Follow-up
- Note any treatment plans or follow-up instructions

Focus on medical accuracy and completeness. Use bullet points for clarity."""
        
        elif style == "patient":
            prompt += """Create a patient-friendly summary that:
- Uses simple, non-technical language
- Explains medical terms when necessary
- Focuses on what the patient needs to know
- Organizes information clearly
- Avoids medical jargon

Structure:
1. Main Health Issues
2. Medications You're Taking
3. Tests or Procedures Done
4. What to Do Next"""
        
        elif style == "research":
            prompt += """Create a research-oriented summary that:
- Emphasizes data and measurements
- Includes all quantitative findings
- Notes methodologies if mentioned
- Preserves technical terminology
- Highlights statistical significance

Structure:
1. Clinical Presentation
2. Diagnostic Findings
3. Interventions
4. Outcomes/Results
5. Data Points"""
        
        else:  # administrative or default
            prompt += """Create an administrative summary focusing on:
- Procedures performed (with codes if available)
- Diagnoses (primary and secondary)
- Medications prescribed
- Follow-up requirements
- Documentation completeness"""
        
        return prompt