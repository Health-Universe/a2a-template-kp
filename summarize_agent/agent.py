"""
Document Summarizer Agent
Analyzes content chunks and generates structured summaries.
Health Universe compatible with A2A compliance.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a.types import AgentSkill
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from base import A2AAgent
from utils.logging import get_logger


logger = get_logger(__name__)


class DocumentSummarizerAgent(A2AAgent):
    """
    LLM-powered agent that analyzes content chunks and generates structured summaries.
    """

    # --- A2A Metadata ---
    def get_agent_name(self) -> str:
        return "Document Summarizer"

    def get_agent_description(self) -> str:
        return (
            "Analyzes content chunks to generate comprehensive document summaries. "
            "Creates structured analysis including key findings, themes, and insights. "
            "Provides human-readable summaries suitable for review and decision-making."
        )

    def get_agent_version(self) -> str:
        return "1.0.0"

    def get_agent_skills(self) -> List[AgentSkill]:
        return [
            AgentSkill(
                id="summarize_content",
                name="Content Summarization",
                description="Analyze content chunks and generate structured summaries with key insights",
                tags=["summary", "analysis", "nlp", "insights"],
                inputModes=["application/json"],
                outputModes=["text/plain", "text/markdown"],
            )
        ]

    def supports_streaming(self) -> bool:
        return True  # Required by Health Universe platform

    def get_system_instruction(self) -> str:
        return (
            "You are a document analysis and summarization specialist. Your role is to "
            "analyze content chunks and create comprehensive summaries. Focus on: "
            "1) Key findings and important information "
            "2) Main themes and patterns "
            "3) Actionable insights "
            "4) Clear, structured presentation "
            "Create summaries that are informative, well-organized, and easy to understand."
        )

    # --- Core Processing ---
    async def process_message(self, message: str) -> str:
        """
        Analyze chunks and generate summary.
        Returns string summary (will be wrapped in TextPart).
        """
        try:
            # Parse input - expect structured data with chunks
            chunk_data = self._parse_chunk_input(message)
            
            if "error" in chunk_data:
                return f"Error parsing input: {chunk_data['error']}"
            
            # Extract chunks
            chunks = chunk_data.get("chunks", [])
            
            if not chunks:
                return "No content chunks provided for summarization."
            
            # Generate summary using LLM
            summary = await self._generate_summary(chunks)
            
            # Return string directly - base agent will wrap in TextPart
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._create_fallback_summary(chunk_data.get("chunks", []))

    def _parse_chunk_input(self, message: str) -> Dict[str, Any]:
        """Parse input message to extract chunks."""
        try:
            # Try to parse as JSON first
            data = json.loads(message)
            
            if isinstance(data, dict):
                # Extract chunks array
                chunks = data.get("chunks", [])
                
                return {
                    "chunks": chunks,
                    "metadata": data.get("metadata", {})
                }
            else:
                return {"error": "Invalid input format - expected structured data"}
                
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input"}

    async def _generate_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate summary using LLM with structured output."""
        
        # Prepare content for LLM
        chunk_contents = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "").strip()
            if content:
                match_info = chunk.get("match_info", {})
                matched_text = match_info.get("matched_text", "")
                
                chunk_text = f"Chunk {i}"
                if matched_text:
                    chunk_text += f" (Match: '{matched_text}')"
                chunk_text += f":\n{content}\n"
                
                chunk_contents.append(chunk_text)
        
        combined_content = "\n---\n".join(chunk_contents)
        
        # Limit content length for LLM processing
        MAX_CONTENT_LENGTH = int(os.getenv("MAX_SUMMARY_INPUT", "4000"))
        if len(combined_content) > MAX_CONTENT_LENGTH:
            combined_content = combined_content[:MAX_CONTENT_LENGTH] + "\n\n[Content truncated...]"
        
        prompt = f"""Analyze the following content chunks and create a comprehensive summary.

Content Chunks:
{combined_content}

Please provide:
1. **Executive Summary**: A brief overview of the main findings (2-3 sentences)
2. **Key Findings**: The most important information discovered (bullet points)
3. **Themes and Patterns**: Any recurring themes or patterns identified
4. **Detailed Analysis**: More detailed insights from the content
5. **Recommendations**: Any actionable recommendations or next steps (if applicable)

Format your response in clear markdown with appropriate headers."""
        
        try:
            # Use LLM utility if available
            from utils.llm_utils import generate_text
            
            summary = await generate_text(
                prompt=prompt,
                system_instruction=self.get_system_instruction(),
                temperature=0.5,  # Balanced creativity for summary writing
                max_tokens=2000
            )
            
            if not summary:
                return self._create_fallback_summary(chunks)
            
            return summary
            
        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}, using fallback")
            return self._create_fallback_summary(chunks)

    def _create_fallback_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """Create basic summary without LLM."""
        if not chunks:
            return "## Document Summary\n\nNo content was available for summarization."
        
        summary_lines = ["# Document Analysis Summary\n"]
        
        # Basic statistics
        total_chunks = len(chunks)
        total_chars = sum(len(c.get("content", "")) for c in chunks)
        
        summary_lines.append(f"**Analysis Overview:**")
        summary_lines.append(f"- Processed {total_chunks} content chunks")
        summary_lines.append(f"- Total content: {total_chars:,} characters")
        summary_lines.append("")
        
        # Extract key information
        matches = []
        for chunk in chunks:
            match_info = chunk.get("match_info", {})
            matched_text = match_info.get("matched_text")
            if matched_text and matched_text.strip():
                matches.append(matched_text.strip())
        
        if matches:
            # Remove duplicates while preserving order
            unique_matches = []
            seen = set()
            for match in matches:
                if match.lower() not in seen:
                    seen.add(match.lower())
                    unique_matches.append(match)
            
            summary_lines.append("## Key Terms Found")
            for match in unique_matches[:10]:  # Top 10 matches
                summary_lines.append(f"- {match}")
            summary_lines.append("")
        
        # Content preview
        summary_lines.append("## Content Preview")
        for i, chunk in enumerate(chunks[:3], 1):  # First 3 chunks
            content = chunk.get("content", "").strip()
            if content:
                # Truncate long content
                if len(content) > 200:
                    content = content[:200] + "..."
                
                summary_lines.append(f"**Chunk {i}:**")
                summary_lines.append(content)
                summary_lines.append("")
        
        if len(chunks) > 3:
            summary_lines.append(f"*... and {len(chunks) - 3} more chunks*")
            summary_lines.append("")
        
        # Analysis notes
        summary_lines.append("## Analysis Notes")
        summary_lines.append("This is a basic summary created without advanced language processing.")
        summary_lines.append("For more detailed analysis, please review the individual content chunks.")
        summary_lines.append("")
        
        # Line coverage information
        all_lines = set()
        for chunk in chunks:
            start = chunk.get("start_line")
            end = chunk.get("end_line")
            if start and end:
                all_lines.update(range(start, end + 1))
        
        if all_lines:
            min_line = min(all_lines)
            max_line = max(all_lines)
            summary_lines.append(f"**Document Coverage:** Lines {min_line}-{max_line} ({len(all_lines)} total lines)")
        
        return "\n".join(summary_lines)

    def _extract_themes(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract themes using simple keyword analysis."""
        themes = []
        
        # Combine all content
        all_content = " ".join(c.get("content", "") for c in chunks).lower()
        
        # Common medical/document themes
        theme_keywords = {
            "Medical Findings": ["diagnosis", "symptoms", "condition", "disease", "disorder"],
            "Test Results": ["test", "result", "lab", "blood", "normal", "abnormal"],
            "Treatment": ["treatment", "therapy", "medication", "prescription", "dosage"],
            "Patient Information": ["patient", "age", "history", "background"],
            "Recommendations": ["recommend", "suggest", "should", "consider", "follow-up"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in all_content for keyword in keywords):
                themes.append(theme)
        
        return themes

    def _create_detailed_analysis(self, chunks: List[Dict[str, Any]]) -> str:
        """Create detailed analysis section."""
        analysis_lines = []
        
        # Analyze chunk distribution
        line_ranges = []
        for chunk in chunks:
            start = chunk.get("start_line")
            end = chunk.get("end_line")
            if start and end:
                line_ranges.append((start, end))
        
        if line_ranges:
            line_ranges.sort()
            analysis_lines.append("**Content Distribution:**")
            analysis_lines.append(f"- Content spans from line {line_ranges[0][0]} to line {line_ranges[-1][1]}")
            analysis_lines.append(f"- Found relevant content in {len(line_ranges)} sections")
            analysis_lines.append("")
        
        # Match type analysis
        match_types = {}
        for chunk in chunks:
            match_info = chunk.get("match_info", {})
            match_type = match_info.get("type", "unknown")
            match_types[match_type] = match_types.get(match_type, 0) + 1
        
        if match_types:
            analysis_lines.append("**Match Types:**")
            for match_type, count in match_types.items():
                analysis_lines.append(f"- {match_type.title()}: {count} chunks")
            analysis_lines.append("")
        
        return "\n".join(analysis_lines) if analysis_lines else ""


# --- Module-level app creation for Health Universe deployment ---
agent = DocumentSummarizerAgent()
agent_card = agent.create_agent_card()
task_store = InMemoryTaskStore()
request_handler = DefaultRequestHandler(
    agent_executor=agent,
    task_store=task_store
)

app = A2AStarletteApplication(
    agent_card=agent_card,  # A2A Spec: MUST make AgentCard available
    http_handler=request_handler  # Handles RPC methods
).build()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8005))
    print(f"🚀 Starting {agent.get_agent_name()}")
    print(f"📍 Available at: http://localhost:{port}")
    print(f"🔍 Agent Card: http://localhost:{port}/.well-known/agentcard.json")
    uvicorn.run(app, host="0.0.0.0", port=port)