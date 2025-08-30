#!/usr/bin/env python3
"""
Temporal Tagging Agent - Extracts temporal information from medical documents.
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
from tools.temporal_tools import (
    extract_temporal_information,
    consolidate_temporal_data,
    analyze_temporal_patterns,
    tag_timeline_segments,
    normalize_dates
)
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill


class TemporalTaggingAgent(A2AAgent):
    """Agent that extracts temporal information from medical text."""
    
    def get_agent_name(self) -> str:
        return "Temporal Tagging Agent"
    
    def get_agent_description(self) -> str:
        return (
            "Extracts and analyzes temporal information from medical documents, "
            "identifying dates, temporal relationships, and event sequences."
        )
    
    def get_agent_version(self) -> str:
        return "1.0.0"
    
    def get_system_instruction(self) -> str:
        return """You are an expert at identifying and extracting temporal information from medical documents. 
        
Your role is to analyze medical records and extract ALL temporal information while maintaining data integrity.

IMPORTANT: Some documents may not contain any dates. In such cases, still extract the clinical content but mark them as having "Unknown Date" rather than creating fake dates.

When analyzing documents, you should:

1. **IDENTIFY ALL DATES:**
   - Explicit dates (MM/DD/YYYY, DD-Mon-YYYY, etc.)
   - Partial dates (Month YYYY, YYYY only)
   - Relative dates ("last month", "3 weeks ago", "previous visit")
   - Referenced dates ("on the previous scan", "at initial diagnosis")
   - Date ranges ("from May to July 2023")
   - If NO DATES found, mark content as "Unknown Date"

2. **CLASSIFY EACH DATE:**
   - ENCOUNTER DATE: When patient actually had visit/procedure/test
   - REFERENCE DATE: Date mentioned but referring to past events
   - REPORT DATE: When a report was generated
   - COLLECTION DATE: When specimen/data was collected
   - UNKNOWN: When no date information is available

3. **EXTRACT TEXT SEGMENTS:**
   For each meaningful piece of clinical information, identify:
   - The text content
   - Its primary associated date (when it happened) - use "Unknown Date" if no date found
   - Any referenced dates it mentions
   - Whether it's reporting new information or referencing previous findings
   - Temporal relationships and indicators

4. **TEMPORAL RELATIONSHIPS:**
   - Identify when text refers to previous events ("previously showed", "prior scan")
   - Note carry-forward information (repeated from earlier reports)
   - Identify temporal sequences ("following chemotherapy", "before surgery")
   - Maintain temporal context even when dates are missing

5. **TAG AND ORGANIZE:**
   - Tag timeline segments for chronological ordering
   - Normalize various date formats
   - Create structured timeline

CRITICAL RULES:
- NEVER create synthetic or fake dates
- Always preserve "Unknown Date" for content without temporal information
- Extract clinical content even when no dates are present
- Maintain data integrity - no assumptions about missing dates

Use the provided tools to:
1. extract_temporal_information - Extract dates and temporal data from text
2. consolidate_temporal_data - Consolidate multiple temporal extractions
3. analyze_temporal_patterns - Identify patterns in temporal data
4. tag_timeline_segments - Tag and organize timeline segments
5. normalize_dates - Normalize various date formats to standard format

Return structured temporal data that accurately represents the document's temporal information or lack thereof."""
    
    def get_tools(self) -> List:
        """Return the temporal extraction tools."""
        return [
            FunctionTool(func=extract_temporal_information),
            FunctionTool(func=consolidate_temporal_data),
            FunctionTool(func=analyze_temporal_patterns),
            FunctionTool(func=tag_timeline_segments),
            FunctionTool(func=normalize_dates)
        ]
    
    def get_agent_skills(self) -> List[AgentSkill]:
        """Define the temporal tagging skills."""
        return [
            AgentSkill(
                id="temporal_extraction",
                name="Temporal Information Extraction",
                description="Extract dates, times, and temporal relationships from medical text",
                tags=["dates", "temporal", "timeline", "chronology"],
                examples=[
                    "Extract all dates from this discharge summary",
                    "Identify temporal relationships between medical events",
                    "Find all encounter dates in this patient record"
                ]
            ),
            AgentSkill(
                id="date_classification",
                name="Date Classification",
                description="Classify dates as encounter, reference, report, or collection dates",
                tags=["classification", "dates", "temporal"],
                examples=[
                    "Classify which dates are actual visits vs references",
                    "Identify report generation dates vs clinical event dates"
                ]
            ),
            AgentSkill(
                id="temporal_consolidation",
                name="Temporal Data Consolidation",
                description="Consolidate temporal information from multiple sources",
                tags=["consolidation", "merging", "temporal"],
                examples=[
                    "Merge temporal data from multiple documents",
                    "Consolidate duplicate temporal references"
                ]
            )
        ]
    
    def supports_streaming(self) -> bool:
        """Enable streaming for real-time temporal extraction."""
        return True
    
    async def process_message(self, message: str) -> str:
        # This won't be called when tools are provided
        return "Processing temporal information..."


# Module-level app creation (required for deployment)
agent = TemporalTaggingAgent()
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
    port = int(os.getenv("PORT", 8010))
    print(f"🚀 Starting {agent.get_agent_name()}")
    print(f"📍 Server: http://localhost:{port}")
    print(f"📋 Agent Card: http://localhost:{port}/.well-known/agent-card.json")
    uvicorn.run(app, host="0.0.0.0", port=port)