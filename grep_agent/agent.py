"""
Document Search Agent (Grep)
Searches documents using keywords and patterns to find relevant content.
Health Universe compatible with A2A compliance.
"""

import json
import os
import sys
import re
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


class DocumentSearchAgent(A2AAgent):
    """
    Agent that searches documents using keywords and regex patterns.
    Returns structured data with matches and locations.
    """
    
    # Configuration constants
    MAX_MATCHES_PER_PATTERN = int(os.getenv("MAX_GREP_MATCHES", "100"))
    CONTEXT_LINES_BEFORE = int(os.getenv("CONTEXT_LINES_BEFORE", "3"))
    CONTEXT_LINES_AFTER = int(os.getenv("CONTEXT_LINES_AFTER", "3"))

    # --- A2A Metadata ---
    def get_agent_name(self) -> str:
        return "Document Search Agent"

    def get_agent_description(self) -> str:
        return (
            "Searches documents using provided keywords and regex patterns. "
            "Finds relevant content matches with line numbers and context. "
            "Returns structured results for further processing by downstream agents."
        )

    def get_agent_version(self) -> str:
        return "1.0.0"

    def get_agent_skills(self) -> List[AgentSkill]:
        return [
            AgentSkill(
                id="search_document",
                name="Document Search",
                description="Search documents using keywords and patterns to find relevant content",
                tags=["search", "grep", "text", "pattern"],
                inputModes=["application/json"],
                outputModes=["application/json"],
            )
        ]

    def supports_streaming(self) -> bool:
        return True  # Required by Health Universe platform

    def get_system_instruction(self) -> str:
        return (
            "You are a document search specialist. Your role is to find relevant content "
            "in documents using provided keywords and patterns. Focus on: "
            "1) Accurate pattern matching "
            "2) Context preservation "
            "3) Comprehensive coverage "
            "4) Structured result reporting "
            "Be thorough but efficient in your search operations."
        )

    # --- Core Processing ---
    async def process_message(self, message: str) -> Union[Dict[str, Any], str]:
        """
        Search document using provided keywords and patterns.
        Returns dict with matches (will be wrapped in DataPart).
        """
        try:
            # Parse input - expect structured data with keywords and document
            search_data = self._parse_search_input(message)
            
            if "error" in search_data:
                return search_data
            
            # Extract components
            keywords = search_data.get("keywords", {})
            document = search_data.get("document", "")
            patterns = search_data.get("patterns", [])
            
            # Perform search
            matches = await self._search_document(keywords, patterns, document)
            
            # Structure results
            results = {
                "matches": matches,
                "metadata": {
                    "searcher": "grep_agent_v1",
                    "document_length": len(document),
                    "total_matches": len(matches),
                    "keywords_used": len(keywords.get("keywords", [])),
                    "patterns_used": len(patterns)
                }
            }
            
            # Return dict directly - base agent will wrap in DataPart
            return results
            
        except Exception as e:
            logger.error(f"Error searching document: {e}")
            return {
                "error": str(e),
                "matches": []
            }

    def _parse_search_input(self, message: str) -> Dict[str, Any]:
        """Parse input message to extract search parameters."""
        try:
            # Try to parse as JSON first
            data = json.loads(message)
            
            if isinstance(data, dict):
                # Extract keywords object
                keywords = data.get("keywords", {})
                
                # Extract document
                document = data.get("document", "")
                
                # Extract patterns from keywords if available
                patterns = []
                if isinstance(keywords, dict):
                    patterns = keywords.get("patterns", [])
                    # Also look for patterns in top level
                    patterns.extend(data.get("patterns", []))
                
                return {
                    "keywords": keywords,
                    "document": document,
                    "patterns": patterns
                }
            else:
                return {"error": "Invalid input format - expected structured data"}
                
        except json.JSONDecodeError:
            # If not JSON, try to extract from text
            return self._parse_text_input(message)
    
    def _parse_text_input(self, message: str) -> Dict[str, Any]:
        """Fallback parser for text input."""
        # Simple text input - treat as keywords to search in the same text
        words = message.split()
        keywords = {"keywords": words[:10]}  # Limit to first 10 words as keywords
        
        return {
            "keywords": keywords,
            "document": message,
            "patterns": []
        }

    async def _search_document(
        self, 
        keywords: Dict[str, Any], 
        patterns: List[str], 
        document: str
    ) -> List[Dict[str, Any]]:
        """
        Search document using keywords and patterns.
        Returns list of match objects with context.
        """
        matches = []
        doc_lines = document.split('\n')
        
        # Extract keyword list
        keyword_list = keywords.get("keywords", [])
        if not isinstance(keyword_list, list):
            keyword_list = []
        
        # Search using keywords
        for keyword in keyword_list:
            if not isinstance(keyword, str) or not keyword.strip():
                continue
                
            keyword = keyword.strip()
            matches.extend(self._search_keyword(keyword, doc_lines))
        
        # Search using patterns
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
                
            try:
                matches.extend(self._search_pattern(pattern, doc_lines))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                continue
        
        # Remove duplicates while preserving order
        unique_matches = []
        seen_locations = set()
        
        for match in matches:
            location_key = (match.get("line_number"), match.get("start_pos"))
            if location_key not in seen_locations:
                seen_locations.add(location_key)
                unique_matches.append(match)
        
        # Sort by line number, then by position
        unique_matches.sort(key=lambda x: (x.get("line_number", 0), x.get("start_pos", 0)))
        
        # Limit results to prevent overwhelming downstream agents
        MAX_MATCHES = int(os.getenv("MAX_GREP_MATCHES", "100"))
        if len(unique_matches) > MAX_MATCHES:
            logger.info(f"Limiting results to {MAX_MATCHES} matches (found {len(unique_matches)})")
            unique_matches = unique_matches[:MAX_MATCHES]
        
        return unique_matches

    def _search_keyword(self, keyword: str, doc_lines: List[str]) -> List[Dict[str, Any]]:
        """Search for a keyword in document lines."""
        matches = []
        
        for line_num, line in enumerate(doc_lines, 1):
            # Case-insensitive search for keywords
            line_lower = line.lower()
            keyword_lower = keyword.lower()
            
            start_pos = 0
            while True:
                pos = line_lower.find(keyword_lower, start_pos)
                if pos == -1:
                    break
                
                match = {
                    "type": "keyword",
                    "keyword": keyword,
                    "line_number": line_num,
                    "line_content": line.strip(),
                    "start_pos": pos,
                    "end_pos": pos + len(keyword),
                    "matched_text": line[pos:pos + len(keyword)],
                    "context_before": line[:pos].strip()[-50:] if pos > 0 else "",
                    "context_after": line[pos + len(keyword):].strip()[:50],
                }
                
                matches.append(match)
                start_pos = pos + 1  # Continue searching for overlapping matches
        
        return matches

    def _search_pattern(self, pattern: str, doc_lines: List[str]) -> List[Dict[str, Any]]:
        """Search for a regex pattern in document lines."""
        matches = []
        
        try:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}': {e}")
            return matches
        
        for line_num, line in enumerate(doc_lines, 1):
            for match in regex.finditer(line):
                match_obj = {
                    "type": "pattern",
                    "pattern": pattern,
                    "line_number": line_num,
                    "line_content": line.strip(),
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "matched_text": match.group(0),
                    "context_before": line[:match.start()].strip()[-50:] if match.start() > 0 else "",
                    "context_after": line[match.end():].strip()[:50],
                }
                
                # Add capture groups if present
                if match.groups():
                    match_obj["groups"] = match.groups()
                
                matches.append(match_obj)
        
        return matches

    def _validate_search_results(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean search results."""
        valid_matches = []
        
        for match in matches:
            # Ensure required fields
            if not isinstance(match, dict):
                continue
                
            # Validate match structure
            required_fields = ["type", "line_number", "line_content", "matched_text"]
            if not all(field in match for field in required_fields):
                logger.warning(f"Invalid match structure, skipping: {match}")
                continue
            
            # Clean and validate data types
            try:
                match["line_number"] = int(match["line_number"])
                match["start_pos"] = int(match.get("start_pos", 0))
                match["end_pos"] = int(match.get("end_pos", 0))
                
                # Ensure string fields are strings
                for field in ["type", "line_content", "matched_text"]:
                    match[field] = str(match[field])
                
                valid_matches.append(match)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid match data types, skipping: {e}")
                continue
        
        return valid_matches

    # --- Helper Methods for Complex Searches ---
    def _search_multi_line_patterns(self, pattern: str, document: str) -> List[Dict[str, Any]]:
        """Search for patterns that may span multiple lines."""
        matches = []
        
        try:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            for match in regex.finditer(document):
                # Calculate line number for match start
                lines_before = document[:match.start()].count('\n')
                line_number = lines_before + 1
                
                match_obj = {
                    "type": "multiline_pattern",
                    "pattern": pattern,
                    "line_number": line_number,
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "matched_text": match.group(0),
                    "context_before": document[max(0, match.start()-100):match.start()],
                    "context_after": document[match.end():match.end()+100],
                }
                
                # Add capture groups if present
                if match.groups():
                    match_obj["groups"] = match.groups()
                
                matches.append(match_obj)
                
        except re.error as e:
            logger.warning(f"Invalid multiline regex pattern '{pattern}': {e}")
        
        return matches

    def _create_search_summary(self, matches: List[Dict[str, Any]]) -> str:
        """Create a text summary of search results."""
        if not matches:
            return "No matches found in the document."
        
        summary_lines = [f"Found {len(matches)} matches:"]
        
        # Group by type
        keyword_matches = [m for m in matches if m.get("type") == "keyword"]
        pattern_matches = [m for m in matches if m.get("type") == "pattern"]
        
        if keyword_matches:
            summary_lines.append(f"\nKeyword matches: {len(keyword_matches)}")
            # Show top keywords
            keyword_counts = {}
            for match in keyword_matches:
                keyword = match.get("keyword", "unknown")
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
            top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for keyword, count in top_keywords:
                summary_lines.append(f"  - '{keyword}': {count} times")
        
        if pattern_matches:
            summary_lines.append(f"\nPattern matches: {len(pattern_matches)}")
        
        # Show line distribution
        lines_with_matches = set(m.get("line_number") for m in matches)
        summary_lines.append(f"\nMatches found on {len(lines_with_matches)} different lines")
        
        return "\n".join(summary_lines)


# --- Module-level app creation for Health Universe deployment ---
agent = DocumentSearchAgent()
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
    port = int(os.getenv("PORT", 8003))
    print(f"🚀 Starting {agent.get_agent_name()}")
    print(f"📍 Available at: http://localhost:{port}")
    print(f"🔍 Agent Card: http://localhost:{port}/.well-known/agentcard.json")
    uvicorn.run(app, host="0.0.0.0", port=port)