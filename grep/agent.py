"""
Pattern Search (Grep) Agent
Searches documents using regex patterns with intelligent error handling.
This is a pure algorithmic agent - no LLM needed for the core search functionality.
"""

import json
import re
from typing import List, Dict, Any, Optional, Union

from a2a.types import AgentSkill
from base import A2AAgent
from utils.logging import get_logger

logger = get_logger(__name__)


class GrepAgent(A2AAgent):
    """
    Pattern search agent that searches documents using regex patterns.
    Pure algorithmic implementation - no LLM required for searching.
    """

    # Configuration
    MAX_MATCHES_PER_PATTERN = 100  # Limit matches per pattern to avoid memory issues
    CONTEXT_LINES_BEFORE = 2
    CONTEXT_LINES_AFTER = 2

    # --- A2A Metadata ---
    def get_agent_name(self) -> str:
        return "CS Pipeline - Grep"

    def get_agent_description(self) -> str:
        return (
            "High-performance regex search agent for medical documents. "
            "Handles pattern validation, error recovery, and context extraction."
        )

    def get_agent_version(self) -> str:
        return "2.0.0"  # Template-based version

    def get_agent_skills(self) -> List[AgentSkill]:
        return [
            AgentSkill(
                id="search_patterns",
                name="Search Document Patterns",
                description="Search for regex patterns in documents with error handling",
                tags=["grep", "search", "pattern", "regex", "medical"],
                inputModes=["application/json"],
                outputModes=["application/json"],
            ),
            AgentSkill(
                id="validate_patterns",
                name="Validate Regex Patterns",
                description="Validate and test regex patterns before searching",
                tags=["regex", "validation"],
                inputModes=["application/json"],
                outputModes=["application/json"],
            )
        ]

    def supports_streaming(self) -> bool:
        return False

    def get_system_instruction(self) -> str:
        return (
            "You are a high-performance pattern search agent. "
            "Execute regex searches efficiently with proper error handling."
        )

    # --- Core Processing ---
    async def process_message(self, message: str) -> Union[Dict[str, Any], str]:
        """
        Search document with regex patterns.
        Input: JSON with patterns, document_content, and optional case_sensitive flag
        Output: Dict with matches array and search metadata (will be wrapped in DataPart)
        """
        try:
            # Parse input
            data = self._parse_input(message)
            
            # Extract parameters
            patterns = data.get("patterns", [])
            document = data.get("document_content", data.get("document", ""))
            case_sensitive = data.get("case_sensitive", False)
            
            # Validate inputs
            if not patterns:
                return {
                    "error": "No patterns provided",
                    "matches": [],
                    "total_matches": 0
                }
            
            if not document:
                return {
                    "error": "No document content provided",
                    "matches": [],
                    "total_matches": 0
                }
            
            # Normalize patterns input
            if isinstance(patterns, str):
                patterns = [patterns]
            elif isinstance(patterns, dict):
                # Handle structured pattern objects from keyword agent
                patterns = self._extract_patterns_from_structured(patterns)
            
            # Perform search
            results = self._search_document(patterns, document, case_sensitive)
            
            return results  # Return dict for DataPart
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                "error": str(e),
                "matches": [],
                "total_matches": 0
            }

    def _parse_input(self, message: str) -> Dict[str, Any]:
        """Parse input message."""
        try:
            data = json.loads(message)
            if isinstance(data, dict):
                return data
            return {"document": message}
        except:
            return {"document": message}

    def _extract_patterns_from_structured(self, patterns_data: Dict) -> List[str]:
        """Extract pattern strings from structured pattern data."""
        patterns = []
        
        # Handle categorized patterns from keyword agent
        for category in ["section_patterns", "clinical_patterns", "medication_patterns", 
                        "temporal_patterns", "vital_patterns", "term_patterns"]:
            if category in patterns_data:
                for pattern_obj in patterns_data[category]:
                    if isinstance(pattern_obj, dict) and "pattern" in pattern_obj:
                        patterns.append(pattern_obj["pattern"])
                    elif isinstance(pattern_obj, str):
                        patterns.append(pattern_obj)
        
        # Also check for direct patterns list
        if "patterns" in patterns_data:
            for p in patterns_data["patterns"]:
                if isinstance(p, str):
                    patterns.append(p)
        
        return patterns

    def _search_document(self, patterns: List[str], document: str, case_sensitive: bool) -> Dict[str, Any]:
        """
        Perform the actual search operation.
        Returns structured results with matches and metadata.
        """
        matches = []
        errors = []
        patterns_searched = 0
        
        # Split document into lines for line-based searching
        # If no newlines, try to intelligently split on sentence boundaries
        lines = document.splitlines()
        if len(lines) == 1 and len(document) > 500:
            # Document is one long line - try to split on periods followed by space and capital
            import re
            # Split on periods followed by space, but preserve the period
            sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', document)
            if len(sentences) > 1:
                lines = sentences
                logger.info(f"Document had no newlines, split into {len(lines)} sentences")
            else:
                # Last resort: split on any period followed by space
                lines = re.split(r'(?<=\.)\s+', document)
                if len(lines) == 1:
                    # Still one line - leave as is
                    logger.warning("Document is one continuous line without clear sentence breaks")
        
        # Create line index for faster context extraction
        line_index = {i: line for i, line in enumerate(lines)}
        
        # Search each pattern
        for pattern in patterns:
            patterns_searched += 1
            pattern_matches = self._search_single_pattern(
                pattern, lines, line_index, case_sensitive
            )
            
            if isinstance(pattern_matches, dict) and "error" in pattern_matches:
                errors.append(pattern_matches)
            else:
                matches.extend(pattern_matches)
            
            # Limit total matches to prevent memory issues
            if len(matches) > 1000:
                logger.warning(f"Truncating matches at 1000 (searched {patterns_searched}/{len(patterns)} patterns)")
                break
        
        # Sort matches by line number
        matches.sort(key=lambda x: x.get("line_number", 0))
        
        # Build result with type identifier
        result = {
            "type": "grep.result.v1",  # Type identifier for explicit contract
            "source": "grep",
            "matches": matches,
            "total_matches": len(matches),
            "patterns_searched": patterns_searched,
            "patterns_total": len(patterns),
            "document_lines": len(lines)
        }
        
        if errors:
            result["errors"] = errors
        
        # Debug logging
        logger.info(f"Grep search complete: {len(matches)} total matches from {patterns_searched} patterns across {len(lines)} lines")
        if matches:
            # Log sample of unique line numbers
            unique_lines = set(m.get("line_number", 0) for m in matches[:20])
            logger.debug(f"Sample of match line numbers: {sorted(unique_lines)[:10]}")
        
        return result

    def _search_single_pattern(self, pattern: str, lines: List[str], 
                              line_index: Dict[int, str], case_sensitive: bool) -> List[Dict[str, Any]]:
        """Search for a single pattern and return matches or error."""
        matches = []
        
        try:
            # Compile regex with appropriate flags
            flags = re.IGNORECASE | re.MULTILINE if not case_sensitive else re.MULTILINE
            regex = re.compile(pattern, flags)
            
            # Search through lines
            for line_num, line in enumerate(lines):
                # Find all matches in this line
                for match in regex.finditer(line):
                    # Create match info
                    match_info = {
                        "pattern": pattern,
                        "line_number": line_num + 1,  # 1-based line numbers
                        "match_text": match.group(),
                        "line_content": line,
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                        "file_path": "document.txt",
                    }
                    
                    # Add context lines
                    match_info["context_before"] = self._get_context_before(
                        line_num, line_index, self.CONTEXT_LINES_BEFORE
                    )
                    match_info["context_after"] = self._get_context_after(
                        line_num, line_index, self.CONTEXT_LINES_AFTER
                    )
                    
                    # Add the document content for chunk agent (but not in the matches)
                    # This will be passed separately in the orchestrator
                    match_info["document"] = "\n".join(lines)
                    
                    matches.append(match_info)
                    
                    # Limit matches per pattern
                    if len(matches) >= self.MAX_MATCHES_PER_PATTERN:
                        logger.debug(f"Pattern '{pattern}' hit match limit of {self.MAX_MATCHES_PER_PATTERN}")
                        return matches
            
            return matches
            
        except re.error as e:
            # Return error info for this pattern
            logger.warning(f"Invalid regex pattern '{pattern}': {e}")
            return {
                "error": f"Invalid regex pattern",
                "pattern": pattern,
                "details": str(e),
                "suggestion": self._suggest_pattern_fix(pattern, str(e))
            }
        except Exception as e:
            logger.error(f"Unexpected error searching pattern '{pattern}': {e}")
            return {
                "error": "Search failed",
                "pattern": pattern,
                "details": str(e)
            }

    def _get_context_before(self, line_num: int, line_index: Dict[int, str], num_lines: int) -> List[str]:
        """Get context lines before the match."""
        context = []
        for i in range(max(0, line_num - num_lines), line_num):
            if i in line_index:
                context.append(line_index[i])
        return context

    def _get_context_after(self, line_num: int, line_index: Dict[int, str], num_lines: int) -> List[str]:
        """Get context lines after the match."""
        context = []
        for i in range(line_num + 1, min(len(line_index), line_num + num_lines + 1)):
            if i in line_index:
                context.append(line_index[i])
        return context

    def _suggest_pattern_fix(self, pattern: str, error: str) -> str:
        """Suggest a fix for an invalid regex pattern."""
        error_lower = error.lower()
        
        if "unbalanced parenthesis" in error_lower or "missing )" in error_lower:
            open_count = pattern.count("(")
            close_count = pattern.count(")")
            if open_count > close_count:
                return f"Missing {open_count - close_count} closing parenthesis ')'. Check: {pattern}"
            elif close_count > open_count:
                return f"Extra closing parenthesis. Check: {pattern}"
            else:
                return f"Check parentheses pairing and escaping in: {pattern}"
                
        elif "bad escape" in error_lower or "bad character" in error_lower:
            return f"Invalid escape sequence. Use raw strings (r'...') or double backslashes. Pattern: {pattern}"
            
        elif "nothing to repeat" in error_lower:
            # Find the problematic quantifier
            for q in ["*", "+", "?", "{"]:
                if q in pattern:
                    idx = pattern.find(q)
                    if idx == 0 or (idx > 0 and pattern[idx-1] in "*+?{"):
                        return f"Quantifier '{q}' at position {idx} needs a pattern before it"
            return f"Quantifier (*,+,?,{{}}) needs something before it in: {pattern}"
            
        elif "bad character range" in error_lower:
            return f"Invalid character range in [...]. Check dash placement: {pattern}"
            
        elif "multiple repeat" in error_lower:
            return f"Cannot stack quantifiers (like **). Simplify: {pattern}"
            
        else:
            # Generic suggestion
            simplified = self._simplify_pattern(pattern)
            if simplified != pattern:
                return f"Try simplified version: {simplified}"
            return "Check regex syntax. Test at regex101.com with Python flavor"

    def _simplify_pattern(self, pattern: str) -> str:
        """Attempt to simplify a complex pattern."""
        # Remove excessive escaping
        simplified = pattern.replace(r"\\", "\\")
        
        # Simplify common issues
        simplified = re.sub(r'\*+', '*', simplified)  # Multiple * to single *
        simplified = re.sub(r'\++', '+', simplified)  # Multiple + to single +
        simplified = re.sub(r'\?+', '?', simplified)  # Multiple ? to single ?
        
        # Remove problematic lookarounds if present
        simplified = re.sub(r'\(\?[=!<].*?\)', '', simplified)
        
        return simplified if simplified != pattern else pattern