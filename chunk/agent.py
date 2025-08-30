"""
Text Chunk Extractor Agent
Extracts and formats text chunks from documents around match locations
with intelligent medical context preservation.
"""

import json
import re
from typing import Dict, Any, List, Optional

from a2a.types import AgentSkill
from base import A2AAgent
from utils.logging import get_logger

logger = get_logger(__name__)


class ChunkAgent(A2AAgent):
    """
    Text chunk extraction agent that extracts and formats chunks around matches.
    Pure algorithmic implementation - no LLM required for chunk extraction.
    """

    # Configuration
    DEFAULT_LINES_BEFORE = 3
    DEFAULT_LINES_AFTER = 3
    MAX_CHUNK_SIZE = 50  # Maximum lines per chunk

    # --- A2A Metadata ---
    def get_agent_name(self) -> str:
        return "CS Pipeline - Chunker"

    def get_agent_description(self) -> str:
        return (
            "Extracts and formats text chunks from medical documents around match locations "
            "with intelligent context preservation and medical term highlighting."
        )

    def get_agent_version(self) -> str:
        return "2.0.0"  # Template-based version

    def get_agent_skills(self) -> List[AgentSkill]:
        return [
            AgentSkill(
                id="extract_chunk",
                name="Extract Text Chunk",
                description="Extract and format text chunk with context around a match",
                tags=["chunk", "extract", "context", "text", "medical"],
                inputModes=["application/json"],
                outputModes=["text/plain", "text/markdown"],
            ),
            AgentSkill(
                id="smart_boundaries",
                name="Smart Boundary Detection",
                description="Intelligently adjust chunk boundaries to preserve medical context",
                tags=["medical", "context", "boundaries"],
                inputModes=["application/json"],
                outputModes=["text/plain"],
            )
        ]

    def supports_streaming(self) -> bool:
        return False

    def get_system_instruction(self) -> str:
        return (
            "You are a text chunk extraction specialist for medical documents. "
            "Extract meaningful chunks with appropriate context around matches."
        )

    # --- Core Processing ---
    async def process_message(self, message: str) -> str:
        """
        Extract and format a text chunk around a match.
        Input: JSON with match_info, optional lines_before/after
        Output: Formatted text chunk with medical context
        """
        try:
            # Parse input
            data = self._parse_input(message)
            
            # Extract parameters
            match_info = data.get("match_info", {})
            lines_before = data.get("lines_before", self.DEFAULT_LINES_BEFORE)
            lines_after = data.get("lines_after", self.DEFAULT_LINES_AFTER)
            
            # Validate input
            if not match_info:
                return "Error: No match_info provided"
            
            # Extract and format chunk
            chunk = self._extract_chunk(match_info, lines_before, lines_after)
            
            return chunk
            
        except Exception as e:
            logger.error(f"Chunk extraction error: {e}")
            return f"Error extracting chunk: {str(e)}"

    def _parse_input(self, message: str) -> Dict[str, Any]:
        """Parse input message."""
        try:
            data = json.loads(message)
            if isinstance(data, dict):
                return data
            return {"match_info": {}}
        except:
            return {"match_info": {}}

    def _extract_chunk(self, match_info: Dict[str, Any], 
                      lines_before: int, lines_after: int) -> str:
        """
        Extract and format a text chunk around the match.
        """
        # Get document content
        document = match_info.get("document", match_info.get("file_content", ""))
        
        # If no full document, use simple format
        if not document:
            return self._format_simple_chunk(match_info)
        
        # Get match details
        line_number = match_info.get("line_number", 1)
        pattern = match_info.get("pattern", "unknown")
        match_text = match_info.get("match_text", "")
        
        # Split document into lines
        lines = document.splitlines()
        
        # Calculate smart boundaries
        start_line, end_line = self._calculate_smart_boundaries(
            lines, line_number, lines_before, lines_after, match_text
        )
        
        # Build the formatted chunk
        return self._format_chunk(
            lines, start_line, end_line, line_number,
            pattern, match_text, match_info
        )

    def _calculate_smart_boundaries(self, lines: List[str], line_number: int,
                                   lines_before: int, lines_after: int,
                                   match_text: str) -> tuple:
        """
        Calculate smart chunk boundaries that preserve medical context.
        """
        # Basic boundaries (convert to 0-based indexing)
        line_idx = line_number - 1
        start_idx = max(0, line_idx - lines_before)
        end_idx = min(len(lines), line_idx + lines_after + 1)
        
        # Adjust start to sentence/section boundary
        if start_idx > 0:
            # Look backwards for a natural break
            for i in range(start_idx, max(0, start_idx - 3), -1):
                if i < len(lines):
                    line = lines[i].strip()
                    # Check for section headers or sentence starts
                    if (line and (
                        line[0].isupper() or
                        line.startswith(('•', '-', '*', '1.', '2.', '3.')) or
                        any(header in line.lower() for header in 
                            ['history:', 'assessment:', 'plan:', 'diagnosis:', 'medications:'])
                    )):
                        start_idx = i
                        break
        
        # Extend for medical context
        if self._contains_medical_info(match_text):
            # Check if we need more context for medications
            if any(unit in match_text.lower() for unit in ['mg', 'ml', 'mcg', 'units', 'iu']):
                # Look for medication name before dosage
                if start_idx > 0 and line_idx > 0:
                    prev_line = lines[line_idx - 1] if line_idx > 0 else ""
                    if any(char.isalpha() for char in prev_line):
                        start_idx = max(0, start_idx - 1)
                
                # Look for frequency/instructions after dosage
                if end_idx < len(lines) - 1:
                    next_line = lines[line_idx + 1] if line_idx < len(lines) - 1 else ""
                    if any(freq in next_line.lower() for freq in 
                          ['daily', 'twice', 'tid', 'bid', 'qid', 'prn', 'as needed']):
                        end_idx = min(len(lines), end_idx + 1)
            
            # Check for vital signs context
            if any(vital in match_text.lower() for vital in 
                  ['blood pressure', 'bp', 'heart rate', 'hr', 'temperature', 'temp']):
                # Include surrounding vital signs
                end_idx = min(len(lines), end_idx + 2)
        
        # Ensure we don't exceed max chunk size
        if end_idx - start_idx > self.MAX_CHUNK_SIZE:
            # Center around the match
            half_size = self.MAX_CHUNK_SIZE // 2
            start_idx = max(0, line_idx - half_size)
            end_idx = min(len(lines), line_idx + half_size + 1)
        
        return start_idx, end_idx

    def _format_chunk(self, lines: List[str], start_idx: int, end_idx: int,
                      match_line_num: int, pattern: str, match_text: str,
                      match_info: Dict[str, Any]) -> str:
        """
        Format the extracted chunk with proper highlighting and context.
        """
        chunk_lines = []
        
        # Header
        chunk_lines.append("=" * 50)
        chunk_lines.append("📄 Medical Document Chunk")
        chunk_lines.append("=" * 50)
        chunk_lines.append(f"🔍 Pattern: {pattern}")
        chunk_lines.append(f"✓ Match: '{match_text}' at line {match_line_num}")
        chunk_lines.append("-" * 50)
        chunk_lines.append("")
        
        # Context indicator if starting mid-document
        if start_idx > 0:
            chunk_lines.append(f"[...context from line {start_idx}...]")
            chunk_lines.append("")
        
        # Add lines with formatting
        match_idx = match_line_num - 1
        for i in range(start_idx, end_idx):
            if i >= len(lines):
                break
            
            line_num = i + 1
            line_text = lines[i]
            
            # Format based on whether it's the match line
            if i == match_idx:
                # Highlight the matching line
                chunk_lines.append(f">>> {line_num:4d}: {line_text}")
            else:
                # Regular context line with medical term detection
                formatted = self._format_context_line(line_text)
                chunk_lines.append(f"    {line_num:4d}: {formatted}")
        
        # Context indicator if ending mid-document
        if end_idx < len(lines):
            chunk_lines.append("")
            chunk_lines.append(f"[...continues at line {end_idx + 1}...]")
        
        chunk_lines.append("")
        chunk_lines.append("-" * 50)
        
        # Add medical information summary
        medical_summary = self._extract_medical_summary(
            lines[start_idx:end_idx], match_idx - start_idx
        )
        if medical_summary:
            chunk_lines.append("")
            chunk_lines.append("📊 Key Medical Information:")
            for item in medical_summary:
                chunk_lines.append(f"  • {item}")
        
        chunk_lines.append("=" * 50)
        
        return "\n".join(chunk_lines)

    def _format_simple_chunk(self, match_info: Dict[str, Any]) -> str:
        """
        Format a simple chunk when full document is not available.
        """
        pattern = match_info.get("pattern", "unknown")
        match_text = match_info.get("match_text", "")
        line_content = match_info.get("line_content", "No content available")
        line_number = match_info.get("line_number", "unknown")
        context_before = match_info.get("context_before", [])
        context_after = match_info.get("context_after", [])
        
        chunk_lines = [
            "=" * 50,
            "📄 Text Chunk (Limited Context)",
            "=" * 50,
            f"🔍 Pattern: {pattern}",
            f"✓ Match: '{match_text}'",
            f"📍 Line {line_number}",
            "-" * 50,
            ""
        ]
        
        # Add context before
        if context_before:
            chunk_lines.append("Context before:")
            for line in context_before:
                chunk_lines.append(f"    {line}")
            chunk_lines.append("")
        
        # Add match line
        chunk_lines.append(f">>> {line_content}")
        chunk_lines.append("")
        
        # Add context after
        if context_after:
            chunk_lines.append("Context after:")
            for line in context_after:
                chunk_lines.append(f"    {line}")
        
        chunk_lines.append("=" * 50)
        
        return "\n".join(chunk_lines)

    def _format_context_line(self, line_text: str) -> str:
        """
        Format a context line, potentially highlighting medical terms.
        """
        # For now, just return the line as-is
        # Could add medical term detection/highlighting here
        return line_text

    def _contains_medical_info(self, text: str) -> bool:
        """
        Check if text contains medical information that needs context.
        """
        medical_indicators = [
            r'\b\d+\s*(?:mg|mcg|ml|cc|units?|iu)\b',  # Dosages
            r'(?:blood\s+pressure|bp|heart\s+rate|hr|temperature|temp)',  # Vitals
            r'(?:diagnosis|diagnosed|treatment|prescribed)',  # Clinical terms
            r'(?:daily|bid|tid|qid|prn)',  # Frequencies
        ]
        
        text_lower = text.lower()
        for pattern in medical_indicators:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def _extract_medical_summary(self, chunk_lines: List[str], 
                                match_idx: int) -> List[str]:
        """
        Extract key medical information from the chunk.
        """
        summary = []
        chunk_text = ' '.join(chunk_lines).lower()
        
        # Look for medications with dosages
        med_pattern = r'\b(\w+)\s+(\d+(?:\.\d+)?\s*(?:mg|mcg|ml|units?|iu))\b'
        medications = re.findall(med_pattern, chunk_text, re.IGNORECASE)
        if medications:
            unique_meds = list(dict.fromkeys([f"{med[0].title()} {med[1]}" for med in medications]))
            if unique_meds:
                summary.append(f"Medications: {', '.join(unique_meds[:5])}")
        
        # Look for vital signs
        vitals = []
        
        # Blood pressure
        bp_pattern = r'(?:blood\s+pressure|bp)[\s:]+(\d{2,3}/\d{2,3})'
        bp_matches = re.findall(bp_pattern, chunk_text, re.IGNORECASE)
        if bp_matches:
            vitals.append(f"BP: {bp_matches[0]}")
        
        # Heart rate
        hr_pattern = r'(?:heart\s+rate|hr|pulse)[\s:]+(\d{2,3})'
        hr_matches = re.findall(hr_pattern, chunk_text, re.IGNORECASE)
        if hr_matches:
            vitals.append(f"HR: {hr_matches[0]}")
        
        # Temperature
        temp_pattern = r'(?:temperature|temp)[\s:]+(\d{2,3}(?:\.\d)?)'
        temp_matches = re.findall(temp_pattern, chunk_text, re.IGNORECASE)
        if temp_matches:
            vitals.append(f"Temp: {temp_matches[0]}°")
        
        if vitals:
            summary.append(f"Vital Signs: {', '.join(vitals)}")
        
        # Look for diagnoses
        diag_keywords = ['diagnosis', 'diagnosed with', 'assessment']
        for keyword in diag_keywords:
            if keyword in chunk_text:
                summary.append("Contains diagnostic information")
                break
        
        # Look for dates
        date_pattern = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'
        dates = re.findall(date_pattern, chunk_text)
        if dates:
            summary.append(f"Dates mentioned: {', '.join(list(dict.fromkeys(dates))[:3])}")
        
        return summary