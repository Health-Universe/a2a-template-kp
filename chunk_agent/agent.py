"""
Content Chunk Extraction Agent
Extracts contextual chunks around search matches for further analysis.
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


class ChunkExtractionAgent(A2AAgent):
    """
    Agent that extracts contextual chunks around search matches.
    Returns structured data with content chunks and metadata.
    """

    # Configuration constants
    DEFAULT_LINES_BEFORE = 3
    DEFAULT_LINES_AFTER = 3
    MAX_CHUNK_SIZE = 50  # Maximum lines per chunk

    # --- A2A Metadata ---
    def get_agent_name(self) -> str:
        return "Content Chunk Extractor"

    def get_agent_description(self) -> str:
        return (
            "Extracts contextual chunks of content around search matches. "
            "Provides relevant context windows for better understanding of found content. "
            "Returns structured chunks with metadata for downstream analysis."
        )

    def get_agent_version(self) -> str:
        return "1.0.0"

    def get_agent_skills(self) -> List[AgentSkill]:
        return [
            AgentSkill(
                id="extract_chunks",
                name="Content Chunk Extraction",
                description="Extract contextual content chunks around search matches",
                tags=["chunk", "context", "extraction", "text"],
                inputModes=["application/json"],
                outputModes=["application/json"],
            )
        ]

    def supports_streaming(self) -> bool:
        return True  # Required by Health Universe platform

    def get_system_instruction(self) -> str:
        return (
            "You are a content chunk extraction specialist. Your role is to extract "
            "meaningful contextual chunks around search matches. Focus on: "
            "1) Preserving relevant context "
            "2) Maintaining readability "
            "3) Avoiding content duplication "
            "4) Providing useful metadata "
            "Extract chunks that provide complete context for understanding the matches."
        )

    # --- Core Processing ---
    async def process_message(self, message: str) -> Union[Dict[str, Any], str]:
        """
        Extract chunks from search results and document.
        Returns dict with chunks (will be wrapped in DataPart).
        """
        try:
            # Parse input - expect structured data with search results and document
            chunk_data = self._parse_chunk_input(message)
            
            if "error" in chunk_data:
                return chunk_data
            
            # Extract components
            search_results = chunk_data.get("search_results", {})
            document = chunk_data.get("document", "")
            
            # Extract matches from search results
            matches = search_results.get("matches", [])
            
            # Extract chunks
            chunks = await self._extract_chunks(matches, document)
            
            # Structure results
            results = {
                "chunks": chunks,
                "metadata": {
                    "extractor": "chunk_agent_v1",
                    "document_length": len(document),
                    "total_chunks": len(chunks),
                    "matches_processed": len(matches),
                    "avg_chunk_size": sum(len(c.get("content", "")) for c in chunks) / max(len(chunks), 1)
                }
            }
            
            # Return dict directly - base agent will wrap in DataPart
            return results
            
        except Exception as e:
            logger.error(f"Error extracting chunks: {e}")
            return {
                "error": str(e),
                "chunks": []
            }

    def _parse_chunk_input(self, message: str) -> Dict[str, Any]:
        """Parse input message to extract chunk parameters."""
        try:
            # Try to parse as JSON first
            data = json.loads(message)
            
            if isinstance(data, dict):
                # Extract search results
                search_results = data.get("search_results", {})
                
                # Extract document
                document = data.get("document", "")
                
                return {
                    "search_results": search_results,
                    "document": document
                }
            else:
                return {"error": "Invalid input format - expected structured data"}
                
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input"}
    
    async def _extract_chunks(self, matches: List[Dict[str, Any]], document: str) -> List[Dict[str, Any]]:
        """
        Extract contextual chunks around matches.
        """
        if not matches:
            # If no matches, create a summary chunk of the document
            return self._create_summary_chunks(document)
        
        chunks = []
        doc_lines = document.split('\n')
        
        # Configuration
        CONTEXT_LINES = int(os.getenv("CHUNK_CONTEXT_LINES", "3"))  # Lines before/after match
        MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "50"))   # Minimum chunk character length
        MAX_CHUNK_LENGTH = int(os.getenv("MAX_CHUNK_LENGTH", "1000")) # Maximum chunk character length
        
        processed_lines = set()  # Track lines already included in chunks to avoid duplication
        
        for match in matches:
            chunk = self._extract_single_chunk(
                match, doc_lines, CONTEXT_LINES, MIN_CHUNK_LENGTH, MAX_CHUNK_LENGTH, processed_lines
            )
            
            if chunk:
                chunks.append(chunk)
        
        # Merge overlapping chunks
        chunks = self._merge_overlapping_chunks(chunks)
        
        # Sort by line number
        chunks.sort(key=lambda x: x.get("start_line", 0))
        
        # Limit number of chunks
        MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "50"))
        if len(chunks) > MAX_CHUNKS:
            logger.info(f"Limiting chunks to {MAX_CHUNKS} (found {len(chunks)})")
            chunks = chunks[:MAX_CHUNKS]
        
        return chunks

    def _extract_single_chunk(
        self,
        match: Dict[str, Any],
        doc_lines: List[str],
        context_lines: int,
        min_length: int,
        max_length: int,
        processed_lines: set
    ) -> Dict[str, Any]:
        """Extract a single chunk around a match."""
        
        line_num = match.get("line_number", 1)
        
        # Calculate chunk boundaries
        start_line = max(1, line_num - context_lines)
        end_line = min(len(doc_lines), line_num + context_lines)
        
        # Extract lines (convert to 0-based indexing)
        chunk_lines = doc_lines[start_line-1:end_line]
        content = '\n'.join(chunk_lines)
        
        # Skip if too short
        if len(content.strip()) < min_length:
            return None
        
        # Truncate if too long
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        # Create chunk object
        chunk = {
            "content": content.strip(),
            "start_line": start_line,
            "end_line": end_line,
            "match_line": line_num,
            "match_info": {
                "type": match.get("type", "unknown"),
                "matched_text": match.get("matched_text", ""),
                "keyword": match.get("keyword"),
                "pattern": match.get("pattern")
            },
            "metadata": {
                "lines_count": end_line - start_line + 1,
                "char_count": len(content),
                "context_lines_before": line_num - start_line,
                "context_lines_after": end_line - line_num
            }
        }
        
        # Mark lines as processed
        for line in range(start_line, end_line + 1):
            processed_lines.add(line)
        
        return chunk

    def _create_summary_chunks(self, document: str) -> List[Dict[str, Any]]:
        """Create summary chunks when no matches are found."""
        doc_lines = document.split('\n')
        chunks = []
        
        CHUNK_SIZE = int(os.getenv("SUMMARY_CHUNK_SIZE", "10"))  # Lines per chunk
        
        for i in range(0, len(doc_lines), CHUNK_SIZE):
            chunk_lines = doc_lines[i:i + CHUNK_SIZE]
            content = '\n'.join(chunk_lines).strip()
            
            if content:
                chunk = {
                    "content": content,
                    "start_line": i + 1,
                    "end_line": min(i + CHUNK_SIZE, len(doc_lines)),
                    "match_line": None,
                    "match_info": {"type": "summary", "matched_text": None},
                    "metadata": {
                        "lines_count": len(chunk_lines),
                        "char_count": len(content),
                        "is_summary": True
                    }
                }
                chunks.append(chunk)
        
        # Limit summary chunks
        MAX_SUMMARY_CHUNKS = int(os.getenv("MAX_SUMMARY_CHUNKS", "5"))
        return chunks[:MAX_SUMMARY_CHUNKS]

    def _merge_overlapping_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge chunks that overlap significantly."""
        if len(chunks) <= 1:
            return chunks
        
        merged_chunks = []
        
        # Sort by start line
        chunks.sort(key=lambda x: x.get("start_line", 0))
        
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            current_end = current_chunk.get("end_line", 0)
            next_start = next_chunk.get("start_line", 0)
            
            # Check if chunks overlap significantly (at least 2 lines)
            overlap = max(0, current_end - next_start + 1)
            
            if overlap >= 2:
                # Merge chunks
                merged_content = self._merge_chunk_content(current_chunk, next_chunk)
                
                current_chunk = {
                    "content": merged_content,
                    "start_line": current_chunk.get("start_line"),
                    "end_line": next_chunk.get("end_line"),
                    "match_line": current_chunk.get("match_line"),  # Keep first match line
                    "match_info": self._merge_match_info(current_chunk, next_chunk),
                    "metadata": {
                        "lines_count": next_chunk.get("end_line", 0) - current_chunk.get("start_line", 0) + 1,
                        "char_count": len(merged_content),
                        "merged": True,
                        "original_chunks": 2
                    }
                }
            else:
                # No overlap, keep current chunk and move to next
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        # Add the last chunk
        merged_chunks.append(current_chunk)
        
        return merged_chunks

    def _merge_chunk_content(self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]) -> str:
        """Merge content from two overlapping chunks."""
        content1 = chunk1.get("content", "")
        content2 = chunk2.get("content", "")
        
        # Simple merge - combine unique lines
        lines1 = content1.split('\n')
        lines2 = content2.split('\n')
        
        # Create a set to track unique lines while preserving order
        seen_lines = set()
        merged_lines = []
        
        for line in lines1 + lines2:
            line_key = line.strip().lower()
            if line_key and line_key not in seen_lines:
                seen_lines.add(line_key)
                merged_lines.append(line)
        
        return '\n'.join(merged_lines)

    def _merge_match_info(self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge match information from two chunks."""
        match1 = chunk1.get("match_info", {})
        match2 = chunk2.get("match_info", {})
        
        # Combine matched texts
        texts = [match1.get("matched_text"), match2.get("matched_text")]
        texts = [t for t in texts if t and t.strip()]
        
        return {
            "type": "merged",
            "matched_text": " | ".join(texts) if texts else None,
            "original_types": [match1.get("type"), match2.get("type")]
        }

    def _validate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean chunk results."""
        valid_chunks = []
        
        for chunk in chunks:
            # Ensure required fields
            if not isinstance(chunk, dict):
                continue
                
            # Validate chunk structure
            required_fields = ["content", "start_line", "end_line"]
            if not all(field in chunk for field in required_fields):
                logger.warning(f"Invalid chunk structure, skipping: {list(chunk.keys())}")
                continue
            
            # Validate content is not empty
            if not chunk.get("content", "").strip():
                continue
            
            # Clean and validate data types
            try:
                chunk["start_line"] = int(chunk["start_line"])
                chunk["end_line"] = int(chunk["end_line"])
                chunk["content"] = str(chunk["content"])
                
                # Ensure reasonable line numbers
                if chunk["start_line"] <= 0 or chunk["end_line"] < chunk["start_line"]:
                    logger.warning(f"Invalid line numbers in chunk: {chunk['start_line']}-{chunk['end_line']}")
                    continue
                
                valid_chunks.append(chunk)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid chunk data types, skipping: {e}")
                continue
        
        return valid_chunks

    def _create_chunk_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """Create a text summary of extracted chunks."""
        if not chunks:
            return "No content chunks were extracted."
        
        summary_lines = [f"Extracted {len(chunks)} content chunks:"]
        
        total_chars = sum(len(c.get("content", "")) for c in chunks)
        avg_size = total_chars / len(chunks)
        
        summary_lines.append(f"Total content: {total_chars} characters")
        summary_lines.append(f"Average chunk size: {avg_size:.1f} characters")
        
        # Show line coverage
        all_lines = set()
        for chunk in chunks:
            start = chunk.get("start_line", 0)
            end = chunk.get("end_line", 0)
            all_lines.update(range(start, end + 1))
        
        if all_lines:
            min_line = min(all_lines)
            max_line = max(all_lines)
            summary_lines.append(f"Line coverage: {min_line}-{max_line} ({len(all_lines)} lines)")
        
        return "\n".join(summary_lines)


# --- Module-level app creation for Health Universe deployment ---
agent = ChunkExtractionAgent()
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
    port = int(os.getenv("PORT", 8004))
    print(f"🚀 Starting {agent.get_agent_name()}")
    print(f"📍 Available at: http://localhost:{port}")
    print(f"🔍 Agent Card: http://localhost:{port}/.well-known/agentcard.json")
    uvicorn.run(app, host="0.0.0.0", port=port)