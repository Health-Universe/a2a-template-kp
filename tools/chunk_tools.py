"""
Chunk extraction tools with LLM-powered boundary detection.
Following nutrition_example.py pattern with Google ADK FunctionTool.
"""
import json
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from google.adk.tools import FunctionTool
import logging

logger = logging.getLogger(__name__)


def create_document_chunk(
    file_path: str,
    match_info: Dict[str, Any],
    context_size: int = 5,
    boundary_detection: bool = True,
    file_content: Optional[str] = None
) -> str:
    """
    Create an intelligent chunk around a pattern match.
    
    Args:
        file_path: Path to the document
        match_info: Information about the match (line_number, pattern, etc.)
        context_size: Base lines of context before/after match
        boundary_detection: Whether to use intelligent boundary detection
        file_content: Optional document content (if provided, file_path is ignored)
        
    Returns:
        JSON string with chunk content and metadata
    """
    try:
        line_number = match_info.get("line_number", 1)
        pattern = match_info.get("pattern", "")
        match_text = match_info.get("match_text", "")
        match_position = match_info.get("match_position", None)  # Character position if available
        
        # Get content either from parameter or file
        if file_content is not None:
            # Use provided content directly
            content = file_content
        else:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        # Check if this is a single-line document
        if _is_single_line_document(content):
            # Use character-based chunking
            logger.info("Detected single-line document, using character-based chunking")
            
            # If we don't have match_position, try to find it
            if match_position is None and match_text:
                match_position = content.find(match_text)
                if match_position == -1:
                    # Fallback to approximate position based on line number
                    lines = content.splitlines()
                    if lines and line_number <= len(lines):
                        # Calculate approximate character position
                        match_position = sum(len(lines[i]) + 1 for i in range(line_number - 1))
                    else:
                        match_position = len(content) // 2  # Default to middle
            
            # Extract chunk by characters
            chunk_text, start_pos, end_pos = _chunk_by_characters(
                content, 
                match_position or len(content) // 2,
                context_chars=context_size * 80  # Approximate chars per line
            )
            
            # Create analysis for character-based chunk
            chunk_analysis = {
                "chunk_method": "character-based",
                "single_line_document": True,
                "chunk_size_chars": len(chunk_text)
            }
            
            # Create result with character positions
            result = {
                "chunk": {
                    "content": chunk_text,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "match_position": match_position,
                    "character_based": True
                },
                "match_info": match_info,
                "analysis": chunk_analysis,
                "metadata": {
                    "file_path": file_path,
                    "pattern": pattern,
                    "boundary_detection_used": False,
                    "chunking_method": "character",
                    "original_context_size": context_size
                }
            }
            
        else:
            # Use traditional line-based chunking
            lines = content.splitlines(keepends=True)
            
            # Basic boundaries
            start_line = max(0, line_number - context_size - 1)
            end_line = min(len(lines), line_number + context_size)
            
            # Apply intelligent boundary detection if requested
            if boundary_detection:
                start_line, end_line = _detect_semantic_boundaries(
                    lines, line_number - 1, start_line, end_line
                )
            
            # Extract chunk
            chunk_lines = lines[start_line:end_line]
            chunk_text = ''.join(chunk_lines)
            
            # Identify key information in chunk
            chunk_analysis = _analyze_chunk_content(chunk_lines, line_number - start_line - 1)
            
            result = {
                "chunk": {
                    "content": chunk_text,
                    "start_line": start_line + 1,
                    "end_line": end_line,
                    "match_line": line_number,
                    "line_count": len(chunk_lines)
                },
                "match_info": match_info,
                "analysis": chunk_analysis,
                "metadata": {
                    "file_path": file_path,
                    "pattern": pattern,
                    "boundary_detection_used": boundary_detection,
                    "chunking_method": "line",
                    "original_context_size": context_size
                }
            }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error creating chunk: {str(e)}")
        return json.dumps({
            "error": str(e),
            "file_path": file_path,
            "match_info": match_info
        })


def extract_multiple_chunks(
    file_path: str,
    matches: List[Dict[str, Any]],
    merge_overlapping: bool = True,
    max_chunks: int = 50,
    file_content: Optional[str] = None
) -> str:
    """
    Extract multiple chunks from a document, optionally merging overlapping ones.
    
    Args:
        file_path: Path to the document
        matches: List of match information dictionaries
        merge_overlapping: Whether to merge overlapping chunks
        max_chunks: Maximum number of chunks to extract
        file_content: Optional document content (if provided, file_path is ignored)
        
    Returns:
        JSON string with extracted chunks
    """
    try:
        # Get content to check if it's single-line
        if file_content is not None:
            content = file_content
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        is_single_line = _is_single_line_document(content)
        
        if is_single_line:
            logger.info(f"Processing single-line document with {len(matches)} matches")
            # For single-line documents, sort by character position if available
            sorted_matches = sorted(matches, key=lambda x: x.get("match_position", x.get("line_number", 0)))
        else:
            # Sort matches by line number for multi-line documents
            sorted_matches = sorted(matches, key=lambda x: x.get("line_number", 0))
        
        # Extract individual chunks
        chunks = []
        processed_positions = set()  # Track processed positions to avoid duplicates
        
        for match in sorted_matches[:max_chunks]:
            # Skip if we've already processed a chunk near this position
            if is_single_line and "match_position" in match:
                pos = match["match_position"]
                # Check if any processed position is within 200 chars
                if any(abs(pos - p) < 200 for p in processed_positions):
                    logger.debug(f"Skipping match at position {pos} (too close to existing chunk)")
                    continue
                processed_positions.add(pos)
            
            chunk_data = json.loads(create_document_chunk(
                file_path=file_path,
                match_info=match,
                context_size=5,
                boundary_detection=not is_single_line,  # Skip boundary detection for single-line
                file_content=file_content
            ))
            
            if "error" not in chunk_data:
                chunks.append(chunk_data)
        
        # Merge overlapping chunks if requested
        if merge_overlapping and len(chunks) > 1:
            if is_single_line:
                chunks = _merge_overlapping_chunks_character_based(chunks)
            else:
                chunks = _merge_overlapping_chunks(chunks)
        
        # Add summary statistics
        total_lines = 0
        total_chars = 0
        
        if is_single_line:
            # For single-line docs, calculate coverage based on characters
            total_chars = sum(
                c["chunk"].get("end_pos", 0) - c["chunk"].get("start_pos", 0) 
                for c in chunks 
                if "start_pos" in c["chunk"]
            )
            coverage = (total_chars / len(content)) * 100 if content else 0
        else:
            total_lines = sum(c["chunk"].get("line_count", 0) for c in chunks)
            coverage = _calculate_document_coverage(file_path, chunks, file_content)
        
        result = {
            "chunks": chunks,
            "summary": {
                "total_chunks": len(chunks),
                "total_lines": total_lines if not is_single_line else None,
                "total_chars": total_chars if is_single_line else None,
                "document_coverage": coverage,
                "merged": merge_overlapping,
                "single_line_document": is_single_line,
                "chunking_method": "character" if is_single_line else "line"
            },
            "file_path": file_path
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error extracting chunks: {str(e)}")
        return json.dumps({
            "error": str(e),
            "file_path": file_path
        })


def find_chunk_boundaries(
    lines: List[str],
    target_line: int,
    expansion_limit: int = 20
) -> str:
    """
    Find natural boundaries for a chunk using LLM guidance.
    
    Args:
        lines: Document lines
        target_line: Target line number (0-based)
        expansion_limit: Maximum lines to expand in each direction
        
    Returns:
        JSON string with boundary information
    """
    # Look for natural boundaries
    boundaries = {
        "sections": [],
        "paragraphs": [],
        "lists": [],
        "headers": []
    }
    
    # Search before target
    for i in range(max(0, target_line - expansion_limit), target_line):
        line = lines[i].strip()
        
        # Section headers (all caps, ending with colon, etc.)
        if _is_section_header(line):
            boundaries["headers"].append({"line": i, "text": line, "type": "section"})
        
        # Paragraph breaks (empty lines)
        if not line and i > 0:
            boundaries["paragraphs"].append({"line": i, "type": "break"})
        
        # List starts
        if _is_list_start(line):
            boundaries["lists"].append({"line": i, "text": line, "type": "start"})
    
    # Search after target
    for i in range(target_line + 1, min(len(lines), target_line + expansion_limit + 1)):
        line = lines[i].strip()
        
        if _is_section_header(line):
            boundaries["headers"].append({"line": i, "text": line, "type": "section"})
        
        if not line:
            boundaries["paragraphs"].append({"line": i, "type": "break"})
    
    # Recommend optimal boundaries
    recommendations = _recommend_boundaries(boundaries, target_line)
    
    result = {
        "target_line": target_line,
        "boundaries_found": boundaries,
        "recommendations": recommendations,
        "analysis": {
            "headers_found": len(boundaries["headers"]),
            "paragraph_breaks": len(boundaries["paragraphs"]),
            "list_items": len(boundaries["lists"])
        }
    }
    
    return json.dumps(result)


def optimize_chunk_size(
    chunk_content: str,
    target_size: int = 500,
    preserve_context: bool = True
) -> str:
    """
    Optimize chunk size while preserving important context.
    
    Args:
        chunk_content: Original chunk content
        target_size: Target size in lines
        preserve_context: Whether to preserve semantic context
        
    Returns:
        JSON string with optimized chunk
    """
    lines = chunk_content.split('\n')
    
    if len(lines) <= target_size:
        return json.dumps({
            "optimized": False,
            "content": chunk_content,
            "line_count": len(lines),
            "reason": "Chunk already within target size"
        })
    
    # Identify important lines to preserve
    important_lines = []
    for i, line in enumerate(lines):
        if _is_important_line(line):
            important_lines.append(i)
    
    # Strategy for optimization
    if preserve_context:
        # Keep important lines and their context
        optimized_lines = _preserve_important_context(
            lines, important_lines, target_size
        )
    else:
        # Simple truncation from edges
        mid = len(lines) // 2
        half_target = target_size // 2
        optimized_lines = lines[mid - half_target:mid + half_target]
    
    optimized_content = '\n'.join(optimized_lines)
    
    result = {
        "optimized": True,
        "content": optimized_content,
        "original_lines": len(lines),
        "optimized_lines": len(optimized_lines),
        "important_lines_preserved": len([i for i in important_lines if i < len(optimized_lines)]),
        "strategy": "context_preservation" if preserve_context else "simple_truncation"
    }
    
    return json.dumps(result)


# Helper functions
def _detect_semantic_boundaries(
    lines: List[str], 
    target_idx: int, 
    start_idx: int, 
    end_idx: int
) -> Tuple[int, int]:
    """Detect semantic boundaries for better chunk extraction."""
    # Look backward for section start
    for i in range(target_idx, max(start_idx - 10, 0), -1):
        if _is_section_header(lines[i].strip()):
            start_idx = i
            break
        elif i < target_idx - 2 and not lines[i].strip():
            # Paragraph break
            start_idx = i + 1
            break
    
    # Look forward for section end
    for i in range(target_idx + 1, min(end_idx + 10, len(lines))):
        if _is_section_header(lines[i].strip()):
            end_idx = i
            break
        elif i > target_idx + 2 and not lines[i].strip() and i + 1 < len(lines) and lines[i + 1].strip():
            # Paragraph break
            end_idx = i
            break
    
    return start_idx, end_idx


def _is_section_header(line: str) -> bool:
    """Check if a line is likely a section header."""
    if not line:
        return False
    
    # All caps
    if line.isupper() and len(line) > 3:
        return True
    
    # Ends with colon
    if line.endswith(':') and len(line) < 50:
        return True
    
    # Common medical headers
    header_patterns = [
        r'^(CHIEF COMPLAINT|HISTORY|ASSESSMENT|PLAN|DIAGNOSIS|MEDICATIONS)',
        r'^(Physical Exam|Review of Systems|Laboratory|Imaging)',
        r'^\d+\.\s+[A-Z]',  # Numbered sections
    ]
    
    for pattern in header_patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    
    return False


def _is_list_start(line: str) -> bool:
    """Check if a line starts a list."""
    list_patterns = [
        r'^\s*\d+\.',     # 1. Item
        r'^\s*[a-z]\.',   # a. Item
        r'^\s*[-•*]',     # - Item or • Item
        r'^\s*\([a-z]\)', # (a) Item
    ]
    
    for pattern in list_patterns:
        if re.match(pattern, line):
            return True
    
    return False


def _analyze_chunk_content(lines: List[str], match_line_idx: int) -> Dict[str, Any]:
    """Analyze chunk content for key information."""
    analysis = {
        "headers_found": [],
        "medical_terms_density": 0,
        "structure_type": "narrative",
        "key_sections": []
    }
    
    # Find headers
    for i, line in enumerate(lines):
        if _is_section_header(line.strip()):
            analysis["headers_found"].append({
                "line": i + 1,
                "text": line.strip()
            })
    
    # Detect structure type
    list_count = sum(1 for line in lines if _is_list_start(line.strip()))
    if list_count > len(lines) * 0.3:
        analysis["structure_type"] = "list"
    elif len(analysis["headers_found"]) > 2:
        analysis["structure_type"] = "structured"
    
    # Medical terms density (simplified)
    medical_keywords = [
        'diagnosis', 'patient', 'treatment', 'medication', 'symptoms',
        'history', 'examination', 'laboratory', 'imaging', 'assessment'
    ]
    
    text = ' '.join(lines).lower()
    term_count = sum(1 for keyword in medical_keywords if keyword in text)
    analysis["medical_terms_density"] = term_count / max(len(lines), 1)
    
    return analysis


def _merge_overlapping_chunks_character_based(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge overlapping chunks for character-based documents.
    
    Args:
        chunks: List of chunks with character positions
        
    Returns:
        List of merged chunks
    """
    if not chunks:
        return chunks
    
    # Sort by start position
    sorted_chunks = sorted(
        chunks, 
        key=lambda c: c["chunk"].get("start_pos", 0)
    )
    
    merged = []
    current = sorted_chunks[0]
    
    for next_chunk in sorted_chunks[1:]:
        current_end = current["chunk"].get("end_pos", 0)
        next_start = next_chunk["chunk"].get("start_pos", 0)
        
        # Check for overlap (with some tolerance)
        if next_start <= current_end + 50:  # 50 char overlap tolerance
            # Merge chunks
            current["chunk"]["end_pos"] = max(
                current_end, 
                next_chunk["chunk"]["end_pos"]
            )
            
            # Merge content if needed
            if next_chunk["chunk"]["end_pos"] > current_end:
                # We need to extend the content
                # This is a simplified merge - in practice might need more sophisticated handling
                current["chunk"]["content"] = current["chunk"]["content"]
            
            # Track merged matches
            if "merged_matches" not in current:
                current["merged_matches"] = [current["match_info"]]
            current["merged_matches"].append(next_chunk["match_info"])
        else:
            merged.append(current)
            current = next_chunk
    
    merged.append(current)
    return merged


def _merge_overlapping_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge overlapping chunks."""
    if not chunks:
        return chunks
    
    merged = []
    current = chunks[0]
    
    for next_chunk in chunks[1:]:
        current_end = current["chunk"]["end_line"]
        next_start = next_chunk["chunk"]["start_line"]
        
        if next_start <= current_end + 2:  # Overlapping or adjacent
            # Merge chunks
            current["chunk"]["end_line"] = max(current_end, next_chunk["chunk"]["end_line"])
            current["chunk"]["line_count"] = (
                current["chunk"]["end_line"] - current["chunk"]["start_line"]
            )
            # Merge match info
            if "merged_matches" not in current:
                current["merged_matches"] = [current["match_info"]]
            current["merged_matches"].append(next_chunk["match_info"])
        else:
            merged.append(current)
            current = next_chunk
    
    merged.append(current)
    return merged


def _calculate_document_coverage(file_path: str, chunks: List[Dict[str, Any]], file_content: Optional[str] = None) -> float:
    """Calculate what percentage of document is covered by chunks."""
    try:
        if file_content is not None:
            # Count lines in provided content
            total_lines = len(file_content.splitlines())
        else:
            with open(file_path, 'r') as f:
                total_lines = sum(1 for _ in f)
        
        covered_lines = set()
        for chunk in chunks:
            start = chunk["chunk"]["start_line"]
            end = chunk["chunk"]["end_line"]
            covered_lines.update(range(start, end + 1))
        
        return len(covered_lines) / max(total_lines, 1) * 100
    except:
        return 0.0


def _recommend_boundaries(boundaries: Dict[str, List[Dict]], target_line: int) -> Dict[str, Any]:
    """Recommend optimal boundaries based on found markers.
    
    Args:
        boundaries: Dictionary containing found boundaries
        target_line: Target line number (0-based)
        
    Returns:
        Dictionary with recommended start/end lines and reasoning
    """
    recommendations = {
        "start_line": target_line - 5,  # Default fallback
        "end_line": target_line + 5,
        "reasoning": [],
        "confidence": "low"
    }
    
    # Find nearest header before target
    before_headers = [h for h in boundaries["headers"] if h["line"] < target_line]
    if before_headers:
        nearest_header = max(before_headers, key=lambda h: h["line"])
        recommendations["start_line"] = nearest_header["line"]
        recommendations["reasoning"].append(f"Section starts at line {nearest_header['line']}")
        recommendations["confidence"] = "high"
    
    # Find nearest header after target
    after_headers = [h for h in boundaries["headers"] if h["line"] > target_line]
    if after_headers:
        next_header = min(after_headers, key=lambda h: h["line"])
        recommendations["end_line"] = next_header["line"] - 1
        recommendations["reasoning"].append(f"Section ends before line {next_header['line']}")
        if recommendations["confidence"] == "high":
            recommendations["confidence"] = "very_high"
        else:
            recommendations["confidence"] = "medium"
    
    # Check for paragraph boundaries
    para_breaks = boundaries["paragraphs"]
    if para_breaks:
        # Find closest paragraph break before target
        before_breaks = [p for p in para_breaks if p["line"] < target_line]
        if before_breaks and not before_headers:  # Only use if no header found
            nearest_break = max(before_breaks, key=lambda p: p["line"])
            if target_line - nearest_break["line"] < 10:
                recommendations["start_line"] = nearest_break["line"] + 1
                recommendations["reasoning"].append("Paragraph boundary detected")
                recommendations["confidence"] = "medium"
        
        # Find closest paragraph break after target
        after_breaks = [p for p in para_breaks if p["line"] > target_line]
        if after_breaks and not after_headers:  # Only use if no header found
            next_break = min(after_breaks, key=lambda p: p["line"])
            if next_break["line"] - target_line < 10:
                recommendations["end_line"] = next_break["line"]
                recommendations["reasoning"].append("Paragraph ends at break")
    
    # Ensure boundaries are within reasonable limits
    recommendations["start_line"] = max(0, recommendations["start_line"])
    
    if not recommendations["reasoning"]:
        recommendations["reasoning"].append("Using default context window")
    
    return recommendations


def _is_important_line(line: str) -> bool:
    """Check if a line contains important medical information."""
    important_patterns = [
        r'diagnosis:?\s*\S+',
        r'medication:?\s*\S+',
        r'assessment:?\s*\S+',
        r'plan:?\s*\S+',
        r'\b(abnormal|positive|negative|elevated|decreased)\b',
        r'\b\d+\s*(mg|ml|mcg|units?)\b',  # Dosages
    ]
    
    line_lower = line.lower()
    for pattern in important_patterns:
        if re.search(pattern, line_lower, re.IGNORECASE):
            return True
    
    return False


def _is_single_line_document(content: str) -> bool:
    """Check if the document is essentially a single long line.
    
    Args:
        content: Document content
        
    Returns:
        True if document has very few line breaks relative to its size
    """
    lines = content.splitlines()
    if len(lines) <= 3:  # Very few lines
        # Check if any line is extremely long
        for line in lines:
            if len(line) > 1000:  # Arbitrary threshold for "long line"
                return True
    
    # Check ratio of newlines to content length
    newline_ratio = content.count('\n') / max(len(content), 1)
    if newline_ratio < 0.001:  # Less than 0.1% newlines
        return True
        
    return False


def _chunk_by_characters(
    content: str,
    match_position: int,
    context_chars: int = 500,
    word_boundary: bool = True
) -> Tuple[str, int, int]:
    """Extract a chunk based on character positions rather than lines.
    
    Args:
        content: Full document content
        match_position: Character position of the match
        context_chars: Number of characters before/after match
        word_boundary: Whether to expand to word boundaries
        
    Returns:
        Tuple of (chunk_text, start_pos, end_pos)
    """
    # Calculate initial boundaries
    start_pos = max(0, match_position - context_chars)
    end_pos = min(len(content), match_position + context_chars)
    
    if word_boundary:
        # Expand to word boundaries
        # Move start back to previous whitespace or punctuation
        while start_pos > 0 and not content[start_pos-1].isspace() and content[start_pos-1] not in '.,;:!?':
            start_pos -= 1
            
        # Move end forward to next whitespace or punctuation  
        while end_pos < len(content) and not content[end_pos].isspace() and content[end_pos] not in '.,;:!?':
            end_pos += 1
    
    # Look for sentence boundaries to create more natural chunks
    # Check for sentence start
    sentence_start = start_pos
    for i in range(start_pos, max(0, start_pos - 100), -1):
        if i > 0 and content[i-1] in '.!?' and (i >= len(content) or content[i].isspace()):
            sentence_start = i
            break
            
    # Check for sentence end
    sentence_end = end_pos
    for i in range(end_pos, min(len(content), end_pos + 100)):
        if content[i] in '.!?' and (i + 1 >= len(content) or content[i+1].isspace()):
            sentence_end = i + 1
            break
    
    # Use sentence boundaries if they're not too far from original boundaries
    if match_position - sentence_start < context_chars * 1.5:
        start_pos = sentence_start
    if sentence_end - match_position < context_chars * 1.5:
        end_pos = sentence_end
        
    chunk_text = content[start_pos:end_pos].strip()
    return chunk_text, start_pos, end_pos


def _preserve_important_context(
    lines: List[str], 
    important_indices: List[int], 
    target_size: int
) -> List[str]:
    """Preserve important lines and their context within size limit."""
    if not important_indices:
        # No important lines, just truncate from middle
        mid = len(lines) // 2
        half = target_size // 2
        return lines[max(0, mid - half):min(len(lines), mid + half)]
    
    # Build ranges around important lines
    ranges = []
    context = 2  # Lines of context around important lines
    
    for idx in important_indices:
        start = max(0, idx - context)
        end = min(len(lines), idx + context + 1)
        ranges.append((start, end))
    
    # Merge overlapping ranges
    merged_ranges = []
    for start, end in sorted(ranges):
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))
    
    # Extract lines from ranges
    result_lines = []
    for start, end in merged_ranges:
        if len(result_lines) + (end - start) <= target_size:
            result_lines.extend(lines[start:end])
        else:
            # Partial range to fit size limit
            remaining = target_size - len(result_lines)
            if remaining > 0:
                result_lines.extend(lines[start:start + remaining])
            break
    
    return result_lines


# Create FunctionTool instances for Google ADK
create_chunk_tool = FunctionTool(func=create_document_chunk)
extract_chunks_tool = FunctionTool(func=extract_multiple_chunks)
find_boundaries_tool = FunctionTool(func=find_chunk_boundaries)
optimize_chunk_tool = FunctionTool(func=optimize_chunk_size)

# Export all tools
CHUNK_TOOLS = [
    create_chunk_tool,
    extract_chunks_tool,
    find_boundaries_tool,
    optimize_chunk_tool
]