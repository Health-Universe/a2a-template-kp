"""
Smart chunk processing that deduplicates and limits chunk extraction.
Handles single-line documents specially.
"""

import json
import logging
from typing import List, Dict, Any
from utils.a2a_client import A2AAgentClient, AgentRegistry

logger = logging.getLogger("SmartChunkProcessor")


async def process_chunks_intelligently(
    grep_results: str,
    document_content: str,
    max_chunks: int = 5
) -> List[str]:
    """
    Process grep results intelligently to avoid duplicate chunking.
    
    For single-line documents (like Eleanor's text):
    - Don't chunk the same line multiple times
    - Consider splitting the line into segments instead
    
    Args:
        grep_results: Results from grep agent
        document_content: Original document content
        max_chunks: Maximum number of chunks to extract
        
    Returns:
        List of extracted chunks
    """
    logger.info("="*80)
    logger.info("🧠 SMART CHUNK PROCESSING")
    logger.info("="*80)
    
    # Parse grep results
    try:
        matches = json.loads(grep_results)
        if isinstance(matches, dict) and "matches" in matches:
            matches = matches["matches"]
    except:
        logger.warning("Could not parse grep results as JSON")
        return [grep_results[:1000]]  # Return truncated result as single chunk
    
    if not matches:
        logger.info("No matches to process")
        return []
    
    logger.info(f"📊 Processing {len(matches)} matches")
    
    # Check if this is a single-line document
    lines = document_content.split('\n')
    is_single_line = len(lines) == 1 or (len(lines) == 2 and not lines[1].strip())
    
    if is_single_line:
        logger.info("⚠️ Single-line document detected")
        logger.info(f"   Document length: {len(document_content)} characters")
        logger.info(f"   Number of matches: {len(matches)}")
        
        # For single-line documents, we have different strategies:
        # 1. If document is short (<2000 chars), just return the whole thing once
        if len(document_content) < 2000:
            logger.info("   Strategy: Return entire document as single chunk")
            return [document_content]
        
        # 2. If document is long, split into segments around key matches
        else:
            logger.info("   Strategy: Split into segments around key terms")
            # Find positions of key medical terms
            segments = []
            segment_size = 500  # Characters per segment
            
            # Extract segments around first few matches
            for i, match in enumerate(matches[:max_chunks]):
                match_text = match.get("match_text", "")
                # Find position in document
                pos = document_content.find(match_text)
                if pos >= 0:
                    start = max(0, pos - segment_size // 2)
                    end = min(len(document_content), pos + len(match_text) + segment_size // 2)
                    segment = document_content[start:end]
                    
                    # Avoid duplicate segments
                    if not any(segment in s or s in segment for s in segments):
                        segments.append(f"...{segment}...")
                        logger.info(f"   Segment {i+1}: {len(segment)} chars around '{match_text[:30]}'")
            
            return segments[:max_chunks]
    
    else:
        # Multi-line document - deduplicate by line number
        logger.info("📄 Multi-line document detected")
        logger.info(f"   Document has {len(lines)} lines")
        
        # Group matches by line number
        unique_lines = {}
        for match in matches:
            line_num = match.get("line_number", 1)
            if line_num not in unique_lines:
                unique_lines[line_num] = match
        
        logger.info(f"   Found matches on {len(unique_lines)} unique lines")
        
        # Process unique lines only
        chunks = []
        registry = AgentRegistry()
        chunk_url = registry.get_agent_url("chunk")
        if not chunk_url:
            chunk_url = "http://localhost:8004"
        
        matches_to_process = list(unique_lines.values())[:max_chunks]
        logger.info(f"   Will extract {len(matches_to_process)} chunks")
        
        async with A2AAgentClient(timeout=60.0) as client:
            for i, match in enumerate(matches_to_process, 1):
                logger.info(f"   [{i}/{len(matches_to_process)}] Extracting chunk for line {match.get('line_number')}...")
                
                chunk_message = json.dumps({
                    "match_info": match,
                    "lines_before": 2,
                    "lines_after": 2
                })
                
                try:
                    response = await client.call_agent(chunk_url, chunk_message)
                    chunks.append(response)
                except Exception as e:
                    logger.error(f"   Error extracting chunk: {e}")
        
        return chunks


async def extract_single_comprehensive_chunk(
    document_content: str,
    matches: List[Dict[str, Any]]
) -> str:
    """
    For single-line documents, create one comprehensive chunk
    that includes all relevant information.
    
    Args:
        document_content: The full document
        matches: All grep matches
        
    Returns:
        A single comprehensive chunk
    """
    logger.info("📦 Creating comprehensive single chunk")
    
    # Collect all match positions
    match_positions = []
    for match in matches[:20]:  # Limit to first 20 matches
        match_text = match.get("match_text", "")
        pos = document_content.find(match_text)
        if pos >= 0:
            match_positions.append({
                "position": pos,
                "text": match_text,
                "pattern": match.get("pattern", "")
            })
    
    # Sort by position
    match_positions.sort(key=lambda x: x["position"])
    
    # Create a summary of what was found
    summary_parts = []
    summary_parts.append(f"Document Analysis ({len(document_content)} characters, {len(matches)} matches found)")
    summary_parts.append("\nKey terms identified:")
    
    # Group matches by pattern/category
    pattern_groups = {}
    for pos_info in match_positions:
        pattern = pos_info["pattern"]
        if pattern not in pattern_groups:
            pattern_groups[pattern] = []
        pattern_groups[pattern].append(pos_info["text"])
    
    for pattern, texts in list(pattern_groups.items())[:10]:
        summary_parts.append(f"• {pattern}: {', '.join(texts[:5])}")
    
    summary_parts.append(f"\nFull document content:\n{document_content}")
    
    return "\n".join(summary_parts)