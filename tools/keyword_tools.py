"""
Keyword generation tools for LLM-based pattern extraction.
Following nutrition_example.py pattern with Google ADK FunctionTool.
"""
import json
import re
from typing import Dict, List, Any, Optional
from google.adk.tools import FunctionTool


def generate_document_patterns(
    preview: str,
    doc_type: str = "medical",
    focus_areas: Optional[List[str]] = None,
    max_patterns: int = 25
) -> str:
    """
    Generate regex patterns from document preview.
    
    This function prepares the input for LLM pattern generation.
    The actual LLM call happens in the agent executor.
    
    Args:
        preview: First 20 lines of the document
        doc_type: Type of medical document (medical, oncology, pathology)
        focus_areas: Specific areas to focus on (diagnosis, treatment, etc.)
        max_patterns: Maximum patterns to generate
        
    Returns:
        JSON string with pattern generation request
    """
    if not focus_areas:
        focus_areas = ["diagnosis", "treatment", "medications", "procedures"]
    
    # Validate preview
    lines = preview.split('\n')
    preview_lines = '\n'.join(lines[:20])  # Limit to first 20 lines
    
    # Build request for LLM processing
    request = {
        "action": "generate_patterns",
        "preview": preview_lines,
        "doc_type": doc_type,
        "focus_areas": focus_areas,
        "max_patterns": max_patterns,
        "instructions": f"""Analyze this medical document preview and generate regex patterns.
        
Document Type: {doc_type}
Focus Areas: {', '.join(focus_areas)}

Generate ripgrep-compatible regex patterns that will help identify:
1. Section boundaries (headers, transitions)
2. Clinical events and findings
3. Relevant medical terms and their common variations
4. Temporal markers that indicate when events occurred

Requirements:
- Use case-insensitive patterns where appropriate: (?i)
- Include common abbreviations and full terms
- Patterns should be specific enough to avoid false positives
- Consider medical synonyms and variations
- Include both structured (e.g., "CBC:") and narrative formats
- Maximum {max_patterns} patterns total
- Focus especially on: {', '.join(focus_areas)}"""
    }
    
    return json.dumps(request)


def generate_focused_patterns(
    preview: str,
    focus_type: str,
    max_patterns: int = 10
) -> str:
    """
    Generate patterns focused on specific medical aspects.
    
    Args:
        preview: Document preview
        focus_type: Type to focus on (medications, procedures, labs, etc.)
        max_patterns: Maximum patterns to generate
        
    Returns:
        JSON string with focused pattern generation request
    """
    focus_instructions = {
        "medications": "medication names, dosages, routes, frequencies, and drug classes",
        "procedures": "surgical procedures, interventions, and medical operations",
        "labs": "laboratory tests, results, reference ranges, and abnormal values",
        "diagnosis": "diagnoses, ICD codes, disease names, and conditions",
        "vitals": "vital signs, measurements, and physiological parameters",
        "allergies": "allergies, reactions, and sensitivities"
    }
    
    instruction = focus_instructions.get(focus_type, "relevant medical information")
    
    request = {
        "action": "generate_focused_patterns",
        "preview": preview.split('\n')[:20],
        "focus_type": focus_type,
        "max_patterns": max_patterns,
        "instructions": f"""Generate regex patterns specifically for {focus_type}.
        
Focus on identifying: {instruction}

Create patterns that:
- Are highly specific to {focus_type}
- Include common abbreviations and variations
- Work with both structured and narrative text
- Use case-insensitive matching where appropriate"""
    }
    
    return json.dumps(request)


def analyze_preview_structure(preview: str) -> str:
    """
    Analyze document structure to help pattern generation.
    
    Args:
        preview: Document preview
        
    Returns:
        JSON string with structure analysis
    """
    lines = preview.split('\n')[:20]
    
    # Basic structure detection
    headers = []
    date_lines = []
    structured_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect headers (lines ending with colon, all caps, etc.)
        if stripped and (
            stripped.endswith(':') or
            stripped.isupper() or
            (len(stripped) < 50 and stripped[0].isupper() and ':' in stripped)
        ):
            headers.append({
                "line": i + 1,
                "text": stripped,
                "type": "header" if stripped.endswith(':') else "section"
            })
        
        # Detect dates
        if re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', line):
            date_lines.append(i + 1)
        
        # Detect structured data (key: value format)
        if ':' in line and len(line) < 100:
            structured_lines.append(i + 1)
    
    analysis = {
        "action": "structure_analysis",
        "total_lines": len(lines),
        "headers_found": headers,
        "date_lines": date_lines,
        "structured_lines": structured_lines,
        "preview_snippet": '\n'.join(lines[:5])
    }
    
    return json.dumps(analysis)


def validate_patterns(patterns: List[str]) -> str:
    """
    Validate regex patterns for correctness.
    
    Args:
        patterns: List of regex patterns to validate
        
    Returns:
        JSON string with validation results
    """
    results = []
    
    for pattern in patterns:
        try:
            # Try to compile the regex
            re.compile(pattern)
            results.append({
                "pattern": pattern,
                "valid": True,
                "error": None
            })
        except re.error as e:
            results.append({
                "pattern": pattern,
                "valid": False,
                "error": str(e)
            })
    
    validation = {
        "action": "pattern_validation",
        "total_patterns": len(patterns),
        "valid_patterns": sum(1 for r in results if r["valid"]),
        "invalid_patterns": sum(1 for r in results if not r["valid"]),
        "results": results
    }
    
    return json.dumps(validation)


# Create FunctionTool instances for Google ADK
generate_patterns_tool = FunctionTool(func=generate_document_patterns)
focused_patterns_tool = FunctionTool(func=generate_focused_patterns)
structure_analysis_tool = FunctionTool(func=analyze_preview_structure)
validate_patterns_tool = FunctionTool(func=validate_patterns)

# Export all tools
KEYWORD_TOOLS = [
    generate_patterns_tool,
    focused_patterns_tool,
    structure_analysis_tool,
    validate_patterns_tool
]