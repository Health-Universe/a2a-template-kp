"""
Summarization tools for medical text analysis with LLM.
Following nutrition_example.py pattern with Google ADK FunctionTool.
"""
import json
import re
from typing import Dict, List, Any, Optional
from google.adk.tools import FunctionTool
import logging

logger = logging.getLogger(__name__)


def summarize_medical_chunk(
    chunk_content: str,
    chunk_metadata: Dict[str, Any],
    summary_style: str = "concise",
    extract_entities: bool = True,
    max_key_points: int = 5
) -> str:
    """
    Summarize a medical text chunk using LLM analysis.
    
    Args:
        chunk_content: The text content to summarize
        chunk_metadata: Metadata about the chunk (file_path, line numbers, etc.)
        summary_style: Style of summary (concise, detailed, clinical)
        extract_entities: Whether to extract medical entities
        max_key_points: Maximum number of key points to extract
        
    Returns:
        JSON string with summary, key points, entities, and relevance score
    """
    # Prepare the analysis request
    analysis_request = {
        "action": "summarize_medical_chunk",
        "content": chunk_content,
        "metadata": chunk_metadata,
        "style": summary_style,
        "extract_entities": extract_entities,
        "instructions": f"""Analyze this medical text chunk and provide:

1. Summary ({summary_style} style):
   - {"2-3 sentences capturing key medical information" if summary_style == "concise" else ""}
   - {"Detailed paragraph with all relevant medical details" if summary_style == "detailed" else ""}
   - {"Clinical-style summary focusing on diagnoses and treatments" if summary_style == "clinical" else ""}

2. Key Points (maximum {max_key_points}):
   - Most important medical findings
   - Critical diagnoses or conditions
   - Significant treatments or medications
   - Important test results or observations

3. Relevance Score (0-10):
   - 9-10: Critical medical information (primary diagnosis, treatment plans)
   - 7-8: Important medical data (medications, significant findings)
   - 5-6: Relevant medical context (history, exam findings)
   - 3-4: Background information (social history, administrative)
   - 1-2: Minimal medical relevance

4. Medical Entities:
   - Diagnoses/Conditions
   - Treatments/Procedures
   - Medications (with dosages)
   - Laboratory values
   - Anatomical locations

Focus on extracting actionable medical information."""
    }
    
    return json.dumps(analysis_request)


def extract_medical_entities(
    text_content: str,
    entity_types: Optional[List[str]] = None,
    include_context: bool = True
) -> str:
    """
    Extract specific medical entities from text.
    
    Args:
        text_content: Text to analyze
        entity_types: Specific entity types to extract (None for all)
        include_context: Whether to include surrounding context
        
    Returns:
        JSON string with categorized medical entities
    """
    if not entity_types:
        entity_types = [
            "diagnoses", "treatments", "medications", 
            "lab_results", "symptoms", "anatomy"
        ]
    
    extraction_request = {
        "action": "extract_medical_entities",
        "content": text_content,
        "entity_types": entity_types,
        "include_context": include_context,
        "instructions": f"""Extract the following medical entities from the text:

Entity Types to Extract:
{chr(10).join(f'- {et.title()}' for et in entity_types)}

For each entity found:
1. Extract the exact mention from the text
2. Categorize it correctly
3. Include any associated details (dosages, frequencies, values)
4. {"Include 10-20 words of surrounding context" if include_context else "Extract entity only"}

Special Instructions:
- For medications: Include drug name, dose, route, and frequency
- For diagnoses: Include any staging, grading, or severity
- For lab results: Include the value and reference range if present
- For procedures: Include the date or timing if mentioned

Return entities grouped by type with their details."""
    }
    
    return json.dumps(extraction_request)


def score_medical_relevance(
    text_content: str,
    search_patterns: List[str],
    scoring_criteria: Optional[Dict[str, Any]] = None
) -> str:
    """
    Score text relevance based on medical importance and pattern matches.
    
    Args:
        text_content: Text to score
        search_patterns: Original search patterns that found this text
        scoring_criteria: Custom scoring criteria (optional)
        
    Returns:
        JSON string with relevance score and justification
    """
    default_criteria = {
        "critical_terms": ["diagnosis", "cancer", "stage", "grade", "treatment"],
        "important_terms": ["medication", "symptom", "laboratory", "imaging"],
        "relevant_terms": ["history", "examination", "assessment", "plan"],
        "pattern_weight": 0.3,  # How much pattern matches affect score
        "content_weight": 0.7   # How much medical content affects score
    }
    
    if scoring_criteria:
        default_criteria.update(scoring_criteria)
    
    scoring_request = {
        "action": "score_medical_relevance",
        "content": text_content,
        "patterns": search_patterns,
        "criteria": default_criteria,
        "instructions": """Score this text's medical relevance on a 0-10 scale.

Consider these factors:
1. Presence of critical medical information
2. Match quality with search patterns
3. Completeness of information
4. Clinical actionability
5. Temporal relevance (current vs historical)

Scoring Guidelines:
- 9-10: Contains primary diagnosis with staging/grading OR active treatment plan
- 7-8: Contains current medications with dosages OR significant test results
- 5-6: Contains relevant medical history OR physical exam findings
- 3-4: Contains background/social history OR administrative information
- 1-2: Minimal medical content OR mostly irrelevant to patterns

Provide:
1. Numeric score (0-10)
2. Primary factors that influenced the score
3. Specific medical elements found
4. Pattern match quality assessment"""
    }
    
    return json.dumps(scoring_request)


def batch_summarize_chunks(
    chunks: List[Dict[str, Any]],
    combine_related: bool = True,
    max_summaries: int = 20
) -> str:
    """
    Summarize multiple chunks, optionally combining related ones.
    
    Args:
        chunks: List of chunks with content and metadata
        combine_related: Whether to combine related chunks before summarizing
        max_summaries: Maximum number of summaries to generate
        
    Returns:
        JSON string with batch summarization results
    """
    batch_request = {
        "action": "batch_summarize_chunks",
        "chunks": chunks[:max_summaries],
        "combine_related": combine_related,
        "instructions": """Process multiple text chunks efficiently:

1. Analyze chunks for relationships:
   - Same section or topic
   - Sequential information
   - Related medical findings

2. If combining related chunks:
   - Merge chunks that discuss the same medical event
   - Combine sequential procedural steps
   - Group related lab results or medications

3. For each chunk/group:
   - Generate concise summary
   - Extract key medical information
   - Score relevance (0-10)
   - Note relationships to other chunks

4. Provide overall synthesis:
   - Main medical themes across chunks
   - Most critical findings
   - Recommended priority order for review

Optimize for clinical utility and efficiency."""
    }
    
    return json.dumps(batch_request)


def generate_clinical_summary(
    summaries: List[Dict[str, Any]],
    focus_areas: Optional[List[str]] = None,
    summary_format: str = "narrative"
) -> str:
    """
    Generate a clinical summary from multiple chunk summaries.
    
    Args:
        summaries: List of individual chunk summaries
        focus_areas: Specific areas to focus on (diagnosis, treatment, etc.)
        summary_format: Format of output (narrative, structured, bullet)
        
    Returns:
        JSON string with clinical summary
    """
    if not focus_areas:
        focus_areas = ["diagnosis", "treatment", "medications", "follow-up"]
    
    clinical_request = {
        "action": "generate_clinical_summary",
        "summaries": summaries,
        "focus_areas": focus_areas,
        "format": summary_format,
        "instructions": f"""Create a comprehensive clinical summary from the individual summaries.

Focus Areas: {', '.join(focus_areas)}
Output Format: {summary_format}

Requirements:
1. Synthesize information across all summaries
2. Prioritize most recent and relevant information
3. Resolve any contradictions or inconsistencies
4. Highlight critical findings and actions needed

Structure for {summary_format} format:
{"- Flowing narrative paragraphs" if summary_format == "narrative" else ""}
{"- Organized sections with headers" if summary_format == "structured" else ""}
{"- Bulleted key points under categories" if summary_format == "bullet" else ""}

Include:
- Primary diagnoses with current status
- Active treatments and responses
- Current medications with dosages
- Recent test results and significance
- Follow-up plans and recommendations

Ensure clinical accuracy and completeness."""
    }
    
    return json.dumps(clinical_request)


def analyze_medical_terminology(
    text_content: str,
    include_definitions: bool = False,
    categorize_terms: bool = True
) -> str:
    """
    Analyze and extract medical terminology from text.
    
    Args:
        text_content: Text to analyze
        include_definitions: Whether to include term definitions
        categorize_terms: Whether to categorize terms by type
        
    Returns:
        JSON string with medical terminology analysis
    """
    terminology_request = {
        "action": "analyze_medical_terminology",
        "content": text_content,
        "include_definitions": include_definitions,
        "categorize_terms": categorize_terms,
        "instructions": """Identify and analyze medical terminology in the text.

Extract:
1. Medical terms and abbreviations
2. {"Brief definitions for each term" if include_definitions else "Terms only"}
3. {"Categories: anatomical, pathological, procedural, pharmaceutical" if categorize_terms else ""}

For each term:
- Identify the full form if abbreviated
- Note the context of usage
- {"Provide a concise medical definition" if include_definitions else ""}
- {"Assign to appropriate category" if categorize_terms else ""}

Special attention to:
- Disease names and classifications
- Medical procedures and techniques
- Drug names (generic and brand)
- Laboratory tests and values
- Anatomical locations
- Medical devices and equipment

Organize results for clinical reference."""
    }
    
    return json.dumps(terminology_request)


# Create FunctionTool instances for Google ADK
summarize_chunk_tool = FunctionTool(func=summarize_medical_chunk)
extract_entities_tool = FunctionTool(func=extract_medical_entities)
score_relevance_tool = FunctionTool(func=score_medical_relevance)
batch_summarize_tool = FunctionTool(func=batch_summarize_chunks)
clinical_summary_tool = FunctionTool(func=generate_clinical_summary)
terminology_tool = FunctionTool(func=analyze_medical_terminology)

# Export all tools
SUMMARIZE_TOOLS = [
    summarize_chunk_tool,
    extract_entities_tool,
    score_relevance_tool,
    batch_summarize_tool,
    clinical_summary_tool,
    terminology_tool
]