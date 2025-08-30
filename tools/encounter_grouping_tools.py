"""
Encounter grouping tools for LLM-based temporal organization.
Following keyword_tools.py pattern with Google ADK FunctionTool.
"""
import json
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from google.adk.tools import FunctionTool


def group_encounters(temporal_data: str) -> str:
    """
    Group temporal tagged data by true encounter dates.
    Content with unknown dates is grouped separately.
    
    This function prepares the input for LLM encounter grouping.
    The actual LLM call happens in the agent executor.
    
    Args:
        temporal_data: JSON string containing:
            - encounter_dates: List of identified encounter dates
            - text_segments: List of text segments with temporal tags
            - date_metadata: Metadata about each date
            - has_unknown_dates: Boolean indicating presence of unknown dates
            
    Returns:
        JSON string with encounter grouping request
    """
    # Parse the JSON string if it's a string
    if isinstance(temporal_data, str):
        temporal_data = json.loads(temporal_data)
    
    # Extract key data for processing
    encounter_dates = temporal_data.get("encounter_dates", [])
    text_segments = temporal_data.get("text_segments", [])
    date_metadata = temporal_data.get("date_metadata", {})
    has_unknown_dates = temporal_data.get("has_unknown_dates", False)
    
    # Prepare segment summaries for LLM processing
    segment_summaries = []
    for i, segment in enumerate(text_segments[:50]):  # Limit to first 50 for LLM context
        # Handle case where segment might be a string or dict
        if isinstance(segment, str):
            # If segment is just a string, create a minimal structure
            summary = {
                "index": i,
                "primary_date": "Unknown Date",
                "referenced_dates": [],
                "is_carry_forward": False,
                "text_preview": segment[:200] if len(segment) > 200 else segment,
                "source_page": None,
                "source_document": None
            }
        else:
            # Normal case - segment is a dictionary
            summary = {
                "index": i,
                "primary_date": segment.get("primary_date", "Unknown Date"),
                "referenced_dates": segment.get("referenced_dates", []),
                "is_carry_forward": segment.get("is_carry_forward", False),
                "text_preview": segment.get("text", "")[:200],  # First 200 chars
                "source_page": segment.get("source_page"),
                "source_document": segment.get("source_document")
            }
        segment_summaries.append(summary)
    
    request = {
        "action": "group_encounters",
        "encounter_dates": encounter_dates,
        "has_unknown_dates": has_unknown_dates,
        "total_segments": len(text_segments),
        "segment_summaries": segment_summaries,
        "date_metadata": date_metadata,
        "instructions": """Analyze the temporal data and group content by actual encounter dates.

Key Tasks:
1. Create encounter groups for each true encounter date
2. Assign text segments to appropriate encounter groups based on:
   - Primary date (the encounter this content is FROM)
   - Whether it's carry-forward content (referenced from another encounter)
   - Clinical context and relevance
3. Create a special "Unknown Date" group if has_unknown_dates is true
4. Track source documents and pages for each encounter
5. Maintain distinction between primary content and referenced content

For each encounter group, determine:
- encounter_date: The date of this encounter
- encounter_type: visit/procedure/test/report/unknown
- primary_content: Segments that are FROM this encounter
- referenced_content: Segments that REFERENCE this encounter
- source_pages: List of pages containing this encounter's content
- source_documents: List of documents containing this encounter's content

Return the grouped encounters organized by date."""
    }
    
    return json.dumps(request)


def identify_encounter_relationships(encounter_groups: str) -> str:
    """
    Identify and mark relationships between encounters.
    
    Args:
        encounter_groups: JSON string containing list of encounter groups to analyze
        
    Returns:
        JSON string with relationship identification request
    """
    # Parse the JSON string if it's a string
    if isinstance(encounter_groups, str):
        encounter_groups = json.loads(encounter_groups)
    
    # Prepare encounter summaries for relationship analysis
    encounter_summaries = []
    for group in encounter_groups:
        if isinstance(group, dict) and group.get("encounter_date") != "Unknown Date":  # Skip unknown date groups
            summary = {
                "date": group.get("encounter_date"),
                "type": group.get("encounter_type"),
                "primary_content_count": len(group.get("primary_content", [])),
                "sample_content": ""
            }
            
            # Get sample content from primary content
            primary_content = group.get("primary_content", [])
            if primary_content and len(primary_content) > 0:
                first_segment = primary_content[0]
                if isinstance(first_segment, dict):
                    summary["sample_content"] = first_segment.get("text", "")[:200]
                elif isinstance(first_segment, str):
                    summary["sample_content"] = first_segment[:200]
            
            encounter_summaries.append(summary)
    
    request = {
        "action": "identify_relationships",
        "encounters": encounter_summaries,
        "instructions": """Analyze these clinical encounters and identify important relationships.

Identify the following types of relationships:
1. Follow-up relationships: Which encounters are follow-ups to others?
2. Test-result relationships: Which results belong to which test orders?
3. Treatment sequences: Pre-op, surgery, post-op patterns
4. Baseline vs. comparison: Which studies are comparisons to baseline?

For each relationship found, provide:
- from_date: The earlier encounter date
- to_date: The later encounter date
- relationship_type: follow-up|result-of|comparison-to|part-of-sequence
- description: Brief description of the relationship
- confidence: high|medium|low

Consider temporal proximity, clinical context, and explicit references when identifying relationships."""
    }
    
    return json.dumps(request)


def classify_encounter_type(
    encounter_date: str,
    date_metadata: str,
    sample_content: str = ""
) -> str:
    """
    Determine the type of encounter based on metadata and context.
    
    Args:
        encounter_date: The date to classify
        date_metadata: JSON string containing metadata about this date
        sample_content: Optional sample text from this encounter
        
    Returns:
        JSON string with classification request
    """
    # Parse the JSON string if it's a string
    if isinstance(date_metadata, str):
        date_metadata = json.loads(date_metadata) if date_metadata else []
    
    # Collect all contexts for this date
    contexts = []
    date_types = []
    
    for meta in date_metadata:
        if meta.get("context"):
            contexts.append(meta["context"])
        if meta.get("type"):
            date_types.append(meta["type"])
    
    request = {
        "action": "classify_encounter",
        "encounter_date": encounter_date,
        "contexts": contexts,
        "date_types": date_types,
        "sample_content": sample_content[:500] if sample_content else None,
        "instructions": """Classify this encounter based on the available information.

Encounter types:
- "procedure": Surgical procedures, operations, interventions
  Keywords: surgery, surgical, operation, resection, procedure, intervention
  
- "test": Diagnostic tests, imaging, labs, pathology
  Keywords: CT, MRI, PET, scan, imaging, radiology, pathology, biopsy, lab, test
  
- "report": Reports, addendums, amendments
  Keywords: report, addendum, amended, findings, results
  
- "visit": Clinical visits, consultations, appointments
  Keywords: visit, consultation, clinic, appointment, seen, evaluated
  
- "unknown": Cannot determine type from available information

Analyze the contexts and content to determine the most appropriate type.
Consider:
1. Explicit keywords in the contexts
2. Date type metadata (ENCOUNTER vs COLLECTION)
3. Content structure and terminology

Return the most likely encounter type with confidence level."""
    }
    
    return json.dumps(request)


def merge_encounter_groups(
    groups: str,
    merge_threshold_days: int = 0
) -> str:
    """
    Merge encounter groups that likely represent the same clinical event.
    
    Args:
        groups: JSON string containing list of encounter groups to potentially merge
        merge_threshold_days: Days threshold for considering merge (0 = same day only)
        
    Returns:
        JSON string with merge analysis request
    """
    # Parse the JSON string if it's a string
    if isinstance(groups, str):
        groups = json.loads(groups)
    request = {
        "action": "merge_encounters",
        "groups": groups,
        "merge_threshold_days": merge_threshold_days,
        "instructions": f"""Analyze encounter groups and identify which should be merged.

Encounters should be merged if they:
1. Occur on the same day (or within {merge_threshold_days} days)
2. Represent the same clinical event split across documents
3. Have complementary content (e.g., pre-op note + op report + post-op note)

Do NOT merge if they:
1. Are clearly separate clinical events
2. Have conflicting information
3. Represent follow-up or comparison encounters

For each merge recommendation, provide:
- dates_to_merge: List of encounter dates that should be merged
- merged_date: The primary date to use for the merged group
- merge_reason: Why these should be merged
- confidence: high|medium|low"""
    }
    
    return json.dumps(request)


def validate_encounter_groups(encounter_groups: str) -> str:
    """
    Validate the quality and completeness of encounter grouping.
    
    Args:
        encounter_groups: JSON string containing list of encounter groups to validate
        
    Returns:
        JSON string with validation results
    """
    # Parse the JSON string if it's a string
    if isinstance(encounter_groups, str):
        encounter_groups = json.loads(encounter_groups)
    # Basic validation checks
    total_groups = len(encounter_groups)
    groups_with_content = sum(1 for g in encounter_groups 
                            if g.get("primary_content") or g.get("referenced_content"))
    unknown_date_groups = sum(1 for g in encounter_groups 
                            if g.get("is_date_unknown", False))
    
    # Check for potential issues
    issues = []
    
    # Check for empty groups
    empty_groups = [g.get("encounter_date") for g in encounter_groups 
                   if not g.get("primary_content") and not g.get("referenced_content")]
    if empty_groups:
        issues.append({
            "type": "empty_groups",
            "severity": "warning",
            "details": f"Empty groups found: {empty_groups}"
        })
    
    # Check for groups with only referenced content
    reference_only = [g.get("encounter_date") for g in encounter_groups 
                     if not g.get("primary_content") and g.get("referenced_content")]
    if reference_only:
        issues.append({
            "type": "reference_only_groups",
            "severity": "info",
            "details": f"Groups with only referenced content: {reference_only}"
        })
    
    validation = {
        "action": "validation_results",
        "total_groups": total_groups,
        "groups_with_content": groups_with_content,
        "unknown_date_groups": unknown_date_groups,
        "issues": issues,
        "validation_passed": len([i for i in issues if i["severity"] == "error"]) == 0
    }
    
    return json.dumps(validation)


# Create FunctionTool instances for Google ADK
group_encounters_tool = FunctionTool(func=group_encounters)
identify_relationships_tool = FunctionTool(func=identify_encounter_relationships)
classify_encounter_tool = FunctionTool(func=classify_encounter_type)
merge_groups_tool = FunctionTool(func=merge_encounter_groups)
validate_groups_tool = FunctionTool(func=validate_encounter_groups)

# Export all tools
ENCOUNTER_TOOLS = [
    group_encounters_tool,
    identify_relationships_tool,
    classify_encounter_tool,
    merge_groups_tool,
    validate_groups_tool
]
