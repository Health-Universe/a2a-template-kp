"""
Narrative synthesis tools for LLM-based patient narrative generation.
Following nutrition_example.py pattern with Google ADK FunctionTool.
"""
import json
from typing import Dict, List, Any, Optional
from google.adk.tools import FunctionTool


def synthesize_patient_narrative(
    timeline_events: List[Dict[str, Any]],
    diagnosis_treatment_data: Optional[Dict[str, Any]] = None,
    patient_headline: Optional[str] = None,
    include_sections: Optional[List[str]] = None
) -> str:
    """
    Synthesize a complete patient narrative from timeline events.
    
    This function prepares the input for LLM narrative synthesis.
    The actual LLM call happens in the agent executor.
    
    Args:
        timeline_events: List of timeline event dictionaries
        diagnosis_treatment_data: Optional diagnosis and treatment information
        patient_headline: Optional headline for the patient narrative
        include_sections: Sections to include (diagnosis, timeline, treatments, etc.)
        
    Returns:
        JSON string with narrative synthesis request
    """
    if not include_sections:
        include_sections = ["diagnosis", "timeline", "treatments", "complications", "response_metrics"]
    
    # Separate known date events from unknown date events
    known_date_events = []
    unknown_date_events = []
    
    for event in timeline_events:
        if event.get('date', '').lower() in ['unknown date', 'background information']:
            unknown_date_events.append(event)
        else:
            known_date_events.append(event)
    
    # Format events for LLM processing
    formatted_known_events = []
    for event in known_date_events:
        sources_str = ", ".join(event.get('source_documents', []))
        pages = event.get('source_pages', [])
        pages_str = ", ".join(str(p) for p in pages) if pages else ""
        verification_status = "VERIFIED" if event.get('verified', False) else "UNVERIFIED"
        
        if pages_str:
            source_citation = f"{sources_str} (Pages: {pages_str})" if len(pages) > 1 else f"{sources_str} (Page: {pages_str})"
        else:
            source_citation = sources_str
        
        event_str = {
            "date": event.get('date', 'Unknown'),
            "summary": event.get('summary', ''),
            "verified": verification_status,
            "confidence": event.get('confidence_score', 0.0),
            "sources": source_citation
        }
        formatted_known_events.append(event_str)
    
    # Format unknown date events
    formatted_unknown_events = []
    for event in unknown_date_events:
        sources_str = ", ".join(event.get('source_documents', []))
        pages = event.get('source_pages', [])
        pages_str = ", ".join(str(p) for p in pages) if pages else ""
        verification_status = "VERIFIED" if event.get('verified', False) else "UNVERIFIED"
        
        if pages_str:
            source_citation = f"{sources_str} (Pages: {pages_str})" if len(pages) > 1 else f"{sources_str} (Page: {pages_str})"
        else:
            source_citation = sources_str
        
        event_str = {
            "date": "UNKNOWN",
            "summary": event.get('summary', ''),
            "verified": verification_status,
            "confidence": event.get('confidence_score', 0.0),
            "sources": source_citation
        }
        formatted_unknown_events.append(event_str)
    
    # Build request for LLM processing
    request = {
        "action": "synthesize_narrative",
        "patient_headline": patient_headline,
        "known_date_events": formatted_known_events,
        "unknown_date_events": formatted_unknown_events,
        "diagnosis_treatment_data": diagnosis_treatment_data,
        "include_sections": include_sections,
        "total_events": len(timeline_events),
        "instructions": "Create a final, polished patient narrative following strict formatting rules"
    }
    
    return json.dumps(request)


def synthesize_focused_narrative(
    timeline_events: List[Dict[str, Any]],
    focus_type: str,
    patient_headline: Optional[str] = None
) -> str:
    """
    Generate a narrative focused on specific aspects.
    
    Args:
        timeline_events: List of timeline event dictionaries
        focus_type: Type to focus on (diagnosis, treatment, complications, etc.)
        patient_headline: Optional headline for the patient narrative
        
    Returns:
        JSON string with focused narrative synthesis request
    """
    focus_instructions = {
        "diagnosis": "Focus only on diagnoses, staging, and disease progression",
        "treatment": "Focus only on treatments, procedures, and interventions",
        "complications": "Focus only on adverse effects and complications",
        "response": "Focus only on treatment response and outcome metrics",
        "timeline": "Focus only on chronological sequence of events"
    }
    
    instruction = focus_instructions.get(focus_type, "all relevant medical information")
    
    # Filter events based on focus type
    filtered_events = []
    for event in timeline_events:
        summary_lower = event.get('summary', '').lower()
        
        if focus_type == "diagnosis" and any(term in summary_lower for term in ['diagnos', 'stage', 'grade']):
            filtered_events.append(event)
        elif focus_type == "treatment" and any(term in summary_lower for term in ['treatment', 'therapy', 'surgery', 'procedure']):
            filtered_events.append(event)
        elif focus_type == "complications" and any(term in summary_lower for term in ['complication', 'adverse', 'toxicity']):
            filtered_events.append(event)
        elif focus_type == "response" and any(term in summary_lower for term in ['response', 'recist', 'remission']):
            filtered_events.append(event)
        elif focus_type == "timeline":
            filtered_events.append(event)
    
    request = {
        "action": "synthesize_focused_narrative",
        "focus_type": focus_type,
        "patient_headline": patient_headline,
        "events": filtered_events,
        "total_events": len(filtered_events),
        "instructions": f"Create a narrative focusing on: {instruction}"
    }
    
    return json.dumps(request)


def format_timeline_events(
    timeline_events: List[Dict[str, Any]],
    format_type: str = "chronological"
) -> str:
    """
    Format timeline events for narrative synthesis.
    
    Args:
        timeline_events: List of timeline event dictionaries
        format_type: How to format (chronological, by_category, by_source)
        
    Returns:
        JSON string with formatted events
    """
    formatted_result = {
        "action": "format_timeline_events",
        "format_type": format_type,
        "total_events": len(timeline_events)
    }
    
    if format_type == "chronological":
        # Sort by date, with unknown dates at the end
        known_events = [e for e in timeline_events if e.get('date', '').lower() not in ['unknown date', 'background information']]
        unknown_events = [e for e in timeline_events if e.get('date', '').lower() in ['unknown date', 'background information']]
        
        # Sort known events by date
        try:
            known_events.sort(key=lambda x: x.get('date', ''))
        except:
            pass
        
        formatted_result["known_events"] = known_events
        formatted_result["unknown_events"] = unknown_events
        
    elif format_type == "by_category":
        # Group by event type
        categories = {
            "diagnoses": [],
            "treatments": [],
            "complications": [],
            "response_metrics": [],
            "other": []
        }
        
        for event in timeline_events:
            summary_lower = event.get('summary', '').lower()
            
            if any(term in summary_lower for term in ['diagnos', 'stage', 'grade']):
                categories["diagnoses"].append(event)
            elif any(term in summary_lower for term in ['treatment', 'therapy', 'surgery']):
                categories["treatments"].append(event)
            elif any(term in summary_lower for term in ['complication', 'adverse', 'toxicity']):
                categories["complications"].append(event)
            elif any(term in summary_lower for term in ['response', 'recist', 'remission']):
                categories["response_metrics"].append(event)
            else:
                categories["other"].append(event)
        
        formatted_result["categories"] = categories
        
    elif format_type == "by_source":
        # Group by source document
        by_source = {}
        for event in timeline_events:
            for source in event.get('source_documents', ['Unknown']):
                if source not in by_source:
                    by_source[source] = []
                by_source[source].append(event)
        
        formatted_result["by_source"] = by_source
    
    return json.dumps(formatted_result)


def validate_narrative_structure(
    narrative: str,
    required_sections: Optional[List[str]] = None
) -> str:
    """
    Validate that a narrative contains required sections and formatting.
    
    Args:
        narrative: The generated narrative text
        required_sections: Sections that must be present
        
    Returns:
        JSON string with validation results
    """
    if not required_sections:
        required_sections = ["DIAGNOSIS:", "TIMELINE:", "TREATMENTS:", "COMPLICATIONS:", "RESPONSE METRICS:"]
    
    validation_results = {
        "action": "validate_narrative",
        "narrative_length": len(narrative),
        "sections_found": [],
        "sections_missing": [],
        "formatting_issues": []
    }
    
    # Check for required sections
    narrative_upper = narrative.upper()
    for section in required_sections:
        if section.upper() in narrative_upper:
            validation_results["sections_found"].append(section)
        else:
            validation_results["sections_missing"].append(section)
    
    # Check for formatting issues
    lines = narrative.split('\n')
    for i, line in enumerate(lines):
        # Check date format
        if '/' in line and not any(d in line for d in ['mm/dd/yyyy', 'Source:']):
            # Look for dates not in mm/dd/yyyy format
            import re
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
            dates = re.findall(date_pattern, line)
            for date in dates:
                if not re.match(r'\d{2}/\d{2}/\d{4}', date):
                    validation_results["formatting_issues"].append({
                        "line": i + 1,
                        "issue": f"Date format issue: {date}",
                        "expected": "mm/dd/yyyy"
                    })
        
        # Check source citation format
        if 'source:' in line.lower() and not line.strip().startswith('Source:'):
            validation_results["formatting_issues"].append({
                "line": i + 1,
                "issue": "Source citation not on new line",
                "expected": "Source: on new line"
            })
    
    validation_results["valid"] = (
        len(validation_results["sections_missing"]) == 0 and
        len(validation_results["formatting_issues"]) == 0
    )
    
    return json.dumps(validation_results)


# Create FunctionTool instances for Google ADK
synthesize_narrative_tool = FunctionTool(func=synthesize_patient_narrative)
focused_narrative_tool = FunctionTool(func=synthesize_focused_narrative)
format_events_tool = FunctionTool(func=format_timeline_events)
validate_narrative_tool = FunctionTool(func=validate_narrative_structure)

# Export all tools
NARRATIVE_SYNTHESIS_TOOLS = [
    synthesize_narrative_tool,
    focused_narrative_tool,
    format_events_tool,
    validate_narrative_tool
]
