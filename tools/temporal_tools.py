"""
Temporal tagging tools for date extraction and timeline analysis.
Following keyword_tools.py pattern with Google ADK FunctionTool.
"""
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
from google.adk.tools import FunctionTool


def extract_temporal_information(
    content: str,
    page_number: int = 1,
    source_document: str = "document"
) -> str:
    """
    Extract temporal information from document content.
    
    This function prepares the input for LLM temporal extraction.
    The actual LLM call happens in the agent executor.
    
    Args:
        content: Document text to analyze
        page_number: Page number of the content
        source_document: Name of source document
        
    Returns:
        JSON string with temporal extraction request
    """
    request = {
        "action": "extract_temporal",
        "content": content,
        "page_number": page_number,
        "source_document": source_document,
        "instructions": """Analyze this medical document and extract ALL temporal information.
        
CRITICAL TASKS:
1. IDENTIFY ALL DATES:
   - Explicit dates (MM/DD/YYYY, DD-Mon-YYYY, etc.)
   - Partial dates (Month YYYY, YYYY only)
   - Relative dates ("last month", "3 weeks ago", "previous visit")
   - Referenced dates ("on the previous scan", "at initial diagnosis")
   - Date ranges ("from May to July 2023")
   - If NO DATES found, mark segments as "NO_DATE"

2. CLASSIFY EACH DATE:
   - ENCOUNTER: When patient had visit/procedure/test
   - REFERENCE: Date mentioned referring to past events
   - REPORT: When report was generated
   - COLLECTION: When specimen/data was collected

3. EXTRACT TEXT SEGMENTS:
   - The clinical information
   - Primary associated date (use "NO_DATE" if none)
   - Referenced dates mentioned
   - Whether it's new information or carry-forward

4. TEMPORAL RELATIONSHIPS:
   - Previous events ("previously showed", "prior scan")
   - Carry-forward information
   - Temporal sequences ("following chemotherapy")

Return in format:
{
    "dates_found": [...],
    "text_segments": [...],
    "relative_dates": [...]
}"""
    }
    
    return json.dumps(request)


def consolidate_temporal_data(
    temporal_extractions: List[Dict[str, Any]],
    merge_duplicates: bool = True
) -> str:
    """
    Consolidate temporal information from multiple extractions.
    
    Args:
        temporal_extractions: List of temporal extraction results
        merge_duplicates: Whether to merge duplicate dates
        
    Returns:
        JSON string with consolidated temporal data
    """
    # Collect all dates and segments
    all_dates = defaultdict(list)
    encounter_dates = set()
    referenced_dates = set()
    all_segments = []
    
    for extraction in temporal_extractions:
        # Process dates
        for date_info in extraction.get("dates_found", []):
            date_str = date_info["date_str"]
            date_type = date_info["date_type"]
            
            all_dates[date_str].append({
                "type": date_type,
                "context": date_info["context"],
                "source_page": extraction.get("page_number", 1),
                "source_doc": extraction.get("source_document", "unknown")
            })
            
            if date_type in ["ENCOUNTER", "COLLECTION"]:
                encounter_dates.add(date_str)
            else:
                referenced_dates.add(date_str)
        
        # Process text segments
        for segment in extraction.get("text_segments", []):
            enhanced_segment = {
                "text": segment["text"],
                "primary_date": segment.get("primary_date", "Unknown Date"),
                "primary_date_type": segment.get("primary_date_type", "UNKNOWN"),
                "referenced_dates": segment.get("referenced_dates", []),
                "is_new_information": segment.get("is_new_information", True),
                "is_carry_forward": segment.get("is_carry_forward", False),
                "temporal_indicators": segment.get("temporal_indicators", []),
                "source_page": extraction.get("page_number", 1),
                "source_document": extraction.get("source_document", "unknown")
            }
            
            # Convert NO_DATE to Unknown Date
            if enhanced_segment["primary_date"] == "NO_DATE":
                enhanced_segment["primary_date"] = "Unknown Date"
                enhanced_segment["primary_date_type"] = "UNKNOWN"
            
            all_segments.append(enhanced_segment)
    
    # Identify true encounter dates
    true_encounter_dates = _identify_true_encounter_dates(all_dates, encounter_dates)
    
    # Sort dates (excluding "Unknown Date")
    sorted_encounter_dates = sorted([d for d in true_encounter_dates if d != "Unknown Date"])
    sorted_referenced_dates = sorted([d for d in (referenced_dates - true_encounter_dates) if d != "Unknown Date"])
    
    result = {
        "encounter_dates": sorted_encounter_dates,
        "referenced_dates": sorted_referenced_dates,
        "text_segments": all_segments,
        "date_metadata": dict(all_dates),
        "has_unknown_dates": any(seg["primary_date"] == "Unknown Date" for seg in all_segments),
        "summary": {
            "total_encounter_dates": len(sorted_encounter_dates),
            "total_referenced_dates": len(sorted_referenced_dates),
            "total_segments": len(all_segments),
            "unknown_date_segments": sum(1 for seg in all_segments if seg["primary_date"] == "Unknown Date")
        }
    }
    
    return json.dumps(result)


def analyze_temporal_patterns(
    consolidated_data: Dict[str, Any],
    pattern_types: Optional[List[str]] = None
) -> str:
    """
    Analyze temporal patterns in consolidated data.
    
    Args:
        consolidated_data: Consolidated temporal data
        pattern_types: Types to analyze (frequency, gaps, clusters, progression)
        
    Returns:
        JSON string with pattern analysis
    """
    if pattern_types is None:
        pattern_types = ["frequency", "gaps", "clusters"]
    
    all_dates = consolidated_data.get("encounter_dates", []) + consolidated_data.get("referenced_dates", [])
    all_dates = [d for d in all_dates if d != "Unknown Date"]
    
    patterns = {}
    
    if "frequency" in pattern_types and len(all_dates) > 1:
        # Calculate frequency of encounters
        date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in sorted(all_dates)]
        intervals = [(date_objs[i+1] - date_objs[i]).days for i in range(len(date_objs)-1)]
        
        patterns["frequency"] = {
            "average_interval_days": sum(intervals) / len(intervals) if intervals else 0,
            "min_interval_days": min(intervals) if intervals else 0,
            "max_interval_days": max(intervals) if intervals else 0,
            "total_encounters": len(all_dates)
        }
    
    if "gaps" in pattern_types and len(all_dates) > 1:
        # Identify significant gaps
        date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in sorted(all_dates)]
        gaps = []
        
        for i in range(len(date_objs)-1):
            interval = (date_objs[i+1] - date_objs[i]).days
            if interval > 90:  # Gap > 3 months
                gaps.append({
                    "start_date": all_dates[i],
                    "end_date": all_dates[i+1],
                    "gap_days": interval
                })
        
        patterns["gaps"] = gaps
    
    if "clusters" in pattern_types:
        # Identify clusters of activity
        clusters = []
        if len(all_dates) > 2:
            date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in sorted(all_dates)]
            cluster_threshold = 30  # Days
            
            current_cluster = [date_objs[0]]
            for i in range(1, len(date_objs)):
                if (date_objs[i] - current_cluster[-1]).days <= cluster_threshold:
                    current_cluster.append(date_objs[i])
                else:
                    if len(current_cluster) > 1:
                        clusters.append({
                            "start_date": current_cluster[0].strftime("%Y-%m-%d"),
                            "end_date": current_cluster[-1].strftime("%Y-%m-%d"),
                            "event_count": len(current_cluster)
                        })
                    current_cluster = [date_objs[i]]
            
            if len(current_cluster) > 1:
                clusters.append({
                    "start_date": current_cluster[0].strftime("%Y-%m-%d"),
                    "end_date": current_cluster[-1].strftime("%Y-%m-%d"),
                    "event_count": len(current_cluster)
                })
        
        patterns["clusters"] = clusters
    
    result = {
        "patterns": patterns,
        "date_range": {
            "earliest": min(all_dates) if all_dates else None,
            "latest": max(all_dates) if all_dates else None,
            "span_days": (datetime.strptime(max(all_dates), "%Y-%m-%d") - 
                         datetime.strptime(min(all_dates), "%Y-%m-%d")).days if len(all_dates) > 1 else 0
        },
        "total_dates_analyzed": len(all_dates)
    }
    
    return json.dumps(result)


def tag_timeline_segments(
    text_segments: List[Dict[str, Any]],
    group_by: str = "date"
) -> str:
    """
    Tag and organize text segments by timeline.
    
    Args:
        text_segments: List of text segments with temporal info
        group_by: How to group segments (date, type, source)
        
    Returns:
        JSON string with tagged timeline
    """
    if group_by == "date":
        # Group by primary date
        timeline = defaultdict(list)
        for segment in text_segments:
            date = segment.get("primary_date", "Unknown Date")
            timeline[date].append({
                "text": segment["text"],
                "type": segment.get("primary_date_type", "UNKNOWN"),
                "source": f"Page {segment.get('source_page', '?')}",
                "is_new": segment.get("is_new_information", True)
            })
        
        # Sort timeline
        sorted_timeline = []
        for date in sorted(timeline.keys()):
            if date != "Unknown Date":
                sorted_timeline.append({
                    "date": date,
                    "segments": timeline[date]
                })
        
        # Add unknown dates at the end
        if "Unknown Date" in timeline:
            sorted_timeline.append({
                "date": "Unknown Date",
                "segments": timeline["Unknown Date"]
            })
        
        result = {
            "timeline": sorted_timeline,
            "total_dates": len(timeline),
            "grouped_by": group_by
        }
    
    else:
        # Simple list format
        result = {
            "segments": text_segments,
            "total_segments": len(text_segments),
            "grouped_by": "none"
        }
    
    return json.dumps(result)


def normalize_dates(date_strings: List[str]) -> str:
    """
    Normalize various date formats to YYYY-MM-DD.
    
    Args:
        date_strings: List of date strings in various formats
        
    Returns:
        JSON string with normalized dates
    """
    results = []
    
    for date_str in date_strings:
        normalized = _normalize_date_string(date_str)
        results.append({
            "original": date_str,
            "normalized": normalized,
            "success": normalized is not None
        })
    
    return json.dumps({
        "results": results,
        "success_count": sum(1 for r in results if r["success"]),
        "total_count": len(results)
    })


# Helper functions (private)
def _identify_true_encounter_dates(all_dates: Dict, initial_encounter_dates: set) -> set:
    """Identify true encounter dates by analyzing context."""
    true_encounters = set()
    
    for date_str, occurrences in all_dates.items():
        if date_str == "Unknown Date":
            continue
            
        type_counts = defaultdict(int)
        for occ in occurrences:
            type_counts[occ["type"]] += 1
        
        # Rules for true encounter dates
        if type_counts["ENCOUNTER"] > 0 or type_counts["COLLECTION"] > 0:
            true_encounters.add(date_str)
        elif date_str in initial_encounter_dates:
            contexts = [occ["context"] for occ in occurrences]
            if any("performed" in ctx or "underwent" in ctx or "showed" in ctx 
                  or "revealed" in ctx or "demonstrated" in ctx for ctx in contexts):
                true_encounters.add(date_str)
    
    return true_encounters


def _normalize_date_string(date_str: str) -> Optional[str]:
    """Normalize various date formats to YYYY-MM-DD."""
    if not date_str:
        return None
    
    # Common date formats
    formats = [
        '%Y-%m-%d',
        '%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y',
        '%m/%d/%y', '%m-%d-%y', '%m.%d.%y',
        '%B %d, %Y', '%b %d, %Y',
        '%d %B %Y', '%d %b %Y',
        '%Y/%m/%d', '%Y.%m.%d'
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    # Handle partial dates
    if re.match(r'^\d{4}$', date_str.strip()):
        return f"{date_str}-01-01"
    
    # Month Year
    month_year_match = re.match(r'^(\w+)\s+(\d{4})$', date_str.strip())
    if month_year_match:
        try:
            dt = datetime.strptime(date_str.strip(), '%B %Y')
            return dt.strftime('%Y-%m-01')
        except ValueError:
            try:
                dt = datetime.strptime(date_str.strip(), '%b %Y')
                return dt.strftime('%Y-%m-01')
            except ValueError:
                pass
    
    return None


# Create FunctionTool instances for Google ADK
extract_temporal_tool = FunctionTool(func=extract_temporal_information)
consolidate_temporal_tool = FunctionTool(func=consolidate_temporal_data)
analyze_patterns_tool = FunctionTool(func=analyze_temporal_patterns)
tag_timeline_tool = FunctionTool(func=tag_timeline_segments)
normalize_dates_tool = FunctionTool(func=normalize_dates)

# Export all tools
TEMPORAL_TOOLS = [
    extract_temporal_tool,
    consolidate_temporal_tool,
    analyze_patterns_tool,
    tag_timeline_tool,
    normalize_dates_tool
]
