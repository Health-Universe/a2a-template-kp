"""
Unified verification tools for verifying extracted clinical data.
Following nutrition_example.py pattern with Google ADK FunctionTool.
"""
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from google.adk.tools import FunctionTool


def verify_diagnoses(
    diagnosis_data: Dict[str, Any],
    timeline_events: List[Dict[str, Any]]
) -> str:
    """
    Verify diagnoses with special attention to cancer staging.
    
    This function prepares the input for LLM diagnosis verification.
    The actual LLM call happens in the agent executor.
    
    Args:
        diagnosis_data: Extracted diagnosis data to verify
        timeline_events: Source timeline events
        
    Returns:
        JSON string with verification request
    """
    # Create timeline lookup
    timeline_lookup = create_timeline_lookup(timeline_events)
    
    # Prepare verification requests
    verification_requests = []
    for diagnosis in diagnosis_data.get("diagnoses", []):
        date_str = diagnosis.get("date", "")
        matching_events = find_matching_events(date_str, timeline_lookup, search_nearby=False)
        
        if matching_events:
            combined_source, actual_sources, actual_pages = get_source_info(matching_events)
            
            # Check if this appears to be a cancer diagnosis
            cancer_keywords = ["cancer", "carcinoma", "adenocarcinoma", "squamous cell", "lymphoma",
                              "leukemia", "melanoma", "sarcoma", "tumor", "neoplasm", "malignancy"]
            summary_lower = diagnosis.get("summary", "").lower()
            is_cancer = any(keyword in summary_lower or keyword in combined_source.lower()
                           for keyword in cancer_keywords)
            
            verification_requests.append({
                "diagnosis": diagnosis,
                "source_text": combined_source,
                "actual_sources": list(actual_sources),
                "actual_pages": list(actual_pages),
                "is_cancer": is_cancer,
                "has_matching_events": True
            })
        else:
            verification_requests.append({
                "diagnosis": diagnosis,
                "source_text": "",
                "actual_sources": [],
                "actual_pages": [],
                "is_cancer": False,
                "has_matching_events": False
            })
    
    request = {
        "action": "verify_diagnoses",
        "total_diagnoses": len(diagnosis_data.get("diagnoses", [])),
        "verification_requests": verification_requests,
        "instructions": """Verify each diagnosis extraction against the source timeline events.
        
        For each diagnosis:
        1. Is the diagnosis accurately extracted from the timeline events?
        2. Is the summary factually correct based on the source?
        3. Are the source documents and pages correct?
        4. FOR CANCER DIAGNOSES: Is staging information included if available?
           - Look for: Stage I-IV, TNM classification (e.g., T2N1M0), descriptors like "locally advanced"
           - If staging is mentioned in source but missing from extraction, ADD IT
        5. Is the diagnosis type correctly categorized?
        
        CRITICAL: If this is a cancer diagnosis and staging information exists in the source,
        it MUST be included in the verified diagnosis.
        
        Return verified diagnoses with corrections where needed."""
    }
    
    return json.dumps(request)


def verify_treatments(
    treatment_data: Dict[str, Any],
    timeline_events: List[Dict[str, Any]]
) -> str:
    """
    Verify treatments against source timeline events.
    
    Args:
        treatment_data: Extracted treatment data to verify
        timeline_events: Source timeline events
        
    Returns:
        JSON string with verification request
    """
    # Create timeline lookup
    timeline_lookup = create_timeline_lookup(timeline_events)
    
    # Prepare verification requests
    verification_requests = []
    for treatment in treatment_data.get("treatments", []):
        date_str = treatment.get("date", "")
        # Handle date ranges
        lookup_date = date_str.split(" - ")[0] if " - " in date_str else date_str
        
        matching_events = find_matching_events(lookup_date, timeline_lookup, search_nearby=False)
        
        if matching_events:
            combined_source, actual_sources, actual_pages = get_source_info(matching_events)
            verification_requests.append({
                "treatment": treatment,
                "source_text": combined_source,
                "actual_sources": list(actual_sources),
                "actual_pages": list(actual_pages),
                "has_matching_events": True
            })
        else:
            verification_requests.append({
                "treatment": treatment,
                "source_text": "",
                "actual_sources": [],
                "actual_pages": [],
                "has_matching_events": False
            })
    
    request = {
        "action": "verify_treatments",
        "total_treatments": len(treatment_data.get("treatments", [])),
        "verification_requests": verification_requests,
        "instructions": """Verify each treatment extraction against the source timeline events.
        
        For each treatment:
        1. Is the treatment accurately extracted from the timeline events?
        2. Is the summary factually correct with all important details?
        3. Are the source documents and pages correct?
        4. For treatments: Is the date range accurate if provided?
        5. Is the category correctly assigned?
        6. Are specific regimens/protocols correctly identified?
        7. Are cycle numbers accurate for chemotherapy?
        
        IMPORTANT DETAILS TO VERIFY:
        - Drug names and combinations (e.g., FOLFOX, carboplatin/paclitaxel)
        - Dosing if mentioned
        - Duration of treatment
        - Number of cycles for chemotherapy
        - Surgical procedure names
        - Radiation dose and fractions
        
        Return verified treatments with corrections where needed."""
    }
    
    return json.dumps(request)


def verify_complications(
    complications_data: Dict[str, Any],
    timeline_events: List[Dict[str, Any]]
) -> str:
    """
    Verify complications against source timeline events.
    
    Args:
        complications_data: Extracted complications data to verify
        timeline_events: Source timeline events
        
    Returns:
        JSON string with verification request
    """
    # Create timeline lookup
    timeline_lookup = create_timeline_lookup(timeline_events)
    
    # Prepare verification requests
    verification_requests = []
    for complication in complications_data.get("complications", []):
        date_str = complication.get("date", "")
        matching_events = find_matching_events(date_str, timeline_lookup, search_nearby=True)
        
        if matching_events:
            combined_source, actual_sources, actual_pages = get_source_info(matching_events)
            verification_requests.append({
                "complication": complication,
                "source_text": combined_source,
                "actual_sources": list(actual_sources),
                "actual_pages": list(actual_pages),
                "has_matching_events": True
            })
        else:
            verification_requests.append({
                "complication": complication,
                "source_text": "",
                "actual_sources": [],
                "actual_pages": [],
                "has_matching_events": False
            })
    
    request = {
        "action": "verify_complications",
        "total_complications": len(complications_data.get("complications", [])),
        "verification_requests": verification_requests,
        "instructions": """Verify each complication extraction against the source timeline events.
        
        For each complication:
        1. Is this truly a complication (adverse effect or unexpected issue from disease/treatment)?
        2. Is the summary factually correct based on the source?
        3. Are the source documents and pages correct?
        4. Is the date accurate?
        5. Does it include severity/grade information if available in the source?
        
        WHAT QUALIFIES AS A COMPLICATION:
        - Treatment side effects, toxicities, or adverse reactions
        - Infections or secondary conditions
        - Hospitalizations due to adverse events
        - Disease-related complications
        - Any unexpected negative outcome
        
        Return verified complications, filtering out any non-complications."""
    }
    
    return json.dumps(request)


def verify_response_metrics(
    response_metrics_data: Dict[str, Any],
    timeline_events: List[Dict[str, Any]]
) -> str:
    """
    Verify response metrics against source timeline events.
    
    Args:
        response_metrics_data: Extracted response metrics to verify
        timeline_events: Source timeline events
        
    Returns:
        JSON string with verification request
    """
    # Create timeline lookup
    timeline_lookup = create_timeline_lookup(timeline_events)
    
    # Prepare verification requests
    verification_requests = []
    for metric in response_metrics_data.get("response_metrics", []):
        date_str = metric.get("date", "")
        matching_events = find_matching_events(date_str, timeline_lookup, search_nearby=True)
        
        if matching_events:
            combined_source, actual_sources, actual_pages = get_source_info(matching_events)
            verification_requests.append({
                "metric": metric,
                "source_text": combined_source,
                "actual_sources": list(actual_sources),
                "actual_pages": list(actual_pages),
                "has_matching_events": True
            })
        else:
            verification_requests.append({
                "metric": metric,
                "source_text": "",
                "actual_sources": [],
                "actual_pages": [],
                "has_matching_events": False
            })
    
    request = {
        "action": "verify_response_metrics",
        "total_metrics": len(response_metrics_data.get("response_metrics", [])),
        "verification_requests": verification_requests,
        "instructions": """Verify each response metric extraction against the source timeline events.
        
        For each metric:
        1. Is this truly a response metric (measurement of treatment response)?
        2. Does it include specific criteria (RECIST, pathologic, percentage change)?
        3. Are the measurements/values accurate based on the source?
        4. Are the source documents and pages correct?
        5. Is the date accurate?
        
        WHAT QUALIFIES AS A RESPONSE METRIC:
        - RECIST criteria assessments (CR, PR, SD, PD)
        - Tumor size percentage changes
        - Pathologic response assessments (pCR, etc.)
        - Biomarker level changes indicating response
        - Specific response evaluations after treatment
        
        Return verified response metrics, filtering out baseline measurements."""
    }
    
    return json.dumps(request)


def verify_demographics(
    demographics_data: Dict[str, Any],
    timeline_events: List[Dict[str, Any]]
) -> str:
    """
    Verify demographics against source timeline events.
    
    Args:
        demographics_data: Extracted demographics to verify
        timeline_events: Source timeline events
        
    Returns:
        JSON string with verification request
    """
    # Combine all timeline text for verification
    all_timeline_text = []
    for event in timeline_events:
        all_timeline_text.append(f"Date: {event.get('date', '')}\nSummary: {event.get('summary', '')}")
    
    combined_text = "\n\n".join(all_timeline_text)
    
    # Extract data to verify
    age = demographics_data.get("age")
    gender = demographics_data.get("gender")
    dob = demographics_data.get("date_of_birth")
    sources = demographics_data.get("sources", {})
    
    request = {
        "action": "verify_demographics",
        "demographics": {
            "age": age,
            "gender": gender,
            "date_of_birth": dob
        },
        "sources": sources,
        "combined_timeline_text": combined_text[:3000],  # Limit for context
        "instructions": """Verify the extracted demographic information against the source medical records.
        
        For each demographic item:
        1. Is the information accurately extracted from the source?
        2. Is the value correctly interpreted and formatted?
        3. Are there any errors or inconsistencies?
        4. If missing, can it be found in the source text?
        
        VERIFICATION RULES:
        - Only accept demographics explicitly stated in the records
        - For age: prefer explicit age statements over calculated ages
        - For gender: use terms as stated in medical records
        - For DOB: ensure proper MM/DD/YYYY format
        - Flag any inconsistencies or unclear references
        
        If any demographic was not initially found, search for it in the source text.
        
        Return verified demographics with corrections or additions where needed."""
    }
    
    return json.dumps(request)


def verify_patient_headline(
    headline: str,
    timeline_events: List[Dict[str, Any]],
    verified_diagnoses: List[Dict[str, Any]],
    demographics_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Verify patient headline against source records and verified demographics.
    
    Args:
        headline: Generated headline to verify
        timeline_events: Source timeline events
        verified_diagnoses: Already verified diagnoses
        demographics_data: Optional demographics information
        
    Returns:
        JSON string with verification request
    """
    # Gather timeline summaries
    all_summaries = []
    for event in timeline_events[:20]:
        all_summaries.append(f"{event.get('date', '')}: {event.get('summary', '')}")
    
    timeline_text = "\n".join(all_summaries)
    
    # Include verified diagnoses
    diagnosis_text = "\n".join([
        f"{d['date']}: {d['summary']}"
        for d in verified_diagnoses[:5]
    ])
    
    # Add demographics information if available
    demographics_text = ""
    if demographics_data:
        age = demographics_data.get("age")
        gender = demographics_data.get("gender")
        dob = demographics_data.get("date_of_birth")
        
        demographics_text = "\nVERIFIED DEMOGRAPHICS:\n"
        if age:
            demographics_text += f"Age: {age}\n"
        if gender:
            demographics_text += f"Gender: {gender}\n"
        if dob:
            demographics_text += f"DOB: {dob}\n"
    
    request = {
        "action": "verify_patient_headline",
        "headline": headline,
        "timeline_text": timeline_text,
        "diagnosis_text": diagnosis_text,
        "demographics_text": demographics_text,
        "has_demographics": demographics_data is not None,
        "instructions": """Verify this patient headline for accuracy against the source medical records and verified demographics.
        
        VERIFICATION RULES:
        1. Age and gender can come from verified demographics data (preferred) OR timeline events
        2. If demographics data provides age/gender, those should be considered verified
        3. Primary condition must match the verified diagnoses
        4. No information should be inferred beyond verified demographics and timeline events
        5. Maximum 2 lines
        6. Demographic accuracy should prioritize verified demographics data
        
        DEMOGRAPHIC VERIFICATION:
        - If demographics data shows age/gender, headline should use those values
        - Only flag demographic errors if they contradict verified demographics
        - Accept demographics from verified extraction even if not visible in timeline snippets
        
        COMMON ERRORS TO CHECK:
        - Incorrect primary diagnosis
        - Missing verified demographics when available
        - Added information not in records or verified data
        - Inconsistency with verified demographics
        
        If the headline contains any unverified information or misses verified demographics, provide a corrected version."""
    }
    
    return json.dumps(request)


# Helper functions
def create_timeline_lookup(timeline_events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create a lookup dictionary of timeline events by date.
    
    Args:
        timeline_events: List of timeline events
        
    Returns:
        Dictionary mapping dates to lists of events
    """
    timeline_lookup = {}
    for event in timeline_events:
        date = event.get("date", "")
        if date not in timeline_lookup:
            timeline_lookup[date] = []
        timeline_lookup[date].append(event)
    
    return timeline_lookup


def find_matching_events(
    date_str: str,
    timeline_lookup: Dict[str, List[Dict[str, Any]]],
    search_nearby: bool = True
) -> List[Dict[str, Any]]:
    """
    Find timeline events matching a date, optionally searching nearby dates.
    
    Args:
        date_str: Date to search for
        timeline_lookup: Timeline lookup dictionary
        search_nearby: Whether to search nearby dates
        
    Returns:
        List of matching events
    """
    matching_events = []
    
    # Try exact date match first
    if date_str in timeline_lookup:
        matching_events.extend(timeline_lookup[date_str])
    
    # Check Background Information for undated items
    if "Background Information" in timeline_lookup:
        matching_events.extend(timeline_lookup["Background Information"])
    
    # Search nearby dates if requested
    if search_nearby and not matching_events:
        try:
            target_date = datetime.strptime(date_str, "%m/%d/%Y")
            for days_offset in [-1, 1, -2, 2, -3, 3]:
                nearby_date = target_date + timedelta(days=days_offset)
                nearby_date_str = nearby_date.strftime("%m/%d/%Y")
                if nearby_date_str in timeline_lookup:
                    matching_events.extend(timeline_lookup[nearby_date_str])
        except:
            pass
    
    return matching_events


def get_source_info(events: List[Dict[str, Any]]) -> Tuple[str, Set[str], Set[int]]:
    """
    Extract combined source text and metadata from events.
    
    Args:
        events: List of timeline events
        
    Returns:
        Tuple of (combined_source_text, source_documents_set, source_pages_set)
    """
    source_texts = []
    actual_sources = set()
    actual_pages = set()
    
    for event in events:
        source_texts.append(event.get("summary", ""))
        actual_sources.update(event.get("source_documents", []))
        actual_pages.update(event.get("source_pages", []))
    
    combined_source = " ".join(source_texts)
    return combined_source, actual_sources, actual_pages


def search_for_missing_demographic(
    demo_type: str,
    source_text: str
) -> str:
    """
    Search for a demographic that wasn't initially found.
    
    Args:
        demo_type: Type of demographic to search for (age, gender, date_of_birth)
        source_text: Text to search in
        
    Returns:
        JSON string with search request
    """
    search_patterns = {
        "age": "- Age: '58 year old', 'age 45', '58yo', 'patient is 67'",
        "gender": "- Gender: 'male', 'female', 'man', 'woman', 'he/him', 'she/her'",
        "date_of_birth": "- DOB: 'DOB:', 'Date of birth:', 'Born:', birth date"
    }
    
    pattern = search_patterns.get(demo_type, "")
    
    request = {
        "action": "search_missing_demographic",
        "demo_type": demo_type,
        "source_text": source_text[:3000],  # Limit for context
        "search_pattern": pattern,
        "instructions": f"""Search for {demo_type} information that may have been missed in this medical record.
        
        SEARCH FOR {demo_type.upper()}:
        - Look for explicit mentions of patient {demo_type}
        - Check consultation notes and patient presentations
        - Find information in provider assessments
        - Look for demographic information in any clinical context
        
        SEARCH PATTERNS:
        {pattern}
        
        Return found value if present, otherwise null."""
    }
    
    return json.dumps(request)


def create_verification_summary(
    verification_results: Dict[str, Any]
) -> str:
    """
    Create a summary of all verification results.
    
    Args:
        verification_results: Dictionary containing all verification results
        
    Returns:
        JSON string with verification summary
    """
    summary = {
        "action": "create_summary",
        "total_items_verified": 0,
        "total_corrections": 0,
        "verification_breakdown": {}
    }
    
    # Count diagnoses
    if "diagnoses" in verification_results:
        diag_stats = verification_results.get("diagnosis_stats", {})
        summary["verification_breakdown"]["diagnoses"] = {
            "total": diag_stats.get("total_diagnoses", 0),
            "corrected": diag_stats.get("diagnosis_corrections", 0),
            "staging_added": diag_stats.get("staging_corrections", 0)
        }
        summary["total_items_verified"] += diag_stats.get("total_diagnoses", 0)
        summary["total_corrections"] += diag_stats.get("diagnosis_corrections", 0)
    
    # Count treatments
    if "treatments" in verification_results:
        treat_stats = verification_results.get("treatment_stats", {})
        summary["verification_breakdown"]["treatments"] = {
            "total": treat_stats.get("total_treatments", 0),
            "corrected": treat_stats.get("treatment_corrections", 0)
        }
        summary["total_items_verified"] += treat_stats.get("total_treatments", 0)
        summary["total_corrections"] += treat_stats.get("treatment_corrections", 0)
    
    # Count complications
    if "complications" in verification_results:
        comp_stats = verification_results.get("complication_stats", {})
        summary["verification_breakdown"]["complications"] = {
            "total": comp_stats.get("total_complications", 0),
            "corrected": comp_stats.get("complications_corrections", 0)
        }
        summary["total_items_verified"] += comp_stats.get("total_complications", 0)
        summary["total_corrections"] += comp_stats.get("complications_corrections", 0)
    
    # Count response metrics
    if "response_metrics" in verification_results:
        resp_stats = verification_results.get("response_metrics_stats", {})
        summary["verification_breakdown"]["response_metrics"] = {
            "total": resp_stats.get("total_response_metrics", 0),
            "corrected": resp_stats.get("response_metrics_corrections", 0)
        }
        summary["total_items_verified"] += resp_stats.get("total_response_metrics", 0)
        summary["total_corrections"] += resp_stats.get("response_metrics_corrections", 0)
    
    # Count demographics
    if "demographics" in verification_results:
        demo_stats = verification_results.get("demographics_stats", {})
        summary["verification_breakdown"]["demographics"] = {
            "found": demo_stats.get("demographics_found", 0),
            "corrected": demo_stats.get("demographics_corrected", 0)
        }
        summary["total_items_verified"] += 3  # age, gender, dob
        summary["total_corrections"] += demo_stats.get("demographics_corrected", 0)
    
    # Headline verification
    if "headline_corrected" in verification_results:
        summary["verification_breakdown"]["headline"] = {
            "corrected": verification_results["headline_corrected"]
        }
        summary["total_items_verified"] += 1
        if verification_results["headline_corrected"]:
            summary["total_corrections"] += 1
    
    return json.dumps(summary)


# Create FunctionTool instances for Google ADK
verify_diagnoses_tool = FunctionTool(func=verify_diagnoses)
verify_treatments_tool = FunctionTool(func=verify_treatments)
verify_complications_tool = FunctionTool(func=verify_complications)
verify_response_metrics_tool = FunctionTool(func=verify_response_metrics)
verify_demographics_tool = FunctionTool(func=verify_demographics)
verify_headline_tool = FunctionTool(func=verify_patient_headline)
create_timeline_lookup_tool = FunctionTool(func=create_timeline_lookup)
find_matching_events_tool = FunctionTool(func=find_matching_events)
get_source_info_tool = FunctionTool(func=get_source_info)
search_demographic_tool = FunctionTool(func=search_for_missing_demographic)
create_summary_tool = FunctionTool(func=create_verification_summary)

# Export all tools
UNIFIED_VERIFIER_TOOLS = [
    verify_diagnoses_tool,
    verify_treatments_tool,
    verify_complications_tool,
    verify_response_metrics_tool,
    verify_demographics_tool,
    verify_headline_tool,
    create_timeline_lookup_tool,
    find_matching_events_tool,
    get_source_info_tool,
    search_demographic_tool,
    create_summary_tool
]
