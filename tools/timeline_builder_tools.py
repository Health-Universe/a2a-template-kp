"""
Timeline builder tools for creating verified clinical timelines.
Following nutrition_example.py pattern with Google ADK FunctionTool.
"""
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from google.adk.tools import FunctionTool


def build_timeline(
    facts: List[Dict[str, Any]],
    verification_mode: str = "full",
    max_retries: int = 3
) -> str:
    """
    Build a clinical timeline from extracted facts.
    
    This function prepares the input for LLM timeline building.
    The actual LLM call happens in the agent executor.
    
    Args:
        facts: List of extracted facts with metadata
        verification_mode: Mode for verification (full, batch, selective)
        max_retries: Maximum retry attempts for verification
        
    Returns:
        JSON string with timeline building request
    """
    # Group facts by date, considering reconciliation metadata
    facts_by_date = defaultdict(list)
    duplicate_count = 0
    carry_forward_count = 0
    
    for fact in facts:
        date_key = fact.get("date_str", "Unknown Date")
        
        # Check reconciliation metadata
        provenance = fact.get("provenance", "")
        status = fact.get("status", "")
        confidence = fact.get("confidence", 1.0)
        
        if provenance == "Previously Reported" and confidence < 0.7:
            carry_forward_count += 1
            continue
            
        if status == "Duplicate":
            duplicate_count += 1
            continue
        
        facts_by_date[date_key].append(fact)
    
    # Prepare event data for each date
    event_data_list = []
    for date, associated_facts in facts_by_date.items():
        event_data_list.append({
            "date": date,
            "facts": associated_facts,
            "fact_count": len(associated_facts)
        })
    
    request = {
        "action": "build_timeline",
        "verification_mode": verification_mode,
        "max_retries": max_retries,
        "total_facts": len(facts),
        "duplicate_count": duplicate_count,
        "carry_forward_count": carry_forward_count,
        "date_groups": len(facts_by_date),
        "event_data": event_data_list,
        "instructions": """Build a clinical timeline from the grouped facts.

For each date group:
1. Create a clinical summary combining all facts
2. Prioritize facts with "Final" or "Corrected" status
3. Remove redundant date information from summaries
4. Ensure clinical relevance and accuracy
5. Apply verification based on the specified mode

Return timeline events with verified summaries and metadata."""
    }
    
    return json.dumps(request)


def verify_with_context(
    event_data: Dict[str, Any],
    max_retries: int = 3
) -> str:
    """
    Verify timeline event with context and retry mechanism.
    
    Args:
        event_data: Event data including summary and source text
        max_retries: Maximum retry attempts
        
    Returns:
        JSON string with verification request
    """
    request = {
        "action": "verify_with_context",
        "event_date": event_data.get("date", ""),
        "initial_summary": event_data.get("initial_summary", ""),
        "source_text": event_data.get("source_text", "")[:2000],  # Limit for context
        "max_retries": max_retries,
        "metadata": event_data.get("metadata", {}),
        "instructions": """Verify this clinical timeline summary for accuracy and completeness.
        
        CURRENT SUMMARY: {initial_summary}
        
        SOURCE TEXT: {source_text}...
        
        VERIFICATION FOCUS:
        1. Ensure all clinically significant information is included
        2. Verify factual accuracy against source
        3. Check for completeness of key medical details
        4. Identify any errors or missing information
        5. Assess clinical relevance and clarity
        
        Return verification results with confidence score and any identified issues."""
    }
    
    return json.dumps(request)


def build_contextual_prompt(
    event_data: Dict[str, Any],
    current_summary: str,
    context_history: List[Dict[str, Any]],
    attempt: int,
    max_retries: int
) -> str:
    """
    Build verification prompt with context from previous attempts.
    
    Args:
        event_data: Original event data
        current_summary: Current version of summary
        context_history: History of previous verification attempts
        attempt: Current attempt number
        max_retries: Maximum allowed retries
        
    Returns:
        JSON string with contextual prompt
    """
    base_prompt = f"""
        Verify this clinical timeline summary for accuracy and completeness.
        
        CURRENT SUMMARY: {current_summary}
        
        SOURCE TEXT: {event_data.get('source_text', '')[:2000]}...
        """
    
    if attempt == 0:
        # First attempt - standard verification
        verification_focus = """
        
        VERIFICATION FOCUS:
        1. Ensure all clinically significant information is included
        2. Verify factual accuracy against source
        3. Check for completeness of key medical details
        4. Identify any errors or missing information
        5. Assess clinical relevance and clarity
        """
    else:
        # Subsequent attempts - include context history
        context_section = "\nPREVIOUS ATTEMPTS AND ISSUES:\n"
        for ctx in context_history:
            context_section += f"\nAttempt {ctx['attempt']}:\n"
            context_section += f"- Summary: {ctx['summary']}\n"
            context_section += f"- Confidence: {ctx['confidence']:.2f}\n"
            context_section += f"- Issues: {ctx.get('verification_summary', '')}\n"
            
            issues = ctx.get('issues_found', [])
            for issue in issues[:3]:  # Top 3 issues
                context_section += f"  • {issue.get('type', '')}: {issue.get('description', '')}\n"
        
        verification_focus = context_section + f"""
        
        VERIFICATION FOCUS (Attempt {attempt + 1}/{max_retries + 1}):
        This is a revised summary that attempted to address the previous issues.
        Please verify if the issues have been resolved and the summary is now accurate.
        
        Pay special attention to:
        1. Whether previous issues have been addressed
        2. No new errors were introduced
        3. Clinical accuracy and completeness
        4. All key information from source is included
        """
    
    request = {
        "action": "build_contextual_prompt",
        "prompt": base_prompt + verification_focus,
        "attempt": attempt,
        "has_context": len(context_history) > 0
    }
    
    return json.dumps(request)


def generate_contextual_correction(
    current_summary: str,
    source_text: str,
    context_history: List[Dict[str, Any]]
) -> str:
    """
    Generate improved summary using context from all previous attempts.
    
    Args:
        current_summary: Current version needing improvement
        source_text: Original source text
        context_history: History of all verification attempts
        
    Returns:
        JSON string with correction request
    """
    # Build a comprehensive view of all issues
    all_issues = []
    for ctx in context_history:
        all_issues.extend(ctx.get('issues', []))
    
    # Group issues by type
    issue_groups = defaultdict(list)
    for issue in all_issues:
        issue_type = issue.get('type', 'other')
        description = issue.get('description', '')
        issue_groups[issue_type].append(description)
    
    # Create improvement prompt
    prompt = f"""
        Improve this clinical summary based on the verification feedback from multiple attempts.
        
        CURRENT SUMMARY: {current_summary}
        
        SOURCE TEXT: {source_text[:2000]}...
        
        CONTEXT FROM {len(context_history)} VERIFICATION ATTEMPTS:
        """
    
    # Add grouped issues
    for issue_type, descriptions in issue_groups.items():
        prompt += f"\n{issue_type.upper()} ISSUES:\n"
        for desc in set(descriptions):  # Unique descriptions
            prompt += f"- {desc}\n"
    
    # Add specific guidance based on patterns
    guidance = ""
    if 'missing_info' in issue_groups:
        guidance += "\nIMPORTANT: Add the missing clinical information identified above.\n"
    if 'factual_error' in issue_groups:
        guidance += "\nIMPORTANT: Correct the factual errors using only information from the source.\n"
    if 'date_error' in issue_groups:
        guidance += "\nIMPORTANT: Ensure all dates match the source exactly.\n"
    
    prompt += guidance + """
        
        Generate an improved summary that:
        1. Addresses ALL identified issues
        2. Maintains factual accuracy from source
        3. Includes all clinically relevant information
        4. Is clear and concise (1-2 sentences)
        5. Does not introduce new errors
        
        IMPROVED SUMMARY:
        """
    
    request = {
        "action": "generate_correction",
        "current_summary": current_summary,
        "issue_count": len(all_issues),
        "issue_types": list(issue_groups.keys()),
        "prompt": prompt
    }
    
    return json.dumps(request)


def create_clinical_summary(
    fact_list: List[str],
    facts_metadata: List[Dict[str, Any]]
) -> str:
    """
    Create a clinically-focused summary from multiple facts.
    
    Args:
        fact_list: List of fact summaries
        facts_metadata: Metadata for each fact (status, provenance, etc.)
        
    Returns:
        JSON string with summary creation request
    """
    if not fact_list:
        return json.dumps({"action": "create_summary", "summary": "", "fact_count": 0})
    
    # Prioritize facts based on metadata
    prioritized_facts = []
    other_facts = []
    
    for i, fact in enumerate(fact_list):
        metadata = facts_metadata[i] if i < len(facts_metadata) else {}
        status = metadata.get("status", "")
        
        if status in ["Final", "Corrected"]:
            prioritized_facts.append(fact)
        else:
            other_facts.append(fact)
    
    all_facts = prioritized_facts + other_facts
    
    request = {
        "action": "create_clinical_summary",
        "fact_count": len(fact_list),
        "prioritized_count": len(prioritized_facts),
        "facts": all_facts,
        "instructions": """Create a clinically-focused summary from these facts.

Guidelines:
1. Combine related facts into a coherent summary
2. Prioritize Final and Corrected status facts
3. Remove redundant date information
4. Focus on clinically significant information
5. Ensure proper sentence structure
6. Keep summary concise (1-2 sentences)

If single fact: enhance for clinical relevance
If multiple facts: combine intelligently without redundancy"""
    }
    
    return json.dumps(request)


def enhance_clinical_fact(
    fact: str,
    fact_metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Enhance a single fact for clinical relevance.
    
    Args:
        fact: Single fact text
        fact_metadata: Optional metadata about the fact
        
    Returns:
        JSON string with enhancement request
    """
    # Clean up the fact
    fact_clean = re.sub(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}:?\s*', '', fact)
    fact_clean = re.sub(r'\s*\(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\)\s*', '', fact_clean)
    fact_clean = re.sub(r'^On \d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s*', '', fact_clean, flags=re.IGNORECASE)
    fact_clean = fact_clean.replace('...', '').strip()
    
    if fact_clean and not fact_clean.endswith(('.', '!', '?')):
        fact_clean += '.'
    
    request = {
        "action": "enhance_fact",
        "original_fact": fact,
        "cleaned_fact": fact_clean,
        "metadata": fact_metadata or {},
        "enhanced": fact_clean.strip()
    }
    
    return json.dumps(request)


def prepare_event_data(
    date: str,
    associated_facts: List[Dict[str, Any]]
) -> str:
    """
    Prepare all data needed for verification of a single event.
    
    Args:
        date: Event date
        associated_facts: List of facts for this date
        
    Returns:
        JSON string with prepared event data
    """
    raw_fact_summaries = [f.get("summary", "") for f in associated_facts]
    source_texts = [f.get("source_chunk", {}).get("content", "") for f in associated_facts]
    source_text_combined = "\n\n".join(source_texts)
    
    source_docs = sorted(list(set(
        f.get("source_chunk", {}).get("source_document", "") 
        for f in associated_facts if f.get("source_chunk", {}).get("source_document")
    )))
    
    source_pages = sorted(list(set(
        f.get("source_chunk", {}).get("page_number", 0) 
        for f in associated_facts if f.get("source_chunk", {}).get("page_number")
    )))
    
    display_date = "Background Information" if date == "Unknown Date" else date
    
    metadata = {
        "has_reconciliation": any(f.get("status") for f in associated_facts),
        "statuses": list(set(f.get("status", "") for f in associated_facts if f.get("status"))),
        "provenances": list(set(f.get("provenance", "") for f in associated_facts if f.get("provenance")))
    }
    
    event_data = {
        "date": date,
        "display_date": display_date,
        "fact_summaries": raw_fact_summaries,
        "source_text": source_text_combined[:3000],  # Limit for processing
        "source_docs": source_docs,
        "source_pages": source_pages,
        "metadata": metadata,
        "fact_count": len(associated_facts)
    }
    
    return json.dumps(event_data)


# Create FunctionTool instances for Google ADK
build_timeline_tool = FunctionTool(func=build_timeline)
verify_context_tool = FunctionTool(func=verify_with_context)
contextual_prompt_tool = FunctionTool(func=build_contextual_prompt)
generate_correction_tool = FunctionTool(func=generate_contextual_correction)
create_summary_tool = FunctionTool(func=create_clinical_summary)
enhance_fact_tool = FunctionTool(func=enhance_clinical_fact)
prepare_event_tool = FunctionTool(func=prepare_event_data)

# Export all tools
TIMELINE_BUILDER_TOOLS = [
    build_timeline_tool,
    verify_context_tool,
    contextual_prompt_tool,
    generate_correction_tool,
    create_summary_tool,
    enhance_fact_tool,
    prepare_event_tool
]
