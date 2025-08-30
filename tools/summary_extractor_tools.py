"""
Summary extraction tools for fact extraction from clinical documents.
Following nutrition_example.py pattern with Google ADK FunctionTool.
"""
import json
import re
from typing import Dict, List, Any, Optional
from google.adk.tools import FunctionTool


def extract_from_reconciled_groups(
    reconciled_groups: List[Dict[str, Any]],
    page_to_chunk_mapping: Optional[Dict[int, Dict[str, Any]]] = None
) -> str:
    """
    Extract facts from reconciled encounter groups.
    
    This function prepares the input for LLM fact extraction.
    The actual LLM call happens in the agent executor.
    
    Args:
        reconciled_groups: List of reconciled encounter groups
        page_to_chunk_mapping: Optional mapping of page numbers to document chunks
        
    Returns:
        JSON string with extraction request
    """
    # Process reconciled groups for extraction
    extraction_requests = []
    
    for group in reconciled_groups:
        encounter_date = group.get("encounter_date", "")
        reconciled_facts = group.get("reconciled_facts", [])
        
        # Filter out carry-forward facts
        unique_facts = []
        for fact in reconciled_facts:
            # Skip carry-forward facts to avoid duplicates
            if fact.get("is_carry_forward") and fact.get("provenance") == "Previously Reported":
                continue
            
            unique_facts.append({
                "content": fact.get("content", ""),
                "status": fact.get("status", ""),
                "provenance": fact.get("provenance", ""),
                "confidence": fact.get("confidence", 0.9),
                "source_pages": fact.get("source_pages", []),
                "source_documents": fact.get("source_documents", [])
            })
        
        if unique_facts:
            extraction_requests.append({
                "encounter_date": encounter_date,
                "encounter_type": group.get("encounter_type", "unknown"),
                "facts": unique_facts,
                "fact_count": len(unique_facts)
            })
    
    request = {
        "action": "extract_from_reconciled",
        "total_groups": len(reconciled_groups),
        "extraction_requests": extraction_requests,
        "instructions": """Process these reconciled facts for final extraction.

For each fact:
1. Preserve the exact clinical content
2. Maintain the encounter date
3. Determine medical specialty based on content
4. Include status and provenance metadata
5. Skip any previously reported (carry-forward) facts

Return extracted facts organized by encounter with appropriate metadata."""
    }
    
    return json.dumps(request)


def extract_events_by_date(
    chunk_content: str,
    page_number: int,
    source_document: str
) -> str:
    """
    Extract all dated clinical events from a document chunk.
    
    Args:
        chunk_content: The text content of the chunk
        page_number: Page number of the chunk
        source_document: Source document name
        
    Returns:
        JSON string with extraction request
    """
    # Clean content
    clean_content = re.sub(r'Case #\d+.*?To Top.*?\n', '', chunk_content, flags=re.IGNORECASE)
    
    request = {
        "action": "extract_events_by_date",
        "page_number": page_number,
        "source_document": source_document,
        "content": clean_content,
        "instructions": f"""You are an expert clinical data extractor. Your task is to scan the provided page of a medical record and extract all dated clinical events that would be relevant for medical decision-making and patient care.

        **Text from Page {page_number} of document '{source_document}':**
        ---
        {clean_content}
        ---

        **CRITICAL RULES FOR EXTRACTION:**

        1.  **IGNORE DEMOGRAPHIC DATES:** Ignore dates labeled 'Date of Birth' or 'DOB'.
        2.  **USE EVENT DATES:** Use dates labeled 'Exam Date', 'Report Date', 'Date of Service', 'Collection Date', or dates next to a procedure.
        3.  **BE COMPREHENSIVE:** Extract EVERY distinct finding or result, even if they seem similar. Create separate events for each distinct piece of information.
        4.  **EXHAUSTIVE EXTRACTION:** Extract ALL medically relevant events including:
                - EVERY diagnostic test result and finding
                - EVERY measurement, value, or quantitative result
                - ALL test interpretations and conclusions
                - ALL treatment decisions and interventions
                - ALL clinical assessments and evaluations
                - ALL disease status evaluations
                - ALL consultant recommendations and referrals
                - ALL significant clinical changes or developments
                - ALL therapeutic procedures and their outcomes
                - ALL laboratory results with clinical significance
                - ALL care coordination and planning
                - ALL biomarker or molecular testing results
                - ALL staging or classification information
                - ALL follow-up recommendations and care plans
                - ALL confirmation or repeat testing results
                - ALL diagnostic clarifications or updates

        **CLINICAL RELEVANCE CRITERIA:**
        - Information that affects diagnosis, staging, or treatment decisions
        - ALL test results showing disease characteristics or extent
        - ALL interventions that impact patient care
        - ALL findings that require follow-up or monitoring
        - ALL recommendations from healthcare providers
        - ALL significant changes in patient condition
        - ALL biomarker or molecular testing results
        - ALL receptor status or genetic testing information
        - ALL staging or disease extent assessments
        - ANY information that would influence clinical management

        **EXTRACTION PRINCIPLES:**
        - Extract EVERY piece of medically relevant information - be exhaustive, not selective
        - If a single report contains multiple findings, create separate events for EACH distinct finding
        - Include ALL test results, even if they confirm previous findings
        - Capture ALL diagnostic details, including measurements, grades, staging components, receptor status
        - Do not consolidate or combine findings - extract each separately for maximum granularity
        - Extract complete pathology findings including ALL markers tested and ALL results
        - Include ALL treatment recommendations, even preliminary discussions
        - Capture ALL clinical observations and assessments, including repeat evaluations
        - Extract ALL imaging findings, including measurements and specific anatomical details
        - Include follow-up results that confirm, modify, or add to previous findings
        - Do not assume anything is redundant - extract everything for completeness

        **Output Format (CRITICAL):**
        Respond with a JSON list of objects. Each object represents a SINGLE DATE found on the page and must contain:
        - `date_str`: The date you found, normalized to `YYYY-MM-DD`.
        - `events`: A list of strings, where each string is a **brief, one-sentence summary** of a single clinically relevant event from that day.
        - `specialty`: The single most relevant medical specialty for the events of that day (e.g., "Radiology", "Pathology", "Medical Oncology", "Surgery", "Internal Medicine", "Cardiology").

        **Example Response:**
        [
          {{
            "date_str": "2024-09-12",
            "events": [
              "Primary diagnostic procedure performed on target tissue.",
              "Tissue measurements and dimensions documented.",
              "First biomarker test result reported.",
              "Second biomarker test result reported.", 
              "Third biomarker test result reported.",
              "Proliferation marker level determined.",
              "Regional tissue sampling showed specific findings.",
              "Diagnostic classification assigned.",
              "Clinical assessment recommendation provided."
            ],
            "specialty": "Pathology"
          }},
          {{
            "date_str": "2024-09-15",
            "events": [
              "Imaging showed primary lesion with specific measurements.",
              "Second area of concern identified with measurements.",
              "Regional tissue shows abnormal characteristics.",
              "Assessment of local extension completed.",
              "Additional diagnostic procedures recommended."
            ],
            "specialty": "Radiology"
          }}
        ]"""
    }
    
    return json.dumps(request)


def determine_specialty(content: str) -> str:
    """
    Determine medical specialty based on content analysis.
    
    Args:
        content: Clinical content to analyze
        
    Returns:
        JSON string with specialty determination
    """
    content_lower = content.lower()
    
    # Specialty keywords mapping
    specialty_keywords = {
        "Radiology": ["ct", "mri", "scan", "imaging", "contrast", "enhancement", "lesion", "mass"],
        "Pathology": ["biopsy", "specimen", "histology", "grade", "differentiated", "necrosis", "margins"],
        "Medical Oncology": ["chemotherapy", "folfox", "cisplatin", "carboplatin", "cycles", "regimen"],
        "Surgery": ["resection", "surgery", "operative", "incision", "dissection", "anastomosis"],
        "Radiation Oncology": ["radiation", "radiotherapy", "gray", "fractions", "boost", "imrt"],
        "Laboratory": ["lab", "blood", "wbc", "hemoglobin", "platelet", "creatinine", "liver"],
        "Cardiology": ["ecg", "echo", "ejection fraction", "cardiac", "heart", "coronary"],
        "Internal Medicine": ["admission", "discharge", "consultation", "physical exam", "review"]
    }
    
    # Score each specialty
    specialty_scores = {}
    for specialty, keywords in specialty_keywords.items():
        score = sum(1 for keyword in keywords if keyword in content_lower)
        if score > 0:
            specialty_scores[specialty] = score
    
    # Determine the best match
    if specialty_scores:
        best_specialty = max(specialty_scores.items(), key=lambda x: x[1])[0]
        confidence = specialty_scores[best_specialty] / len(specialty_keywords[best_specialty])
    else:
        best_specialty = "Internal Medicine"
        confidence = 0.5
    
    result = {
        "action": "determine_specialty",
        "content_preview": content[:200],
        "specialty_scores": specialty_scores,
        "selected_specialty": best_specialty,
        "confidence": confidence
    }
    
    return json.dumps(result)


def process_extraction_batch(
    chunks: List[Dict[str, Any]],
    batch_size: int = 5
) -> str:
    """
    Process multiple chunks in a batch for efficient extraction.
    
    Args:
        chunks: List of document chunks to process
        batch_size: Number of chunks to process together
        
    Returns:
        JSON string with batch processing request
    """
    # Prepare chunks for batch processing
    chunk_summaries = []
    for i, chunk in enumerate(chunks[:batch_size]):
        chunk_summaries.append({
            "index": i,
            "page_number": chunk.get("page_number", 0),
            "source_document": chunk.get("source_document", "Unknown"),
            "content_length": len(chunk.get("content", "")),
            "content_preview": chunk.get("content", "")[:500]
        })
    
    request = {
        "action": "process_batch",
        "batch_size": len(chunk_summaries),
        "total_chunks": len(chunks),
        "chunk_summaries": chunk_summaries,
        "instructions": """Process this batch of document chunks for fact extraction.

For each chunk:
1. Extract all dated clinical events
2. Categorize by medical specialty
3. Be exhaustive and granular
4. Create separate events for each finding

Return a consolidated list of all extracted facts from the batch."""
    }
    
    return json.dumps(request)


def validate_extracted_facts(
    extracted_facts: List[Dict[str, Any]]
) -> str:
    """
    Validate the quality and completeness of extracted facts.
    
    Args:
        extracted_facts: List of extracted facts to validate
        
    Returns:
        JSON string with validation results
    """
    # Analyze extracted facts
    total_facts = len(extracted_facts)
    facts_by_date = {}
    facts_by_specialty = {}
    
    for fact in extracted_facts:
        # Group by date
        date_str = fact.get("date_str", "Unknown")
        if date_str not in facts_by_date:
            facts_by_date[date_str] = 0
        facts_by_date[date_str] += 1
        
        # Group by specialty
        specialty = fact.get("specialty", "Unknown")
        if specialty not in facts_by_specialty:
            facts_by_specialty[specialty] = 0
        facts_by_specialty[specialty] += 1
    
    # Check for potential issues
    issues = []
    
    # Check for facts without dates
    if "Unknown" in facts_by_date:
        issues.append({
            "type": "missing_dates",
            "severity": "warning",
            "details": f"{facts_by_date['Unknown']} facts have unknown dates"
        })
    
    # Check for unbalanced specialty distribution
    if facts_by_specialty and max(facts_by_specialty.values()) / total_facts > 0.8:
        dominant_specialty = max(facts_by_specialty.items(), key=lambda x: x[1])[0]
        issues.append({
            "type": "specialty_imbalance",
            "severity": "info",
            "details": f"{dominant_specialty} accounts for >80% of facts"
        })
    
    validation = {
        "action": "validation_results",
        "total_facts": total_facts,
        "unique_dates": len(facts_by_date),
        "facts_by_date": facts_by_date,
        "facts_by_specialty": facts_by_specialty,
        "issues": issues,
        "validation_passed": len([i for i in issues if i["severity"] == "error"]) == 0
    }
    
    return json.dumps(validation)


# Create FunctionTool instances for Google ADK
extract_reconciled_tool = FunctionTool(func=extract_from_reconciled_groups)
extract_events_tool = FunctionTool(func=extract_events_by_date)
determine_specialty_tool = FunctionTool(func=determine_specialty)
batch_process_tool = FunctionTool(func=process_extraction_batch)
validate_facts_tool = FunctionTool(func=validate_extracted_facts)

# Export all tools
SUMMARY_EXTRACTOR_TOOLS = [
    extract_reconciled_tool,
    extract_events_tool,
    determine_specialty_tool,
    batch_process_tool,
    validate_facts_tool
]
