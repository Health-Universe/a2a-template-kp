"""
Unified extraction tools for comprehensive clinical data extraction.
Following nutrition_example.py pattern with Google ADK FunctionTool.
"""
import json
from typing import Dict, List, Any, Optional
from google.adk.tools import FunctionTool


def extract_diagnoses(timeline_events: List[Dict[str, Any]]) -> str:
    """
    Extract diagnosis information with special emphasis on cancer staging.
    
    This function prepares the input for LLM diagnosis extraction.
    The actual LLM call happens in the agent executor.
    
    Args:
        timeline_events: List of timeline events to extract from
        
    Returns:
        JSON string with diagnosis extraction request
    """
    if not timeline_events:
        return json.dumps({"action": "extract_diagnoses", "diagnoses": []})
    
    # Format events for processing
    formatted_events = []
    for event in timeline_events:
        event_str = f"Date: {event.get('date', '')}\nSummary: {event.get('summary', '')}\nSources: {', '.join(event.get('source_documents', []))} (Pages: {', '.join(str(p) for p in event.get('source_pages', []))})"
        formatted_events.append(event_str)
    
    events_text = "\n\n".join(formatted_events)
    
    request = {
        "action": "extract_diagnoses",
        "timeline_events_count": len(timeline_events),
        "events_text": events_text,
        "instructions": """Extract all DIAGNOSIS events with comprehensive cancer information:
        
        DIAGNOSIS CATEGORIES TO IDENTIFY:
        1. ANY pathology or biopsy result establishing or confirming diagnosis
        2. Initial cancer diagnosis with complete pathologic information
        3. ALL staging information (any staging system used) - initial and restaging
        4. Histologic classification and grade/differentiation from ANY source
        5. ALL molecular and biomarker results relevant to the cancer type
        6. Disease recurrence, progression, or new primary cancers
        7. Metastatic disease identification and sites
        8. ALL restaging events showing disease status changes
        9. ALL pathologic findings that establish or modify diagnosis
        10. Secondary malignancies or treatment-related cancers
        11. Imaging findings that contribute to staging or diagnosis
        12. ANY diagnostic confirmation or clarification
        
        CLINICAL INFORMATION TO CAPTURE:
        - ALL pathologic diagnoses with histologic type (from any biopsy)
        - ALL staging information (regardless of staging system) - initial and updates
        - ALL grade or differentiation level when available
        - ALL molecular markers, mutations, or biomarkers mentioned
        - Anatomical location and extent of disease
        - ALL lymph node involvement status
        - ALL metastatic sites if present
        - ANY prognostic or predictive factors
        - ALL imaging findings that establish or modify diagnosis
        - ALL confirmatory testing results
        
        RULES:
        - Include ALL events that provide diagnostic information
        - Extract EVERY pathology result, even repeat biopsies of same lesion
        - Include ALL staging information whenever mentioned  
        - Capture ALL diagnostic imaging findings that affect diagnosis or staging
        - Include molecular/biomarker information from ANY source
        - Extract ALL restaging or confirmatory diagnostic events
        - Include diagnostic information from consultation notes
        - Extract ALL grading information (histologic, nuclear, etc.)
        - Include ALL receptor status results from any test
        - Focus on COMPLETENESS - do not filter out diagnostic information
        - Include initial biopsies, repeat biopsies, confirmatory tests
        - Extract imaging findings that establish metastases or staging
        - Include ALL diagnostic details that would influence treatment decisions
        
        OUTPUT FORMAT (JSON):
        {
            "diagnoses": [
                {
                    "date": "MM/DD/YYYY",
                    "summary": "complete diagnostic information including how diagnosed, staging, and molecular features",
                    "type": "primary|staging|recurrence|progression|secondary|metastasis|restaging",
                    "staging_info": "specific staging details if available",
                    "molecular_info": "biomarkers or molecular characteristics if available",
                    "diagnostic_method": "how diagnosis was established (biopsy, imaging, pathology, etc.)",
                    "sources": "document name",
                    "pages": "page numbers"
                }
            ]
        }
        
        SUMMARY FORMAT REQUIREMENTS:
        Each diagnosis summary should include:
        1. The diagnosis itself (cancer type, histology, grade)
        2. How it was diagnosed (e.g., "confirmed by biopsy", "diagnosed via CT imaging", "established by pathology")
        3. Staging information when available
        4. Key molecular/biomarker results when present"""
    }
    
    return json.dumps(request)


def extract_treatments(timeline_events: List[Dict[str, Any]]) -> str:
    """
    Extract treatment information with focus on oncology therapies.
    
    Args:
        timeline_events: List of timeline events to extract from
        
    Returns:
        JSON string with treatment extraction request
    """
    if not timeline_events:
        return json.dumps({"action": "extract_treatments", "treatments": []})
    
    # Format events for processing
    formatted_events = []
    for event in timeline_events:
        event_str = f"Date: {event.get('date', '')}\nSummary: {event.get('summary', '')}\nSources: {', '.join(event.get('source_documents', []))} (Pages: {', '.join(str(p) for p in event.get('source_pages', []))})"
        formatted_events.append(event_str)
    
    events_text = "\n\n".join(formatted_events)
    
    request = {
        "action": "extract_treatments",
        "timeline_events_count": len(timeline_events),
        "events_text": events_text,
        "instructions": """Extract all TREATMENT events with special focus on oncology therapies INCLUDING PLANNED/RECOMMENDED treatments:
        1. Oncologic surgery (resection, debulking, lymphadenectomy, metastasectomy)
        2. Chemotherapy regimens with specific drugs and cycles (including planned regimens)
        3. Radiation therapy (EBRT, SBRT, SRS, brachytherapy) with dose/fractions
        4. Immunotherapy (checkpoint inhibitors: pembrolizumab, nivolumab, atezolizumab)
        5. Targeted therapy (TKIs, mAbs: erlotinib, bevacizumab, trastuzumab)
        6. Hormone therapy (tamoxifen, aromatase inhibitors, ADT)
        7. Bone marrow/stem cell transplant
        8. Clinical trial enrollment with protocol numbers (including planned enrollment)
        9. Supportive oncology care (antiemetics, growth factors, bisphosphonates)
        10. Treatment planning and recommendations (even if not yet started)
        11. Preparatory procedures (port placement, genetic counseling for treatment planning)
        
        RULES:
        - Include actual treatments AND planned/recommended treatments
        - Include treatment planning discussions and recommendations
        - Look for keywords: "planned", "recommended", "suggested", "discussed", "will start", "to begin"
        - For duration, use "MM/DD/YYYY - MM/DD/YYYY" format for completed treatments
        - Use single date for planned treatments or treatment discussions
        - Include SPECIFIC drug names and combinations when mentioned
        - For chemo: include regimen name (FOLFOX, FOLFIRINOX, R-CHOP, etc.)
        - For radiation: include total dose, fractions, and site if specified
        - Include cycle numbers for systemic therapy
        - Note dose reductions or modifications
        - Include clinical trial protocol numbers when available
        - Include preparatory procedures related to treatment (port placement, etc.)
        - Extract from consultation notes discussing treatment plans
        
        TREATMENT CATEGORIES:
        - "planned": for recommended or discussed treatments not yet started
        - "active": for ongoing treatments
        - "completed": for finished treatments
        - "preparatory": for procedures to enable treatment (port placement, genetic counseling)
        
        OUTPUT FORMAT (JSON):
        {
            "treatments": [
                {
                    "date": "MM/DD/YYYY or MM/DD/YYYY - MM/DD/YYYY",
                    "summary": "treatment with specific drugs/dose/technique or treatment plan",
                    "category": "surgery|chemotherapy|radiation|immunotherapy|targeted|hormone|transplant|supportive|preparatory",
                    "regimen": "specific protocol name or drug combination",
                    "status": "planned|active|completed|preparatory",
                    "cycles": "number of cycles if applicable",
                    "sources": "document name",
                    "pages": "page numbers"
                }
            ]
        }"""
    }
    
    return json.dumps(request)


def extract_complications(timeline_events: List[Dict[str, Any]]) -> str:
    """
    Extract complications and adverse events with focus on oncology toxicities.
    
    Args:
        timeline_events: List of timeline events to extract from
        
    Returns:
        JSON string with complications extraction request
    """
    if not timeline_events:
        return json.dumps({"action": "extract_complications", "complications": []})
    
    # Format events for processing
    formatted_events = []
    for event in timeline_events:
        event_str = f"Date: {event.get('date', '')}\nSummary: {event.get('summary', '')}\nSources: {', '.join(event.get('source_documents', []))} (Pages: {', '.join(str(p) for p in event.get('source_pages', []))})"
        formatted_events.append(event_str)
    
    events_text = "\n\n".join(formatted_events)
    
    request = {
        "action": "extract_complications",
        "timeline_events_count": len(timeline_events),
        "events_text": events_text,
        "instructions": """Identify all COMPLICATIONS and adverse events related to cancer treatment or disease:
        
        GENERAL COMPLICATION PRINCIPLES:
        1. Any negative health outcome related to treatment
        2. Unexpected results from medical procedures
        3. Symptoms or conditions caused by the underlying disease
        4. Medical issues requiring intervention or monitoring
        5. Events requiring hospitalization or emergency care
        6. Treatment modifications due to adverse effects
        7. Long-term effects from previous interventions
        8. Secondary medical conditions
        9. Significant symptoms affecting patient wellbeing
        10. Any medically concerning development
        
        PRINCIPLES FOR IDENTIFICATION:
        - Include any significant medical concern or adverse outcome
        - Exclude routine disease monitoring unless complications arise
        - Include severity information when available (any description)
        - Note if issue led to treatment changes or medical intervention
        - Include events requiring additional medical care
        - Consider both immediate and delayed medical concerns
        - Include any symptom or condition requiring clinical attention
        - Look for patient-reported concerns with medical significance
        - Include any development requiring medical evaluation or intervention
        
        WHAT QUALIFIES AS A COMPLICATION:
        - Any negative outcome from medical treatment
        - Unexpected results from procedures
        - Symptoms or conditions requiring medical attention
        - Events leading to hospitalization or emergency care
        - Medical issues requiring treatment modification
        - Any graded adverse event (regardless of system used)
        - Treatment interruptions due to medical concerns
        - Patient-reported symptoms of clinical significance
        - Any medically concerning development
        - Conditions affecting patient function or wellbeing
        
        OUTPUT FORMAT (JSON):
        {
            "complications": [
                {
                    "date": "MM/DD/YYYY",
                    "summary": "specific complication with severity if available",
                    "category": "treatment-related|procedural|disease-related|medical-intervention|other",
                    "severity": "grade or description if available",
                    "outcome": "resolved|ongoing|led to treatment change|hospitalization|unknown",
                    "sources": "document name",
                    "pages": "page numbers"
                }
            ]
        }"""
    }
    
    return json.dumps(request)


def extract_response_metrics(timeline_events: List[Dict[str, Any]]) -> str:
    """
    Extract cancer treatment response metrics including RECIST criteria.
    
    Args:
        timeline_events: List of timeline events to extract from
        
    Returns:
        JSON string with response metrics extraction request
    """
    if not timeline_events:
        return json.dumps({"action": "extract_response_metrics", "response_metrics": []})
    
    # Format events for processing
    formatted_events = []
    for event in timeline_events:
        event_str = f"Date: {event.get('date', '')}\nSummary: {event.get('summary', '')}\nSources: {', '.join(event.get('source_documents', []))} (Pages: {', '.join(str(p) for p in event.get('source_pages', []))})"
        formatted_events.append(event_str)
    
    events_text = "\n\n".join(formatted_events)
    
    request = {
        "action": "extract_response_metrics",
        "timeline_events_count": len(timeline_events),
        "events_text": events_text,
        "instructions": """Extract all RESPONSE METRICS and tumor assessments specific to oncology:
        1. RECIST criteria assessments (CR, PR, SD, PD)
        2. Tumor size measurements and percentage changes
        3. Pathologic response (pCR, MPR, near-complete response, % viable tumor)
        4. Radiographic response assessments
        5. Biomarker responses (CEA, CA19-9, PSA, ctDNA changes)
        6. PET/CT metabolic response (SUV changes, Deauville scores)
        7. Liquid biopsy/ctDNA clearance
        8. Disease-free survival milestones
        9. Time to progression metrics
        10. Brain metastases response (RANO criteria)
        
        RULES:
        - Include ONLY objective response assessments with specific criteria
        - Must include percentage changes or specific measurements
        - Include the assessment method/criteria used
        - Do NOT include baseline measurements without comparison
        - Include both radiographic and pathologic responses
        - Note if response led to treatment decisions
        
        ONCOLOGY-SPECIFIC METRICS:
        - RECIST 1.1: "PR with 45% reduction", "PD with 25% increase"
        - Pathologic: "90% treatment effect", "pCR (ypT0N0)", "MPR with <10% viable tumor"
        - Biomarkers: "CEA decreased from 125 to 15 ng/mL", "PSA nadir 0.1"
        - PET: "SUVmax decreased from 12.5 to 3.2", "Deauville score 2"
        - Brain: "RANO-BM partial response"
        - Liquid biopsy: "ctDNA clearance achieved", "VAF decreased from 2.1% to 0%"
        
        OUTPUT FORMAT (JSON):
        {
            "response_metrics": [
                {
                    "date": "MM/DD/YYYY",
                    "summary": "specific response with measurements and criteria",
                    "criteria": "RECIST|pathologic|biomarker|metabolic|RANO|other",
                    "response_category": "CR|PR|SD|PD|pCR|MPR|other",
                    "measurement": "specific percentage or value change",
                    "sources": "document name",
                    "pages": "page numbers"
                }
            ]
        }"""
    }
    
    return json.dumps(request)


def extract_demographics(timeline_events: List[Dict[str, Any]]) -> str:
    """
    Extract patient demographics (age, gender, date of birth).
    
    Args:
        timeline_events: List of timeline events to search
        
    Returns:
        JSON string with demographics extraction request
    """
    if not timeline_events:
        return json.dumps({
            "action": "extract_demographics",
            "age": None,
            "gender": None,
            "date_of_birth": None
        })
    
    # Format events for processing
    formatted_events = []
    for event in timeline_events:
        event_str = f"Date: {event.get('date', '')}\nSummary: {event.get('summary', '')}\nSources: {', '.join(event.get('source_documents', []))} (Pages: {', '.join(str(p) for p in event.get('source_pages', []))})"
        formatted_events.append(event_str)
    
    events_text = "\n\n".join(formatted_events)
    
    request = {
        "action": "extract_demographics",
        "timeline_events_count": len(timeline_events),
        "events_text": events_text,
        "instructions": """Extract patient demographic information from the medical records:
        
        DEMOGRAPHIC INFORMATION TO FIND:
        1. Patient age (in years) - look for explicit age statements
        2. Patient gender/sex - look for gender references or pronouns
        3. Date of birth - look for DOB or birth date information
        
        SEARCH STRATEGIES:
        - Look for explicit age statements ("58 year old", "age 45", "58yo", "58-year-old")
        - Find age in patient information headers or demographics sections
        - Search for age in consultation notes ("This 67-year-old", "67yo patient")
        - Look for age in intake documentation ("Patient is a 45 year old")
        - Find age in medical history sections
        - Check provider assessments that mention patient age
        - Look for age in referral information or transfer notes
        - Search for age in any clinical context or patient descriptions
        - Find gender references ("male", "female", "man", "woman") 
        - Search for pronouns consistently used ("he/him", "she/her")
        - Look for gender in patient information or demographics
        - Find gender references in clinical descriptions
        - Look for date of birth entries ("DOB:", "Date of birth:")
        - Check patient information sections or headers
        - Look in consultation notes or intake documentation
        - Find demographic information in clinical assessments
        
        RULES:
        - Only extract information explicitly stated in the records
        - Do not infer or assume demographics
        - If multiple sources provide same information, use the most recent/reliable
        - For age, prefer explicit age statements over calculated ages
        - For gender, use the terms as stated in the medical record
        - Return None for any demographic not found
        
        OUTPUT FORMAT (JSON):
        {
            "age": "age in years as string or null",
            "gender": "gender as stated in records or null", 
            "date_of_birth": "date of birth in MM/DD/YYYY format or null",
            "sources": {
                "age_source": "document and page where age was found",
                "gender_source": "document and page where gender was found",
                "dob_source": "document and page where DOB was found"
            }
        }"""
    }
    
    return json.dumps(request)


def generate_patient_headline(
    timeline_events: List[Dict[str, Any]],
    diagnosis_treatment_data: Dict[str, Any],
    demographics_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a concise 1-2 line patient headline for oncology patients.
    
    Args:
        timeline_events: List of timeline events
        diagnosis_treatment_data: Extracted diagnoses and treatments
        demographics_data: Optional demographics information
        
    Returns:
        JSON string with headline generation request
    """
    if not timeline_events:
        return json.dumps({
            "action": "generate_headline",
            "headline": "Patient summary unavailable due to insufficient data."
        })
    
    # Get demographics if available
    age = None
    gender = None
    if demographics_data:
        age = demographics_data.get("age")
        gender = demographics_data.get("gender")
    
    # Gather diagnosis information
    diagnoses = diagnosis_treatment_data.get("diagnoses", [])
    
    # Format events
    all_events = []
    for event in timeline_events[:20]:
        all_events.append(f"{event.get('date', '')}: {event.get('summary', '')}")
    events_text = "\n".join(all_events)
    
    # Include all diagnoses with staging
    primary_diagnoses_with_staging = []
    for d in diagnoses:
        diagnosis_text = f"{d['date']}: {d['summary']}"
        if d.get('staging_info'):
            diagnosis_text += f" ({d['staging_info']})"
        primary_diagnoses_with_staging.append(diagnosis_text)
    
    primary_diagnoses = "\n".join(primary_diagnoses_with_staging)
    
    request = {
        "action": "generate_patient_headline",
        "age": age,
        "gender": gender,
        "diagnoses_count": len(diagnoses),
        "primary_diagnoses": primary_diagnoses if primary_diagnoses else "No specific diagnoses identified",
        "events_text": events_text,
        "instructions": """Create a 1-2 line patient headline focused on oncology that captures ALL essential patient diagnoses.
        
        FORMAT: "Patient is a [AGE] year old [SEX] with [ALL CANCER DIAGNOSES, STAGES, KEY MOLECULAR FEATURES]"
        
        DIAGNOSTIC COMPLETENESS RULES:
        - Include ALL primary cancer diagnoses from the diagnosis data
        - If multiple cancers, list all with their individual stages and characteristics
        - For each cancer, include: location, histology, stage, and key molecular features
        - Use connecting words like "and", "with concurrent", "and synchronous" for multiple cancers
        - Prioritize the most advanced stage cancer first, then list others
        - Include metastatic sites if present (hepatic, pulmonary, nodal, etc.)
        
        DEMOGRAPHIC USAGE RULES:
        - ALWAYS use provided age and gender if available - do not search timeline events for demographics when already provided
        - If age is provided, use it in the headline format
        - If gender is provided, use it in the headline format  
        - If both age and gender provided: "Patient is a [age] year old [gender] with..."
        - If only gender provided: "Patient is a [gender] with..."
        - If only age provided: "Patient is a [age] year old individual with..."
        - If neither provided: "Patient with..." or search clinical events as fallback
        
        GENERAL PRINCIPLES:
        - Prioritize provided demographics over searching timeline events
        - Include ALL cancer diagnoses with their staging when available
        - Include key molecular markers if relevant (MMR status, receptor status, etc.)
        - Include histology when important (adenocarcinoma, invasive ductal carcinoma, etc.)
        - For metastatic disease, mention key metastatic sites
        - Maximum 2 lines but use both lines if needed for multiple cancers or complex disease
        - Be comprehensive but concise
        - Use present tense
        - Don't include dates or treatment details"""
    }
    
    return json.dumps(request)


def format_timeline_events(timeline_events: List[Any]) -> str:
    """
    Helper function to format timeline events for LLM processing.
    
    Args:
        timeline_events: List of timeline event objects
        
    Returns:
        JSON string with formatted events
    """
    formatted_events = []
    for event in timeline_events:
        # Handle both dict and object formats
        if isinstance(event, dict):
            event_str = f"Date: {event.get('date', '')}\nSummary: {event.get('summary', '')}\nSources: {', '.join(event.get('source_documents', []))} (Pages: {', '.join(str(p) for p in event.get('source_pages', []))})"
        else:
            # Assume it's an object with attributes
            event_str = f"Date: {getattr(event, 'date', '')}\nSummary: {getattr(event, 'summary', '')}\nSources: {', '.join(getattr(event, 'source_documents', []))} (Pages: {', '.join(str(p) for p in getattr(event, 'source_pages', []))})"
        formatted_events.append(event_str)
    
    return json.dumps({
        "action": "format_events",
        "formatted_events": formatted_events,
        "count": len(formatted_events)
    })


# Create FunctionTool instances for Google ADK
extract_diagnoses_tool = FunctionTool(func=extract_diagnoses)
extract_treatments_tool = FunctionTool(func=extract_treatments)
extract_complications_tool = FunctionTool(func=extract_complications)
extract_response_metrics_tool = FunctionTool(func=extract_response_metrics)
extract_demographics_tool = FunctionTool(func=extract_demographics)
generate_headline_tool = FunctionTool(func=generate_patient_headline)
format_events_tool = FunctionTool(func=format_timeline_events)

# Export all tools
UNIFIED_EXTRACTOR_TOOLS = [
    extract_diagnoses_tool,
    extract_treatments_tool,
    extract_complications_tool,
    extract_response_metrics_tool,
    extract_demographics_tool,
    generate_headline_tool,
    format_events_tool
]
