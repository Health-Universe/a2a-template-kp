"""
Checker tools for single-pass verification of clinical summaries.
Following nutrition_example.py pattern with Google ADK FunctionTool.
"""
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from google.adk.tools import FunctionTool


def comprehensive_verification(
    summary: str,
    source_text: str,
    verification_prompt: Optional[str] = None
) -> str:
    """
    Perform comprehensive single-pass verification with detailed context.
    
    This function prepares the input for LLM verification.
    The actual LLM call happens in the agent executor.
    
    Args:
        summary: Summary to verify
        source_text: Source text to verify against
        verification_prompt: Optional custom verification prompt
        
    Returns:
        JSON string with verification request
    """
    # Limit source text for context
    source_text_limited = source_text[:3000] + "..." if len(source_text) > 3000 else source_text
    
    if verification_prompt:
        # Use custom prompt
        prompt = f"""
            {verification_prompt}
            
            SUMMARY TO VERIFY: {summary}
            
            Analyze the summary and provide comprehensive feedback in JSON format:
            {{
                "is_verified": true/false,
                "confidence": 0.0-1.0,
                "claim_analysis": [
                    {{
                        "claim": "specific claim from summary",
                        "status": "verified|unverified|needs_clarification",
                        "source_support": "relevant quote from source or null",
                        "issue": "issue description if not verified"
                    }}
                ],
                "suggested_corrections": {{
                    "has_corrections": true/false,
                    "corrected_summary": "suggested correction if needed",
                    "correction_notes": "what was changed and why"
                }},
                "verification_summary": "overall assessment"
            }}
            """
    else:
        # Default verification prompt
        prompt = f"""
            You are an expert medical fact-checker. Perform a comprehensive single-pass verification of this summary.

            VERIFICATION CRITERIA:
            1. FACTUAL ACCURACY: Every fact must be explicitly stated or clearly evident in source
            2. CLINICAL RELEVANCE: Focus on medically significant information
            3. DATE ACCURACY: Verify dates match source exactly
            4. SOURCE FIDELITY: Ensure summary reflects source content accurately
            5. COMPLETENESS: Check if key clinical information is included

            SOURCE TEXT:
            {source_text_limited}

            SUMMARY TO VERIFY:
            {summary}

            Provide comprehensive verification feedback in JSON format:
            {{
                "is_verified": true/false,
                "confidence": 0.0-1.0,
                "claim_analysis": [
                    {{
                        "claim": "specific claim from summary",
                        "status": "verified|unverified|needs_clarification",
                        "source_support": "relevant quote from source or null",
                        "issue": "issue description if not verified"
                    }}
                ],
                "issues_found": [
                    {{
                        "type": "factual_error|missing_info|date_error|terminology|inference",
                        "description": "specific issue description",
                        "severity": "high|medium|low"
                    }}
                ],
                "suggested_corrections": {{
                    "has_corrections": true/false,
                    "corrected_summary": "suggested correction if needed",
                    "correction_notes": "what was changed and why"
                }},
                "clinical_completeness": {{
                    "key_findings_included": true/false,
                    "missing_elements": ["list of missing clinical elements"]
                }},
                "verification_summary": "overall assessment in 1-2 sentences"
            }}
            """
    
    request = {
        "action": "comprehensive_verification",
        "summary": summary,
        "source_text_length": len(source_text),
        "has_custom_prompt": verification_prompt is not None,
        "prompt": prompt
    }
    
    return json.dumps(request)


def analyze_claims(
    summary: str,
    source_text: str
) -> str:
    """
    Break down summary into individual claims for detailed analysis.
    
    Args:
        summary: Summary to analyze
        source_text: Source text for verification
        
    Returns:
        JSON string with claim analysis request
    """
    request = {
        "action": "analyze_claims",
        "summary": summary,
        "source_text_preview": source_text[:500] + "..." if len(source_text) > 500 else source_text,
        "instructions": """Break down this summary into individual verifiable claims.

For each claim:
1. Extract the specific factual assertion
2. Determine if it can be verified from the source
3. Find supporting evidence if available
4. Identify any issues or discrepancies

Focus on:
- Medical facts and findings
- Dates and temporal information
- Measurements and values
- Clinical assessments
- Treatment information

Return detailed claim-by-claim analysis."""
    }
    
    return json.dumps(request)


def suggest_corrections(
    summary: str,
    issues_found: List[Dict[str, Any]],
    source_text: str
) -> str:
    """
    Generate specific corrections based on identified issues.
    
    Args:
        summary: Original summary with issues
        issues_found: List of identified issues
        source_text: Source text for reference
        
    Returns:
        JSON string with correction request
    """
    # Group issues by type
    issue_types = {}
    for issue in issues_found:
        issue_type = issue.get("type", "other")
        if issue_type not in issue_types:
            issue_types[issue_type] = []
        issue_types[issue_type].append(issue.get("description", ""))
    
    prompt = f"""Generate a corrected version of this summary based on the identified issues.

ORIGINAL SUMMARY: {summary}

IDENTIFIED ISSUES:
"""
    
    for issue_type, descriptions in issue_types.items():
        prompt += f"\n{issue_type.upper()}:\n"
        for desc in descriptions:
            prompt += f"- {desc}\n"
    
    prompt += f"""

SOURCE TEXT (for reference):
{source_text[:1500]}...

Generate a corrected summary that:
1. Addresses all identified issues
2. Maintains factual accuracy from source
3. Preserves all correct information
4. Uses appropriate medical terminology
5. Remains concise and clear

Provide the correction with explanation of changes."""
    
    request = {
        "action": "suggest_corrections",
        "original_summary": summary,
        "issue_count": len(issues_found),
        "issue_types": list(issue_types.keys()),
        "prompt": prompt
    }
    
    return json.dumps(request)


def assess_clinical_completeness(
    summary: str,
    source_text: str,
    clinical_focus: Optional[str] = None
) -> str:
    """
    Assess if key clinical information is included in the summary.
    
    Args:
        summary: Summary to assess
        source_text: Source text containing full information
        clinical_focus: Optional specific clinical area to focus on
        
    Returns:
        JSON string with completeness assessment request
    """
    focus_areas = {
        "diagnosis": ["diagnoses", "staging", "grading", "classification"],
        "treatment": ["medications", "procedures", "surgeries", "therapies"],
        "results": ["lab results", "imaging findings", "pathology", "biomarkers"],
        "timeline": ["dates", "sequence of events", "follow-up timing"],
        "general": ["key findings", "clinical status", "recommendations"]
    }
    
    selected_focus = focus_areas.get(clinical_focus, focus_areas["general"])
    
    request = {
        "action": "assess_completeness",
        "summary": summary,
        "source_text_preview": source_text[:1000] + "..." if len(source_text) > 1000 else source_text,
        "clinical_focus": clinical_focus or "general",
        "focus_elements": selected_focus,
        "instructions": f"""Assess the clinical completeness of this summary.

Check if the summary includes key clinical elements:
{', '.join(selected_focus)}

Identify:
1. What key clinical information is included
2. What important information is missing
3. Whether the summary captures the clinical essence
4. Any critical omissions that affect understanding

Return a completeness assessment with specific missing elements."""
    }
    
    return json.dumps(request)


def validate_verification_result(
    verification_result: Dict[str, Any]
) -> str:
    """
    Validate and normalize verification results.
    
    Args:
        verification_result: Raw verification result to validate
        
    Returns:
        JSON string with validated result
    """
    # Ensure all required fields
    validated = {
        "is_verified": verification_result.get("is_verified", False),
        "confidence": float(verification_result.get("confidence", 0.0)),
        "claim_analysis": verification_result.get("claim_analysis", []),
        "issues_found": verification_result.get("issues_found", []),
        "suggested_corrections": verification_result.get("suggested_corrections", {
            "has_corrections": False,
            "corrected_summary": "",
            "correction_notes": ""
        }),
        "clinical_completeness": verification_result.get("clinical_completeness", {
            "key_findings_included": True,
            "missing_elements": []
        }),
        "verification_summary": verification_result.get("verification_summary", "Verification completed")
    }
    
    # Validate confidence score
    validated["confidence"] = max(0.0, min(1.0, validated["confidence"]))
    
    # Check for high-severity issues
    high_severity_issues = [
        i for i in validated["issues_found"]
        if i.get("severity") == "high"
    ]
    
    # Adjust verification status based on issues
    if high_severity_issues and validated["is_verified"]:
        validated["is_verified"] = False
        validated["confidence"] = min(validated["confidence"], 0.5)
    
    # Count issues by type
    issue_counts = {}
    for issue in validated["issues_found"]:
        issue_type = issue.get("type", "other")
        issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
    
    result = {
        "action": "validate_result",
        "validated_result": validated,
        "high_severity_count": len(high_severity_issues),
        "total_issues": len(validated["issues_found"]),
        "issue_counts": issue_counts,
        "validation_passed": True
    }
    
    return json.dumps(result)


# Create FunctionTool instances for Google ADK
comprehensive_verification_tool = FunctionTool(func=comprehensive_verification)
analyze_claims_tool = FunctionTool(func=analyze_claims)
suggest_corrections_tool = FunctionTool(func=suggest_corrections)
assess_completeness_tool = FunctionTool(func=assess_clinical_completeness)
validate_result_tool = FunctionTool(func=validate_verification_result)

# Export all tools
CHECKER_TOOLS = [
    comprehensive_verification_tool,
    analyze_claims_tool,
    suggest_corrections_tool,
    assess_completeness_tool,
    validate_result_tool
]
