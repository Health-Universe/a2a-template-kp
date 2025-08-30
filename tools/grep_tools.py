"""
Grep tools for pattern searching with LLM error handling.
Following nutrition_example.py pattern with Google ADK FunctionTool.
"""
import json
import re
import os
from typing import Dict, List, Any, Optional
from google.adk.tools import FunctionTool
import logging

logger = logging.getLogger(__name__)


def search_medical_patterns(
    file_path: str,
    patterns: List[str],
    case_sensitive: bool = False,
    max_matches: int = 50,
    context_lines: int = 3,
    file_content: Optional[str] = None
) -> str:
    """
    Search for multiple regex patterns in a medical document.
    
    Args:
        file_path: Path to the document to search
        patterns: List of regex patterns to search for
        case_sensitive: Whether search is case sensitive
        max_matches: Maximum matches per pattern
        context_lines: Lines of context before/after match
        file_content: Optional document content (if provided, file_path is ignored)
        
    Returns:
        JSON string with search results and any errors
    """
    results = {
        "file_path": file_path,
        "search_results": [],
        "errors": [],
        "summary": {
            "total_patterns": len(patterns),
            "successful_searches": 0,
            "total_matches": 0
        }
    }
    
    # Get content either from parameter or file
    full_content = None
    if file_content is not None:
        # Use provided content directly
        full_content = file_content
        lines = file_content.splitlines(keepends=True)
    else:
        # Read from file system
        if not os.path.exists(file_path):
            results["errors"].append({
                "type": "file_not_found",
                "message": f"File not found: {file_path}",
                "severity": "critical"
            })
            return json.dumps(results)
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                full_content = f.read()
                lines = full_content.splitlines(keepends=True)
        except Exception as e:
            results["errors"].append({
                "type": "file_read_error",
                "message": f"Error reading file: {str(e)}",
                "severity": "critical"
            })
            return json.dumps(results)
    
    # Check if this is a single-line document
    is_single_line = len(lines) <= 3 and any(len(line) > 1000 for line in lines)
    
    # Search each pattern
    for pattern in patterns:
        pattern_result = {
            "pattern": pattern,
            "matches": [],
            "error": None
        }
        
        try:
            # Handle case sensitivity and (?i) prefix
            flags = 0 if case_sensitive else re.IGNORECASE
            clean_pattern = pattern
            
            if pattern.startswith("(?i)"):
                clean_pattern = pattern[4:]
                flags = re.IGNORECASE
            
            # Compile regex
            regex = re.compile(clean_pattern, flags)
            
            # Search through lines
            match_count = 0
            for i, line in enumerate(lines):
                if match_count >= max_matches:
                    break
                    
                matches = list(regex.finditer(line))
                for match in matches:
                    if match_count >= max_matches:
                        break
                        
                    # Get context
                    context_start = max(0, i - context_lines)
                    context_end = min(len(lines), i + context_lines + 1)
                    context = lines[context_start:context_end]
                    
                    match_info = {
                        "line_number": i + 1,
                        "line_text": line.strip(),
                        "match_text": match.group(),
                        "match_start": match.start(),
                        "match_end": match.end(),
                        "context": [l.strip() for l in context],
                        "context_start_line": context_start + 1
                    }
                    
                    # Add character position for single-line documents
                    if is_single_line and full_content:
                        # Calculate absolute character position in document
                        char_pos = sum(len(lines[j]) for j in range(i)) + match.start()
                        match_info["match_position"] = char_pos
                        match_info["single_line_doc"] = True
                    
                    pattern_result["matches"].append(match_info)
                    match_count += 1
            
            results["summary"]["successful_searches"] += 1
            results["summary"]["total_matches"] += len(pattern_result["matches"])
            
        except re.error as e:
            pattern_result["error"] = {
                "type": "regex_error",
                "message": f"Invalid regex pattern: {str(e)}",
                "pattern": pattern
            }
            results["errors"].append(pattern_result["error"])
        except Exception as e:
            pattern_result["error"] = {
                "type": "search_error",
                "message": f"Error searching pattern: {str(e)}",
                "pattern": pattern
            }
            results["errors"].append(pattern_result["error"])
        
        results["search_results"].append(pattern_result)
    
    return json.dumps(results)


def validate_and_fix_patterns(patterns: List[str]) -> str:
    """
    Validate regex patterns and suggest fixes using LLM guidance.
    
    Args:
        patterns: List of patterns to validate
        
    Returns:
        JSON string with validation results and fix suggestions
    """
    validation_results = {
        "patterns": [],
        "summary": {
            "total": len(patterns),
            "valid": 0,
            "invalid": 0,
            "fixed": 0
        }
    }
    
    for pattern in patterns:
        result = {
            "original": pattern,
            "valid": False,
            "error": None,
            "suggested_fix": None,
            "fix_explanation": None
        }
        
        try:
            # Try to compile the pattern
            re.compile(pattern)
            result["valid"] = True
            validation_results["summary"]["valid"] += 1
        except re.error as e:
            result["error"] = str(e)
            validation_results["summary"]["invalid"] += 1
            
            # Suggest common fixes
            if "unbalanced parenthesis" in str(e).lower():
                result["suggested_fix"] = pattern.replace("(", r"\(").replace(")", r"\)")
                result["fix_explanation"] = "Escaped parentheses that should be literal"
            elif "bad escape" in str(e).lower():
                result["suggested_fix"] = pattern.replace("\\", "\\\\")
                result["fix_explanation"] = "Fixed escape sequences"
            elif "nothing to repeat" in str(e).lower():
                result["suggested_fix"] = re.escape(pattern)
                result["fix_explanation"] = "Escaped special characters"
            
            # Verify the fix works
            if result["suggested_fix"]:
                try:
                    re.compile(result["suggested_fix"])
                    validation_results["summary"]["fixed"] += 1
                except:
                    result["suggested_fix"] = None
                    result["fix_explanation"] = "Could not automatically fix pattern"
        
        validation_results["patterns"].append(result)
    
    return json.dumps(validation_results)


def search_with_error_recovery(
    file_path: str,
    pattern: str,
    fallback_patterns: Optional[List[str]] = None,
    file_content: Optional[str] = None
) -> str:
    """
    Search with automatic error recovery and fallback patterns.
    
    Args:
        file_path: Document to search
        pattern: Primary pattern to search
        fallback_patterns: Alternative patterns if primary fails
        file_content: Optional document content (if provided, file_path is ignored)
        
    Returns:
        JSON string with search results
    """
    if not fallback_patterns:
        fallback_patterns = []
    
    all_patterns = [pattern] + fallback_patterns
    
    for i, current_pattern in enumerate(all_patterns):
        try:
            # Try searching with current pattern
            results = json.loads(search_medical_patterns(
                file_path=file_path,
                patterns=[current_pattern],
                max_matches=100,
                file_content=file_content
            ))
            
            # Check if search was successful
            if results["summary"]["successful_searches"] > 0:
                results["recovery_info"] = {
                    "pattern_used": current_pattern,
                    "attempt_number": i + 1,
                    "total_attempts": len(all_patterns)
                }
                return json.dumps(results)
                
        except Exception as e:
            logger.error(f"Error with pattern {current_pattern}: {str(e)}")
            continue
    
    # All patterns failed
    return json.dumps({
        "error": "All patterns failed",
        "attempted_patterns": all_patterns,
        "file_path": file_path
    })


def analyze_search_performance(search_results: Dict[str, Any]) -> str:
    """
    Analyze search results to identify performance issues and suggest improvements.
    
    Args:
        search_results: Results from previous search
        
    Returns:
        JSON string with performance analysis
    """
    analysis = {
        "performance_metrics": {},
        "issues_found": [],
        "recommendations": []
    }
    
    # Analyze results
    total_patterns = search_results.get("summary", {}).get("total_patterns", 0)
    successful = search_results.get("summary", {}).get("successful_searches", 0)
    total_matches = search_results.get("summary", {}).get("total_matches", 0)
    errors = search_results.get("errors", [])
    
    # Calculate metrics
    analysis["performance_metrics"] = {
        "success_rate": (successful / total_patterns * 100) if total_patterns > 0 else 0,
        "average_matches_per_pattern": total_matches / successful if successful > 0 else 0,
        "error_count": len(errors)
    }
    
    # Identify issues
    if len(errors) > 0:
        analysis["issues_found"].append({
            "type": "pattern_errors",
            "count": len(errors),
            "severity": "high"
        })
        analysis["recommendations"].append(
            "Review and fix regex patterns with errors"
        )
    
    if total_matches == 0 and successful > 0:
        analysis["issues_found"].append({
            "type": "no_matches",
            "severity": "medium"
        })
        analysis["recommendations"].append(
            "Consider broadening search patterns or checking document content"
        )
    
    if total_matches > 1000:
        analysis["issues_found"].append({
            "type": "too_many_matches",
            "count": total_matches,
            "severity": "low"
        })
        analysis["recommendations"].append(
            "Consider more specific patterns to reduce match count"
        )
    
    # Pattern-specific analysis
    for result in search_results.get("search_results", []):
        pattern = result.get("pattern", "")
        matches = result.get("matches", [])
        
        if len(matches) > 100:
            analysis["recommendations"].append(
                f"Pattern '{pattern}' has {len(matches)} matches - consider making it more specific"
            )
    
    return json.dumps(analysis)


# Create FunctionTool instances for Google ADK
search_patterns_tool = FunctionTool(func=search_medical_patterns)
validate_patterns_tool = FunctionTool(func=validate_and_fix_patterns)
search_with_recovery_tool = FunctionTool(func=search_with_error_recovery)
analyze_performance_tool = FunctionTool(func=analyze_search_performance)

# Export all tools
GREP_TOOLS = [
    search_patterns_tool,
    validate_patterns_tool,
    search_with_recovery_tool,
    analyze_performance_tool
]