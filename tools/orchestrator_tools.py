"""
Orchestrator tools for coordinating the medical document analysis pipeline.
Following nutrition_example.py pattern with Google ADK FunctionTool.
"""
import json
import os
import httpx
from typing import Dict, List, Any, Optional
from google.adk.tools import FunctionTool
import logging

logger = logging.getLogger(__name__)


def understand_user_request(
    user_message: str,
    available_capabilities: Optional[List[str]] = None
) -> str:
    """
    Understand and parse the user's natural language request.
    
    Args:
        user_message: The user's request in natural language
        available_capabilities: List of available agent capabilities
        
    Returns:
        JSON string with request understanding
    """
    if not available_capabilities:
        available_capabilities = [
            "pattern_generation", "document_search", "chunk_extraction", 
            "medical_summarization", "entity_extraction", "terminology_analysis"
        ]
    
    understanding_request = {
        "action": "understand_user_request",
        "user_message": user_message,
        "available_capabilities": available_capabilities,
        "instructions": """Analyze the user's request and determine:

1. Primary Intent:
   - What is the user trying to accomplish?
   - Is this a document analysis, search, or information extraction task?

2. Key Requirements:
   - What specific information are they looking for?
   - Are there particular medical concepts of interest?
   - Do they need summaries, entities, or specific data?

3. Document Type:
   - Can you infer the type of medical document?
   - Is it a clinical note, lab report, imaging, etc?

4. Focus Areas:
   - What medical aspects should we prioritize?
   - Are they interested in diagnoses, treatments, medications, etc?

5. Processing Strategy:
   - Should we search for specific patterns?
   - Do we need comprehensive analysis or targeted extraction?

Provide a structured understanding that will guide the pipeline."""
    }
    
    return json.dumps(understanding_request)


def plan_pipeline_execution(
    request_understanding: Dict[str, Any],
    document_preview: str,
    available_agents: Optional[Dict[str, Any]] = None
) -> str:
    """
    Plan the pipeline execution based on request understanding.
    
    Args:
        request_understanding: Parsed understanding of user request
        document_preview: Preview of the document to analyze
        available_agents: Information about available agents
        
    Returns:
        JSON string with execution plan
    """
    if not available_agents:
        available_agents = {
            "keyword": "Pattern generation from preview",
            "grep": "Search document for patterns",
            "chunk": "Extract context around matches",
            "summarize": "Summarize and analyze chunks"
        }
    
    planning_request = {
        "action": "plan_pipeline_execution",
        "understanding": request_understanding,
        "preview": document_preview[:500],  # First 500 chars
        "agents": available_agents,
        "instructions": """Create an execution plan for the document analysis pipeline.

Based on the request understanding and document preview, determine:

1. Pattern Generation Strategy:
   - What types of patterns should we generate?
   - Should we focus on specific medical concepts?
   - How many patterns are appropriate?

2. Search Configuration:
   - What search parameters to use?
   - Should we prioritize certain pattern types?
   - How to handle no matches or too many matches?

3. Chunk Extraction:
   - What context size is appropriate?
   - Should we merge overlapping chunks?
   - How to ensure we capture complete medical concepts?

4. Summarization Approach:
   - What summary style fits the request?
   - Which entities to extract?
   - How to score relevance?

5. Result Optimization:
   - How many results to return?
   - How to rank and prioritize findings?
   - What metadata to include?

Create a detailed plan that maximizes the chance of satisfying the user's request."""
    }
    
    return json.dumps(planning_request)




def synthesize_final_response(
    user_request: str,
    pipeline_results: Dict[str, Any],
    response_format: str = "natural"
) -> str:
    """
    Synthesize the final response from pipeline results.
    
    Args:
        user_request: Original user request
        pipeline_results: Results from all pipeline stages
        response_format: Format for response (natural, structured, clinical)
        
    Returns:
        JSON string with synthesis request
    """
    synthesis_request = {
        "action": "synthesize_final_response",
        "user_request": user_request,
        "pipeline_results": pipeline_results,
        "format": response_format,
        "instructions": f"""Create a {response_format} language response that addresses the user's request.

Based on the pipeline results:

1. Answer the User's Question:
   - Directly address what they asked for
   - Use information from the summaries
   - Be specific and accurate

2. Highlight Key Findings:
   - Most relevant medical information
   - Critical diagnoses or treatments
   - Important dates or values

3. Provide Context:
   - Explain where information was found
   - Note confidence levels
   - Mention any limitations

4. Format Appropriately:
   {"- Natural flowing text" if response_format == "natural" else ""}
   {"- Structured sections with headers" if response_format == "structured" else ""}
   {"- Clinical format with standard sections" if response_format == "clinical" else ""}

5. Include Metadata:
   - Number of relevant sections found
   - Coverage of the document
   - Processing statistics

Make the response helpful, accurate, and easy to understand."""
    }
    
    return json.dumps(synthesis_request)


def handle_pipeline_errors(
    error_info: Dict[str, Any],
    partial_results: Optional[Dict[str, Any]] = None,
    recovery_options: Optional[List[str]] = None
) -> str:
    """
    Handle errors in the pipeline execution with recovery strategies.
    
    Args:
        error_info: Information about the error
        partial_results: Any partial results available
        recovery_options: Possible recovery strategies
        
    Returns:
        JSON string with error handling plan
    """
    if not recovery_options:
        recovery_options = [
            "retry_with_defaults",
            "use_partial_results",
            "fallback_patterns",
            "simplified_analysis"
        ]
    
    error_request = {
        "action": "handle_pipeline_error",
        "error": error_info,
        "partial_results": partial_results,
        "recovery_options": recovery_options,
        "instructions": """Determine the best error recovery strategy:

1. Analyze the Error:
   - What stage failed?
   - Is it recoverable?
   - Do we have partial results?

2. Choose Recovery Strategy:
   - Can we retry with different parameters?
   - Should we use default patterns?
   - Can we proceed with partial results?
   - Should we simplify the analysis?

3. Construct Recovery Plan:
   - Specific steps to recover
   - Alternative approaches
   - Graceful degradation options

4. User Communication:
   - How to explain the issue
   - What results we can still provide
   - Suggestions for better results

Prioritize providing useful results even if incomplete."""
    }
    
    return json.dumps(error_request)


def optimize_pipeline_performance(
    execution_history: List[Dict[str, Any]],
    current_request: Dict[str, Any]
) -> str:
    """
    Optimize pipeline performance based on execution history.
    
    Args:
        execution_history: History of previous executions
        current_request: Current request details
        
    Returns:
        JSON string with optimization suggestions
    """
    optimization_request = {
        "action": "optimize_pipeline_performance",
        "history": execution_history[-5:] if execution_history else [],  # Last 5 executions
        "current_request": current_request,
        "instructions": """Analyze execution history to optimize the pipeline:

1. Pattern Optimization:
   - Which patterns yield best results?
   - How many patterns are optimal?
   - Should we adjust pattern types?

2. Search Efficiency:
   - Are we getting too many/few matches?
   - Should we adjust search parameters?
   - Can we parallelize searches?

3. Chunk Quality:
   - Are chunks capturing complete concepts?
   - Is context size appropriate?
   - Should we merge more aggressively?

4. Summary Effectiveness:
   - Are summaries addressing user needs?
   - Is entity extraction comprehensive?
   - Are relevance scores accurate?

5. Performance Metrics:
   - Where are bottlenecks?
   - Can we cache common patterns?
   - Should we adjust timeouts?

Provide specific optimizations for this request."""
    }
    
    return json.dumps(optimization_request)


# Create FunctionTool instances for Google ADK
understand_request_tool = FunctionTool(func=understand_user_request)
plan_execution_tool = FunctionTool(func=plan_pipeline_execution)
synthesize_response_tool = FunctionTool(func=synthesize_final_response)
handle_errors_tool = FunctionTool(func=handle_pipeline_errors)
optimize_performance_tool = FunctionTool(func=optimize_pipeline_performance)

# Export all tools
ORCHESTRATOR_TOOLS = [
    understand_request_tool,
    plan_execution_tool,
    synthesize_response_tool,
    handle_errors_tool,
    optimize_performance_tool
]