"""
Reconciliation tools for fact deduplication and status tagging.
Following nutrition_example.py pattern with Google ADK FunctionTool.
"""
import json
import hashlib
from typing import Dict, List, Any, Optional
from collections import defaultdict
from google.adk.tools import FunctionTool


def reconcile_encounter_group(
    encounter_group: Dict[str, Any],
    fact_registry: Optional[Dict[str, Any]] = None
) -> str:
    """
    Reconcile facts within a single encounter group.
    
    This function prepares the input for LLM reconciliation.
    The actual LLM call happens in the agent executor.
    
    Args:
        encounter_group: Dictionary containing encounter data with primary and referenced content
        fact_registry: Optional registry of previously seen facts for deduplication
        
    Returns:
        JSON string with reconciliation request
    """
    if fact_registry is None:
        fact_registry = {}
    
    # Process all content from this encounter
    all_content = []
    
    # Add primary content
    primary_content = encounter_group.get("primary_content", [])
    for content in primary_content:
        all_content.append({
            **content,
            "content_type": "primary",
            "is_carry_forward": content.get("is_carry_forward", False)
        })
    
    # Add referenced content
    referenced_content = encounter_group.get("referenced_content", [])
    for content in referenced_content:
        all_content.append({
            **content,
            "content_type": "referenced",
            "is_carry_forward": True  # Referenced content is typically carry-forward
        })
    
    # Prepare content for LLM analysis
    content_list = []
    for i, item in enumerate(all_content):
        content_list.append({
            "id": i,
            "text": item.get("text", ""),
            "content_type": item.get("content_type", "unknown"),
            "is_carry_forward": item.get("is_carry_forward", False),
            "temporal_indicators": item.get("temporal_indicators", [])
        })
    
    request = {
        "action": "reconcile_encounter_group",
        "encounter_date": encounter_group.get("encounter_date", ""),
        "encounter_type": encounter_group.get("encounter_type", ""),
        "content_items": content_list,
        "fact_registry_size": len(fact_registry),
        "instructions": f"""Analyze and reconcile these clinical facts from encounter date {encounter_group.get('encounter_date', '')}.
        Identify duplicates, carry-forward information, and assign appropriate status tags.
        
        CONTENT ITEMS:
        {json.dumps(content_list, indent=2)}
        
        For each unique fact, determine:
        
        1. STATUS (choose one):
           - "Final": Confirmed, finalized result or finding
           - "Corrected": Amended or corrected information
           - "In Process": Preliminary or pending result
           - "Ordered": Test/procedure ordered but not completed
           
        2. PROVENANCE (choose one):
           - "Primary": New information from this encounter
           - "Updated": Updated version of previous information
           - "Previously Reported": Carry-forward from earlier encounter
        
        3. DEDUPLICATION:
           - Identify items that represent the same fact
           - Keep the most complete/recent version
           - Note which items are duplicates
        
        Return a JSON list of reconciled facts:
        {{
            "reconciled_facts": [
                {{
                    "content": "the clinical fact text",
                    "status": "Final|Corrected|In Process|Ordered",
                    "provenance": "Primary|Updated|Previously Reported",
                    "confidence": 0.95,
                    "is_carry_forward": false,
                    "source_item_ids": [0, 3],  // which input items this comes from
                    "duplicate_of": null  // or ID of fact this duplicates
                }}
            ]
        }}
        
        Focus on:
        - Detecting carry-forward notes (e.g., "previously noted", "unchanged")
        - Identifying status indicators (e.g., "preliminary", "final", "amended")
        - Recognizing duplicates even with slight wording differences
        - Preserving all unique clinical information"""
    }
    
    return json.dumps(request)


def cross_encounter_reconciliation(
    reconciled_groups: List[Dict[str, Any]]
) -> str:
    """
    Perform cross-encounter reconciliation to identify facts across encounters.
    
    Args:
        reconciled_groups: List of reconciled encounter groups
        
    Returns:
        JSON string with cross-encounter analysis
    """
    # Build a map of content hashes to facts across all encounters
    content_map = defaultdict(list)
    
    for group in reconciled_groups:
        for fact in group.get("reconciled_facts", []):
            # Create content hash if not present
            content = fact.get("content", "")
            content_hash = fact.get("content_hash") or hashlib.md5(content.encode()).hexdigest()
            
            content_map[content_hash].append({
                "fact": fact,
                "encounter_date": group.get("encounter_date", ""),
                "group_id": group.get("id", "")
            })
    
    # Identify facts that appear in multiple encounters
    cross_encounter_facts = []
    for content_hash, occurrences in content_map.items():
        if len(occurrences) > 1:
            # Sort by encounter date
            occurrences.sort(key=lambda x: x["encounter_date"])
            
            cross_encounter_facts.append({
                "content_hash": content_hash,
                "content": occurrences[0]["fact"].get("content", ""),
                "occurrences": len(occurrences),
                "first_encounter": occurrences[0]["encounter_date"],
                "last_encounter": occurrences[-1]["encounter_date"],
                "encounter_dates": [occ["encounter_date"] for occ in occurrences]
            })
    
    analysis = {
        "action": "cross_encounter_reconciliation",
        "total_unique_facts": len(content_map),
        "facts_in_multiple_encounters": len(cross_encounter_facts),
        "cross_encounter_facts": cross_encounter_facts,
        "summary": {
            "total_groups": len(reconciled_groups),
            "total_facts": sum(len(g.get("reconciled_facts", [])) for g in reconciled_groups)
        }
    }
    
    return json.dumps(analysis)


def llm_reconciliation(
    content_items: List[Dict[str, Any]],
    encounter_date: str
) -> str:
    """
    Use LLM to analyze and reconcile content items.
    
    Args:
        content_items: List of content items to reconcile
        encounter_date: Date of the encounter
        
    Returns:
        JSON string with LLM reconciliation request
    """
    # Prepare content for LLM analysis
    content_list = []
    for i, item in enumerate(content_items):
        content_list.append({
            "id": i,
            "text": item.get("text", ""),
            "content_type": item.get("content_type", "unknown"),
            "is_carry_forward": item.get("is_carry_forward", False),
            "temporal_indicators": item.get("temporal_indicators", [])
        })
    
    request = {
        "action": "llm_reconciliation",
        "encounter_date": encounter_date,
        "content_items": content_list,
        "instructions": f"""Analyze and reconcile these clinical facts from encounter date {encounter_date}.
        Identify duplicates, carry-forward information, and assign appropriate status tags.
        
        CONTENT ITEMS:
        {json.dumps(content_list, indent=2)}
        
        For each unique fact, determine:
        
        1. STATUS (choose one):
           - "Final": Confirmed, finalized result or finding
           - "Corrected": Amended or corrected information
           - "In Process": Preliminary or pending result
           - "Ordered": Test/procedure ordered but not completed
           
        2. PROVENANCE (choose one):
           - "Primary": New information from this encounter
           - "Updated": Updated version of previous information
           - "Previously Reported": Carry-forward from earlier encounter
        
        3. DEDUPLICATION:
           - Identify items that represent the same fact
           - Keep the most complete/recent version
           - Note which items are duplicates
        
        Return a JSON list of reconciled facts:
        {{
            "reconciled_facts": [
                {{
                    "content": "the clinical fact text",
                    "status": "Final|Corrected|In Process|Ordered",
                    "provenance": "Primary|Updated|Previously Reported",
                    "confidence": 0.95,
                    "is_carry_forward": false,
                    "source_item_ids": [0, 3],  // which input items this comes from
                    "duplicate_of": null  // or ID of fact this duplicates
                }}
            ]
        }}
        
        Focus on:
        - Detecting carry-forward notes (e.g., "previously noted", "unchanged")
        - Identifying status indicators (e.g., "preliminary", "final", "amended")
        - Recognizing duplicates even with slight wording differences
        - Preserving all unique clinical information"""
    }
    
    return json.dumps(request)


def generate_reconciliation_summary(
    reconciled_groups: List[Dict[str, Any]]
) -> str:
    """
    Generate a summary of the reconciliation process.
    
    Args:
        reconciled_groups: List of reconciled encounter groups
        
    Returns:
        JSON string with reconciliation summary
    """
    total_facts = sum(len(g.get("reconciled_facts", [])) for g in reconciled_groups)
    total_duplicates = sum(g.get("duplicate_count", 0) for g in reconciled_groups)
    total_carry_forward = sum(g.get("carry_forward_count", 0) for g in reconciled_groups)
    
    # Aggregate status counts
    status_summary = defaultdict(int)
    for group in reconciled_groups:
        for status, count in group.get("status_summary", {}).items():
            status_summary[status] += count
    
    # Find facts that appear across multiple encounters
    fact_occurrences = defaultdict(set)
    for group in reconciled_groups:
        for fact in group.get("reconciled_facts", []):
            content_hash = fact.get("content_hash", "")
            if content_hash:
                fact_occurrences[content_hash].add(group.get("encounter_date", ""))
    
    facts_in_multiple_encounters = sum(1 for dates in fact_occurrences.values() if len(dates) > 1)
    
    summary = {
        "action": "reconciliation_summary",
        "total_encounter_groups": len(reconciled_groups),
        "total_facts_processed": total_facts,
        "duplicates_identified": total_duplicates,
        "carry_forward_notes": total_carry_forward,
        "facts_in_multiple_encounters": facts_in_multiple_encounters,
        "status_distribution": dict(status_summary),
        "encounter_dates": [g.get("encounter_date", "") for g in reconciled_groups],
        "average_facts_per_encounter": total_facts / len(reconciled_groups) if reconciled_groups else 0
    }
    
    return json.dumps(summary)


# Create FunctionTool instances for Google ADK
reconcile_group_tool = FunctionTool(func=reconcile_encounter_group)
cross_encounter_tool = FunctionTool(func=cross_encounter_reconciliation)
llm_reconciliation_tool = FunctionTool(func=llm_reconciliation)
summary_tool = FunctionTool(func=generate_reconciliation_summary)

# Export all tools
RECONCILIATION_TOOLS = [
    reconcile_group_tool,
    cross_encounter_tool,
    llm_reconciliation_tool,
    summary_tool
]
