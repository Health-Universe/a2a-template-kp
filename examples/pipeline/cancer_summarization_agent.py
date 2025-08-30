#!/usr/bin/env python3
"""
Simple Cancer Summarization Agent - Fixed-sequence pipeline for cancer document analysis.
Executes agents in a predefined order with special checker loop logic.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from base import A2AAgent
from google.adk.tools import FunctionTool
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Global agent instance for tool access
_agent_instance = None

async def run_cancer_pipeline(document: str) -> str:
    """
    Execute the complete cancer document analysis pipeline.
    
    This tool processes medical documents through a 12-step pipeline
    specifically designed for cancer-related information extraction.
    
    Args:
        document: The medical document text to analyze
        
    Returns:
        Comprehensive cancer-focused narrative
    """
    global _agent_instance
    if _agent_instance:
        return await _agent_instance.execute_pipeline(document)
    return "Error: Agent not initialized"


class CancerSummarizationAgent(A2AAgent):
    """Simple orchestrator for cancer document analysis using fixed pipeline sequence."""
    
    def __init__(self):
        """Initialize the agent and set global instance for tool access."""
        super().__init__()
        global _agent_instance
        _agent_instance = self
    
    def get_agent_name(self) -> str:
        """Return the agent's name."""
        return "Cancer Summarization Pipeline"
    
    def get_agent_description(self) -> str:
        """Return agent description."""
        return (
            "Fixed-sequence orchestrator specifically designed for cancer document analysis. "
            "Executes a comprehensive pipeline: keyword → grep → chunk → temporal → encounter → "
            "reconciliation → summary → timeline → checker (with retry) → unified extraction → "
            "verification → narrative synthesis."
        )
    
    def get_agent_version(self) -> str:
        """Return the agent version."""
        return "1.0.0"
    
    def get_agent_skills(self) -> list[AgentSkill]:
        """Return the agent's skills."""
        return [
            AgentSkill(
                id="cancer_analysis",
                name="Cancer Document Analysis",
                description="Comprehensive analysis of cancer-related medical documents",
                tags=["cancer", "oncology", "medical", "pipeline"],
                examples=[
                    "Analyze this oncology report",
                    "Summarize cancer patient history",
                    "Extract cancer staging and treatment information"
                ]
            )
        ]
    
    def supports_streaming(self) -> bool:
        """Enable streaming for meta-orchestrator compatibility."""
        return True
    
    def get_system_instruction(self) -> str:
        """System instruction for the LLM when using tools."""
        return """You are a cancer document analysis pipeline coordinator.

When you receive a medical document, use the run_cancer_pipeline tool to process it through the complete 12-step analysis pipeline.

The pipeline will:
1. Generate cancer-specific search patterns
2. Search the document for matches
3. Extract context around matches
4. Extract temporal information
5. Group by clinical encounters
6. Reconcile conflicts
7. Extract structured summary
8. Build timeline
9. Check quality (with retries)
10. Extract all medical entities
11. Verify final data
12. Synthesize narrative

Always use the run_cancer_pipeline tool for any medical document analysis."""
    
    def get_tools(self) -> list:
        """Return the pipeline execution tool."""
        return [FunctionTool(func=run_cancer_pipeline)]
    
    async def execute_pipeline(self, message: str) -> str:
        """
        Execute the cancer document analysis pipeline.
        
        This method is called by the tool function to do the actual work.
        
        Args:
            message: The input medical document text
            
        Returns:
            Final cancer-focused narrative summary
        """
        start_time = time.time()
        self.logger.info("="*80)
        self.logger.info("🔬 CANCER SUMMARIZATION PIPELINE STARTING")
        self.logger.info("="*80)
        
        pipeline_results = {}
        
        try:
            # ========== STEP 1: KEYWORD GENERATION ==========
            self.logger.info("\n📍 STEP 1: Keyword Generation (Cancer Focus)")
            self.logger.info("-"*40)
            
            keyword_message = f"""Generate comprehensive regex patterns for finding cancer-related information in this medical document:

{message[:1500]}

Focus on:
- Cancer types, stages, grades
- Oncology treatments (chemotherapy, radiation, surgery)
- Tumor markers and genetic mutations
- Metastasis and progression
- Response to treatment
- Side effects and complications

Generate patterns for all cancer-related medical information."""
            
            self.logger.info(f"📤 Sending to keyword agent: {len(keyword_message)} characters")
            keyword_response = await self.call_other_agent("keyword", keyword_message)
            self.logger.info(f"📥 Received patterns from keyword agent")
            pipeline_results["keyword"] = keyword_response
            
            # Extract patterns (simplified parsing)
            patterns = self._extract_patterns(keyword_response)
            self.logger.info(f"   Extracted {len(patterns)} cancer-related patterns")
            
            # ========== STEP 2: GREP SEARCH ==========
            self.logger.info("\n📍 STEP 2: Pattern Search (Grep)")
            self.logger.info("-"*40)
            
            grep_message = json.dumps({
                "patterns": patterns,
                "document_content": message,
                "case_sensitive": False,
                "focus": "cancer-related matches"
            })
            
            self.logger.info(f"📤 Searching with {len(patterns)} patterns")
            grep_response = await self.call_other_agent("grep", grep_message)
            self.logger.info(f"📥 Found matches from grep agent")
            pipeline_results["grep"] = grep_response
            
            # Parse matches
            matches = self._parse_grep_results(grep_response)
            self.logger.info(f"   Found {len(matches)} cancer-related matches")
            
            # ========== STEP 3: CHUNK EXTRACTION ==========
            self.logger.info("\n📍 STEP 3: Context Extraction (Chunking)")
            self.logger.info("-"*40)
            
            # Deduplicate and limit chunks
            unique_chunks = self._deduplicate_matches(matches)
            chunks_to_process = unique_chunks[:10]  # Process up to 10 unique chunks
            
            chunks = []
            for i, match in enumerate(chunks_to_process, 1):
                chunk_message = json.dumps({
                    "match_info": match,
                    "lines_before": 3,
                    "lines_after": 3,
                    "focus": "cancer context"
                })
                
                self.logger.info(f"   [{i}/{len(chunks_to_process)}] Extracting chunk...")
                chunk_response = await self.call_other_agent("chunk", chunk_message)
                chunks.append(chunk_response)
            
            combined_chunks = "\n\n".join(chunks)
            self.logger.info(f"   Extracted {len(chunks)} context chunks")
            pipeline_results["chunks"] = combined_chunks
            
            # ========== STEP 4: TEMPORAL TAGGING ==========
            self.logger.info("\n📍 STEP 4: Temporal Information Extraction")
            self.logger.info("-"*40)
            
            temporal_message = f"""Extract all temporal information from this cancer patient data:

{combined_chunks}

Focus on:
- Diagnosis dates
- Treatment start/end dates
- Follow-up appointments
- Progression timelines
- Remission periods"""
            
            self.logger.info("📤 Extracting temporal information...")
            temporal_response = await self.call_other_agent("temporal_tagging", temporal_message)
            self.logger.info("📥 Temporal data extracted")
            pipeline_results["temporal"] = temporal_response
            
            # ========== STEP 5: ENCOUNTER GROUPING ==========
            self.logger.info("\n📍 STEP 5: Encounter Grouping")
            self.logger.info("-"*40)
            
            encounter_message = json.dumps({
                "temporal_data": temporal_response,
                "clinical_content": combined_chunks,
                "focus": "oncology visits and treatments"
            })
            
            self.logger.info("📤 Grouping by clinical encounters...")
            encounter_response = await self.call_other_agent("encounter_grouping", encounter_message)
            self.logger.info("📥 Encounters grouped")
            pipeline_results["encounters"] = encounter_response
            
            # ========== STEP 6: RECONCILIATION ==========
            self.logger.info("\n📍 STEP 6: Data Reconciliation")
            self.logger.info("-"*40)
            
            reconciliation_message = json.dumps({
                "encounter_groups": encounter_response,
                "temporal_data": temporal_response,
                "resolve": "conflicting dates and information"
            })
            
            self.logger.info("📤 Reconciling conflicts...")
            reconciled_data = await self.call_other_agent("reconciliation", reconciliation_message)
            self.logger.info("📥 Data reconciled")
            pipeline_results["reconciled"] = reconciled_data
            
            # ========== STEP 7: SUMMARY EXTRACTION ==========
            self.logger.info("\n📍 STEP 7: Summary Extraction")
            self.logger.info("-"*40)
            
            summary_message = json.dumps({
                "reconciled_data": reconciled_data,
                "focus": "cancer diagnosis, staging, treatment, outcomes",
                "extract": "structured medical summary"
            })
            
            self.logger.info("📤 Extracting structured summary...")
            summary = await self.call_other_agent("summary_extractor", summary_message)
            self.logger.info("📥 Summary extracted")
            pipeline_results["summary"] = summary
            
            # ========== STEP 8: TIMELINE BUILDING ==========
            self.logger.info("\n📍 STEP 8: Timeline Construction")
            self.logger.info("-"*40)
            
            timeline_message = json.dumps({
                "summary": summary,
                "temporal_data": temporal_response,
                "encounters": encounter_response,
                "build": "chronological cancer journey"
            })
            
            self.logger.info("📤 Building timeline...")
            timeline = await self.call_other_agent("timeline_builder", timeline_message)
            self.logger.info("📥 Timeline built")
            pipeline_results["timeline"] = timeline
            
            # ========== STEP 9: CHECKER LOOP (Max 3 attempts) ==========
            self.logger.info("\n📍 STEP 9: Quality Check with Retry Loop")
            self.logger.info("-"*40)
            
            checked_summary = summary
            checker_attempts = 0
            max_attempts = 3
            
            for attempt in range(1, max_attempts + 1):
                checker_attempts = attempt
                self.logger.info(f"   Attempt {attempt}/{max_attempts}")
                
                checker_message = json.dumps({
                    "summary": checked_summary,
                    "timeline": timeline,
                    "original_data": reconciled_data,
                    "check_for": "accuracy, completeness, consistency"
                })
                
                self.logger.info("   📤 Sending to checker...")
                checker_response = await self.call_other_agent("checker", checker_message)
                self.logger.info("   📥 Checker response received")
                
                # Check if issues were found
                checker_lower = checker_response.lower()
                has_issues = any(word in checker_lower for word in 
                               ["issue", "error", "incorrect", "missing", "fix", "problem"])
                
                if not has_issues:
                    self.logger.info("   ✅ Checker approved - no issues found")
                    break
                else:
                    self.logger.info(f"   ⚠️ Issues found - sending back to summary extractor")
                    
                    if attempt < max_attempts:
                        # Send back to summary extractor with feedback
                        fix_message = json.dumps({
                            "original_data": reconciled_data,
                            "checker_feedback": checker_response,
                            "instruction": "Please fix the issues identified by the checker",
                            "attempt": attempt
                        })
                        
                        self.logger.info("   📤 Requesting summary fixes...")
                        checked_summary = await self.call_other_agent("summary_extractor", fix_message)
                        self.logger.info("   📥 Updated summary received")
                        pipeline_results["summary"] = checked_summary
                    else:
                        self.logger.info("   ⚠️ Max attempts reached - proceeding with current summary")
            
            self.logger.info(f"   Checker loop completed after {checker_attempts} attempt(s)")
            
            # ========== STEP 10: UNIFIED EXTRACTION ==========
            self.logger.info("\n📍 STEP 10: Unified Medical Entity Extraction")
            self.logger.info("-"*40)
            
            unified_message = json.dumps({
                "summary": checked_summary,
                "timeline": timeline,
                "reconciled_data": reconciled_data,
                "extract_all": "medications, procedures, diagnoses, labs, imaging"
            })
            
            self.logger.info("📤 Extracting all medical entities...")
            unified_data = await self.call_other_agent("unified_extractor", unified_message)
            self.logger.info("📥 Entities extracted")
            pipeline_results["unified"] = unified_data
            
            # ========== STEP 11: UNIFIED VERIFICATION ==========
            self.logger.info("\n📍 STEP 11: Final Verification")
            self.logger.info("-"*40)
            
            verify_message = json.dumps({
                "extracted_data": unified_data,
                "original_summary": checked_summary,
                "timeline": timeline,
                "verify": "completeness, accuracy, medical validity"
            })
            
            self.logger.info("📤 Performing final verification...")
            verified_data = await self.call_other_agent("unified_verifier", verify_message)
            self.logger.info("📥 Verification complete")
            pipeline_results["verified"] = verified_data
            
            # ========== STEP 12: NARRATIVE SYNTHESIS ==========
            self.logger.info("\n📍 STEP 12: Final Narrative Synthesis")
            self.logger.info("-"*40)
            
            narrative_message = json.dumps({
                "verified_data": verified_data,
                "summary": checked_summary,
                "timeline": timeline,
                "synthesize": "comprehensive cancer patient narrative",
                "focus": "oncology journey from diagnosis to current status"
            })
            
            self.logger.info("📤 Synthesizing final narrative...")
            final_narrative = await self.call_other_agent("narrative_synthesis", narrative_message)
            self.logger.info("📥 Narrative complete")
            
            # ========== FINAL SUMMARY ==========
            elapsed_time = time.time() - start_time
            self.logger.info("\n" + "="*80)
            self.logger.info("✅ PIPELINE COMPLETE")
            self.logger.info(f"⏱️ Total time: {elapsed_time:.2f} seconds")
            self.logger.info(f"📊 Checker attempts: {checker_attempts}")
            self.logger.info(f"📄 Final narrative length: {len(final_narrative)} characters")
            self.logger.info("="*80)
            
            return final_narrative
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline error at step: {e}")
            self.logger.error(f"   Completed steps: {list(pipeline_results.keys())}")
            
            # Return partial results if available
            if "narrative_synthesis" in pipeline_results:
                return pipeline_results["narrative_synthesis"]
            elif "verified" in pipeline_results:
                return f"Partial results (verified data):\n\n{pipeline_results['verified']}"
            elif "summary" in pipeline_results:
                return f"Partial results (summary):\n\n{pipeline_results['summary']}"
            else:
                return f"Pipeline failed: {str(e)}\n\nCompleted steps: {list(pipeline_results.keys())}"
    
    def _extract_patterns(self, keyword_response: str) -> list:
        """Extract patterns from keyword agent response."""
        patterns = []
        
        # Simple extraction - look for patterns in quotes or after colons
        lines = keyword_response.split('\n')
        for line in lines:
            if '"' in line:
                # Extract text between quotes
                import re
                quoted = re.findall(r'"([^"]*)"', line)
                patterns.extend(quoted)
            elif ':' in line and not line.strip().endswith(':'):
                # Extract text after colon
                pattern = line.split(':', 1)[1].strip()
                if pattern and len(pattern) > 2:
                    patterns.append(pattern)
        
        # Add default cancer patterns if too few found
        if len(patterns) < 5:
            patterns.extend([
                r"cancer|carcinoma|tumor|malignant",
                r"chemotherapy|radiation|oncolog",
                r"stage [IVX]+|T[0-4]N[0-3]M[0-1]",
                r"metasta|progression|remission",
                r"grade [1-4]|poorly differentiated|well differentiated"
            ])
        
        return patterns[:20]  # Limit to 20 patterns
    
    def _parse_grep_results(self, grep_response: str) -> list:
        """Parse grep agent results."""
        matches = []
        
        try:
            # Try parsing as JSON first
            data = json.loads(grep_response)
            if isinstance(data, list):
                matches = data
            elif isinstance(data, dict) and "matches" in data:
                matches = data["matches"]
        except:
            # Fallback: create simple matches from response
            lines = grep_response.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    matches.append({
                        "line_number": i + 1,
                        "text": line.strip(),
                        "pattern": "cancer-related"
                    })
        
        return matches
    
    def _deduplicate_matches(self, matches: list) -> list:
        """Deduplicate matches by line number."""
        unique = {}
        for match in matches:
            line_num = match.get("line_number", len(unique))
            if line_num not in unique:
                unique[line_num] = match
        
        return list(unique.values())
    
    async def process_message(self, message: str) -> str:
        """
        Process message - required by base class but not used.
        
        The actual processing happens via the tool function.
        """
        # This won't be called when tools are provided
        return "Processing through cancer pipeline..."


# Module-level app creation (required for deployment)
agent = CancerSummarizationAgent()
agent_card = agent.create_agent_card()
task_store = InMemoryTaskStore()
request_handler = DefaultRequestHandler(
    agent_executor=agent,
    task_store=task_store
)

app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler
).build()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8009))
    print(f"🔬 Starting {agent.get_agent_name()}")
    print(f"📍 Server: http://localhost:{port}")
    print(f"📋 Agent Card: http://localhost:{port}/.well-known/agent-card.json")
    print(f"\n📊 Pipeline sequence:")
    print("   1. Keyword (cancer patterns)")
    print("   2. Grep (search)")
    print("   3. Chunk (context)")
    print("   4. Temporal (dates)")
    print("   5. Encounter (grouping)")
    print("   6. Reconciliation")
    print("   7. Summary Extraction")
    print("   8. Timeline Building")
    print("   9. Checker (with retry loop)")
    print("   10. Unified Extraction")
    print("   11. Unified Verification")
    print("   12. Narrative Synthesis")
    print("\n" + "="*60)
    uvicorn.run(app, host="0.0.0.0", port=port)