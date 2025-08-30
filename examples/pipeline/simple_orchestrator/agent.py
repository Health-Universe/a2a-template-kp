"""
Simple Pipeline Orchestrator Agent
Runs a fixed document-analysis pipeline: keyword → grep → chunk → summarize
No branching logic - purely sequential for traceability and simplicity.
"""

import json
import os
import time
import logging
import ast
from typing import List, Dict, Any, Optional

from a2a.types import AgentSkill, Message, DataPart, TextPart, TaskState
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.utils import new_agent_text_message
from base import A2AAgent
from utils.logging import get_logger
from utils.message_utils import create_data_part, create_agent_message

logger = get_logger(__name__)


class SimpleOrchestratorAgent(A2AAgent):
    """
    Dead-simple orchestrator that walks a fixed pipeline:
      keyword -> grep -> chunk -> summarize
    Uses DataPart for structured JSON payloads (spec-preferred).
    """

    # Configuration knobs
    MAX_PATTERNS: int = 20
    MAX_MATCHES_FOR_CHUNKS: int = 5
    LINES_BEFORE: int = 2
    LINES_AFTER: int = 2
    CALL_TIMEOUT_SEC: float = float(os.getenv("ORCH_AGENT_TIMEOUT", "30"))

    def __init__(
        self,
        keyword_agent: Optional[str] = None,
        grep_agent: Optional[str] = None,
        chunk_agent: Optional[str] = None,
        summarize_agent: Optional[str] = None,
    ):
        """Initialize with target agent names."""
        super().__init__()
        
        # Target agents (resolved via registry or direct URL)
        self.keyword_agent = keyword_agent or "keyword"
        self.grep_agent = grep_agent or "grep"
        self.chunk_agent = chunk_agent or "chunk"
        self.summarize_agent = summarize_agent or "summarize"

    # --- A2A Metadata ---
    def get_agent_name(self) -> str:
        return "Cancer Summarization - Simple Orchestrator"

    def get_agent_description(self) -> str:
        return (
            "Runs a fixed document-analysis pipeline (keyword → grep → chunk → summarize). "
            "No branching or tools selection logic; purely sequential for traceability."
        )
    
    def get_agent_version(self) -> str:
        return "2.0.0"  # Template-based version

    def get_agent_skills(self) -> List[AgentSkill]:
        return [
            AgentSkill(
                id="simple_pipeline",
                name="Simple Pipeline Execution",
                description="Execute keyword → grep → chunk → summarize in order.",
                tags=["pipeline", "sequential", "orchestrator"],
                inputModes=["text/plain"],
                outputModes=["text/markdown"],
            )
        ]

    def supports_streaming(self) -> bool:
        return True  # Orchestrator implements execute() with streaming updates

    def get_system_instruction(self) -> str:
        return (
            "You are a medical document analysis pipeline coordinator. "
            "Execute the fixed pipeline sequence and return structured results."
        )

    # --- Streaming Execute for Meta Orchestrator ---
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute with streaming updates for meta orchestrator.
        Emits incremental status updates between pipeline steps.
        """
        task = context.current_task
        if not task:
            # Fallback to parent execute if no task
            await super().execute(context, event_queue)
            return
            
        updater = TaskUpdater(event_queue, task.id, getattr(task, 'context_id', task.id))
        
        try:
            # Extract message from context
            message = self._extract_message_text(context)
            
            # Send initial working status
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("🚀 Starting cancer summarization pipeline...")
            )
            
            # Parse input
            t0 = time.time()
            document = self._extract_document(message)
            
            # --- STEP 1: KEYWORDS ---
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("📝 STEP 1: Generating keyword patterns...")
            )
            patterns = await self._step_keywords(document)
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"✓ Generated {len(patterns)} keyword patterns")
            )
            
            # --- STEP 2: GREP ---
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("🔍 STEP 2: Searching document with patterns...")
            )
            matches = await self._step_grep(patterns, document)
            unique_matches = self._deduplicate_matches(matches)
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"✓ Found {len(unique_matches)} unique matches")
            )
            
            # --- STEP 3: CHUNKS ---
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("📄 STEP 3: Extracting text chunks...")
            )
            chunks = await self._step_chunk(unique_matches[:self.MAX_MATCHES_FOR_CHUNKS], document)
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"✓ Extracted {len(chunks)} text chunks")
            )
            
            # --- STEP 4: SUMMARIZE ---
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("📊 STEP 4: Generating summary...")
            )
            summary = await self._step_summarize(chunks, len(matches))
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("✓ Summary generation complete")
            )
            
            # Format final response
            elapsed = time.time() - t0
            final_text = self._format_final_result(
                summary, patterns, unique_matches, chunks, elapsed
            )
            
            # Send final result as working status, then complete
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(final_text)
            )
            await updater.complete()
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"❌ Pipeline failed: {str(e)}")
            )
            raise

    # --- Core Pipeline Logic ---
    async def process_message(self, message: str) -> str:
        """
        Execute the fixed pipeline on the input document.
        This is the main entry point called by the A2A framework.
        """
        t0 = time.time()
        self.logger.info("🚀 Starting simple pipeline")

        # Parse input if it's JSON
        document = self._extract_document(message)

        # --- STEP 1: KEYWORDS ---
        self.logger.info("STEP 1: Generating keyword patterns")
        patterns = await self._step_keywords(document)
        self.logger.info(f"  Generated {len(patterns)} patterns")

        # --- STEP 2: GREP ---
        self.logger.info("STEP 2: Searching with patterns")
        matches = await self._step_grep(patterns, document)
        self.logger.info(f"  Found {len(matches)} matches")

        # Deduplicate by line number
        unique_matches = self._deduplicate_matches(matches)
        matches_to_chunk = unique_matches[: self.MAX_MATCHES_FOR_CHUNKS]
        self.logger.info(f"  Deduped to {len(matches_to_chunk)} unique matches for chunking")

        # --- STEP 3: CHUNK ---
        self.logger.info("STEP 3: Extracting chunks")
        chunks = await self._step_chunk(matches_to_chunk, document)
        self.logger.info(f"  Extracted {len(chunks)} chunks")

        # --- STEP 4: SUMMARIZE ---
        self.logger.info("STEP 4: Summarizing results")
        summary = await self._step_summarize(chunks, len(matches))
        self.logger.info("  Summarization complete")

        # Build final response
        dt = time.time() - t0
        self.logger.info(f"✅ Pipeline complete in {dt:.2f}s")

        return self._format_final_response(
            patterns, matches, chunks, summary, dt
        )

    # --- Pipeline Steps ---
    async def _step_keywords(self, document: str) -> List[str]:
        """Step 1: Generate keyword patterns using the keyword agent."""
        # Build request with DataPart
        preview = document[:4000]  # First 4000 chars as preview
        
        keyword_msg = self._build_message_with_data({
            "document_preview": preview,
            "focus_areas": ["temporal_events", "dated_diagnoses", "medication_changes", "procedures_with_dates", "admission_discharge_dates"]
        })
        
        # Store diagnostic info for later
        self.keyword_diagnostic = None
        
        # Call keyword agent
        try:
            response = await self.call_other_agent_message(
                self.keyword_agent, 
                keyword_msg,
                timeout=self.CALL_TIMEOUT_SEC
            )
            
            # Debug: Save keyword response to file for analysis
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = f"/tmp/keyword_response_{timestamp}.json"
            try:
                with open(debug_file, 'w') as f:
                    f.write(f"=== KEYWORD AGENT RESPONSE ===\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Response Type: {type(response)}\n")
                    f.write(f"Response Length: {len(str(response))}\n")
                    f.write(f"\n=== RAW RESPONSE ===\n")
                    f.write(str(response))
                    f.write(f"\n\n=== ATTEMPTING JSON PARSE ===\n")
                    try:
                        import json as json_module
                        parsed = json_module.loads(response)
                        f.write("JSON Parse: SUCCESS\n")
                        f.write(f"Keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'Not a dict'}\n")
                        f.write(f"\n=== PRETTY JSON ===\n")
                        f.write(json_module.dumps(parsed, indent=2))
                    except Exception as parse_error:
                        f.write(f"JSON Parse: FAILED - {parse_error}\n")
                self.logger.info(f"📝 DEBUG: Keyword response saved to {debug_file}")
            except Exception as debug_error:
                self.logger.warning(f"Could not save debug file: {debug_error}")
            
            patterns = self._extract_patterns(response)
            self.logger.info(f"📊 Extracted {len(patterns)} patterns from keyword agent response")
            
            # Log and store diagnostic info if present
            try:
                import json as json_module
                response_data = json_module.loads(response) if isinstance(response, str) else response
                if isinstance(response_data, dict) and "diagnostic_info" in response_data:
                    self.keyword_diagnostic = response_data["diagnostic_info"]
                    diag = self.keyword_diagnostic
                    self.logger.info(f"🔍 Keyword Diagnostic: API keys={diag.get('api_keys_detected', {})}, Provider={diag.get('provider_info', {})}, Source={diag.get('source', 'unknown')}")
                    if "error_message" in diag:
                        self.logger.warning(f"⚠️ Keyword LLM Error: {diag.get('error_message', 'unknown error')}")
                # Also check for llm_error from fallback
                if isinstance(response_data, dict) and "llm_error" in response_data:
                    if not self.keyword_diagnostic:
                        self.keyword_diagnostic = {}
                    self.keyword_diagnostic["llm_error"] = response_data["llm_error"]
                    self.logger.warning(f"⚠️ Keyword LLM failed: {response_data['llm_error'].get('error_message', 'unknown')}")
            except:
                pass  # Don't fail if diagnostic parsing fails
                
        except Exception as e:
            self.logger.warning(f"Keyword agent error: {e}, using fallback patterns")
            patterns = self._get_fallback_patterns()
        
        return patterns[: self.MAX_PATTERNS]

    async def _step_grep(self, patterns: List[str], document: str) -> List[Dict[str, Any]]:
        """Step 2: Search document with patterns using grep agent."""
        
        # Debug logging
        self.logger.info(f"Document preview (first 200 chars): {document[:200]!r}")
        self.logger.info(f"Searching with {len(patterns)} patterns")
        
        # Quick self-test to see how many patterns would match
        import re
        test_matches = 0
        for p in patterns[:10]:  # Test first 10 patterns
            try:
                if re.search(p, document, re.IGNORECASE):
                    test_matches += 1
                    self.logger.debug(f"Pattern '{p}' would match")
            except Exception as e:
                self.logger.debug(f"Pattern '{p}' is invalid: {e}")
        self.logger.info(f"Self-test: {test_matches}/{min(10, len(patterns))} patterns would match")
        
        grep_msg = self._build_message_with_data({
            "patterns": patterns,
            "document_content": document,
            "case_sensitive": False
        })
        
        try:
            response = await self.call_other_agent_message(
                self.grep_agent,
                grep_msg,
                timeout=self.CALL_TIMEOUT_SEC
            )
            matches = self._parse_grep_results(response)
            self.logger.info(f"Grep returned {len(matches)} matches")
        except Exception as e:
            self.logger.error(f"Grep agent error: {e}")
            matches = []
        
        return matches

    async def _step_chunk(self, matches: List[Dict[str, Any]], document: str) -> List[str]:
        """Step 3: Extract chunks around matches using chunk agent."""
        chunks = []
        
        for match in matches:
            # Ensure document is in match_info
            if "document" not in match:
                match["document"] = document
                
            chunk_msg = self._build_message_with_data({
                "match_info": match,
                "lines_before": self.LINES_BEFORE,
                "lines_after": self.LINES_AFTER
            })
            
            try:
                chunk_resp = await self.call_other_agent_message(
                    self.chunk_agent,
                    chunk_msg,
                    timeout=self.CALL_TIMEOUT_SEC
                )
                chunks.append(chunk_resp)
            except Exception as e:
                self.logger.warning(f"Chunk extraction error: {e}")
                # Try to extract manually as fallback
                chunks.append(self._extract_fallback_chunk(match, document))
        
        return chunks

    async def _step_summarize(self, chunks: List[str], total_matches: int) -> str:
        """Step 4: Summarize chunks using summarize agent."""
        # Combine chunks for summarization
        combined = "\n\n---\n\n".join(chunks[: self.MAX_MATCHES_FOR_CHUNKS])
        
        sum_msg = self._build_message_with_data({
            "chunk_content": combined,
            "chunk_metadata": {
                "source": "pipeline_analysis",
                "total_matches": total_matches,
                "chunks_extracted": len(chunks),
                "chunks_analyzed": min(len(chunks), self.MAX_MATCHES_FOR_CHUNKS)
            },
            "summary_style": "clinical"
        })
        
        try:
            summary = await self.call_other_agent_message(
                self.summarize_agent,
                sum_msg,
                timeout=self.CALL_TIMEOUT_SEC * 2  # Give more time for summarization
            )
        except Exception as e:
            self.logger.error(f"Summarize agent error: {e}")
            summary = "Summary generation failed. Please review the extracted chunks manually."
        
        return summary

    # --- Helper Methods for Part Extraction ---
    def _iter_messages(self, envelope: Any) -> List[Dict[str, Any]]:
        """Yield all message dicts from Task or Message structures."""
        messages = []
        
        # Handle string representation of dict
        if isinstance(envelope, str) and envelope.strip().startswith('{'):
            try:
                envelope = ast.literal_eval(envelope)
            except:
                return messages
        
        if not isinstance(envelope, dict):
            return messages
            
        # Direct message
        if envelope.get("kind") == "message":
            messages.append(envelope)
            return messages
            
        # Task structure
        if envelope.get("kind") == "task":
            status = envelope.get("status", {})
            msg = status.get("message")
            if isinstance(msg, dict):
                messages.append(msg)
            # Also check history for additional messages
            for h in envelope.get("history", []):
                if isinstance(h, dict) and h.get("kind") == "message":
                    messages.append(h)
        
        return messages

    def _iter_dataparts(self, message: Dict[str, Any]) -> List[Any]:
        """Extract all data from DataParts in a message."""
        data_items = []
        
        for part in message.get("parts", []):
            if isinstance(part, dict):
                if part.get("kind") == "data":
                    data = part.get("data")
                    if data is not None:
                        data_items.append(data)
                # Recovery: try to parse JSON from TextPart
                elif part.get("kind") == "text":
                    text = part.get("text", "")
                    if text.strip().startswith("{"):
                        try:
                            data_items.append(json.loads(text))
                        except:
                            pass
        
        return data_items

    def _extract_all_data(self, envelope: Any) -> Dict[str, Any]:
        """Extract and merge all data from all DataParts across all messages."""
        merged_data = {}
        all_matches = []
        
        for message in self._iter_messages(envelope):
            for data in self._iter_dataparts(message):
                if isinstance(data, dict):
                    # Accumulate matches from all parts
                    if "matches" in data and isinstance(data["matches"], list):
                        all_matches.extend(data["matches"])
                    # Merge other fields (last one wins for non-list fields)
                    for key, value in data.items():
                        if key == "matches":
                            continue  # Handle separately
                        merged_data[key] = value
        
        # Add accumulated matches
        if all_matches:
            merged_data["matches"] = all_matches
            merged_data["total_matches"] = len(all_matches)
        
        return merged_data

    # --- Helper Methods ---
    def _build_message_with_data(self, data: Dict[str, Any]) -> Message:
        """Build A2A Message with DataPart for structured communication."""
        # Use message_utils helper for consistent Part creation
        return create_agent_message(data, role="user")
    
    async def call_other_agent_message(self, agent_name: str, message: Message, timeout: float = 30.0) -> str:
        """
        Call another agent with a structured Message.
        Falls back to text if Message not supported.
        """
        try:
            # Try to use A2AClient with message support
            from utils.a2a_client import A2AClient
            
            # Create client
            if agent_name.startswith(('http://', 'https://')):
                client = A2AClient(agent_name)
            else:
                client = A2AClient.from_registry(agent_name)
            
            try:
                # Call with message using send_message
                result = await client.send_message(
                    message,
                    timeout_sec=timeout
                )
                
                # Parse response
                if isinstance(result, dict):
                    # Check for message response format
                    if "message" in result:
                        msg = result["message"]
                        if isinstance(msg, dict) and "parts" in msg:
                            # Extract text from parts
                            texts = []
                            for part in msg["parts"]:
                                if part.get("kind") == "text":
                                    texts.append(part.get("text", ""))
                            return "\n".join(texts)
                    # Check for direct text field
                    if "text" in result:
                        return result["text"]
                    # Return as JSON if structured
                    return json.dumps(result)
                return str(result)
            finally:
                await client.close()
                
        except Exception as e:
            # Fallback to text-based call
            self.logger.debug(f"Message call failed, falling back to text: {e}")
            
            # Convert message to text
            text_parts = []
            for part in message.parts:
                if hasattr(part, 'kind'):
                    if part.kind == "data":
                        text_parts.append(json.dumps(part.data))
                    elif part.kind == "text":
                        text_parts.append(part.text)
            
            text_payload = "\n".join(text_parts)
            return await self.call_other_agent(agent_name, text_payload, timeout)

    def _extract_message_text(self, context: RequestContext) -> str:
        """Extract message text from A2A RequestContext."""
        if not context.message or not context.message.parts:
            return ""
        
        texts = []
        for part in context.message.parts:
            # Handle discriminated union by kind
            kind = getattr(part, "kind", None)
            if kind == "text":
                text = getattr(part, "text", None)
                if text:
                    texts.append(str(text))
            elif kind == "data":
                data = getattr(part, "data", None)
                if data:
                    if isinstance(data, (dict, list)):
                        texts.append(json.dumps(data))
                    else:
                        texts.append(str(data))
        
        return "\n".join(texts) if texts else ""

    def _extract_document(self, message: str) -> str:
        """Extract document from message (might be JSON or plain text)."""
        try:
            data = json.loads(message)
            if isinstance(data, dict):
                return data.get("document", data.get("text", message))
            return message
        except:
            return message

    def _extract_patterns(self, response: Any) -> List[str]:
        """Extract patterns from keyword agent response (handles all parts)."""
        patterns = []
        source = "unknown"
        
        # Extract all data from all parts
        data = self._extract_all_data(response)
        
        if data:
            source = data.get("source", "unknown")
            
            # Check for flat patterns list
            if "patterns" in data and isinstance(data["patterns"], list):
                patterns = [p for p in data["patterns"] if isinstance(p, str)]
                self.logger.info(f"Using flat patterns list from {source}: {len(patterns)} patterns")
            
            # Extract from categories if no flat list
            if not patterns:
                for category in ["section_patterns", "clinical_patterns", "medication_patterns", 
                               "temporal_patterns", "vital_patterns", "event_patterns", "term_patterns"]:
                    if category in data:
                        for p in data[category]:
                            if isinstance(p, dict) and "pattern" in p:
                                patterns.append(p["pattern"])
                            elif isinstance(p, str):
                                patterns.append(p)
                if patterns:
                    self.logger.info(f"Extracted patterns from categories ({source}): {len(patterns)} patterns")
        
        # Deduplicate
        seen = set()
        deduped = []
        for p in patterns:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        
        if not deduped:
            self.logger.warning("No patterns extracted, using fallbacks")
            deduped = self._get_fallback_patterns()
            source = "orchestrator_fallback"
        
        self.logger.info(f"Pattern source: {source}")
        if deduped:
            self.logger.info(f"First 3 patterns: {deduped[:3]}")
        
        return deduped

    def _get_fallback_patterns(self) -> List[str]:
        """Get fallback patterns for medical documents."""
        return [
            r"(?i)diabetes",
            r"(?i)hypertension",
            r"(?i)diagnosis",
            r"(?i)treatment",
            r"\b\d+\s*(mg|ml|mcg|g|kg|lb)\b",
            r"(?i)blood\s+pressure",
            r"(?i)heart\s+rate",
            r"(?i)temperature",
            r"(?i)medication",
            r"(?i)prescribed",
            r"(?i)allergies",
            r"(?i)symptoms",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # Dates
            r"(?i)vital\s+signs",
            r"(?i)lab\s+results",
        ]

    def _parse_grep_results(self, response: Any) -> List[Dict[str, Any]]:
        """Parse grep agent response to extract ALL matches from ALL parts."""
        
        # Extract and merge all data from all parts
        data = self._extract_all_data(response)
        
        if data and "matches" in data:
            matches = data.get("matches", [])
            self.logger.info(f"Extracted {len(matches)} matches from grep response")
            return matches
        
        self.logger.warning("No matches found in grep response")
        return []

    def _deduplicate_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate matches by line number."""
        unique_by_line = {}
        for match in matches:
            line_num = match.get("line_number", 0)
            if line_num not in unique_by_line:
                unique_by_line[line_num] = match
        return list(unique_by_line.values())

    def _extract_fallback_chunk(self, match: Dict[str, Any], document: str) -> str:
        """Extract a basic chunk as fallback."""
        lines = document.splitlines()
        line_num = match.get("line_number", 1) - 1  # Convert to 0-based
        
        start = max(0, line_num - self.LINES_BEFORE)
        end = min(len(lines), line_num + self.LINES_AFTER + 1)
        
        chunk_lines = []
        for i in range(start, end):
            prefix = ">>>" if i == line_num else "   "
            chunk_lines.append(f"{prefix} {i+1:4d}: {lines[i]}")
        
        return "\n".join(chunk_lines)

    def _format_final_response(
        self,
        patterns: List[str],
        matches: List[Dict[str, Any]],
        chunks: List[str],
        summary: str,
        execution_time: float
    ) -> str:
        """Format the final pipeline response."""
        
        # Add diagnostic info if available
        diagnostic_text = ""
        if hasattr(self, 'keyword_diagnostic') and self.keyword_diagnostic:
            diag = self.keyword_diagnostic
            diagnostic_text = "\n**Keyword Agent Diagnostic:**\n"
            
            # Check API keys
            api_keys = diag.get('api_keys_detected', {})
            if api_keys:
                keys_status = ", ".join([f"{k}: {'✓' if v else '✗'}" for k, v in api_keys.items()])
                diagnostic_text += f"- API Keys: {keys_status}\n"
            
            # Provider info
            provider = diag.get('provider_info', {})
            if provider:
                diagnostic_text += f"- Provider: {provider.get('provider', 'none')}, Model: {provider.get('model', 'none')}\n"
            
            # Pattern source
            source = diag.get('source', 'unknown')
            diagnostic_text += f"- Pattern Source: {source}\n"
            
            # LLM error if present
            if 'error_message' in diag:
                diagnostic_text += f"- Error: {diag['error_message'][:200]}\n"
            elif 'llm_error' in diag:
                llm_err = diag['llm_error']
                diagnostic_text += f"- LLM Error: {llm_err.get('error_type', 'unknown')} - {llm_err.get('error_message', 'unknown')[:200]}\n"
            
            diagnostic_text += "\n"
        
        return (
            f"## Medical Document Analysis Complete\n\n"
            f"**Execution Time:** {execution_time:.2f} seconds\n\n"
            f"**Pipeline Statistics:**\n"
            f"- Patterns generated: {len(patterns)}\n"
            f"- Matches found: {len(matches)}\n"
            f"- Chunks extracted: {len(chunks)}\n"
            f"- Chunks analyzed: {min(len(chunks), self.MAX_MATCHES_FOR_CHUNKS)}\n\n"
            f"{diagnostic_text}"
            f"**Summary:**\n{summary}\n\n"
            f"---\n*Analysis performed by Simple Pipeline Orchestrator v2.0*"
        )