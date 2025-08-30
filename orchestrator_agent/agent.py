"""
Document Processing Pipeline Orchestrator
Coordinates keyword → grep → chunk → summarizer pipeline.
Health Universe compatible with streaming support.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a.types import AgentSkill, TaskState
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.utils import new_agent_text_message
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from base import A2AAgent
from utils.logging import get_logger
from utils.message_utils import create_agent_message


logger = get_logger(__name__)


class DocumentProcessingOrchestratorAgent(A2AAgent):
    """
    Orchestrator that coordinates the document processing pipeline.
    Flow: Document → Keyword → Grep → Chunk → Summarizer → Summary
    """

    def __init__(
        self,
        keyword_agent: Optional[str] = None,
        grep_agent: Optional[str] = None,
        chunk_agent: Optional[str] = None,
        summarizer_agent: Optional[str] = None,
    ):
        """Initialize with target agent names."""
        super().__init__()
        
        # Target agents (resolved via registry or direct URL)
        self.keyword_agent = keyword_agent or "keyword"
        self.grep_agent = grep_agent or "grep"
        self.chunk_agent = chunk_agent or "chunk"
        self.summarizer_agent = summarizer_agent or "summarizer"
        
        # Configuration
        self.CALL_TIMEOUT_SEC = float(os.getenv("ORCH_AGENT_TIMEOUT", "30"))
        self.MAX_PATTERNS = int(os.getenv("MAX_PATTERNS", "10"))
        self.MAX_MATCHES_FOR_CHUNKS = int(os.getenv("MAX_MATCHES_FOR_CHUNKS", "50"))
        self.LINES_BEFORE = int(os.getenv("LINES_BEFORE", "3"))
        self.LINES_AFTER = int(os.getenv("LINES_AFTER", "3"))

    # --- A2A Metadata ---
    def get_agent_name(self) -> str:
        return "Document Processing Pipeline Orchestrator"

    def get_agent_description(self) -> str:
        return (
            "Orchestrates the transformation of documents into structured summaries. "
            "Coordinates Keyword, Grep, Chunk, and Summarizer agents in a sequential pipeline "
            "to extract and analyze relevant information from documents."
        )
    
    def get_agent_version(self) -> str:
        return "1.0.0"

    def get_agent_skills(self) -> List[AgentSkill]:
        return [
            AgentSkill(
                id="document_processing_pipeline",
                name="Document Processing Pipeline",
                description="Transform documents into structured analysis summaries",
                tags=["orchestrator", "document", "analysis", "pipeline"],
                inputModes=["text/plain"],
                outputModes=["text/plain", "application/json"],
            )
        ]

    def supports_streaming(self) -> bool:
        return True  # Enable streaming for real-time progress updates

    def get_system_instruction(self) -> str:
        return (
            "You are a document processing pipeline coordinator. "
            "Your role is to transform raw documents into structured analysis "
            "by coordinating specialized processing agents in sequence."
        )

    # --- Streaming Execute ---
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute with streaming updates for real-time progress tracking.
        """
        task = context.current_task
        if not task:
            await super().execute(context, event_queue)
            return
            
        updater = TaskUpdater(event_queue, task.id, getattr(task, 'context_id', task.id))
        
        try:
            # Extract document from context
            document = self._extract_message_text(context)
            
            if not document:
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message("No document provided. Please provide a document to analyze.")
                )
                return
            
            # Send initial status
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("🔄 Starting document processing pipeline...")
            )
            
            # Step 1: Generate keywords
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("🔍 Step 1: Generating search keywords...")
            )
            
            keywords = await self._generate_keywords(document)
            
            if isinstance(keywords, dict) and keywords.get("error"):
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(f"❌ Keyword generation failed: {keywords['error']}")
                )
                return
            
            # Report keyword results
            keyword_count = len(keywords.get("keywords", [])) if isinstance(keywords, dict) else 0
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"✅ Generated {keyword_count} search keywords")
            )
            
            # Step 2: Search document
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("🔎 Step 2: Searching document with keywords...")
            )
            
            search_results = await self._search_document(keywords, document)
            
            if isinstance(search_results, dict) and search_results.get("error"):
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(f"❌ Document search failed: {search_results['error']}")
                )
                return
            
            # Report search results
            match_count = len(search_results.get("matches", [])) if isinstance(search_results, dict) else 0
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"✅ Found {match_count} matches in document")
            )
            
            # Step 3: Extract chunks
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("📄 Step 3: Extracting contextual chunks...")
            )
            
            chunks = await self._extract_chunks(search_results, document)
            
            if isinstance(chunks, dict) and chunks.get("error"):
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(f"❌ Chunk extraction failed: {chunks['error']}")
                )
                return
            
            # Report chunk results
            chunk_count = len(chunks.get("chunks", [])) if isinstance(chunks, dict) else 0
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"✅ Extracted {chunk_count} contextual chunks")
            )
            
            # Step 4: Generate summary
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("📊 Step 4: Generating analysis summary...")
            )
            
            summary = await self._generate_summary(chunks)
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("✅ Analysis summary completed")
            )
            
            # Final result
            final_message = self._format_final_result(
                document, keywords, search_results, chunks, summary
            )
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(final_message)
            )
            await updater.complete()
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"❌ Pipeline failed: {str(e)}")
            )
            raise

    # --- Core Pipeline Logic ---
    async def process_message(self, message: str) -> str:
        """
        Execute the document processing pipeline.
        """
        start_time = time.time()
        
        logger.info("🔄 Starting document processing pipeline")
        
        # Step 1: Generate keywords
        logger.info("Step 1: Generating keywords")
        keywords = await self._generate_keywords(message)
        
        if isinstance(keywords, dict) and keywords.get("error"):
            return f"Keyword generation failed: {keywords['error']}\n\nPlease check your document format and try again."
        
        keyword_count = len(keywords.get("keywords", [])) if isinstance(keywords, dict) else 0
        logger.info(f"  Generated {keyword_count} keywords")
        
        # Step 2: Search document
        logger.info("Step 2: Searching document")
        search_results = await self._search_document(keywords, message)
        
        if isinstance(search_results, dict) and search_results.get("error"):
            return f"Document search failed: {search_results['error']}\n\nPlease check your document and try again."
        
        match_count = len(search_results.get("matches", [])) if isinstance(search_results, dict) else 0
        logger.info(f"  Found {match_count} matches")
        
        # Step 3: Extract chunks
        logger.info("Step 3: Extracting chunks")
        chunks = await self._extract_chunks(search_results, message)
        
        if isinstance(chunks, dict) and chunks.get("error"):
            return f"Chunk extraction failed: {chunks['error']}\n\nPlease check your document and try again."
        
        chunk_count = len(chunks.get("chunks", [])) if isinstance(chunks, dict) else 0
        logger.info(f"  Extracted {chunk_count} chunks")
        
        # Step 4: Generate summary
        logger.info("Step 4: Generating summary")
        summary = await self._generate_summary(chunks)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Pipeline complete in {elapsed:.2f}s")
        
        # Return the summary directly
        return summary

    # --- Pipeline Steps ---
    async def _generate_keywords(self, document: str) -> Dict[str, Any]:
        """Step 1: Send document to keyword agent for keyword generation."""
        try:
            response = await self.call_other_agent(
                self.keyword_agent,
                document,
                timeout=self.CALL_TIMEOUT_SEC
            )
            
            # Parse response - expect structured data with keywords
            if isinstance(response, dict):
                return response
            elif isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
                
                # If not JSON, assume it's a text response with keywords
                return {"keywords": [response.strip()]}
            
            return {"error": f"Unexpected response format: {type(response)}"}
                
        except Exception as e:
            logger.error(f"Keyword agent error: {e}")
            return {"error": str(e)}

    async def _search_document(self, keywords: Dict[str, Any], document: str) -> Dict[str, Any]:
        """Step 2: Send keywords and document to grep agent."""
        try:
            # Prepare data for grep agent - combine keywords and document
            grep_data = {
                "keywords": keywords,
                "document": document
            }
            
            response = await self.call_other_agent_with_data(
                self.grep_agent,
                grep_data,
                timeout=self.CALL_TIMEOUT_SEC
            )
            
            # Parse response - expect structured data with matches
            if isinstance(response, dict):
                return response
            elif isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
                
                # Fallback - treat as text response
                return {"matches": [{"text": response}]}
            
            return {"error": f"Unexpected response format: {type(response)}"}
                
        except Exception as e:
            logger.error(f"Grep agent error: {e}")
            return {"error": str(e)}

    async def _extract_chunks(self, search_results: Dict[str, Any], document: str) -> Dict[str, Any]:
        """Step 3: Send search results and document to chunk agent."""
        try:
            # Prepare data for chunk agent
            chunk_data = {
                "search_results": search_results,
                "document": document
            }
            
            response = await self.call_other_agent_with_data(
                self.chunk_agent,
                chunk_data,
                timeout=self.CALL_TIMEOUT_SEC
            )
            
            # Parse response - expect structured data with chunks
            if isinstance(response, dict):
                return response
            elif isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
                
                # Fallback - treat as single chunk
                return {"chunks": [{"content": response}]}
            
            return {"error": f"Unexpected response format: {type(response)}"}
                
        except Exception as e:
            logger.error(f"Chunk agent error: {e}")
            return {"error": str(e)}

    async def _generate_summary(self, chunks: Dict[str, Any]) -> str:
        """Step 4: Send chunks to summarizer agent."""
        try:
            response = await self.call_other_agent_with_data(
                self.summarizer_agent,
                chunks,
                timeout=self.CALL_TIMEOUT_SEC
            )
            
            # Response should be text summary
            if isinstance(response, str):
                return response
            elif isinstance(response, dict):
                # Look for summary field
                if "summary" in response:
                    return str(response["summary"])
                else:
                    return json.dumps(response, indent=2)
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Summarizer agent error: {e}")
            return self._create_fallback_summary(chunks)

    # --- Helper Methods ---
    def _extract_message_text(self, context: RequestContext) -> str:
        """Extract message text from A2A RequestContext."""
        if not context.message or not context.message.parts:
            return ""
        
        texts = []
        for part in context.message.parts:
            kind = getattr(part, "kind", None)
            if kind == "text":
                text = getattr(part, "text", None)
                if text:
                    texts.append(str(text))
            elif kind == "data":
                data = getattr(part, "data", None)
                if data:
                    if isinstance(data, str):
                        texts.append(data)
                    elif isinstance(data, dict) and "document" in data:
                        texts.append(data["document"])
        
        return "\n".join(texts) if texts else ""

    def _format_final_result(
        self, 
        original_document: str, 
        keywords: Dict[str, Any],
        search_results: Dict[str, Any],
        chunks: Dict[str, Any],
        summary: str
    ) -> str:
        """Format the complete pipeline result."""
        
        # For the orchestrator, we primarily return the summary
        result = summary
        
        # Add pipeline metadata at the end
        result += "\n\n---\n"
        result += "*This analysis was created by processing your document through "
        result += "our Document Processing Pipeline. The pipeline extracted key information "
        result += "and provided structured analysis of the content.*"
        
        # Add processing stats
        keyword_count = len(keywords.get("keywords", [])) if isinstance(keywords, dict) else 0
        match_count = len(search_results.get("matches", [])) if isinstance(search_results, dict) else 0
        chunk_count = len(chunks.get("chunks", [])) if isinstance(chunks, dict) else 0
        
        result += f"\n\n**Processing Stats:**"
        result += f"\n- Keywords Generated: {keyword_count}"
        result += f"\n- Document Matches: {match_count}"
        result += f"\n- Content Chunks: {chunk_count}"
        
        return result

    def _create_fallback_summary(self, chunks: Dict[str, Any]) -> str:
        """Create basic summary if summarizer agent fails."""
        chunk_list = chunks.get("chunks", [])
        
        lines = ["# Document Analysis Summary\n"]
        lines.append("We processed your document and extracted relevant information:\n")
        
        if chunk_list:
            lines.append("## Key Content Found\n")
            for i, chunk in enumerate(chunk_list[:5], 1):  # Limit to first 5 chunks
                content = chunk.get("content", str(chunk))
                # Truncate long content
                if len(content) > 200:
                    content = content[:200] + "..."
                lines.append(f"{i}. {content}\n")
        else:
            lines.append("No significant content chunks were extracted from the document.\n")
        
        lines.append("## Next Steps\n")
        lines.append("This is a basic summary. For more detailed analysis, please:")
        lines.append("- Check if your document contains the expected content")
        lines.append("- Verify the document format is supported")
        lines.append("- Try with a different document or contact support\n")
        
        lines.append("*Note: This is a fallback summary due to processing limitations.*")
        
        return "\n".join(lines)


# --- Module-level app creation for Health Universe deployment ---
agent = DocumentProcessingOrchestratorAgent()
agent_card = agent.create_agent_card()
task_store = InMemoryTaskStore()
request_handler = DefaultRequestHandler(
    agent_executor=agent,
    task_store=task_store
)

app = A2AStarletteApplication(
    agent_card=agent_card,  # A2A Spec: MUST make AgentCard available
    http_handler=request_handler  # Handles RPC methods
).build()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8006))
    print(f"🚀 Starting {agent.get_agent_name()}")
    print(f"📍 Available at: http://localhost:{port}")
    print(f"🔍 Agent Card: http://localhost:{port}/.well-known/agentcard.json")
    uvicorn.run(app, host="0.0.0.0", port=port)