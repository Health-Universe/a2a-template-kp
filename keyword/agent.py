"""
Keyword Generation Agent
Generates relevant search keywords and patterns from documents.
Health Universe compatible with A2A compliance.
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a.types import AgentSkill
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from base import A2AAgent
from utils.logging import get_logger


logger = get_logger(__name__)


class KeywordGeneratorAgent(A2AAgent):
    """
    LLM-powered agent that generates search keywords and regex patterns from documents.
    Returns structured data with keywords for downstream processing.
    """

    # --- A2A Metadata ---
    def get_agent_name(self) -> str:
        return "Keyword Generator"

    def get_agent_description(self) -> str:
        return (
            "Analyzes documents to generate relevant search keywords and patterns. "
            "Extracts key medical terms, concepts, and entities that can be used "
            "for subsequent document search and analysis operations."
        )

    def get_agent_version(self) -> str:
        return "1.0.0"

    def get_agent_skills(self) -> List[AgentSkill]:
        return [
            AgentSkill(
                id="generate_keywords",
                name="Generate Search Keywords",
                description="Extract and generate relevant keywords and search patterns from documents",
                tags=["keyword", "extraction", "search", "nlp"],
                inputModes=["text/plain"],
                outputModes=["application/json"],
            )
        ]

    def supports_streaming(self) -> bool:
        return True  # Required by Health Universe platform

    def get_system_instruction(self) -> str:
        return (
            "You are a keyword extraction specialist. Your role is to analyze documents "
            "and generate relevant search keywords and patterns. Focus on extracting: "
            "1) Important medical terms and conditions "
            "2) Key concepts and entities "
            "3) Relevant search patterns "
            "4) Alternative terms and synonyms "
            "Be comprehensive but focused on terms that would be useful for document search."
        )

    # --- Core Processing ---
    async def process_message(self, message: str) -> Union[Dict[str, Any], str]:
        """
        Analyze document and generate keywords.
        Returns dict with keywords list (will be wrapped in DataPart).
        """
        try:
            # Extract document content
            document_content = self._extract_document_content(message)
            
            # Generate keywords using hybrid approach (LLM + rule-based)
            keywords = await self._generate_keywords(document_content)
            
            # Ensure we have the expected structure
            if not isinstance(keywords, dict):
                keywords = {"keywords": keywords if isinstance(keywords, list) else []}
            
            # Validate and clean results
            keywords = self._validate_keywords(keywords)
            
            # Add metadata
            keywords["metadata"] = {
                "generator": "keyword_agent_v1",
                "document_length": len(document_content),
                "keyword_count": len(keywords.get("keywords", [])),
                "pattern_count": len(keywords.get("patterns", []))
            }
            
            # Return dict directly - base agent will wrap in DataPart
            return keywords
            
        except Exception as e:
            logger.error(f"Error generating keywords: {e}")
            return {
                "error": str(e),
                "keywords": [],
                "patterns": []
            }

    def _extract_document_content(self, message: str) -> str:
        """Extract document content from message."""
        try:
            data = json.loads(message)
            if isinstance(data, dict):
                return data.get("document", data.get("content", message))
            return message
        except:
            return message

    async def _generate_keywords(self, document_content: str) -> Dict[str, Any]:
        """Generate keywords using LLM with fallback to rule-based extraction."""
        
        # Try LLM-based keyword extraction first
        try:
            from utils.llm_utils import generate_json
            
            prompt = f"""Analyze the following document and extract relevant keywords for search operations.

Document:
{document_content[:2000]}  # Limit to first 2000 chars for efficiency

Generate keywords in these categories:
1. Medical terms and conditions
2. Key concepts and entities
3. Important phrases
4. Alternative terms and synonyms

Focus on terms that would be useful for finding similar content in documents.
Provide 10-20 of the most relevant keywords."""
            
            # Define schema for structured output
            schema = {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "description": "List of relevant keywords and phrases",
                        "items": {
                            "type": "string",
                            "description": "A keyword or key phrase"
                        }
                    },
                    "patterns": {
                        "type": "array",
                        "description": "Regex patterns for finding similar terms",
                        "items": {
                            "type": "string",
                            "description": "A regex pattern"
                        }
                    },
                    "categories": {
                        "type": "object",
                        "description": "Keywords organized by category",
                        "properties": {
                            "medical_terms": {"type": "array", "items": {"type": "string"}},
                            "concepts": {"type": "array", "items": {"type": "string"}},
                            "entities": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "required": ["keywords"]
            }
            
            # Generate structured keywords
            result = await generate_json(
                prompt=prompt,
                system_instruction=self.get_system_instruction(),
                schema=schema,
                temperature=0.4,  # Medium creativity for keyword generation
                max_tokens=1000,
                strict=False
            )
            
            # Enhance with rule-based patterns
            result = self._enhance_with_patterns(result, document_content)
            
            return result
            
        except Exception as e:
            logger.warning(f"LLM keyword generation failed: {e}, using fallback")
            return self._fallback_keyword_extraction(document_content)

    def _enhance_with_patterns(self, llm_result: Dict[str, Any], document: str) -> Dict[str, Any]:
        """Enhance LLM results with rule-based patterns."""
        
        # Ensure we have the patterns field
        if "patterns" not in llm_result:
            llm_result["patterns"] = []
        
        # Add common medical patterns
        medical_patterns = [
            r'\b\d+\.?\d*\s*mg\b',  # Dosages like "5mg", "10.5 mg"
            r'\b\d+\.?\d*\s*ml\b',  # Volumes like "2ml", "1.5 ml"
            r'\b\d+\.?\d*\s*%\b',   # Percentages
            r'\b\d{2,3}/\d{2,3}\b', # Blood pressure like "120/80"
            r'\bT\d+\b',            # T scores like "T1", "T12"
            r'\bL\d+\b',            # Lumbar vertebrae like "L1", "L5"
            r'\bC\d+\b',            # Cervical vertebrae like "C1", "C7"
        ]
        
        # Add patterns that weren't already generated
        existing_patterns = set(llm_result["patterns"])
        for pattern in medical_patterns:
            if pattern not in existing_patterns:
                llm_result["patterns"].append(pattern)
        
        # Extract additional keywords using rule-based methods
        rule_keywords = self._extract_rule_based_keywords(document)
        
        # Merge keywords, avoiding duplicates
        existing_keywords = set([k.lower() for k in llm_result.get("keywords", [])])
        for keyword in rule_keywords:
            if keyword.lower() not in existing_keywords:
                llm_result.setdefault("keywords", []).append(keyword)
        
        return llm_result

    def _extract_rule_based_keywords(self, document: str) -> List[str]:
        """Extract keywords using rule-based patterns."""
        keywords = []
        
        # Common medical abbreviations
        medical_abbrevs = [
            r'\bBP\b', r'\bHR\b', r'\bRR\b', r'\bTemp\b',
            r'\bWBC\b', r'\bRBC\b', r'\bHgb\b', r'\bHct\b',
            r'\bECG\b', r'\bEKG\b', r'\bCT\b', r'\bMRI\b',
            r'\bXR\b', r'\bUS\b'
        ]
        
        for pattern in medical_abbrevs:
            matches = re.findall(pattern, document, re.IGNORECASE)
            keywords.extend([match.upper() for match in matches])
        
        # Medical units and measurements
        unit_patterns = [
            (r'(\d+\.?\d*)\s*(mg|ml|kg|lbs|cm|mm|bpm)', 'measurements'),
            (r'(\d+\.?\d*)\s*degrees?', 'temperature'),
            (r'(\d{2,3}/\d{2,3})', 'blood_pressure'),
        ]
        
        for pattern, category in unit_patterns:
            matches = re.findall(pattern, document, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    keywords.append(' '.join(match))
                else:
                    keywords.append(match)
        
        # Remove duplicates and clean
        keywords = list(set([k.strip() for k in keywords if k.strip()]))
        
        return keywords

    def _fallback_keyword_extraction(self, document_content: str) -> Dict[str, Any]:
        """Simple rule-based keyword extraction as fallback."""
        keywords = []
        patterns = []
        
        # Extract capitalized words (likely proper nouns/medical terms)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', document_content)
        keywords.extend(list(set(capitalized)))
        
        # Extract numbers with units (measurements)
        measurements = re.findall(r'\d+\.?\d*\s*(?:mg|ml|kg|lbs|cm|mm|bpm|%)', document_content, re.IGNORECASE)
        keywords.extend(measurements)
        
        # Extract common medical patterns
        medical_terms = re.findall(r'\b(?:diagnosis|treatment|symptoms?|condition|disease|disorder|syndrome|test|result|normal|abnormal|elevated|decreased|increased)\b', document_content, re.IGNORECASE)
        keywords.extend(list(set(medical_terms)))
        
        # Create basic patterns
        patterns = [
            r'\b\d+\.?\d*\s*(?:mg|ml|kg|lbs|cm|mm|bpm|%)\b',
            r'\b(?:diagnosis|treatment|symptoms?|condition)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        ]
        
        # Clean and deduplicate
        keywords = list(set([k.strip() for k in keywords if k.strip() and len(k) > 2]))
        
        # Limit to most relevant (by frequency or importance)
        keywords = keywords[:20]  # Top 20 keywords
        
        return {
            "keywords": keywords,
            "patterns": patterns,
            "categories": {
                "medical_terms": [k for k in keywords if k.lower() in document_content.lower()],
                "measurements": [k for k in keywords if re.search(r'\d', k)],
                "general": [k for k in keywords if not re.search(r'\d', k)]
            }
        }

    def _validate_keywords(self, keywords: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean keyword results."""
        # Ensure required fields exist
        if "keywords" not in keywords:
            keywords["keywords"] = []
        if "patterns" not in keywords:
            keywords["patterns"] = []
        
        # Ensure they are lists
        if not isinstance(keywords["keywords"], list):
            keywords["keywords"] = []
        if not isinstance(keywords["patterns"], list):
            keywords["patterns"] = []
        
        # Clean keywords
        cleaned_keywords = []
        for keyword in keywords["keywords"]:
            if isinstance(keyword, str) and keyword.strip() and len(keyword.strip()) > 1:
                cleaned_keywords.append(keyword.strip())
        
        keywords["keywords"] = cleaned_keywords[:30]  # Limit to 30 keywords
        
        # Validate patterns
        valid_patterns = []
        for pattern in keywords["patterns"]:
            if isinstance(pattern, str) and pattern.strip():
                try:
                    # Test if it's a valid regex
                    re.compile(pattern)
                    valid_patterns.append(pattern.strip())
                except re.error:
                    logger.warning(f"Invalid regex pattern skipped: {pattern}")
        
        keywords["patterns"] = valid_patterns
        
        return keywords


# --- Module-level app creation for Health Universe deployment ---
agent = KeywordGeneratorAgent()
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
    port = int(os.getenv("PORT", 8002))
    print(f"🚀 Starting {agent.get_agent_name()}")
    print(f"📍 Available at: http://localhost:{port}")
    print(f"🔍 Agent Card: http://localhost:{port}/.well-known/agentcard.json")
    uvicorn.run(app, host="0.0.0.0", port=port)