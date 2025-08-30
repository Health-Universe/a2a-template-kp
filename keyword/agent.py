"""
Keyword Pattern Generator Agent
LLM-powered agent that analyzes medical document previews and generates
regex patterns for identifying key information.
"""

import json
import re
from typing import List, Dict, Any, Optional, Union

from a2a.types import AgentSkill
from base import A2AAgent
from utils.logging import get_logger
from utils.llm_utils import generate_json

logger = get_logger(__name__)


class KeywordAgent(A2AAgent):
    """
    LLM-powered keyword extraction agent that generates regex patterns from documents.
    Uses generate_json() for structured pattern generation.
    """

    # --- A2A Metadata ---
    def get_agent_name(self) -> str:
        return "CS Pipeline - Keyword Generator"

    def get_agent_description(self) -> str:
        return (
            "LLM-powered agent that analyzes medical document previews and generates "
            "regex patterns for identifying key information such as diagnoses, medications, "
            "procedures, and temporal markers."
        )

    def get_agent_version(self) -> str:
        return "2.0.0"  # Template-based version

    def get_agent_skills(self) -> List[AgentSkill]:
        return [
            AgentSkill(
                id="generate_patterns",
                name="Generate Document Patterns",
                description="Analyze document and generate regex patterns using LLM",
                tags=["keyword", "pattern", "extraction", "llm", "medical"],
                inputModes=["text/plain", "application/json"],
                outputModes=["application/json"],
            )
        ]

    def supports_streaming(self) -> bool:
        return False

    def get_system_instruction(self) -> str:
        return (
            "You are a medical timeline extractor specializing in creating ripgrep-compatible "
            "regex patterns. Your primary goal is to build patient timelines by finding ALL "
            "dates and their associated medical events. Focus on: 1) Every date format "
            "(MM/DD/YYYY, YYYY-MM-DD, Month DD YYYY, etc.), 2) Medical events (admissions, "
            "diagnoses, procedures, medication changes), 3) Temporal relationships (before, "
            "after, during, since). Generate patterns that capture the WHEN of medical events. "
            "Use (?i) for case-insensitive matching. Prioritize temporal patterns above all else."
        )

    # --- Core Processing ---
    async def process_message(self, message: str) -> Union[Dict[str, Any], str]:
        """
        Generate regex patterns from document preview.
        Input can be plain text or JSON with document_preview and focus_areas.
        Returns dict with categorized patterns (will be wrapped in DataPart by framework).
        """
        try:
            # Parse input
            data = self._parse_input(message)
            preview = data.get("document_preview", message[:4000])
            focus_areas = data.get("focus_areas", [])
            
            # Generate patterns using LLM
            patterns = await self._generate_patterns(preview, focus_areas)
            
            # Ensure we have fallback patterns if needed
            patterns = self._ensure_minimum_patterns(patterns)
            
            # ALWAYS add diagnostic info
            patterns["diagnostic_info"] = {
                "api_keys_detected": self._check_api_keys(),
                "provider_info": self._get_provider_info(),
                "pattern_count": len(patterns.get("patterns", [])),
                "source": patterns.get("source", "unknown")
            }
            
            # Return as dict (will become DataPart)
            return patterns
            
        except Exception as e:
            logger.error(f"Error generating patterns: {e}")
            # Return fallback patterns with error info
            fallback = self._get_fallback_patterns_json()
            fallback["diagnostic_info"] = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "api_keys_detected": self._check_api_keys(),
                "provider_info": self._get_provider_info(),
                "source": "fallback_due_to_error"
            }
            return fallback

    async def _generate_patterns(self, preview: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Generate patterns using structured LLM output."""
        
        # Build focused prompt
        prompt = self._build_pattern_prompt(preview, focus_areas)
        
        # Define schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "section_patterns": {
                    "type": "array",
                    "description": "Patterns for document section headers",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Regex pattern"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            "description": {"type": "string", "description": "What this pattern matches"}
                        },
                        "required": ["pattern", "priority", "description"]
                    }
                },
                "clinical_patterns": {
                    "type": "array",
                    "description": "Patterns for clinical findings and diagnoses",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            "description": {"type": "string"}
                        },
                        "required": ["pattern", "priority", "description"]
                    }
                },
                "medication_patterns": {
                    "type": "array",
                    "description": "Patterns for medications and dosages",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            "description": {"type": "string"}
                        },
                        "required": ["pattern", "priority", "description"]
                    }
                },
                "temporal_patterns": {
                    "type": "array",
                    "description": "Patterns for dates and time references",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            "description": {"type": "string"}
                        },
                        "required": ["pattern", "priority", "description"]
                    }
                },
                "vital_patterns": {
                    "type": "array",
                    "description": "Patterns for vital signs and measurements",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            "description": {"type": "string"}
                        },
                        "required": ["pattern", "priority", "description"]
                    }
                }
            },
            "required": ["section_patterns", "clinical_patterns", "medication_patterns"]
        }
        
        try:
            # Generate patterns with structured output
            result = await generate_json(
                prompt=prompt,
                system_instruction=self.get_system_instruction(),
                schema=schema,
                temperature=0.3,  # Low temperature for consistency
                max_tokens=2000,
                strict=False  # Be flexible while patterns stabilize
            )
            
            # Validate and clean patterns
            result = self._validate_patterns(result)
            
            # Add source tracking
            result["source"] = "llm"
            result["status"] = "success"
            
            # Also create a flat patterns list for easier consumption
            flat_patterns = []
            for category in result:
                if category != "source" and isinstance(result[category], list):
                    for p in result[category]:
                        if isinstance(p, dict) and "pattern" in p:
                            flat_patterns.append(p["pattern"])
            result["patterns"] = flat_patterns
            
            return result
            
        except Exception as e:
            logger.warning(f"LLM pattern generation failed: {e}, using enhanced fallbacks")
            fallback = self._get_fallback_patterns_json()
            fallback["source"] = "fallback"
            fallback["status"] = "llm_failed"
            fallback["llm_error"] = {
                "error_type": type(e).__name__,
                "error_message": str(e)[:500],  # Limit error message length
                "provider_attempted": self._get_provider_info().get("provider", "unknown")
            }
            
            # Create flat patterns list
            flat_patterns = []
            for category in fallback:
                if category not in ["source", "status", "llm_error"] and isinstance(fallback[category], list):
                    for p in fallback[category]:
                        if isinstance(p, dict) and "pattern" in p:
                            flat_patterns.append(p["pattern"])
            fallback["patterns"] = flat_patterns
            
            return fallback

    def _build_pattern_prompt(self, preview: str, focus_areas: List[str]) -> str:
        """Build prompt for timeline-focused pattern generation."""
        prompt = f"""Generate ripgrep-compatible regex patterns to BUILD A PATIENT TIMELINE from the FULL medical document.
You are seeing only a preview of the first {len(preview)} characters.

DOCUMENT PREVIEW:
{preview}

PRIMARY GOAL: Extract EVERY date and its associated medical events to build a complete patient timeline.

REQUIREMENTS:
1. Find ALL dates in ANY format (MM/DD/YYYY, YYYY-MM-DD, Month DD YYYY, etc.)
2. Find events that happen AT dates (admitted on, diagnosed on, surgery on)
3. Find temporal relationships (since 2020, before surgery, after discharge)
4. Use (?i) for case-insensitive matching
5. Capture medication changes with dates (started, stopped, changed dose)
6. Find procedure dates and appointment dates
7. Prioritize finding WHEN things happened over WHAT happened

"""
        
        if focus_areas:
            prompt += f"TIMELINE FOCUS: {', '.join(focus_areas)}\n"
            prompt += "Extract patterns that show WHEN these events occurred.\n\n"
        
        prompt += """PATTERN CATEGORIES NEEDED (in priority order):
- Temporal patterns (ALL date formats, years, months, relative times)
- Event patterns (admitted, discharged, diagnosed, underwent, started, stopped)
- Medication patterns (focus on changes: started, discontinued, dose changed)
- Clinical patterns (with temporal context: diagnosed with X in/on DATE)
- Vital patterns (timestamped vital signs)

CRITICAL: Prioritize temporal patterns above everything else. We need dates!

Return ONLY valid JSON matching the schema. No additional text."""
        
        return prompt

    def _parse_input(self, message: str) -> Dict[str, Any]:
        """Parse input message which may be JSON or plain text."""
        try:
            data = json.loads(message)
            if isinstance(data, dict):
                return data
            return {"document_preview": message}
        except:
            return {"document_preview": message}

    def _validate_patterns(self, patterns: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
        """Validate that patterns are valid regex."""
        validated = {}
        
        for category, pattern_list in patterns.items():
            validated[category] = []
            for pattern_obj in pattern_list:
                if isinstance(pattern_obj, dict) and "pattern" in pattern_obj:
                    try:
                        # Test compile the pattern
                        re.compile(pattern_obj["pattern"])
                        validated[category].append(pattern_obj)
                    except re.error as e:
                        logger.debug(f"Invalid pattern '{pattern_obj['pattern']}': {e}")
        
        return validated

    def _ensure_minimum_patterns(self, patterns: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
        """Ensure we have at least some patterns in each category."""
        # Count total patterns
        total = sum(len(p) for p in patterns.values())
        
        if total < 5:
            # Merge with fallback patterns
            fallbacks = self._get_fallback_patterns_json()
            for category in fallbacks:
                if category not in patterns:
                    patterns[category] = fallbacks[category]
                elif len(patterns[category]) == 0:
                    patterns[category] = fallbacks[category]
        
        return patterns

    def _get_fallback_patterns_json(self) -> Dict[str, List[Dict[str, str]]]:
        """Get fallback patterns optimized for timeline extraction."""
        return {
            # PRIORITY 1: Temporal patterns for timeline building
            "temporal_patterns": [
                {"pattern": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "priority": "high", "description": "Date MM/DD/YYYY"},
                {"pattern": r"\b\d{4}-\d{2}-\d{2}\b", "priority": "high", "description": "Date YYYY-MM-DD"},
                {"pattern": r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b", "priority": "high", "description": "Date with various separators"},
                {"pattern": r"(?i)(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}", "priority": "high", "description": "Date Month DD, YYYY"},
                {"pattern": r"(?i)\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b", "priority": "high", "description": "Full month names"},
                {"pattern": r"\b(?:19|20)\d{2}\b", "priority": "high", "description": "Year YYYY"},
                {"pattern": r"(?i)in\s+(?:19|20)\d{2}", "priority": "high", "description": "In year"},
                {"pattern": r"(?i)(?:yesterday|today|tomorrow|last\s+(?:week|month|year))", "priority": "medium", "description": "Relative time"},
                {"pattern": r"\b\d+\s*(?:day|week|month|year)s?\s+ago\b", "priority": "high", "description": "Time duration ago"},
                {"pattern": r"(?i)(?:on|at|during|since|until|before|after|following)\s+", "priority": "high", "description": "Temporal prepositions"},
            ],
            # PRIORITY 2: Medical events that happen at specific times
            "event_patterns": [
                {"pattern": r"(?i)(?:admitted|admission|discharged|discharge)", "priority": "high", "description": "Admission/discharge events"},
                {"pattern": r"(?i)(?:diagnosed|diagnosis\s+of|diagnosed\s+with)", "priority": "high", "description": "Diagnosis events"},
                {"pattern": r"(?i)(?:underwent|performed|completed|received|had)", "priority": "high", "description": "Procedure verbs"},
                {"pattern": r"(?i)(?:started|initiated|begun|commenced)", "priority": "high", "description": "Treatment start"},
                {"pattern": r"(?i)(?:stopped|discontinued|ended|completed)", "priority": "high", "description": "Treatment end"},
                {"pattern": r"(?i)(?:presented|arrived|came\s+in|brought\s+in)", "priority": "high", "description": "Presentation events"},
                {"pattern": r"(?i)(?:surgery|procedure|operation|biopsy|scan|imaging)", "priority": "high", "description": "Procedures"},
                {"pattern": r"(?i)(?:emergency|urgent|routine|scheduled|elective)", "priority": "medium", "description": "Event urgency"},
            ],
            # PRIORITY 3: Generic medication patterns (no specific drug names)
            "medication_patterns": [
                {"pattern": r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|μg|g|ml|cc|units?|iu|tablets?|pills?|caps?)\b", "priority": "high", "description": "Medication dosage with units"},
                {"pattern": r"(?i)(?:once|twice|three\s+times|four\s+times)\s+(?:a\s+)?(?:day|daily|week|weekly|month|monthly)", "priority": "high", "description": "Frequency phrases"},
                {"pattern": r"(?i)(?:q\.?\d+h|every\s+\d+\s+hours?)", "priority": "high", "description": "Hourly frequency"},
                {"pattern": r"(?i)(?:q\.?d\.?|b\.?i\.?d\.?|t\.?i\.?d\.?|q\.?i\.?d\.?|prn)", "priority": "high", "description": "Medical frequency abbreviations"},
                {"pattern": r"(?i)(?:daily|weekly|monthly|as\s+needed)", "priority": "high", "description": "Common frequencies"},
                {"pattern": r"(?i)\b[A-Z][a-z]+(?:in|ol|ide|ate|ine|one|pril|artan|statin|zole|cycline|cillin|mycin|azole|pam|pine)\b", "priority": "medium", "description": "Drug name patterns by suffix"},
                {"pattern": r"(?i)(?:medication|med|rx|drug|prescription)s?\s*[:]\s*", "priority": "medium", "description": "Medication section marker"},
                {"pattern": r"(?i)(?:started|changed|increased|decreased|switched\s+to|adjusted)", "priority": "high", "description": "Medication changes"},
            ],
            # Clinical findings (de-prioritized but still useful)
            "clinical_patterns": [
                {"pattern": r"(?i)(?:chief\s+complaint|cc|presenting\s+complaint|reason\s+for\s+visit)", "priority": "high", "description": "Chief complaint"},
                {"pattern": r"(?i)(?:history\s+of|h/o|hx\s+of|past\s+medical)", "priority": "medium", "description": "History markers"},
                {"pattern": r"(?i)(?:diagnosis|diagnoses|dx|impression)[:]\s*", "priority": "high", "description": "Diagnosis section"},
                {"pattern": r"(?i)(?:no\s+evidence\s+of|negative\s+for|denied|denies)", "priority": "medium", "description": "Negative findings"},
            ],
            # Vital signs (often timestamped)
            "vital_patterns": [
                {"pattern": r"(?i)(?:blood\s+pressure|bp)[\s:]+\d{2,3}/\d{2,3}", "priority": "high", "description": "Blood pressure"},
                {"pattern": r"(?i)(?:heart\s+rate|hr|pulse)[\s:]+\d{2,3}", "priority": "high", "description": "Heart rate"},
                {"pattern": r"(?i)(?:temperature|temp)[\s:]+\d{2,3}(?:\.\d)?", "priority": "high", "description": "Temperature"},
                {"pattern": r"(?i)(?:o2\s+sat|oxygen\s+saturation|spo2)[\s:]+\d{2,3}%?", "priority": "high", "description": "Oxygen saturation"},
            ]
        }
    
    def _check_api_keys(self) -> Dict[str, bool]:
        """Check which API keys are present (without exposing values)"""
        import os
        return {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "google": bool(os.getenv("GOOGLE_API_KEY")),
            "gemini": bool(os.getenv("GEMINI_API_KEY"))
        }
    
    def _get_provider_info(self) -> Dict[str, str]:
        """Get info about which provider would be used"""
        import os
        if os.getenv("ANTHROPIC_API_KEY"):
            return {"provider": "anthropic", "model": "claude-3-5-sonnet"}
        elif os.getenv("OPENAI_API_KEY"):
            return {"provider": "openai", "model": "gpt-4o-mini"}
        elif os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            return {"provider": "google", "model": "gemini-2.0-flash-exp"}
        else:
            return {"provider": "none", "model": "none"}