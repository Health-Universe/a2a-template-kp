"""
LLM Utilities for A2A Agents
Multi-provider LLM integration with Health Universe compatibility.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


async def generate_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    model: Optional[str] = None
) -> Optional[str]:
    """
    Generate text using available LLM provider.
    
    Args:
        prompt: Input prompt
        system_instruction: Optional system instruction
        temperature: Generation temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        model: Specific model to use (auto-detected if None)
        
    Returns:
        Generated text or None if failed
    """
    try:
        provider = _detect_llm_provider()
        
        if provider == "google":
            return await _generate_google(prompt, system_instruction, temperature, max_tokens, model)
        elif provider == "openai":
            return await _generate_openai(prompt, system_instruction, temperature, max_tokens, model)
        elif provider == "anthropic":
            return await _generate_anthropic(prompt, system_instruction, temperature, max_tokens, model)
        else:
            logger.error("No LLM provider available")
            return None
            
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return None


async def generate_json(
    prompt: str,
    schema: Dict[str, Any],
    system_instruction: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1000,
    strict: bool = False,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate structured JSON using available LLM provider.
    
    Args:
        prompt: Input prompt
        schema: JSON schema for output validation
        system_instruction: Optional system instruction
        temperature: Generation temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        strict: Whether to enforce strict schema compliance
        model: Specific model to use (auto-detected if None)
        
    Returns:
        Generated structured data as dictionary
    """
    try:
        provider = _detect_llm_provider()
        
        if provider == "google":
            return await _generate_google_json(prompt, schema, system_instruction, temperature, max_tokens, strict, model)
        elif provider == "openai":
            return await _generate_openai_json(prompt, schema, system_instruction, temperature, max_tokens, strict, model)
        elif provider == "anthropic":
            return await _generate_anthropic_json(prompt, schema, system_instruction, temperature, max_tokens, strict, model)
        else:
            logger.error("No LLM provider available")
            return {"error": "No LLM provider available"}
            
    except Exception as e:
        logger.error(f"LLM JSON generation failed: {e}")
        return {"error": str(e)}


def _detect_llm_provider() -> Optional[str]:
    """Detect available LLM provider based on environment variables."""
    if os.getenv("GOOGLE_API_KEY"):
        return "google"
    elif os.getenv("OPENAI_API_KEY"):
        return "openai"
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    else:
        return None


async def _generate_google(
    prompt: str,
    system_instruction: Optional[str],
    temperature: float,
    max_tokens: int,
    model: Optional[str]
) -> Optional[str]:
    """Generate text using Google Gemini."""
    try:
        import google.generativeai as genai
        
        # Configure API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
            
        genai.configure(api_key=api_key)
        
        # Select model
        model_name = model or "gemini-1.5-flash"
        
        # Configure generation
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        # Create model instance
        if system_instruction:
            model_instance = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction=system_instruction
            )
        else:
            model_instance = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config
            )
        
        # Generate content
        response = await model_instance.generate_content_async(prompt)
        
        if response and response.text:
            return response.text
        else:
            return None
            
    except ImportError:
        logger.warning("Google Generative AI not available")
        return None
    except Exception as e:
        logger.error(f"Google generation failed: {e}")
        return None


async def _generate_google_json(
    prompt: str,
    schema: Dict[str, Any],
    system_instruction: Optional[str],
    temperature: float,
    max_tokens: int,
    strict: bool,
    model: Optional[str]
) -> Dict[str, Any]:
    """Generate structured JSON using Google Gemini."""
    try:
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig
        
        # Configure API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {"error": "Google API key not available"}
            
        genai.configure(api_key=api_key)
        
        # Select model
        model_name = model or "gemini-1.5-flash"
        
        # Enhanced prompt for JSON generation
        json_prompt = f"""{prompt}

Please respond with valid JSON that follows this schema:
{json.dumps(schema, indent=2)}

IMPORTANT: Your response must be valid JSON only, no other text."""
        
        if system_instruction:
            json_prompt = f"{system_instruction}\n\n{json_prompt}"
        
        # Configure generation
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if strict else None,
            response_schema=schema if strict else None
        )
        
        # Create model instance
        model_instance = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )
        
        # Generate content
        response = await model_instance.generate_content_async(json_prompt)
        
        if response and response.text:
            try:
                return json.loads(response.text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                return {"error": "Invalid JSON response", "raw_response": response.text}
        else:
            return {"error": "No response generated"}
            
    except ImportError:
        logger.warning("Google Generative AI not available")
        return {"error": "Google Generative AI not available"}
    except Exception as e:
        logger.error(f"Google JSON generation failed: {e}")
        return {"error": str(e)}


async def _generate_openai(
    prompt: str,
    system_instruction: Optional[str],
    temperature: float,
    max_tokens: int,
    model: Optional[str]
) -> Optional[str]:
    """Generate text using OpenAI."""
    try:
        import openai
        
        # Configure API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
            
        client = openai.AsyncOpenAI(api_key=api_key)
        
        # Select model
        model_name = model or "gpt-3.5-turbo"
        
        # Prepare messages
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        # Generate content
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            return None
            
    except ImportError:
        logger.warning("OpenAI not available")
        return None
    except Exception as e:
        logger.error(f"OpenAI generation failed: {e}")
        return None


async def _generate_openai_json(
    prompt: str,
    schema: Dict[str, Any],
    system_instruction: Optional[str],
    temperature: float,
    max_tokens: int,
    strict: bool,
    model: Optional[str]
) -> Dict[str, Any]:
    """Generate structured JSON using OpenAI."""
    try:
        import openai
        
        # Configure API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "OpenAI API key not available"}
            
        client = openai.AsyncOpenAI(api_key=api_key)
        
        # Select model
        model_name = model or "gpt-3.5-turbo"
        
        # Enhanced prompt for JSON generation
        json_prompt = f"""{prompt}

Please respond with valid JSON that follows this schema:
{json.dumps(schema, indent=2)}

IMPORTANT: Your response must be valid JSON only, no other text."""
        
        # Prepare messages
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": json_prompt})
        
        # Configure response format
        response_format = {"type": "json_object"} if strict else None
        
        # Generate content
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
        
        if response.choices and response.choices[0].message.content:
            try:
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                return {"error": "Invalid JSON response", "raw_response": response.choices[0].message.content}
        else:
            return {"error": "No response generated"}
            
    except ImportError:
        logger.warning("OpenAI not available")
        return {"error": "OpenAI not available"}
    except Exception as e:
        logger.error(f"OpenAI JSON generation failed: {e}")
        return {"error": str(e)}


async def _generate_anthropic(
    prompt: str,
    system_instruction: Optional[str],
    temperature: float,
    max_tokens: int,
    model: Optional[str]
) -> Optional[str]:
    """Generate text using Anthropic Claude."""
    try:
        import anthropic
        
        # Configure API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
            
        client = anthropic.AsyncAnthropic(api_key=api_key)
        
        # Select model
        model_name = model or "claude-3-haiku-20240307"
        
        # Generate content
        response = await client.messages.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            system=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if response.content and response.content[0].text:
            return response.content[0].text
        else:
            return None
            
    except ImportError:
        logger.warning("Anthropic not available")
        return None
    except Exception as e:
        logger.error(f"Anthropic generation failed: {e}")
        return None


async def _generate_anthropic_json(
    prompt: str,
    schema: Dict[str, Any],
    system_instruction: Optional[str],
    temperature: float,
    max_tokens: int,
    strict: bool,
    model: Optional[str]
) -> Dict[str, Any]:
    """Generate structured JSON using Anthropic Claude."""
    try:
        import anthropic
        
        # Configure API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return {"error": "Anthropic API key not available"}
            
        client = anthropic.AsyncAnthropic(api_key=api_key)
        
        # Select model
        model_name = model or "claude-3-haiku-20240307"
        
        # Enhanced prompt for JSON generation
        json_prompt = f"""{prompt}

Please respond with valid JSON that follows this schema:
{json.dumps(schema, indent=2)}

IMPORTANT: Your response must be valid JSON only, no other text."""
        
        # Generate content
        response = await client.messages.create(
            model=model_name,
            messages=[{"role": "user", "content": json_prompt}],
            system=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if response.content and response.content[0].text:
            try:
                return json.loads(response.content[0].text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                return {"error": "Invalid JSON response", "raw_response": response.content[0].text}
        else:
            return {"error": "No response generated"}
            
    except ImportError:
        logger.warning("Anthropic not available")
        return {"error": "Anthropic not available"}
    except Exception as e:
        logger.error(f"Anthropic JSON generation failed: {e}")
        return {"error": str(e)}


def get_available_providers() -> List[str]:
    """Get list of available LLM providers."""
    providers = []
    
    if os.getenv("GOOGLE_API_KEY"):
        providers.append("google")
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    
    return providers


async def test_provider(provider: str) -> bool:
    """Test if a specific provider is working."""
    try:
        if provider == "google":
            result = await _generate_google("Say 'OK'", None, 0.1, 10, None)
        elif provider == "openai":
            result = await _generate_openai("Say 'OK'", None, 0.1, 10, None)
        elif provider == "anthropic":
            result = await _generate_anthropic("Say 'OK'", None, 0.1, 10, None)
        else:
            return False
        
        return result is not None and "ok" in result.lower()
        
    except Exception as e:
        logger.error(f"Provider {provider} test failed: {e}")
        return False


# Backward compatibility classes and functions
class LLMProvider:
    """Simple LLM provider wrapper for backward compatibility."""
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
    
    async def generate_text(self, prompt: str, system: Optional[str] = None, **kwargs) -> Optional[str]:
        """Generate text using this provider."""
        return await generate_text(
            prompt=prompt,
            system_instruction=system,
            model=kwargs.get('model'),
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 1000)
        )
    
    async def generate_json(self, prompt: str, schema: Dict[str, Any], system: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate JSON using this provider."""
        return await generate_json(
            prompt=prompt,
            schema=schema,
            system_instruction=system,
            model=kwargs.get('model'),
            temperature=kwargs.get('temperature', 0.3),
            max_tokens=kwargs.get('max_tokens', 1000),
            strict=kwargs.get('strict', False)
        )


def create_llm_agent(provider: Optional[str] = None) -> LLMProvider:
    """Create an LLM agent instance."""
    if provider is None:
        provider = _detect_llm_provider()
    
    if provider is None:
        raise ValueError("No LLM provider available. Please set GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")
    
    return LLMProvider(provider)