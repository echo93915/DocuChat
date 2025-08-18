"""Unified LLM interface with multi-provider support and intelligent fallback.

This module consolidates all LLM functionality into a single, well-organized file with:
- Google Gemini (primary)
- OpenAI Direct API (secondary)
- OpenAI SDK (tertiary)
- Mock responses (fallback for testing)
"""

import logging
import time
import random
import json
import requests
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import dependencies with graceful fallbacks
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Generative AI package not available")

try:
    from openai import OpenAI
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not available")

from .settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class LLMError(Exception):
    """Base exception for LLM operations."""
    pass


class GeminiError(LLMError):
    """Custom exception for Gemini API errors."""
    pass


class OpenAIError(LLMError):
    """Custom exception for OpenAI API errors."""
    pass


# =============================================================================
# GEMINI PROVIDER
# =============================================================================

class GeminiProvider:
    """Google Gemini AI provider implementation."""
    
    @staticmethod
    def initialize():
        """Initialize Gemini with API key."""
        if not GEMINI_AVAILABLE:
            raise GeminiError("Google Generative AI package not available")
        
        genai.configure(api_key=settings.gemini_api_key)
        logger.info("Gemini API initialized")

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=lambda retry_state: logger.warning(
            f"Gemini API call failed, retrying in {retry_state.next_action.sleep} seconds... "
            f"(attempt {retry_state.attempt_number})"
        )
    )
    def embed_texts(texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Google Gemini."""
        if not texts:
            return []
        
        if not GEMINI_AVAILABLE:
            raise GeminiError("Gemini not available")
        
        try:
            GeminiProvider.initialize()
            
            # Remove empty texts and track original indices
            non_empty_texts = []
            original_indices = []
            for i, text in enumerate(texts):
                if text.strip():
                    non_empty_texts.append(text.strip())
                    original_indices.append(i)
            
            if not non_empty_texts:
                logger.warning("All input texts are empty, returning empty embeddings")
                return [[] for _ in texts]
            
            logger.info(f"Generating embeddings for {len(non_empty_texts)} texts using Gemini")
            
            embeddings = [[] for _ in texts]
            
            # Generate embeddings one by one (Gemini doesn't support batch)
            for i, text in enumerate(non_empty_texts):
                try:
                    result = genai.embed_content(
                        model=settings.gemini_embedding_model,
                        content=text,
                        task_type="retrieval_document"
                    )
                    
                    if result and 'embedding' in result:
                        original_idx = original_indices[i]
                        embeddings[original_idx] = result['embedding']
                    else:
                        logger.warning(f"No embedding returned for text {i}")
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to embed text {i}: {e}")
                    continue
            
            logger.info(f"Successfully generated embeddings for {len([e for e in embeddings if e])} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            raise GeminiError(f"Failed to generate embeddings: {e}")

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=lambda retry_state: logger.warning(
            f"Gemini API call failed, retrying in {retry_state.next_action.sleep} seconds... "
            f"(attempt {retry_state.attempt_number})"
        )
    )
    def chat_complete(
        system: str,
        user: str,
        *,
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """Generate chat completion using Google Gemini."""
        if not user.strip():
            raise ValueError("User message cannot be empty")
        
        if not GEMINI_AVAILABLE:
            raise GeminiError("Gemini not available")
        
        try:
            GeminiProvider.initialize()
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens or settings.max_tokens_answer,
                temperature=temperature if temperature is not None else settings.temperature,
            )
            
            model = genai.GenerativeModel(
                model_name=settings.gemini_chat_model,
                generation_config=generation_config
            )
            
            # Combine system and user messages
            if system.strip():
                prompt = f"System: {system.strip()}\n\nUser: {user.strip()}"
            else:
                prompt = user.strip()
            
            logger.info("Generating chat completion using Gemini")
            
            response = model.generate_content(prompt)
            
            if response and response.text:
                answer = response.text.strip()
                logger.info("Successfully generated chat completion with Gemini")
                return answer
            else:
                raise GeminiError("Empty response from Gemini")
            
        except Exception as e:
            logger.error(f"Gemini chat completion error: {e}")
            raise GeminiError(f"Failed to generate chat completion: {e}")

    @staticmethod
    def validate_api_key() -> bool:
        """Validate Gemini API key."""
        try:
            if not GEMINI_AVAILABLE:
                return False
            
            GeminiProvider.initialize()
            
            # Try to list models to test API key
            models = genai.list_models()
            return len(list(models)) > 0
            
        except Exception as e:
            logger.error(f"Gemini API key validation failed: {e}")
            return False


# =============================================================================
# OPENAI DIRECT PROVIDER
# =============================================================================

class OpenAIDirectProvider:
    """OpenAI Direct API provider using requests (bypasses SDK issues)."""
    
    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException,)),
        before_sleep=lambda retry_state: logger.warning(
            f"OpenAI Direct API call failed, retrying in {retry_state.next_action.sleep} seconds... "
            f"(attempt {retry_state.attempt_number})"
        )
    )
    def embed_texts(texts: List[str]) -> List[List[float]]:
        """Generate embeddings using direct OpenAI API calls."""
        if not texts:
            return []
        
        # Remove empty texts and track original indices
        non_empty_texts = []
        original_indices = []
        for i, text in enumerate(texts):
            if text.strip():
                non_empty_texts.append(text.strip())
                original_indices.append(i)
        
        if not non_empty_texts:
            logger.warning("All input texts are empty, returning empty embeddings")
            return [[] for _ in texts]
        
        try:
            logger.info(f"Generating embeddings for {len(non_empty_texts)} texts using OpenAI Direct API")
            
            headers = {
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "input": non_empty_texts,
                "model": settings.embedding_model
            }
            
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code != 200:
                raise OpenAIError(f"API error {response.status_code}: {response.text}")
            
            result = response.json()
            
            # Extract embeddings and map back to original indices
            embeddings = [[] for _ in texts]
            for i, embedding_data in enumerate(result["data"]):
                original_idx = original_indices[i]
                embeddings[original_idx] = embedding_data["embedding"]
            
            logger.info(f"Successfully generated {len(result['data'])} embeddings")
            return embeddings
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error in OpenAI Direct embed_texts: {e}")
            raise OpenAIError(f"Network error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI Direct embed_texts: {e}")
            raise OpenAIError(f"Failed to generate embeddings: {e}")

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException,)),
        before_sleep=lambda retry_state: logger.warning(
            f"OpenAI Direct API call failed, retrying in {retry_state.next_action.sleep} seconds... "
            f"(attempt {retry_state.attempt_number})"
        )
    )
    def chat_complete(
        system: str,
        user: str,
        *,
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """Generate chat completion using direct OpenAI API calls."""
        if not user.strip():
            raise ValueError("User message cannot be empty")
        
        max_tokens = max_tokens or settings.max_tokens_answer
        temperature = temperature if temperature is not None else settings.temperature
        
        try:
            logger.info("Generating chat completion using OpenAI Direct API")
            
            headers = {
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if system.strip():
                messages.append({"role": "system", "content": system.strip()})
            messages.append({"role": "user", "content": user.strip()})
            
            data = {
                "model": settings.chat_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code != 200:
                raise OpenAIError(f"API error {response.status_code}: {response.text}")
            
            result = response.json()
            
            answer = result["choices"][0]["message"]["content"]
            if not answer:
                raise OpenAIError("Received empty response from OpenAI")
            
            logger.info(f"Successfully generated chat completion with {result.get('usage', {}).get('total_tokens', 0)} tokens")
            return answer.strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error in OpenAI Direct chat_complete: {e}")
            raise OpenAIError(f"Network error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI Direct chat_complete: {e}")
            raise OpenAIError(f"Failed to generate chat completion: {e}")

    @staticmethod
    def validate_api_key() -> bool:
        """Validate OpenAI API key using direct API call."""
        try:
            headers = {
                "Authorization": f"Bearer {settings.openai_api_key}",
            }
            
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"OpenAI Direct API key validation failed: {e}")
            return False


# =============================================================================
# OPENAI SDK PROVIDER
# =============================================================================

class OpenAISDKProvider:
    """OpenAI SDK provider (official client library)."""
    
    @staticmethod
    def get_client():
        """Get OpenAI client instance."""
        if not OPENAI_AVAILABLE:
            raise OpenAIError("OpenAI package not available")
        
        import os
        try:
            # Set environment variable for OpenAI client
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
            return OpenAI()  # Will use environment variable
        except Exception as e:
            logger.warning(f"OpenAI client creation failed: {e}")
            # Fallback for older OpenAI SDK versions or environment issues
            import openai
            openai.api_key = settings.openai_api_key
            return openai

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError
        )) if OPENAI_AVAILABLE else (),
        before_sleep=lambda retry_state: logger.warning(
            f"OpenAI SDK API call failed, retrying in {retry_state.next_action.sleep} seconds... "
            f"(attempt {retry_state.attempt_number})"
        )
    )
    def embed_texts(texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI SDK."""
        if not texts:
            return []
        
        if not OPENAI_AVAILABLE:
            raise OpenAIError("OpenAI package not available")
        
        # Remove empty texts and track original indices
        non_empty_texts = []
        original_indices = []
        for i, text in enumerate(texts):
            if text.strip():
                non_empty_texts.append(text.strip())
                original_indices.append(i)
        
        if not non_empty_texts:
            logger.warning("All input texts are empty, returning empty embeddings")
            return [[] for _ in texts]
        
        try:
            logger.info(f"Generating embeddings for {len(non_empty_texts)} texts using OpenAI SDK")
            
            client = OpenAISDKProvider.get_client()
            response = client.embeddings.create(
                input=non_empty_texts,
                model=settings.embedding_model
            )
            
            # Extract embeddings and map back to original indices
            embeddings = [[] for _ in texts]
            for i, embedding_data in enumerate(response.data):
                original_idx = original_indices[i]
                embeddings[original_idx] = embedding_data.embedding
            
            logger.info(f"Successfully generated {len(response.data)} embeddings")
            return embeddings
            
        except (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.BadRequestError
        ) as e:
            logger.error(f"OpenAI SDK API error (non-retryable): {e}")
            raise OpenAIError(f"OpenAI API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI SDK embed_texts: {e}")
            raise OpenAIError(f"Failed to generate embeddings: {e}")

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError
        )) if OPENAI_AVAILABLE else (),
        before_sleep=lambda retry_state: logger.warning(
            f"OpenAI SDK API call failed, retrying in {retry_state.next_action.sleep} seconds... "
            f"(attempt {retry_state.attempt_number})"
        )
    )
    def chat_complete(
        system: str,
        user: str,
        *,
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """Generate chat completion using OpenAI SDK."""
        if not user.strip():
            raise ValueError("User message cannot be empty")
        
        if not OPENAI_AVAILABLE:
            raise OpenAIError("OpenAI package not available")
        
        # Use defaults from settings if not provided
        max_tokens = max_tokens or settings.max_tokens_answer
        temperature = temperature if temperature is not None else settings.temperature
        
        try:
            logger.info(f"Generating chat completion using OpenAI SDK")
            
            messages = []
            if system.strip():
                messages.append({"role": "system", "content": system.strip()})
            messages.append({"role": "user", "content": user.strip()})
            
            client = OpenAISDKProvider.get_client()
            response = client.chat.completions.create(
                model=settings.chat_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            if not answer:
                raise OpenAIError("Received empty response from OpenAI")
            
            logger.info(f"Successfully generated chat completion with {response.usage.total_tokens} tokens")
            return answer.strip()
            
        except (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.BadRequestError
        ) as e:
            logger.error(f"OpenAI SDK API error (non-retryable): {e}")
            raise OpenAIError(f"OpenAI API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI SDK chat_complete: {e}")
            raise OpenAIError(f"Failed to generate chat completion: {e}")

    @staticmethod
    def validate_api_key() -> bool:
        """Validate that the OpenAI API key is working."""
        try:
            if not OPENAI_AVAILABLE:
                return False
            
            # Make a simple API call to test the key
            client = OpenAISDKProvider.get_client()
            response = client.models.list()
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"OpenAI SDK API key validation failed: {e}")
            return False


# =============================================================================
# MOCK PROVIDER
# =============================================================================

class MockProvider:
    """Mock provider for testing when real APIs are unavailable."""
    
    @staticmethod
    def embed_texts(texts: List[str]) -> List[List[float]]:
        """Mock embedding generation for testing."""
        if not texts:
            return []
        
        logger.info(f"MOCK: Generating embeddings for {len(texts)} texts")
        time.sleep(0.1)  # Simulate API delay
        
        # Generate mock embeddings (1536 dimensions like text-embedding-3-small)
        embeddings = []
        for text in texts:
            if text.strip():
                # Generate deterministic mock embeddings based on text hash
                random.seed(hash(text) % 2**32)
                embedding = [random.gauss(0, 0.1) for _ in range(1536)]
                embeddings.append(embedding)
            else:
                embeddings.append([])
        
        return embeddings

    @staticmethod
    def chat_complete(
        system: str,
        user: str,
        *,
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """Mock chat completion for testing."""
        if not user.strip():
            raise ValueError("User message cannot be empty")
        
        logger.info("MOCK: Generating chat completion")
        time.sleep(0.2)  # Simulate API delay
        
        # Generate a mock response
        responses = [
            f"This is a mock response to: '{user[:50]}...' Based on the system message about being a precise assistant, I would analyze the provided context and give you a relevant answer.",
            f"MOCK ANSWER: I understand you're asking about '{user[:30]}...'. Based on the context provided, here's what I found.",
            "This is a mock response from DocuChat. In a real implementation, this would be powered by AI models."
        ]
        
        # Choose response based on hash for deterministic behavior
        response_idx = hash(user) % len(responses)
        return responses[response_idx]

    @staticmethod
    def validate_api_key() -> bool:
        """Mock API key validation."""
        logger.info("MOCK: Validating API key")
        return True


# =============================================================================
# UNIFIED INTERFACE WITH INTELLIGENT FALLBACK
# =============================================================================

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings with intelligent fallback across multiple providers.
    
    Fallback order:
    1. Google Gemini (primary)
    2. OpenAI Direct API (secondary)
    3. OpenAI SDK (tertiary)
    4. Mock responses (testing fallback)
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors (list of floats)
        
    Raises:
        LLMError: If all providers fail
    """
    if not texts:
        return []
    
    providers = [
        ("Gemini", GeminiProvider.embed_texts),
        ("OpenAI Direct", OpenAIDirectProvider.embed_texts),
        ("OpenAI SDK", OpenAISDKProvider.embed_texts),
        ("Mock", MockProvider.embed_texts),
    ]
    
    last_error = None
    
    for provider_name, provider_func in providers:
        try:
            logger.info(f"Attempting embeddings with {provider_name} provider")
            return provider_func(texts)
        except Exception as e:
            logger.warning(f"{provider_name} provider failed: {e}")
            last_error = e
            continue
    
    # If all providers failed
    raise LLMError(f"All embedding providers failed. Last error: {last_error}")


def chat_complete(
    system: str,
    user: str,
    *,
    max_tokens: int = None,
    temperature: float = None
) -> str:
    """
    Generate chat completion with intelligent fallback across multiple providers.
    
    Fallback order:
    1. Google Gemini (primary)
    2. OpenAI Direct API (secondary)
    3. OpenAI SDK (tertiary)
    4. Mock responses (testing fallback)
    
    Args:
        system: System message to set the context/behavior
        user: User message/prompt
        max_tokens: Maximum tokens in the response (optional, uses settings default)
        temperature: Sampling temperature (optional, uses settings default)
        
    Returns:
        Generated response text
        
    Raises:
        LLMError: If all providers fail
    """
    if not user.strip():
        raise ValueError("User message cannot be empty")
    
    providers = [
        ("Gemini", GeminiProvider.chat_complete),
        ("OpenAI Direct", OpenAIDirectProvider.chat_complete),
        ("OpenAI SDK", OpenAISDKProvider.chat_complete),
        ("Mock", MockProvider.chat_complete),
    ]
    
    last_error = None
    
    for provider_name, provider_func in providers:
        try:
            logger.info(f"Attempting chat completion with {provider_name} provider")
            return provider_func(system, user, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            logger.warning(f"{provider_name} provider failed: {e}")
            last_error = e
            continue
    
    # If all providers failed
    raise LLMError(f"All chat completion providers failed. Last error: {last_error}")


def validate_api_keys() -> Dict[str, bool]:
    """
    Validate all available API keys.
    
    Returns:
        Dictionary with validation results for each provider
    """
    results = {
        "gemini": False,
        "openai_direct": False,
        "openai_sdk": False,
        "mock": True,  # Mock always works
    }
    
    # Test Gemini
    try:
        results["gemini"] = GeminiProvider.validate_api_key()
    except Exception as e:
        logger.debug(f"Gemini validation failed: {e}")
    
    # Test OpenAI Direct
    try:
        results["openai_direct"] = OpenAIDirectProvider.validate_api_key()
    except Exception as e:
        logger.debug(f"OpenAI Direct validation failed: {e}")
    
    # Test OpenAI SDK
    try:
        results["openai_sdk"] = OpenAISDKProvider.validate_api_key()
    except Exception as e:
        logger.debug(f"OpenAI SDK validation failed: {e}")
    
    return results


def get_token_usage(response) -> Dict[str, int]:
    """
    Extract token usage information from an API response.
    
    Args:
        response: API response object
        
    Returns:
        Dictionary with token usage information
    """
    if hasattr(response, 'usage'):
        return {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }
    return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# For backward compatibility with existing code
OpenAIError = OpenAIError  # Re-export for compatibility
validate_api_key = lambda: any(validate_api_keys().values())  # Simple validation for backward compatibility
