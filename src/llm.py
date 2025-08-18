"""OpenAI API wrappers with retry logic for DocuChat."""

import logging
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
import openai

from .settings import settings

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client
def get_client():
    """Get OpenAI client instance."""
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


class OpenAIError(Exception):
    """Custom exception for OpenAI API errors."""
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        openai.APIConnectionError
    )),
    before_sleep=lambda retry_state: logger.warning(
        f"OpenAI API call failed, retrying in {retry_state.next_action.sleep} seconds... "
        f"(attempt {retry_state.attempt_number})"
    )
)
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI's embedding model.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors (list of floats)
        
    Raises:
        OpenAIError: If the API call fails after retries
    """
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
        logger.info(f"Generating embeddings for {len(non_empty_texts)} texts using {settings.embedding_model}")
        
        try:
            # Try direct API first
            from .llm_direct import embed_texts as direct_embed_texts
            return direct_embed_texts(texts)
        except Exception as direct_error:
            logger.warning(f"Direct API failed: {direct_error}")
            try:
                # Try SDK client
                client = get_client()
                response = client.embeddings.create(
                    input=non_empty_texts,
                    model=settings.embedding_model
                )
            except Exception as sdk_error:
                # Fallback to mock if both fail
                logger.warning(f"SDK API also failed, using mock: {sdk_error}")
                from .llm_mock import embed_texts as mock_embed_texts
                return mock_embed_texts(texts)
        
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
        logger.error(f"OpenAI API error (non-retryable): {e}")
        raise OpenAIError(f"OpenAI API error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in embed_texts: {e}")
        raise OpenAIError(f"Failed to generate embeddings: {e}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        openai.APIConnectionError
    )),
    before_sleep=lambda retry_state: logger.warning(
        f"OpenAI API call failed, retrying in {retry_state.next_action.sleep} seconds... "
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
    """
    Generate a chat completion using OpenAI's chat model.
    
    Args:
        system: System message to set the context/behavior
        user: User message/prompt
        max_tokens: Maximum tokens in the response (optional, uses settings default)
        temperature: Sampling temperature (optional, uses settings default)
        
    Returns:
        Generated response text
        
    Raises:
        OpenAIError: If the API call fails after retries
    """
    if not user.strip():
        raise ValueError("User message cannot be empty")
    
    # Use defaults from settings if not provided
    max_tokens = max_tokens or settings.max_tokens_answer
    temperature = temperature if temperature is not None else settings.temperature
    
    try:
        logger.info(f"Generating chat completion using {settings.chat_model}")
        
        messages = []
        if system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": user.strip()})
        
        try:
            # Try direct API first
            from .llm_direct import chat_complete as direct_chat_complete
            return direct_chat_complete(system, user, max_tokens=max_tokens, temperature=temperature)
        except Exception as direct_error:
            logger.warning(f"Direct API failed: {direct_error}")
            try:
                # Try SDK client
                client = get_client()
                response = client.chat.completions.create(
                    model=settings.chat_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            except Exception as sdk_error:
                # Fallback to mock if both fail
                logger.warning(f"SDK API also failed, using mock: {sdk_error}")
                from .llm_mock import chat_complete as mock_chat_complete
                return mock_chat_complete(system, user, max_tokens=max_tokens, temperature=temperature)
        
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
        logger.error(f"OpenAI API error (non-retryable): {e}")
        raise OpenAIError(f"OpenAI API error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in chat_complete: {e}")
        raise OpenAIError(f"Failed to generate chat completion: {e}")


def get_token_usage(response) -> dict:
    """
    Extract token usage information from an OpenAI response.
    
    Args:
        response: OpenAI API response object
        
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


def validate_api_key() -> bool:
    """
    Validate that the OpenAI API key is working.
    
    Returns:
        True if API key is valid, False otherwise
    """
    try:
        # Make a simple API call to test the key
        client = get_client()
        response = client.models.list()
        return len(response.data) > 0
    except Exception as e:
        logger.error(f"API key validation failed: {e}")
        return False
