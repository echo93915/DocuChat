"""Direct OpenAI API calls using requests to bypass client issues."""

import logging
import requests
import json
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .settings import settings

logger = logging.getLogger(__name__)


class OpenAIError(Exception):
    """Custom exception for OpenAI API errors."""
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    before_sleep=lambda retry_state: logger.warning(
        f"OpenAI API call failed, retrying in {retry_state.next_action.sleep} seconds... "
        f"(attempt {retry_state.attempt_number})"
    )
)
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using direct API calls.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
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
        logger.info(f"Generating embeddings for {len(non_empty_texts)} texts using direct API")
        
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
        logger.error(f"Network error in embed_texts: {e}")
        raise OpenAIError(f"Network error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in embed_texts: {e}")
        raise OpenAIError(f"Failed to generate embeddings: {e}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
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
    Generate chat completion using direct API calls.
    
    Args:
        system: System message
        user: User message
        max_tokens: Max tokens (optional)
        temperature: Temperature (optional)
        
    Returns:
        Generated response text
    """
    if not user.strip():
        raise ValueError("User message cannot be empty")
    
    max_tokens = max_tokens or settings.max_tokens_answer
    temperature = temperature if temperature is not None else settings.temperature
    
    try:
        logger.info("Generating chat completion using direct API")
        
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
        logger.error(f"Network error in chat_complete: {e}")
        raise OpenAIError(f"Network error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in chat_complete: {e}")
        raise OpenAIError(f"Failed to generate chat completion: {e}")


def validate_api_key() -> bool:
    """
    Validate OpenAI API key using direct API call.
    
    Returns:
        True if API key is valid
    """
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
        logger.error(f"API key validation failed: {e}")
        return False
