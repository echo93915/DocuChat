"""Mock OpenAI API wrappers for testing when the real API is unavailable."""

import logging
import time
from typing import List
import random

from .settings import settings

logger = logging.getLogger(__name__)


class OpenAIError(Exception):
    """Custom exception for OpenAI API errors."""
    pass


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Mock embedding generation for testing.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of mock embedding vectors (list of floats)
    """
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


def chat_complete(
    system: str,
    user: str,
    *,
    max_tokens: int = None,
    temperature: float = None
) -> str:
    """
    Mock chat completion for testing.
    
    Args:
        system: System message to set the context/behavior
        user: User message/prompt
        max_tokens: Maximum tokens in the response (optional)
        temperature: Sampling temperature (optional)
        
    Returns:
        Mock generated response text
    """
    if not user.strip():
        raise ValueError("User message cannot be empty")
    
    logger.info("MOCK: Generating chat completion")
    time.sleep(0.2)  # Simulate API delay
    
    # Generate a mock response
    responses = [
        f"This is a mock response to: '{user[:50]}...' Based on the system message about being a precise assistant, I would analyze the provided context and give you a relevant answer.",
        f"MOCK ANSWER: I understand you're asking about '{user[:30]}...'. Based on the context provided, here's what I found.",
        "This is a mock response from DocuChat. In a real implementation, this would be powered by OpenAI's GPT model."
    ]
    
    # Choose response based on hash for deterministic behavior
    response_idx = hash(user) % len(responses)
    return responses[response_idx]


def validate_api_key() -> bool:
    """
    Mock API key validation.
    
    Returns:
        True (mock success)
    """
    logger.info("MOCK: Validating API key")
    return True


def get_token_usage(response) -> dict:
    """
    Mock token usage information.
    
    Returns:
        Mock dictionary with token usage information
    """
    return {
        'prompt_tokens': 50,
        'completion_tokens': 100,
        'total_tokens': 150
    }
